"""FastAPI 백엔드 서버.

엔드포인트:
  GET  /health      서버/인덱스 상태
  GET  /config      현재 설정 + 선택 가능한 옵션
  POST /config      설정 변경 적용(필요 시 재인덱싱, 설정별 FAISS 캐시)
  GET  /documents   인덱싱된 RFP 문서 목록(메타데이터)
  POST /chat        QA: 질문 -> 답변 + 근거 출처 + 거절 여부
  POST /summarize   요약: RFP를 예산/기관/마감일/핵심요구사항 양식으로 정리

설정 변경 시 비용 구분:
  - 즉시 적용(재인덱싱 불필요): llm/temperature/base_url, retriever.k, use_bm25
  - 재인덱싱 필요: embedding(provider/model), chunk(strategy/size/overlap), parser
설정 시그니처(cache_key)가 같으면 기존 FAISS 캐시를 재사용한다.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from langchain_core.documents import Document
from pydantic import BaseModel

from src import catalog
from src.agents.metadata_agent import extract_metadata
from src.chunking import split_documents
from src.config import load_config, make_embedding, make_llm
from src.embedding import (
    add_to_vectorstore,
    build_vectorstore,
    cache_exists,
    make_cache_key,
)
from src.evaluation.evaluate import REFUSAL_MARK
from src.graph import build_graph
from src.loader import load_documents, parse_file
from src.retriever import (
    build_meta_filtered_retriever,
    build_org_filtered_retriever,
    build_retriever,
    known_orgs,
    match_orgs,
)

app = FastAPI(title="입찰메이트 RFP RAG API")

# 서버 기동 시 1회 조립되고 /config로 갱신되는 전역 상태
_STATE: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    question: str
    # 대화 히스토리: [(사용자질문, 어시스턴트답변), ...]. 멀티턴 문맥 유지용.
    history: List[Tuple[str, str]] = []
    # 특정 RFP 한 건으로 한정하고 싶을 때(원클릭 요약 등) 파일명 지정.
    filename: Optional[str] = None


class ConfigUpdate(BaseModel):
    # 모두 선택적. 들어온 값만 현재 설정에 덮어쓴다.
    # 프리셋(llm/embedding/chunk)은 src/catalog.py에서 선언한 key를 받는다.
    llm_preset: Optional[str] = None
    embedding_preset: Optional[str] = None
    chunk_preset: Optional[str] = None
    parser_hwp_preset: Optional[str] = None
    parser_pdf_preset: Optional[str] = None
    # 프리셋에 포함되지 않는 자유 조정 항목
    llm_temperature: Optional[float] = None
    llm_base_url: Optional[str] = None
    retriever_k: Optional[int] = None
    retriever_use_bm25: Optional[bool] = None


class Source(BaseModel):
    파일명: Optional[str] = None
    기관: Optional[str] = None
    금액: Optional[int] = None
    마감일: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    sources: List[Source]


def _rebuild(cfg: Dict[str, Any]) -> bool:
    """cfg로 임베딩/벡터스토어/검색기/LLM을 다시 조립한다.

    캐시가 없으면 재인덱싱(파싱+청킹+임베딩)한다. 반환값은 재인덱싱 여부.
    """
    embedding = make_embedding(cfg)
    cache_key = make_cache_key(cfg)
    use_bm25 = cfg["retriever"].get("use_bm25", False)
    have_cache = cache_exists(cfg["data"]["faiss_cache"], cache_key)

    # 캐시가 없으면 빌드, BM25는 chunks가 필요하므로 파싱한다.
    chunks = None
    if not have_cache or use_bm25:
        docs = load_documents(cfg)
        chunks = split_documents(docs, cfg)

    vectorstore = build_vectorstore(
        chunks, embedding, cfg["data"]["faiss_cache"], cache_key
    )
    retriever = build_retriever(vectorstore, chunks, cfg)
    llm = make_llm(cfg)
    # 질문별 검색기 선택(_select_retriever)을 그래프에 주입한다. _STATE가 먼저
    # 갱신돼야 _select_retriever가 최신 vectorstore/known_orgs를 참조하므로
    # graph는 update 이후에 빌드한다.
    graph = build_graph(cfg, llm, lambda q, fn: _select_retriever(q, fn))

    _STATE.update(
        cfg=cfg,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        graph=graph,
        embedding=embedding,
        cache_key=cache_key,
        known_orgs=known_orgs(vectorstore),
        n_vectors=vectorstore.index.ntotal,
    )
    return not have_cache


def _apply_updates(cfg: Dict[str, Any], u: ConfigUpdate) -> Dict[str, Any]:
    """ConfigUpdate의 들어온 값만 cfg 사본에 반영한다.

    프리셋은 catalog.apply_preset으로 해당 섹션을 통째로 덮어쓰고,
    프리셋에 없는 항목(temperature/base_url/k/bm25)은 따로 반영한다.
    """
    cfg = copy.deepcopy(cfg)
    if u.llm_preset is not None:
        cfg = catalog.apply_preset(cfg, "llm", u.llm_preset)
    if u.embedding_preset is not None:
        cfg = catalog.apply_preset(cfg, "embedding", u.embedding_preset)
    if u.chunk_preset is not None:
        cfg = catalog.apply_preset(cfg, "chunk", u.chunk_preset)
    if u.parser_hwp_preset is not None:
        cfg = catalog.apply_preset(cfg, "parser_hwp", u.parser_hwp_preset)
    if u.parser_pdf_preset is not None:
        cfg = catalog.apply_preset(cfg, "parser_pdf", u.parser_pdf_preset)
    if u.llm_temperature is not None:
        cfg["llm"]["temperature"] = u.llm_temperature
    if u.llm_base_url is not None:
        cfg["llm"]["base_url"] = u.llm_base_url or None
    if u.retriever_k is not None:
        cfg["retriever"]["k"] = u.retriever_k
    if u.retriever_use_bm25 is not None:
        cfg["retriever"]["use_bm25"] = u.retriever_use_bm25
    return cfg


def _current_settings() -> Dict[str, Any]:
    cfg = _STATE["cfg"]
    return {
        "embedding": cfg["embedding"],
        "llm": cfg["llm"],
        "chunk": cfg["chunk"],
        "parser": cfg.get("parser", {}),
        "retriever": cfg["retriever"],
        "cache_key": _STATE.get("cache_key"),
    }


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config("config.yaml")
    # 기동 시 캐시가 없으면 재인덱싱이 일어날 수 있다(시간 소요).
    _rebuild(cfg)


def _dedup_sources(docs) -> List[Source]:
    """검색된 청크에서 문서(파일명) 단위로 중복 제거한 출처 목록을 만든다."""
    seen = set()
    out: List[Source] = []
    for d in docs:
        fn = d.metadata.get("파일명")
        if fn in seen:
            continue
        seen.add(fn)
        out.append(Source(
            파일명=fn,
            기관=d.metadata.get("기관"),
            금액=d.metadata.get("금액"),
            마감일=d.metadata.get("마감일"),
        ))
    return out


def _select_retriever(question: str, filename: Optional[str] = None):
    """검색 범위를 정한다.

    filename이 있으면 그 문서 한 건으로 한정(원클릭 요약 등),
    없고 질문에 기관명이 있으면 그 기관으로 한정, 아니면 기본 검색기.
    """
    cfg = _STATE["cfg"]
    k = cfg["retriever"].get("k", 4)
    n = _STATE.get("n_vectors", 10000)
    if filename:
        return build_meta_filtered_retriever(
            _STATE["vectorstore"], k, {"파일명": filename}, n
        )
    if cfg["retriever"].get("use_metadata_filter", True):
        matched = match_orgs(question, _STATE.get("known_orgs", set()))
        if matched:
            return build_org_filtered_retriever(_STATE["vectorstore"], k, matched, n)
    return _STATE["retriever"]


def _answer(mode: Optional[str], question: str,
            history: Optional[List[Tuple[str, str]]] = None,
            filename: Optional[str] = None) -> ChatResponse:
    """수퍼바이저 그래프를 거쳐 답변을 생성한다.

    mode가 주어지면 해당 intent("qa"/"summary")를 강제 주입해 분류를 생략하고,
    None이면 그래프의 classify_intent가 의도를 판정해 라우팅한다(/chat).
    """
    if "graph" not in _STATE:
        raise HTTPException(status_code=503, detail="인덱스가 준비되지 않았습니다.")
    state: Dict[str, Any] = {
        "question": question,
        "history": history or [],
        "filename": filename,
    }
    if mode is not None:
        state["intent"] = mode
    result = _STATE["graph"].invoke(state)
    answer = result.get("answer", "")
    return ChatResponse(
        answer=answer,
        refused=REFUSAL_MARK in answer,
        sources=_dedup_sources(result.get("sources") or []),
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    ready = "retriever" in _STATE
    ntotal = _STATE["vectorstore"].index.ntotal if ready else 0
    return {"status": "ok" if ready else "not_ready", "vectors": ntotal}


@app.get("/config")
def get_config() -> Dict[str, Any]:
    return {
        "settings": _current_settings(),
        "catalog": catalog.frontend_catalog(_STATE["cfg"]),
    }


@app.post("/config")
def set_config(update: ConfigUpdate) -> Dict[str, Any]:
    """설정을 적용한다. 재인덱싱이 필요하면 수행한다(시간이 걸릴 수 있음)."""
    new_cfg = _apply_updates(_STATE["cfg"], update)
    try:
        reindexed = _rebuild(new_cfg)
    except Exception as e:  # 잘못된 모델/미설치 패키지/원격 연결 실패 등
        raise HTTPException(status_code=400, detail=f"설정 적용 실패: {e}")
    return {
        "reindexed": reindexed,
        "settings": _current_settings(),
        "catalog": catalog.frontend_catalog(new_cfg),
    }


@app.get("/file")
def get_file(name: str, download: bool = False) -> FileResponse:
    """근거 RFP 원본 파일을 서빙한다.

    download=false면 인라인(PDF는 브라우저에서 바로 보기, HWP는 브라우저가 받음),
    download=true면 첨부(다운로드)로 내려준다.
    name은 data/files 안의 파일명만 허용한다(경로 탈출 차단).
    """
    cfg = _STATE.get("cfg")
    if not cfg:
        raise HTTPException(status_code=503, detail="서버가 준비되지 않았습니다.")
    base = Path(cfg["data"]["files"]).resolve()
    target = (base / name).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    disposition = "attachment" if download else "inline"
    # 한글/공백 파일명은 RFC 5987(filename*)로 인코딩한다.
    encoded = quote(name)
    headers = {"Content-Disposition": f"{disposition}; filename*=UTF-8''{encoded}"}
    return FileResponse(str(target), headers=headers)


@app.get("/documents")
def documents() -> Dict[str, Any]:
    """인덱싱된 문서 목록을 파일명 단위로 반환한다."""
    if "vectorstore" not in _STATE:
        raise HTTPException(status_code=503, detail="인덱스가 준비되지 않았습니다.")
    docstore = _STATE["vectorstore"].docstore
    seen: Dict[str, Dict[str, Any]] = {}
    for doc in docstore._dict.values():
        m = doc.metadata
        fn = m.get("파일명")
        if fn and fn not in seen:
            seen[fn] = {
                "파일명": fn,
                "사업명": m.get("사업명"),
                "기관": m.get("기관"),
                "금액": m.get("금액"),
                "마감일": m.get("마감일"),
                "파일형식": m.get("파일형식"),
                "likely_scanned": m.get("likely_scanned"),
            }
    docs = sorted(seen.values(), key=lambda x: x["파일명"])
    return {"count": len(docs), "documents": docs}


_UPLOAD_EXTS = {".hwp", ".hwpx", ".pdf"}


class UploadResponse(BaseModel):
    파일명: str
    chunks: int           # 새로 추가된 청크 수
    vectors: int          # 추가 후 전체 벡터 수
    기관: Optional[str] = None
    금액: Optional[int] = None
    마감일: Optional[str] = None
    사업명: Optional[str] = None


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)) -> UploadResponse:
    """사용자 문서(HWP/PDF)를 업로드해 파싱·청킹 후 FAISS에 영구 추가한다.

    nara_agent와 동일한 흐름: parse_file → metadata_agent → split_documents
    → add_to_vectorstore. 추가 후 in-memory 검색기/통계를 갱신해 즉시 질문 가능.
    """
    if "vectorstore" not in _STATE:
        raise HTTPException(status_code=503, detail="인덱스가 준비되지 않았습니다.")

    cfg = _STATE["cfg"]
    name = Path(file.filename or "").name  # 경로 탈출 차단(파일명만 사용)
    ext = Path(name).suffix.lower()
    if not name or ext not in _UPLOAD_EXTS:
        raise HTTPException(
            status_code=400, detail="HWP·HWPX·PDF 파일만 업로드할 수 있습니다."
        )

    # data/files에 저장(원본 서빙 /file·목록 /documents와 동일 경로 사용)
    files_dir = Path(cfg["data"]["files"])
    files_dir.mkdir(parents=True, exist_ok=True)
    dest = files_dir / name
    dest.write_bytes(file.file.read())

    text = parse_file(dest, cfg)
    if not text:
        raise HTTPException(
            status_code=422,
            detail="문서에서 텍스트를 추출하지 못했습니다(스캔본/미지원 형식일 수 있음).",
        )

    meta = extract_metadata(text, cfg)
    doc = Document(
        page_content=text,
        metadata={
            "파일명": name,
            "기관": meta.get("발주기관"),
            "금액": meta.get("사업금액"),
            "마감일": meta.get("마감일"),
            "파일형식": ext.lstrip("."),
            "likely_scanned": len(text) < 1000,
            "fallback_used": False,
            "사업명": meta.get("사업명"),
            "uploaded": True,
        },
    )

    chunks = split_documents([doc], cfg)
    vs = add_to_vectorstore(
        chunks, _STATE["embedding"], cfg["data"]["faiss_cache"], _STATE["cache_key"]
    )

    # add_to_vectorstore가 디스크에서 로드/추가/저장한 vs로 in-memory 상태를 교체한다.
    # (use_bm25면 BM25 부분은 재인덱싱 전까지 신규 문서를 포함하지 않는다.)
    retriever = build_retriever(vs, None, cfg)
    _STATE.update(
        vectorstore=vs,
        retriever=retriever,
        known_orgs=known_orgs(vs),
        n_vectors=vs.index.ntotal,
    )

    return UploadResponse(
        파일명=name,
        chunks=len(chunks),
        vectors=vs.index.ntotal,
        기관=meta.get("발주기관"),
        금액=meta.get("사업금액"),
        마감일=meta.get("마감일"),
        사업명=meta.get("사업명"),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # mode=None → 그래프가 의도(qa/summary/metadata/collect)를 분류해 라우팅한다.
    return _answer(None, req.question, req.history, req.filename)


@app.post("/summarize", response_model=ChatResponse)
def summarize(req: ChatRequest) -> ChatResponse:
    return _answer("summary", req.question, req.history, req.filename)
