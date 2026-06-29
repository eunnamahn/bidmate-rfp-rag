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
from src.agents import nara_agent, registry
from src.agents.metadata_agent import append_metadata_row, extract_metadata
from src.chunking import split_documents
from src.config import load_config, make_embedding, make_llm
from src.embedding import (
    add_to_vectorstore,
    build_vectorstore,
    cache_exists,
    make_cache_key,
)
from src.evaluation.evaluate import REFUSAL_MARK
from src.generator import generate_infographic_data
from src.graph import build_graph
from src.infographic import render_infographic_html
from src.loader import load_documents, parse_file
from src.retriever import (
    build_meta_filtered_retriever,
    build_org_filtered_retriever,
    build_retriever,
    known_orgs,
    match_orgs,
)

app = FastAPI(title="입찰메이트 RFP RAG API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 기동 시 1회 조립되고 /config로 갱신되는 전역 상태
_STATE: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    question: str
    # 대화 히스토리: [(사용자질문, 어시스턴트답변), ...]. 멀티턴 문맥 유지용.
    history: List[Tuple[str, str]] = []
    # 특정 RFP 한 건으로 한정하고 싶을 때(원클릭 요약 등) 파일명 지정.
    filename: Optional[str] = None
    # 에이전트 액션 버튼에서 호출 시 registry의 에이전트 id(예: summary/metadata/eligibility).
    # 지정되면 LLM 라우팅 대신 해당 에이전트로 강제 분기한다. None이면 일반 챗(자동 라우팅).
    agent: Optional[str] = None
    # 프론트엔드가 전달하는 회사 정보(needs_company_info=True 에이전트 전용).
    # 이 필드가 있으면 eligibility 노드가 히스토리 LLM 추출 없이 직접 판정에 사용한다.
    company_info: Optional[Dict[str, Any]] = None


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
    retriever_use_reranker: Optional[bool] = None


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
    have_cache = cache_exists(cfg["data"]["faiss_cache"], cache_key)

    # 캐시가 없을 때만 원본을 파싱·청킹한다. BM25를 켜도 청크는 build_retriever가
    # 벡터스토어 docstore에서 복원하므로 재파싱하지 않는다(설정 적용 지연 제거).
    chunks = None
    if not have_cache:
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
    if u.retriever_use_reranker is not None:
        cfg["retriever"]["use_reranker"] = u.retriever_use_reranker
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
            filename: Optional[str] = None,
            company_info: Optional[Dict[str, Any]] = None) -> ChatResponse:
    """수퍼바이저 그래프를 거쳐 답변을 생성한다.

    mode가 주어지면 해당 intent("qa"/"summary")를 강제 주입해 분류를 생략하고,
    None이면 그래프의 classify_intent가 의도를 판정해 라우팅한다(/chat).
    company_info가 있으면 eligibility 노드가 히스토리 추출 없이 직접 판정에 사용한다.
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
    if company_info:
        state["company_info"] = company_info
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


@app.get("/agents")
def agents() -> Dict[str, Any]:
    """챗 UI가 액션 버튼으로 렌더할 에이전트 목록을 반환한다.

    구현 여부(implemented)·회사정보 필요 여부(needs_company_info) 등을 함께 내려,
    프론트가 버튼 활성/안내를 결정한다. (catalog.frontend_catalog와 같은 노출 패턴)
    """
    return {"agents": registry.as_frontend()}


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


def _refresh_index_state(vs, cfg: Dict[str, Any]) -> None:
    """add_to_vectorstore가 돌려준 vs로 in-memory 검색 상태를 갱신한다.

    업로드/나라장터 지연 인덱싱이 공유한다. (use_bm25면 BM25 부분은 재인덱싱
    전까지 신규 문서를 포함하지 않는다.)
    """
    _STATE.update(
        vectorstore=vs,
        retriever=build_retriever(vs, None, cfg),
        known_orgs=known_orgs(vs),
        n_vectors=vs.index.ntotal,
    )


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
    _refresh_index_state(vs, cfg)

    # 메타데이터 에이전트: 추가된 문서의 메타데이터를 data_list.csv 대장에 자동 기록.
    # (업로드 파일은 data/files에 저장돼 추후 전체 재빌드 시에도 행과 일관된다.)
    # 기록 실패가 업로드 자체를 막지 않도록 예외는 흡수한다.
    try:
        append_metadata_row(cfg["data"]["csv"], name, ext.lstrip("."), meta, 텍스트=text)
    except Exception:
        pass

    return UploadResponse(
        파일명=name,
        chunks=len(chunks),
        vectors=vs.index.ntotal,
        기관=meta.get("발주기관"),
        금액=meta.get("사업금액"),
        마감일=meta.get("마감일"),
        사업명=meta.get("사업명"),
    )


class NaraNotice(BaseModel):
    공고번호: Optional[str] = None
    사업명: Optional[str] = None
    기관: Optional[str] = None
    금액: Optional[int] = None
    마감일: Optional[str] = None
    item: Dict[str, Any] = {}     # 지연 인덱싱(/nara/ingest)에 다시 넘길 원본 공고


class NaraIngestRequest(BaseModel):
    item: Dict[str, Any]          # /nara/search가 돌려준 원본 공고 item


@app.get("/nara/search")
def nara_search(q: str = "", date: Optional[str] = None, days: int = 3) -> Dict[str, Any]:
    """나라장터 공고를 메타데이터만 검색한다(다운로드/임베딩 없음 → 인덱스 불변)."""
    cfg = _STATE.get("cfg")
    if not cfg:
        raise HTTPException(status_code=503, detail="서버가 준비되지 않았습니다.")
    target = None
    if date:
        from datetime import datetime
        try:
            target = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="date는 YYYY-MM-DD 형식이어야 합니다.")
    try:
        items = nara_agent.search_notices(cfg, keyword=q or None, target_date=target, days=days)
    except RuntimeError as e:  # NARA_API_KEY 없음
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"나라장터 검색 실패: {e}")

    notices = [
        NaraNotice(**nara_agent.notice_summary(it), item=it) for it in items
    ]
    return {"count": len(notices), "notices": [n.model_dump() for n in notices]}


@app.post("/nara/ingest", response_model=UploadResponse)
def nara_ingest(req: NaraIngestRequest) -> UploadResponse:
    """선택한 공고 1건만 다운로드·파싱·임베딩해 FAISS에 영구 추가한다(지연 인덱싱)."""
    if "vectorstore" not in _STATE:
        raise HTTPException(status_code=503, detail="인덱스가 준비되지 않았습니다.")
    cfg = _STATE["cfg"]
    try:
        result = nara_agent.ingest_notice(
            cfg, req.item, embedding=_STATE["embedding"], cache_key=_STATE["cache_key"]
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"공고 인덱싱 실패: {e}")

    if result["chunks"] == 0:
        raise HTTPException(
            status_code=422,
            detail="첨부가 없거나 텍스트를 추출하지 못해 인덱싱할 내용이 없습니다.",
        )

    _refresh_index_state(result["vectorstore"], cfg)
    return UploadResponse(
        # 실제 문서 파일명(data/files 기준)을 우선 반환해 UI 활성 문서가 인덱스와 일치하게 한다.
        파일명=result.get("파일명") or result.get("사업명") or result.get("공고번호") or "(공고)",
        chunks=result["chunks"],
        vectors=result["vectorstore"].index.ntotal,
        기관=result.get("기관"),
        금액=result.get("금액"),
        마감일=result.get("마감일"),
        사업명=result.get("사업명"),
    )


# 미구현 에이전트 버튼 클릭 시 안내(팀원 PR로 본체가 추가되면 implemented=True가 된다).
_AGENT_PENDING = "이 기능은 준비 중입니다(팀원 PR 예정)."


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # 에이전트 액션 버튼에서 온 요청: registry로 분기.
    if req.agent:
        spec = registry.get(req.agent)
        if spec is None or not spec.implemented or not spec.intent:
            # 미존재/미구현 에이전트 → 거절이 아닌 안내(출처 없음).
            return ChatResponse(answer=_AGENT_PENDING, refused=False, sources=[])
        # 구현된 에이전트는 해당 graph intent를 강제한다. company_info도 함께 전달.
        return _answer(spec.intent, req.question, req.history, req.filename, req.company_info)

    # agent 미지정 → 그래프가 의도(qa/summary/metadata/collect)를 분류해 라우팅한다.
    return _answer(None, req.question, req.history, req.filename, req.company_info)


@app.post("/summarize", response_model=ChatResponse)
def summarize(req: ChatRequest) -> ChatResponse:
    return _answer("summary", req.question, req.history, req.filename)


class InfographicRequest(BaseModel):
    question: str
    filename: Optional[str] = None      # 활성 문서 한정(없으면 전체에서 검색)


class InfographicResponse(BaseModel):
    title: str
    html: str                            # 인라인 렌더용 HTML 카드(빈 문자열이면 실패)
    sources: List[Source]
    parse_error: Optional[str] = None


@app.post("/infographic", response_model=InfographicResponse)
def infographic(req: InfographicRequest) -> InfographicResponse:
    """RFP 문서에서 핵심 정보를 추출해 인포그래픽 HTML 카드를 생성한다.

    챗 UI가 components.html로 인라인 렌더한다(Playwright/PNG 미사용).
    활성 문서가 지정되면 그 문서로 한정해 추출한다(/chat과 동일한 _select_retriever).
    """
    if "graph" not in _STATE:
        raise HTTPException(status_code=503, detail="인덱스가 준비되지 않았습니다.")
    retriever = _select_retriever(req.question, req.filename)
    data, docs = generate_infographic_data(retriever, _STATE["llm"], req.question)
    if data.get("parse_error"):
        return InfographicResponse(
            title=req.question, html="",
            sources=_dedup_sources(docs), parse_error=data["parse_error"],
        )
    return InfographicResponse(
        title=data.get("title") or req.question,
        html=render_infographic_html(data),
        sources=_dedup_sources(docs),
    )
