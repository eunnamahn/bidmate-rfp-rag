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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import catalog
from src.chunking import split_documents
from src.config import load_config, make_embedding, make_llm
from src.embedding import build_vectorstore, cache_exists, make_cache_key
from src.evaluation.evaluate import REFUSAL_MARK
from src.generator import generate_answer
from src.loader import load_documents
from src.retriever import build_retriever

app = FastAPI(title="입찰메이트 RFP RAG API")

# 서버 기동 시 1회 조립되고 /config로 갱신되는 전역 상태
_STATE: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    question: str


class ConfigUpdate(BaseModel):
    # 모두 선택적. 들어온 값만 현재 설정에 덮어쓴다.
    # 프리셋(llm/embedding/chunk)은 src/catalog.py에서 선언한 key를 받는다.
    llm_preset: Optional[str] = None
    embedding_preset: Optional[str] = None
    chunk_preset: Optional[str] = None
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

    _STATE.update(
        cfg=cfg,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        cache_key=cache_key,
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


def _answer(mode: str, question: str) -> ChatResponse:
    if "retriever" not in _STATE:
        raise HTTPException(status_code=503, detail="인덱스가 준비되지 않았습니다.")
    answer, docs = generate_answer(
        _STATE["retriever"], _STATE["llm"], question, mode=mode
    )
    return ChatResponse(
        answer=answer,
        refused=REFUSAL_MARK in answer,
        sources=_dedup_sources(docs),
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
                "기관": m.get("기관"),
                "금액": m.get("금액"),
                "마감일": m.get("마감일"),
                "파일형식": m.get("파일형식"),
                "likely_scanned": m.get("likely_scanned"),
            }
    docs = sorted(seen.values(), key=lambda x: x["파일명"])
    return {"count": len(docs), "documents": docs}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return _answer("qa", req.question)


@app.post("/summarize", response_model=ChatResponse)
def summarize(req: ChatRequest) -> ChatResponse:
    return _answer("summary", req.question)
