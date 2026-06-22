"""전체 파이프라인 조립.

로더 -> 청킹 -> 임베딩 -> 벡터스토어 -> 검색기 -> 생성체인을 config 기준으로 조립한다.
이미 만든 docs/chunks/vectorstore를 인자로 넘기면 재파싱·재임베딩을 건너뛴다
(매 실험마다 비싼 단계를 반복하지 않기 위함).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.chunking import split_documents
from src.config import make_embedding, make_llm
from src.embedding import build_vectorstore, cache_exists, make_cache_key
from src.generator import build_chain
from src.loader import load_documents
from src.retriever import build_retriever


def build_pipeline(
    cfg: Dict[str, Any],
    docs: Optional[List[Document]] = None,
    chunks: Optional[List[Document]] = None,
    vectorstore=None,
    mode: str = "qa",
) -> Tuple[Any, Any]:
    """파이프라인을 조립해 (chain, retriever)를 반환한다.

    재사용 인자:
      docs        : 미리 파싱한 Document 리스트
      chunks      : 미리 청킹한 결과
      vectorstore : 미리 빌드/로드한 FAISS
    """
    embedding = make_embedding(cfg)
    use_bm25 = cfg["retriever"].get("use_bm25", False)
    cache_key = make_cache_key(cfg)

    if vectorstore is None:
        have_cache = cache_exists(cfg["data"]["faiss_cache"], cache_key)
        # 캐시가 없거나 BM25가 필요할 때만 원본을 파싱/청킹한다(불필요한 재파싱 방지).
        if chunks is None and (not have_cache or use_bm25):
            if docs is None:
                docs = load_documents(cfg)
            chunks = split_documents(docs, cfg)
        vectorstore = build_vectorstore(
            chunks,
            embedding,
            cfg["data"]["faiss_cache"],
            cache_key,
        )

    retriever = build_retriever(vectorstore, chunks, cfg)
    llm = make_llm(cfg)
    chain = build_chain(retriever, llm, mode=mode)
    return chain, retriever


def build_components(
    cfg: Dict[str, Any]
) -> Tuple[List[Document], List[Document], Any]:
    """재사용 가능한 무거운 부품(docs, chunks, vectorstore)만 따로 빌드한다.

    QA/요약 두 모드나 여러 실험에서 같은 벡터스토어를 공유할 때 사용한다.
    """
    docs = load_documents(cfg)
    chunks = split_documents(docs, cfg)
    embedding = make_embedding(cfg)
    vectorstore = build_vectorstore(
        chunks, embedding, cfg["data"]["faiss_cache"], make_cache_key(cfg)
    )
    return docs, chunks, vectorstore
