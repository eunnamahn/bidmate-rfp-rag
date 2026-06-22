"""검색기 구성: dense / hybrid(BM25) / rerank(자리).

베이스라인은 dense top-k. config로 BM25 하이브리드를 켤 수 있고,
reranker는 자리만 만들어 둔다.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document


def build_retriever(vectorstore, chunks: List[Document], cfg: Dict[str, Any]):
    """config 기준으로 검색기를 구성해 반환한다."""
    rcfg = cfg["retriever"]
    k = rcfg.get("k", 4)

    dense = vectorstore.as_retriever(search_kwargs={"k": k})

    if rcfg.get("use_bm25", False):
        # BM25 + dense 하이브리드(EnsembleRetriever)
        # langchain 버전에 따라 모듈 경로가 달라 폴백으로 처리한다.
        try:
            from langchain.retrievers import EnsembleRetriever
        except ModuleNotFoundError:
            from langchain_classic.retrievers import EnsembleRetriever
        try:
            from langchain_community.retrievers import BM25Retriever
        except ModuleNotFoundError:
            from langchain_classic.retrievers import BM25Retriever

        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = k
        retriever = EnsembleRetriever(
            retrievers=[bm25, dense], weights=[0.5, 0.5]
        )
    else:
        retriever = dense

    if rcfg.get("use_reranker", False):
        retriever = _wrap_reranker(retriever, cfg)

    return retriever


def _wrap_reranker(retriever, cfg: Dict[str, Any]):
    """reranker(bge-reranker-v2-m3) 적용 자리. 베이스라인은 미적용."""
    # TODO: ContextualCompressionRetriever + CrossEncoderReranker로 구현 예정.
    return retriever
