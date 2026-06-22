"""검색기 구성: dense / hybrid(BM25) / rerank(자리).

베이스라인은 dense top-k. config로 BM25 하이브리드를 켤 수 있고,
reranker는 자리만 만들어 둔다.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

# 기관명 앞에 붙는 법인 형태/괄호 표기를 제거해 매칭을 느슨하게 한다.
_LEGAL_PREFIX = re.compile(r"^(사단법인|재단법인|주식회사|재|사|주)\s*")


def _norm_org(s: Optional[str]) -> str:
    """기관명을 매칭용으로 정규화한다(괄호·법인형태·공백 제거)."""
    s = re.sub(r"[()（）]", "", s or "")
    s = _LEGAL_PREFIX.sub("", s)
    return re.sub(r"\s+", "", s)


def known_orgs(vectorstore) -> set:
    """인덱싱된 청크의 메타데이터에서 기관명 집합을 모은다."""
    return {
        d.metadata.get("기관")
        for d in vectorstore.docstore._dict.values()
        if d.metadata.get("기관")
    }


def match_orgs(query: str, orgs) -> List[str]:
    """질문에 등장하는 기관명을 corpus의 실제 기관값으로 매칭한다(부분일치).

    너무 짧은 이름(오탐 위험)은 제외한다. 매칭이 없으면 빈 리스트.
    """
    nq = _norm_org(query)
    return [o for o in orgs if len(_norm_org(o)) >= 4 and _norm_org(o) in nq]


def build_org_filtered_retriever(vectorstore, k: int, matched_orgs: List[str],
                                 n_vectors: int):
    """기관 메타데이터로 필터링하는 dense 검색기를 만든다.

    fetch_k를 전체 벡터 수로 둬서 '해당 기관 전체'에서 유사도 상위 k를 고른다.
    (FAISS는 기본적으로 가까운 fetch_k개만 뽑은 뒤 필터하므로 fetch_k를 키운다.)
    """
    value = matched_orgs if len(matched_orgs) > 1 else matched_orgs[0]
    return vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"기관": value}, "fetch_k": max(n_vectors, k)}
    )


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
