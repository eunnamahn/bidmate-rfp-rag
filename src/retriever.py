"""검색기 구성: dense / hybrid(BM25) / rerank.

베이스라인은 dense top-k. config로 BM25 하이브리드(EnsembleRetriever)와
CrossEncoder 리랭커(bge-reranker-v2-m3)를 켤 수 있다.

리랭커를 켜면 base 검색기가 후보를 더 많이(rerank_candidates) 뽑고,
CrossEncoder가 질문-청크 쌍을 다시 채점해 최종 k개로 압축한다.
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


def build_meta_filtered_retriever(vectorstore, k: int, filt: Dict[str, Any],
                                  n_vectors: int):
    """메타데이터 동등 필터로 한정 검색하는 dense 검색기를 만든다.

    fetch_k를 전체 벡터 수로 둬서 '필터에 해당하는 전체'에서 유사도 상위 k를 고른다.
    (FAISS는 기본적으로 가까운 fetch_k개만 뽑은 뒤 필터하므로 fetch_k를 키운다.)
    """
    return vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": filt, "fetch_k": max(n_vectors, k)}
    )


def build_org_filtered_retriever(vectorstore, k: int, matched_orgs: List[str],
                                 n_vectors: int):
    """기관 메타데이터로 한정 검색하는 검색기(범용 필터의 기관 전용 래퍼)."""
    value = matched_orgs if len(matched_orgs) > 1 else matched_orgs[0]
    return build_meta_filtered_retriever(vectorstore, k, {"기관": value}, n_vectors)


def build_retriever(vectorstore, chunks: List[Document], cfg: Dict[str, Any]):
    """config 기준으로 검색기를 구성해 반환한다.

    구성 순서: dense(또는 BM25 하이브리드)로 후보를 뽑고, 리랭커를 켜면
    CrossEncoder가 그 후보를 다시 채점해 상위 k개로 압축한다.
    리랭커가 켜져 있으면 base는 k가 아니라 rerank_candidates개를 뽑는다.
    """
    rcfg = cfg["retriever"]
    k = rcfg.get("k", 4)
    use_reranker = rcfg.get("use_reranker", False)
    # 리랭커가 있으면 base 후보 풀을 넓힌다(리랭커가 여기서 k개를 골라낸다).
    fetch_n = rcfg.get("rerank_candidates", 20) if use_reranker else k

    dense = vectorstore.as_retriever(search_kwargs={"k": fetch_n})

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

        # BM25는 청크 본문이 필요하다. 캐시에서 로드한 경우 chunks가 없을 수 있는데,
        # 그 청크는 FAISS 벡터스토어의 docstore에 그대로 보존돼 있으므로 재파싱하지 않고
        # 거기서 복원한다(설정 적용 때마다 원본 전체를 다시 파싱하는 비용을 없앤다).
        if not chunks:
            chunks = list(vectorstore.docstore._dict.values())
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = fetch_n
        # BM25 가중치는 config로 조절(dense 가중치는 1 - bm25_weight).
        w_bm25 = rcfg.get("bm25_weight", 0.5)
        retriever = EnsembleRetriever(
            retrievers=[bm25, dense], weights=[w_bm25, 1.0 - w_bm25]
        )
    else:
        retriever = dense

    if use_reranker:
        retriever = _wrap_reranker(retriever, cfg, top_n=k)

    return retriever


# CrossEncoder 모델은 로딩이 비싸다(bge-reranker-v2-m3 약 2.2GB). 모델명별로 한 번만
# 로드해 캐시하면, k/BM25/온도 등 다른 설정을 바꿔 검색기를 재구성해도 모델을 다시
# 읽지 않는다(설정 적용 지연의 큰 부분을 제거).
_CROSS_ENCODER_CACHE: Dict[str, Any] = {}


def _get_cross_encoder(model_name: str):
    """모델명으로 HuggingFaceCrossEncoder를 가져오되 프로세스 내에서 1회만 로드한다."""
    encoder = _CROSS_ENCODER_CACHE.get(model_name)
    if encoder is not None:
        return encoder

    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    # CUDA GPU가 있으면 GPU로 채점한다(없으면 CPU).
    model_kwargs: Dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            model_kwargs["device"] = "cuda"
    except Exception:
        pass

    encoder = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
    _CROSS_ENCODER_CACHE[model_name] = encoder
    return encoder


def _wrap_reranker(retriever, cfg: Dict[str, Any], top_n: int):
    """base 검색기를 CrossEncoder 리랭커로 감싼다.

    base가 뽑은 후보를 bge-reranker-v2-m3가 질문-청크 쌍으로 다시 채점하고
    상위 top_n개만 남긴다. langchain 버전에 따라 모듈 경로가 달라 폴백 처리한다.
    """
    rcfg = cfg["retriever"]
    model_name = rcfg.get("reranker_model", "BAAI/bge-reranker-v2-m3")

    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
    except ModuleNotFoundError:
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import (
            CrossEncoderReranker,
        )

    encoder = _get_cross_encoder(model_name)
    compressor = CrossEncoderReranker(model=encoder, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
