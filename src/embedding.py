"""임베딩 + FAISS 벡터스토어 빌드/캐시.

캐시 디렉토리 이름을 설정 시그니처(임베딩 모델 + 파서 + 청킹)로 만들어,
임베딩/청킹/파서가 바뀌면 캐시도 분리되고 같은 설정은 캐시를 재사용한다.
저장본이 있으면 로드하고 없으면 새로 빌드 후 save_local로 캐시한다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# 전처리(loader._clean)/청킹 로직이 바뀌면 이 값을 올려 기존 캐시를 무효화한다.
# (캐시 키는 모델/파서/청크 설정만 반영하므로 코드 변경은 별도 버전으로 추적한다.)
PREP_VERSION = "v3"


def _safe_name(name: str) -> str:
    """디렉토리명으로 쓸 수 있게 정리한다."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def make_cache_key(cfg: Dict[str, Any]) -> str:
    """임베딩/파서/청킹 설정으로 FAISS 캐시 키(디렉토리명)를 만든다.

    이 키가 같으면 같은 인덱스를 재사용하고, 하나라도 바뀌면 새로 빌드한다.
    (LLM/검색 k/BM25는 인덱스에 영향 없으므로 키에 넣지 않는다.)
    """
    e = cfg["embedding"]
    c = cfg["chunk"]
    p = cfg.get("parser", {}) or {}
    parts = [
        e.get("provider", "openai"),
        e["model"],
        p.get("hwp", "pyhwp"),
        p.get("pdf", "pymupdf"),
        c.get("strategy", "recursive"),
        str(c["size"]),
        str(c["overlap"]),
        PREP_VERSION,
    ]
    return _safe_name("__".join(parts))


def cache_exists(cache_dir: str, cache_key: str) -> bool:
    """해당 설정의 FAISS 캐시가 이미 있는지 확인한다."""
    return (Path(cache_dir) / _safe_name(cache_key) / "index.faiss").exists()


def build_vectorstore(chunks: List[Document], embedding, cache_dir: str,
                      cache_key: str) -> FAISS:
    """FAISS 벡터스토어를 빌드하거나 캐시에서 로드한다.

    cache_dir/<cache_key>/ 에 저장한다. 설정이 바뀌면 cache_key가 달라진다.
    """
    cache_path = Path(cache_dir) / _safe_name(cache_key)
    index_file = cache_path / "index.faiss"

    if index_file.exists():
        # 저장본 로드. 빌드 시점과 동일한 임베딩 객체를 넘겨야 검색이 일치한다.
        return FAISS.load_local(
            str(cache_path), embedding, allow_dangerous_deserialization=True
        )

    if not chunks:
        raise ValueError(
            "벡터스토어를 만들 청크가 없습니다. 원본 파싱 결과를 확인하세요."
        )

    vs = FAISS.from_documents(chunks, embedding)
    cache_path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(cache_path))
    return vs
