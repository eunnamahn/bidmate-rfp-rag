"""파서 조합 인덱스 정비 스크립트.

1) 기존 pyhwp+pymupdf 인덱스: 저장된 금액 메타데이터를 CSV 기준으로 교정한다.
   (청크 텍스트가 동일하므로 임베딩 재생성 없이 금액만 고쳐 다시 저장 — API 비용 없음)
2) olefile 조합(olefile+pdfplumber, olefile+pymupdf): 캐시가 없으면 새로 빌드한다.

사용: python scripts/reindex_combos.py
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.chunking import split_documents
from src.config import load_config, make_embedding
from src.embedding import (_safe_name, build_vectorstore, cache_exists,
                           make_cache_key)
from src.loader import _parse_amount, load_documents

from langchain_community.vectorstores import FAISS

# 새로 빌드할 조합 (hwp, pdf)
NEW_COMBOS = [
    ("olefile", "pdfplumber"),
    ("olefile", "pymupdf"),
]


def _csv_amount_map(csv_path: str) -> dict:
    """파일명 -> 올바른 금액(수정된 _parse_amount 적용) 매핑."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).replace(" ", "") for c in df.columns]
    return {
        str(r["파일명"]): _parse_amount(r.get("사업금액"))
        for _, r in df.iterrows()
        if pd.notna(r.get("파일명"))
    }


def fix_amounts(base: dict, embedding, hwp: str, pdf: str) -> None:
    """기존 인덱스의 저장된 금액을 CSV 기준으로 교정해 다시 저장한다."""
    cfg = copy.deepcopy(base)
    cfg["parser"] = {"hwp": hwp, "pdf": pdf}
    cache_dir = cfg["data"]["faiss_cache"]
    key = make_cache_key(cfg)
    if not cache_exists(cache_dir, key):
        print(f"[건너뜀] {hwp}+{pdf} 인덱스 없음(교정 대상 아님): {key}", flush=True)
        return

    path = Path(cache_dir) / _safe_name(key)
    vs = FAISS.load_local(str(path), embedding,
                          allow_dangerous_deserialization=True)
    amt = _csv_amount_map(cfg["data"]["csv"])

    fixed = 0
    for doc in vs.docstore._dict.values():
        fn = doc.metadata.get("파일명")
        if fn in amt and doc.metadata.get("금액") != amt[fn]:
            doc.metadata["금액"] = amt[fn]
            fixed += 1
    vs.save_local(str(path))
    print(f"[교정] {hwp}+{pdf}: 청크 {fixed}개 금액 수정 후 저장 -> {key}",
          flush=True)


def build(base: dict, embedding, hwp: str, pdf: str) -> None:
    """파서 조합을 파싱+청킹+임베딩해 FAISS 캐시를 만든다(없을 때만)."""
    cfg = copy.deepcopy(base)
    cfg["parser"] = {"hwp": hwp, "pdf": pdf}
    cache_dir = cfg["data"]["faiss_cache"]
    key = make_cache_key(cfg)
    tag = f"{hwp}+{pdf}"
    if cache_exists(cache_dir, key):
        print(f"[건너뜀] {tag} 캐시 존재: {key}", flush=True)
        return

    t0 = time.time()
    print(f"[빌드 시작] {tag} ...", flush=True)
    docs = load_documents(cfg)
    chunks = split_documents(docs, cfg)
    build_vectorstore(chunks, embedding, cache_dir, key)
    print(f"[완료] {tag}: 문서 {len(docs)}건 / 청크 {len(chunks)}개 "
          f"/ {time.time() - t0:.0f}초 -> {key}", flush=True)


def main() -> None:
    base = load_config("config.yaml")
    embedding = make_embedding(base)  # 조합 간 동일 -> 1회 생성해 재사용

    # 1) 기존 pyhwp+pymupdf 금액 교정(임베딩 재생성 없음)
    fix_amounts(base, embedding, "pyhwp", "pymupdf")

    # 2) olefile 조합 신규 빌드
    for hwp, pdf in NEW_COMBOS:
        build(base, embedding, hwp, pdf)

    print("\n모든 작업 완료.", flush=True)


if __name__ == "__main__":
    main()
