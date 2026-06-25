"""임베딩 모델 × 파서 조합 그리드로 FAISS 인덱스를 빌드한다.

수정된 olefile 파서(서로게이트 제거)와 교정된 금액으로 새로 굽는다.
파서별 파싱/청킹은 1회만 수행하고 임베딩만 바꿔 재사용한다.

대상 그리드(청킹은 recursive 500/50 고정):
  임베딩 : openai-3-small, KURE-v1, bge-m3
  파서   : olefile+pdfplumber, olefile+pymupdf
  (pyhwp+pymupdf × 3임베딩은 이미 정상 빌드되어 있어 건드리지 않는다.)

기존 인덱스는 지우고 새로 굽는다(파서 수정 반영). 사용: python scripts/build_emb_parser_grid.py
"""

from __future__ import annotations

import copy
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import catalog
from src.chunking import split_documents
from src.config import load_config, make_embedding
from src.embedding import _safe_name, build_vectorstore, make_cache_key
from src.loader import load_documents

EMB_PRESETS = ["openai-3-small", "hf-kure-v1", "hf-bge-m3"]
PARSER_COMBOS = [("olefile", "pdfplumber"), ("olefile", "pymupdf")]


def main() -> None:
    base = load_config("config.yaml")
    cache_dir = base["data"]["faiss_cache"]

    # 임베딩 객체는 1회만 생성해 재사용(HF 모델 재로딩 방지).
    embeddings = {
        p: make_embedding(catalog.apply_preset(base, "embedding", p))
        for p in EMB_PRESETS
    }

    for hwp, pdf in PARSER_COMBOS:
        pcfg = copy.deepcopy(base)
        pcfg["parser"] = {"hwp": hwp, "pdf": pdf}
        t0 = time.time()
        print(f"\n[파싱] {hwp}+{pdf} ...", flush=True)
        docs = load_documents(pcfg)
        chunks = split_documents(docs, pcfg)  # 청킹은 임베딩과 무관 -> 1회
        print(f"  문서 {len(docs)}건 / 청크 {len(chunks)}개 / {time.time()-t0:.0f}초",
              flush=True)

        for preset in EMB_PRESETS:
            cfg = catalog.apply_preset(base, "embedding", preset)
            cfg["parser"] = {"hwp": hwp, "pdf": pdf}
            key = make_cache_key(cfg)
            path = Path(cache_dir) / _safe_name(key)
            tag = f"{preset} × {hwp}+{pdf}"
            if path.exists():
                shutil.rmtree(path)  # 파서 수정 반영 위해 항상 새로 굽는다
            t1 = time.time()
            print(f"  [임베딩] {tag} ...", flush=True)
            build_vectorstore(chunks, embeddings[preset], cache_dir, key)
            print(f"  [완료] {tag} / {time.time()-t1:.0f}초 -> {key}", flush=True)

    print("\n그리드 빌드 완료.", flush=True)


if __name__ == "__main__":
    main()
