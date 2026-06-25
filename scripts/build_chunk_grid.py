"""olefile+pdfplumber 파서에 대해 청킹×임베딩 조합 인덱스를 빌드한다.

대상(파서 = olefile+pdfplumber 고정):
  청킹   : recursive 1000/100, section 500/50
  임베딩 : openai-3-small, KURE-v1, bge-m3
  => 2 청킹 × 3 임베딩 = 6칸

- 파싱은 1회만(파서 동일), 청킹은 청크전략별 1회만 수행해 재사용한다.
- 임베딩 모델은 '한 번에 하나만' GPU에 올린다(KURE/bge 동시 상주로 인한 VRAM
  경합을 피해 속도 확보). 각 임베딩 처리 후 메모리를 비운다.
- HF 임베딩은 배치 진행바(show_progress)를 켠다.
- 이미 만들어진 칸(index.faiss 존재)은 건너뛴다(--force로 강제 재빌드).

사용: python scripts/build_chunk_grid.py [--force]
"""

from __future__ import annotations

import copy
import gc
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

PARSER = {"hwp": "olefile", "pdf": "pdfplumber"}
CHUNK_PRESETS = ["recursive-1000-100", "section-500-50", "section-1000-100"]
EMB_PRESETS = ["openai-3-small", "hf-kure-v1", "hf-bge-m3"]


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    force = "--force" in sys.argv
    base = load_config("config.yaml")
    cache_dir = base["data"]["faiss_cache"]

    # 파서 동일 -> 파싱 1회, 청킹은 전략별 1회 미리 계산해 재사용
    pcfg = copy.deepcopy(base)
    pcfg["parser"] = dict(PARSER)
    t0 = time.time()
    print(f"[파싱] {PARSER['hwp']}+{PARSER['pdf']} ...", flush=True)
    docs = load_documents(pcfg)
    print(f"  문서 {len(docs)}건 / {time.time()-t0:.0f}초", flush=True)

    chunks_by = {}
    for cp in CHUNK_PRESETS:
        ccfg = catalog.apply_preset(base, "chunk", cp)
        ccfg["parser"] = dict(PARSER)
        t1 = time.time()
        chunks_by[cp] = split_documents(docs, ccfg)
        print(f"[청킹] {cp}: 청크 {len(chunks_by[cp])}개 / {time.time()-t1:.0f}초",
              flush=True)

    # 임베딩 1개씩 로드(VRAM 경합 회피) -> 해당 임베딩의 청킹 칸들을 모두 빌드
    for preset in EMB_PRESETS:
        cfg0 = catalog.apply_preset(base, "embedding", preset)
        embedding = make_embedding(cfg0)
        if hasattr(embedding, "show_progress"):
            embedding.show_progress = True  # 배치 진행바 켜기
        print(f"\n=== 임베딩 로드: {preset} ===", flush=True)

        for cp in CHUNK_PRESETS:
            cfg = catalog.apply_preset(cfg0, "chunk", cp)
            cfg["parser"] = dict(PARSER)
            key = make_cache_key(cfg)
            path = Path(cache_dir) / _safe_name(key)
            tag = f"{preset} × {PARSER['hwp']}+{PARSER['pdf']} × {cp}"
            if (path / "index.faiss").exists() and not force:
                print(f"  [건너뜀] {tag} (이미 존재)", flush=True)
                continue
            if path.exists():
                shutil.rmtree(path)
            t2 = time.time()
            print(f"  [임베딩] {tag} ...", flush=True)
            build_vectorstore(chunks_by[cp], embedding, cache_dir, key)
            print(f"  [완료] {tag} / {time.time()-t2:.0f}초", flush=True)

        del embedding
        _free_gpu()

    print("\n청킹×임베딩 그리드 빌드 완료.", flush=True)


if __name__ == "__main__":
    main()
