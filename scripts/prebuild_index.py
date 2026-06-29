"""청킹 x 임베딩 조합의 FAISS 인덱스를 미리 빌드(워밍업)한다.

UI/실험에서 임베딩이나 청킹을 바꿀 때마다 재인덱싱을 기다리지 않도록,
카탈로그의 모든 (임베딩 x 청킹) 조합에 대해 설정별 FAISS 캐시를 미리 만들어 둔다.
이미 캐시가 있는 조합은 건너뛴다.

비용/시간:
  - 원본 파싱은 1회만 수행하고 모든 조합이 재사용한다.
  - OpenAI 임베딩은 API라 빠르다. HuggingFace(KURE/bge)는 로컬 CPU라 느리다(특히 bge-m3).
  - 그래서 빠른 임베딩부터 빌드해 먼저 쓸 수 있게 한다.

사용:
  python scripts/prebuild_index.py                       # 모든 조합
  python scripts/prebuild_index.py --embeddings openai-3-small,hf-kure-v1
  python scripts/prebuild_index.py --chunks recursive-1000-100
"""

from __future__ import annotations

import sys
from pathlib import Path

# 저장소 루트를 import 경로에 추가한다(scripts 하위에서 from src... 동작).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
from typing import List

from src import catalog
from src.chunking import split_documents
from src.config import load_config, make_embedding
from src.embedding import build_vectorstore, cache_exists, make_cache_key
from src.loader import load_documents

# 빠른(API) 임베딩 -> 느린(로컬 CPU) 임베딩 순서. 카탈로그에 없는 키는 무시된다.
EMBED_ORDER = ["openai-3-small", "openai-3-large", "hf-kure-v1", "hf-bge-m3"]


def _ordered(keys: List[str]) -> List[str]:
    present = [k for k in EMBED_ORDER if k in keys]
    rest = [k for k in keys if k not in EMBED_ORDER]
    return present + rest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, default="",
                        help="쉼표로 구분한 임베딩 프리셋 키(기본: 전체)")
    parser.add_argument("--chunks", type=str, default="",
                        help="쉼표로 구분한 청킹 프리셋 키(기본: 전체)")
    args = parser.parse_args()

    base = load_config("config.yaml")

    emb_keys = ([s.strip() for s in args.embeddings.split(",") if s.strip()]
                or list(catalog.EMBEDDING_PRESETS))
    chunk_keys = ([s.strip() for s in args.chunks.split(",") if s.strip()]
                  or list(catalog.CHUNK_PRESETS))
    emb_keys = _ordered(emb_keys)

    print(f"임베딩 {len(emb_keys)} x 청킹 {len(chunk_keys)} = {len(emb_keys)*len(chunk_keys)}개 조합")
    print("원본 파싱 중(1회)...", flush=True)
    docs = load_documents(base)

    # 청킹은 임베딩과 무관하므로 조합마다 다시 자르지 않도록 미리 만들어 둔다.
    chunks_by_key = {}
    for ck in chunk_keys:
        cfg_c = catalog.apply_preset(base, "chunk", ck)
        chunks_by_key[ck] = split_documents(docs, cfg_c)
        print(f"  청킹 {ck}: {len(chunks_by_key[ck])}개 청크", flush=True)

    built, skipped, failed = 0, 0, 0
    for ek in emb_keys:
        embedding = None  # HF 모델은 임베딩별 1회만 로드해 청킹 간 재사용
        for ck in chunk_keys:
            cfg = catalog.apply_preset(base, "embedding", ek)
            cfg = catalog.apply_preset(cfg, "chunk", ck)
            key = make_cache_key(cfg)
            if cache_exists(cfg["data"]["faiss_cache"], key):
                print(f"[건너뜀] {ek} x {ck} (캐시 있음)", flush=True)
                skipped += 1
                continue
            print(f"[빌드] {ek} x {ck} ...", flush=True)
            t0 = time.time()
            try:
                if embedding is None:
                    embedding = make_embedding(cfg)
                build_vectorstore(
                    chunks_by_key[ck], embedding,
                    cfg["data"]["faiss_cache"], key,
                )
                built += 1
                print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                failed += 1
                print(f"  실패: {e}", flush=True)

    print(f"\n빌드 {built} | 건너뜀 {skipped} | 실패 {failed}")


if __name__ == "__main__":
    main()
