"""인덱싱 엔트리포인트: 원본 파싱 -> 청킹 -> 임베딩 -> FAISS 저장.

FAISS 저장본(캐시)을 미리 만들어 두면 백엔드/평가/질의가 빠르게 로드한다.
이미 캐시가 있으면 임베딩을 다시 하지 않고 로드만 한다.

사용:
  python main.py
"""

from __future__ import annotations

from src.config import load_config
from src.pipeline import build_components


def main() -> None:
    cfg = load_config("config.yaml")
    print("인덱싱 시작 (파싱 -> 청킹 -> 임베딩 -> FAISS 저장)...", flush=True)

    docs, chunks, vectorstore = build_components(cfg)

    parsed = sum(1 for d in docs if d.page_content.strip())
    scanned = sum(1 for d in docs if d.metadata.get("likely_scanned"))
    fallback = sum(1 for d in docs if d.metadata.get("fallback_used"))

    print("------------------------------------------------------------", flush=True)
    print(f"문서 {len(docs)}건 | 본문 확보 {parsed}건 | 청크 {len(chunks)}개", flush=True)
    print(f"품질 플래그: likely_scanned {scanned}건 | fallback_used {fallback}건",
          flush=True)
    print(f"벡터 수: {vectorstore.index.ntotal}", flush=True)
    print(f"FAISS 저장: {cfg['data']['faiss_cache']}/{cfg['embedding']['model']}",
          flush=True)
    print("인덱싱 완료.", flush=True)


if __name__ == "__main__":
    main()
