"""베이스라인 실행 + 골든셋 채점.

config.yaml 로드 -> build_pipeline -> evaluate -> 점수 출력.
이 점수가 이후 모든 실험의 기준선이 된다.

사용:
  python scripts/eval/run_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 저장소 루트를 import 경로에 추가한다(scripts 하위에서 from src... 동작).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any, Dict

from src.config import load_config
from src.evaluation.evaluate import evaluate, load_golden, summary_score
from src.pipeline import build_pipeline


def _print_report(report: Dict[str, Any]) -> None:
    def fmt(x: Any) -> str:
        return f"{x:.3f}" if isinstance(x, (int, float)) else "N/A"

    r = report["retrieval"]
    m = report["metadata_extraction"]
    f = report["refusal"]

    print("\n==================== 베이스라인 평가 결과 ====================")
    print(f"[검색]        대상 {r['n']}건 | Hit@{r['k']}={fmt(r['hit@k'])} | "
          f"MRR={fmt(r['mrr'])}")
    print(f"[메타 추출]   대상 {m['n']}건 | 정확도={fmt(m['accuracy'])}")
    print(f"[거절]        대상 {f['n']}건 | 거절 정확도={fmt(f['refusal_accuracy'])}")

    body = report.get("body_grounded")
    if body:
        print(f"[서술형]      대상 {body['n']}건 | 평균 점수={fmt(body['avg_score'])}")
    else:
        print("[서술형]      judge_llm 미지정으로 생략")

    print("-------------------------------------------------------------")
    print("요약: " + summary_score(report))
    print("=============================================================\n")


def main() -> None:
    cfg = load_config("config.yaml")

    print("[1/3] 파이프라인 조립 중 (로더->청킹->임베딩->검색기->체인)...")
    chain, retriever = build_pipeline(cfg, mode="qa")

    print("[2/3] 골든셋 로드 중...")
    golden = load_golden(cfg["eval"]["golden_set"])

    print(f"[3/3] 평가 실행 중 (골든셋 {len(golden)}건)...")
    report = evaluate(chain, retriever, golden, k=cfg["eval"]["k"])

    _print_report(report)


if __name__ == "__main__":
    main()
