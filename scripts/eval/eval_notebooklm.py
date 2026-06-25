"""NotebookLM 답변을 기존 실험과 같은 지표로 평가한다.

NotebookLM은 외부 시스템이라 '답변'만 있고 우리 검색 컨텍스트가 없다. 따라서
검색/컨텍스트에 의존하는 지표(Hit@k·MRR·faithfulness·context_precision/recall)는
산출할 수 없고(N/A), 컨텍스트가 필요 없는 답변 품질 지표만 동일 방식으로 채점한다:

  - 사실추출(substring_acc) : 정답 문자열이 답변에 포함되는가          [custom]
  - 거절(refusal_acc)       : unanswerable을 올바르게 거절했는가        [custom]
  - 서술심판(judge_score)    : judge LLM의 0~1 사실일치 점수             [LLM judge]
  - ans_corr(answer_correctness) : RAGAS 답변 정확성                     [RAGAS]

judge는 run_ragas_experiments.py와 동일하게 고정한다(openai-gpt-5-nano + openai-3-small)
→ 18개 조합 실험 결과와 같은 채점기라 직접 비교 가능.

입력:
  - src/evaluation/golden_set.json           (question / ground_truth / sourcefile)
  - src/evaluation/NotebookLM_golden_set.json (question / notebooklm_answer / sourcefile)
  두 파일을 question 기준으로 조인한다.

사용:
  python scripts/eval/eval_notebooklm.py                 # 전체 50문항
  python scripts/eval/eval_notebooklm.py --limit 6       # 앞 6문항만(저비용 점검)
  python scripts/eval/eval_notebooklm.py --no-judge      # RAGAS/서술심판 건너뜀(사실추출·거절만)
  python scripts/eval/eval_notebooklm.py --out notebooklm_eval.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# 저장소 루트를 import 경로에 추가한다(scripts 하위에서 from src... 동작).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import re
from typing import Any, Dict, List

from src import catalog
from src.config import load_config, make_embedding, make_llm
from src.evaluation.ragas_eval import (
    custom_metrics,
    is_unanswerable,
    judge_descriptive,
    load_golden,
    ragas_metrics,
)

# run_ragas_experiments.py와 동일한 고정 judge (비교 가능성 보장).
JUDGE_LLM_PRESET = "openai-gpt-5-nano"

NOTEBOOKLM_PATH = "src/evaluation/NotebookLM_golden_set.json"


def build_records(golden: List[Dict[str, Any]],
                  nlm_path: str) -> List[Dict[str, Any]]:
    """golden_set(ground_truth) + NotebookLM(answer)를 question 기준으로 조인한다.

    NotebookLM은 외부 답변이므로 검색 컨텍스트가 없다(contexts/retrieved_files 비움).
    """
    with open(nlm_path, "r", encoding="utf-8") as f:
        nlm = json.load(f)
    answer_by_q = {x["question"]: x.get("notebooklm_answer", "") for x in nlm}

    records: List[Dict[str, Any]] = []
    missing = 0
    for g in golden:
        q = g["question"]
        if q not in answer_by_q:
            missing += 1
            continue
        records.append({
            "question": q,
            "answer": answer_by_q[q] or "",
            "contexts": [],            # 외부 시스템 → 우리 검색 컨텍스트 없음
            "retrieved_files": [],     # Hit@k/MRR 산출 불가(N/A)
            "ground_truth": g.get("ground_truth", ""),
            "sourcefile": g.get("sourcefile"),
            "unanswerable": is_unanswerable(g),
        })
    if missing:
        print(f"[경고] golden 질문 {missing}개가 NotebookLM 파일에 없어 제외됨")
    return records


# NotebookLM은 거절을 '확인할 수 없습니다' 대신 아래처럼 표현한다.
# 같은 지표(refusal_acc)와 별개로, 표기 차이를 보정한 보조 거절율을 함께 본다.
_LENIENT_REFUSAL = re.compile(
    r"(확인할 수 없|명시되어 있지 않|기재되어 있지 않|포함되어 있지 않|"
    r"제공되지 않|찾을 수 없|나타나 있지 않|언급되어 있지 않|정보가 없)"
)


def _lenient_refusal_acc(records) -> Any:
    """표기 차이를 보정한 거절율(보조 지표 — 실험 표의 refusal_acc와는 다름)."""
    unans = [r for r in records if r["unanswerable"]]
    if not unans:
        return None
    hit = sum(1 for r in unans if _LENIENT_REFUSAL.search(r["answer"] or ""))
    return hit / len(unans)


def evaluate_notebooklm(records, judge_llm, judge_emb, k: int) -> Dict[str, Any]:
    """NotebookLM 답변에 대해 컨텍스트 비의존 지표만 계산한다."""
    cm = custom_metrics(records, k)
    report: Dict[str, Any] = {
        "substring_acc": cm.get("substring_acc"),   # 사실추출
        "refusal_acc": cm.get("refusal_acc"),       # 거절(동일 지표)
        "refusal_acc_lenient": _lenient_refusal_acc(records),  # 거절(표기 보정·보조)
        "n_answerable": cm.get("n_answerable"),
        "n_unanswerable": cm.get("n_unanswerable"),
        # 검색 의존 지표는 외부 시스템이라 산출 불가
        "hit@k": None, "mrr": None,
        "faithfulness": None, "context_precision": None, "context_recall": None,
    }
    if judge_llm is not None:
        report["judge_score"] = judge_descriptive(records, judge_llm)  # 서술심판
        if judge_emb is not None:
            ac = ragas_metrics(records, judge_llm, judge_emb,
                               metric_names=["answer_correctness"])
            report["answer_correctness"] = ac.get("answer_correctness")
    return report


def _f3(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "   N/A"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="앞에서부터 N개 질문만 평가(저비용 점검)")
    ap.add_argument("--no-judge", action="store_true",
                    help="RAGAS/서술심판 건너뛰고 사실추출·거절만 계산")
    ap.add_argument("--out", type=str, default="notebooklm_eval.json")
    args = ap.parse_args()

    base = load_config("config.yaml")
    golden = load_golden(base["eval"]["golden_set"])
    if args.limit > 0:
        golden = golden[:args.limit]
    k = base["eval"]["k"]

    records = build_records(golden, NOTEBOOKLM_PATH)

    judge_llm = judge_emb = None
    if not args.no_judge:
        judge_llm = make_llm(catalog.apply_preset(base, "llm", JUDGE_LLM_PRESET))
        judge_emb = make_embedding(base)  # answer_correctness 의미유사도용(openai-3-small)

    print(f"\n[NotebookLM] {len(records)}문항 평가 중 "
          f"(answerable {sum(not r['unanswerable'] for r in records)} / "
          f"unanswerable {sum(r['unanswerable'] for r in records)})...", flush=True)
    report = evaluate_notebooklm(records, judge_llm, judge_emb, k)

    # 비교표: 실험 표와 같은 컬럼 순서. 검색 의존 지표는 N/A로 표기.
    print("\n■ NotebookLM 품질 지표 (18조합 실험과 동일 지표·동일 judge)")
    print("=" * 64)
    print(f"{'지표':<14} | {'값':>8}   비고")
    print("-" * 64)
    print(f"{'Hit@'+str(k):<14} | {_f3(report['hit@k']):>8}   외부 검색 → 우리 컨텍스트 없음")
    print(f"{'MRR':<14} | {_f3(report['mrr']):>8}   외부 검색 → 우리 컨텍스트 없음")
    print(f"{'사실추출':<13} | {_f3(report['substring_acc']):>8}   정답 문자열 포함율")
    print(f"{'거절':<14} | {_f3(report['refusal_acc']):>8}   '확인할 수 없습니다' 표기 기준(동일 지표)")
    print(f"{'└ 거절(보정)':<13} | {_f3(report.get('refusal_acc_lenient')):>8}   '명시되어 있지 않음' 등 포함(보조)")
    print(f"{'서술심판':<13} | {_f3(report.get('judge_score')):>8}   judge LLM 0~1 사실일치")
    print(f"{'faithfulness':<14} | {_f3(report['faithfulness']):>8}   컨텍스트 없음 → N/A")
    print(f"{'context_prec':<14} | {_f3(report['context_precision']):>8}   컨텍스트 없음 → N/A")
    print(f"{'context_rec':<14} | {_f3(report['context_recall']):>8}   컨텍스트 없음 → N/A")
    print(f"{'ans_corr':<14} | {_f3(report.get('answer_correctness')):>8}   RAGAS 답변 정확성")
    print("=" * 64)
    print(f"answerable {report['n_answerable']} · unanswerable {report['n_unanswerable']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"system": "NotebookLM", "judge": JUDGE_LLM_PRESET,
                   "report": report}, f, ensure_ascii=False, indent=2)
    print(f"\n상세 결과 저장: {args.out}")


if __name__ == "__main__":
    main()
