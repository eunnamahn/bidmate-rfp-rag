"""조합별 RAGAS + 커스텀 지표 평가 러너.

LLM / 임베딩 / 파서 / 청킹 전략을 바꿔가며 같은 골든셋으로 답변을 생성하고,
RAGAS 5종 + 커스텀 4종 지표를 비교표로 출력/저장한다.

공정 비교를 위해 RAGAS 채점기(judge LLM/임베딩)는 조합과 무관하게 고정한다.
같은 파서를 쓰는 실험은 원본 파싱을 1회만 수행해 재사용한다.
임베딩/청킹/파서가 바뀌면 FAISS는 설정별 캐시로 분리된다(캐시 있으면 즉시 로드).

사용:
  python scripts/eval/run_ragas_experiments.py                  # 기본 조합 전부
  python scripts/eval/run_ragas_experiments.py --only baseline,llm_mini
  python scripts/eval/run_ragas_experiments.py --limit 6        # 질문 6개만(저비용 점검)
  python scripts/eval/run_ragas_experiments.py --out results.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# 저장소 루트를 import 경로에 추가한다(scripts 하위에서 from src... 동작).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import copy
import itertools
import json
from typing import Any, Dict, List, Optional, Tuple

from src import catalog
from src.config import load_config, make_embedding, make_llm
from src.evaluation.ragas_eval import evaluate_config, load_golden
from src.loader import load_documents
from src.pipeline import build_pipeline

# 18개 조합 = 청킹(1000) 2종 × 임베딩 3종 × 답변 LLM 3종.
# 청킹/임베딩 변형 인덱스는 olefile+pdfplumber 파서로 빌드돼 있어 그 파서에 고정한다.
# parser는 dict(hwp/pdf), 나머지는 catalog 프리셋 키.
_OLE = {"hwp": "olefile", "pdf": "pdfplumber"}

# (프리셋 키 -> 실험 이름에 쓸 짧은 별칭)
_CHUNK = {"recursive-1000-100": "rec1000", "section-1000-100": "sec1000"}
_EMB = {"openai-3-small": "oa", "hf-kure-v1": "kure", "hf-bge-m3": "bge"}
_ANSWER_LLM = {"openai-gpt-5-mini": "mini",
               "ollama-exaone3.5-7.8b": "exaone",
               "ollama-qwen2.5-14b": "qwen"}

EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": f"{cs}_{es}_{ls}",
        "desc": f"olefile+pdfplumber·{cs}·{es}·{ls}",
        "parser": _OLE,
        "chunk": ckey,
        "embedding": ekey,
        "llm": lkey,
    }
    for (ckey, cs), (ekey, es), (lkey, ls) in itertools.product(
        _CHUNK.items(), _EMB.items(), _ANSWER_LLM.items()
    )
]

# RAGAS 채점기(조합과 무관하게 고정). 사용자 요청으로 nano 사용.
# 주의: nano는 저렴하나 채점 안정성은 mini가 낫다(점수 신뢰도 감안).
JUDGE_LLM_PRESET = "openai-gpt-5-nano"


def _apply(base: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    if "llm" in exp:
        cfg = catalog.apply_preset(cfg, "llm", exp["llm"])
    if "embedding" in exp:
        cfg = catalog.apply_preset(cfg, "embedding", exp["embedding"])
    if "chunk" in exp:
        cfg = catalog.apply_preset(cfg, "chunk", exp["chunk"])
    if "parser" in exp:
        cfg.setdefault("parser", {}).update(exp["parser"])
    return cfg


def _parser_sig(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    p = cfg.get("parser", {}) or {}
    return (p.get("hwp"), p.get("pdf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="앞에서부터 N개 질문만 평가(저비용 점검용)")
    ap.add_argument("--out", type=str, default="ragas_results.json")
    ap.add_argument("--no-judge", action="store_true",
                    help="RAGAS(LLM 채점) 건너뛰고 커스텀+운영 지표만 계산")
    args = ap.parse_args()

    base = load_config("config.yaml")
    golden = load_golden(base["eval"]["golden_set"])
    if args.limit > 0:
        golden = golden[:args.limit]
    k = base["eval"]["k"]

    # 고정 judge: 조합과 무관하게 동일 채점기로 RAGAS 점수를 매긴다.
    # --no-judge면 RAGAS를 건너뛴다(채점기 생성도 생략).
    judge_llm = judge_emb = None
    if not args.no_judge:
        judge_llm = make_llm(catalog.apply_preset(base, "llm", JUDGE_LLM_PRESET))
        judge_emb = make_embedding(base)  # answer_relevancy용(openai-3-small 고정)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    experiments = [e for e in EXPERIMENTS if not only or e["name"] in only]

    results: Dict[str, Dict[str, Any]] = {}
    rows: List[Tuple[str, str, Dict[str, Any]]] = []

    for exp in experiments:
        cfg = _apply(base, exp)
        print(f"\n[{exp['name']}] {exp['desc']} 평가 중...", flush=True)
        try:
            # 모든 인덱스는 사전 빌드되어 있으므로 캐시 로드만 한다(원본 재파싱 없음).
            _, retriever = build_pipeline(cfg, mode="qa")
            llm = make_llm(cfg)
            report = evaluate_config(retriever, llm, golden,
                                     judge_llm=judge_llm, judge_emb=judge_emb,
                                     k=k)
            rows.append((exp["name"], report))
            results[exp["name"]] = {"desc": exp["desc"],
                                    "custom": report["custom"],
                                    "operational": report["operational"],
                                    "ragas": report["ragas"],
                                    "judge_score": report.get("judge_score")}
        except Exception as e:
            rows.append((exp["name"], {"error": str(e)}))
            results[exp["name"]] = {"desc": exp["desc"], "error": str(e)}

    _print_tables(rows)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n상세 결과 저장: {args.out}")


def _table(rows, name_w, title, cols):
    """rows=[(name, report)], cols=[(헤더, 값추출함수, 포맷함수)]."""
    header = f"{'실험':<{name_w}} | " + " | ".join(f"{c[0]:>9}" for c in cols)
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, rep in rows:
        if "error" in rep:
            print(f"{name:<{name_w}} | 실패: {rep['error'][:60]}")
            continue
        cells = " | ".join(f"{c[2](c[1](rep)):>9}" for c in cols)
        print(f"{name:<{name_w}} | {cells}")
    print("=" * len(header))


def _print_tables(rows) -> None:
    name_w = max((len(r[0]) for r in rows), default=8)
    f3 = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "
    fi = lambda x: f"{int(x):,}" if isinstance(x, (int, float)) else "  -  "
    f1 = lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else "  -  "
    f4 = lambda x: f"${x:.4f}" if isinstance(x, (int, float)) else "  -  "

    def cu(key):  # custom
        return lambda rep: rep.get("custom", {}).get(key)

    def rg(key):  # ragas
        return lambda rep: rep.get("ragas", {}).get(key)

    def op(key):  # operational
        return lambda rep: rep.get("operational", {}).get(key)

    js = lambda rep: rep.get("judge_score")  # 서술형 LLM 심판 평균(0~1)

    # 1) 품질 지표 (answer_relevancy는 단답형 정답 저평가로 제외)
    _table(rows, name_w, "■ 품질 지표 (검색·정확성·환각·거절)", [
        ("Hit@4", cu("hit@k"), f3), ("MRR", cu("mrr"), f3),
        ("사실추출", cu("substring_acc"), f3), ("거절", cu("refusal_acc"), f3),
        ("서술심판", js, f3),
        ("faith", rg("faithfulness"), f3),
        ("ctx_prec", rg("context_precision"), f3),
        ("ctx_rec", rg("context_recall"), f3),
        ("ans_corr", rg("answer_correctness"), f3),
    ])
    # 2) 운영 지표
    _table(rows, name_w, "■ 운영 지표 (길이·지연·토큰·비용)", [
        ("답변길이", op("avg_answer_len"), f1),
        ("지연(초)", op("avg_latency_s"), f1),
        ("입력토큰", op("total_in_tokens"), fi),
        ("출력토큰", op("total_out_tokens"), fi),
        ("총비용", op("total_cost_usd"), f4),
    ])


if __name__ == "__main__":
    main()
