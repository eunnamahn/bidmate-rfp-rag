"""실험 비교 러너: 한 번에 한 축만 바꿔 골든셋 점수를 표로 비교한다.

미션상 "다양한 모델로 실험하고 의사결정 과정을 보고서/발표에 드러내기"가 채점 대상이므로,
config 교체 축(파서/청킹/임베딩/LLM/검색)을 한 축씩 바꿔가며 같은 골든셋으로 채점하고
한 줄 지표를 표로 출력한다. 이 표가 곧 보고서의 비교 근거가 된다.

같은 파서를 쓰는 실험은 원본 파싱(가장 느린 단계)을 1회만 수행하고 재사용한다.
임베딩/청킹이 바뀌면 FAISS는 설정별 캐시로 분리되어 재인덱싱은 처음 1회만 든다.

사용:
  python run_experiments.py                 # 아래 EXPERIMENTS 전부 실행
  python run_experiments.py --judge         # 서술형(judge_llm) 채점 포함
  python run_experiments.py --only baseline,chunk_1000   # 일부만
"""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List, Optional, Tuple

from src import catalog
from src.config import load_config, make_llm
from src.evaluation.evaluate import evaluate, load_golden, summary_score
from src.loader import load_documents
from src.pipeline import build_pipeline

# 실험 정의: 한 축만 바꾼다. overrides는 catalog 프리셋 키(llm/embedding/chunk) 또는
# retriever 설정(use_bm25/k)을 받는다. baseline은 현재 config.yaml 그대로.
EXPERIMENTS: List[Dict[str, Any]] = [
    {"name": "baseline", "desc": "현재 config 그대로"},
    {"name": "chunk_1000", "desc": "청크 1000/100", "chunk": "recursive-1000-100"},
    {"name": "hwp_table", "desc": "HWP 표 보존 파서", "parser": {"hwp": "hwp-hwpx-parser"}},
    {"name": "emb_kure", "desc": "임베딩 KURE-v1", "embedding": "hf-kure-v1"},
    {"name": "emb_bge_m3", "desc": "임베딩 bge-m3", "embedding": "hf-bge-m3"},
    {"name": "llm_mini", "desc": "LLM gpt-5-mini", "llm": "openai-gpt-5-mini"},
    {"name": "hybrid_bm25", "desc": "BM25 하이브리드", "use_bm25": True},
]


def _apply(base: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    """실험 정의를 base config에 반영한 새 cfg를 만든다."""
    cfg = copy.deepcopy(base)
    if "llm" in exp:
        cfg = catalog.apply_preset(cfg, "llm", exp["llm"])
    if "embedding" in exp:
        cfg = catalog.apply_preset(cfg, "embedding", exp["embedding"])
    if "chunk" in exp:
        cfg = catalog.apply_preset(cfg, "chunk", exp["chunk"])
    if "parser" in exp:
        cfg.setdefault("parser", {}).update(exp["parser"])
    if "use_bm25" in exp:
        cfg["retriever"]["use_bm25"] = exp["use_bm25"]
    return cfg


def _parser_sig(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    p = cfg.get("parser", {}) or {}
    return (p.get("hwp"), p.get("pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true",
                        help="서술형(body_grounded)을 judge_llm으로 채점")
    parser.add_argument("--only", type=str, default="",
                        help="쉼표로 구분한 실험 이름만 실행")
    args = parser.parse_args()

    base = load_config("config.yaml")
    golden = load_golden(base["eval"]["golden_set"])
    k = base["eval"]["k"]
    judge_llm = make_llm(base) if args.judge else None

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    experiments = [e for e in EXPERIMENTS if not only or e["name"] in only]

    # 같은 파서를 쓰는 실험끼리 파싱 결과를 재사용한다.
    docs_cache: Dict[Tuple, List] = {}
    rows: List[Tuple[str, str, str]] = []

    for exp in experiments:
        cfg = _apply(base, exp)
        sig = _parser_sig(cfg)
        print(f"\n[{exp['name']}] {exp['desc']} 실행 중...", flush=True)
        try:
            if sig not in docs_cache:
                docs_cache[sig] = load_documents(cfg)
            docs = docs_cache[sig]
            chain, retriever = build_pipeline(cfg, docs=docs, mode="qa")
            report = evaluate(chain, retriever, golden, k=k, judge_llm=judge_llm)
            rows.append((exp["name"], exp["desc"], summary_score(report)))
        except Exception as e:
            # 미설치 패키지(로컬 임베딩 등)/키 권한 문제는 해당 실험만 건너뛴다.
            rows.append((exp["name"], exp["desc"], f"실패: {e}"))

    print("\n==================== 실험 비교표 ====================")
    name_w = max(len(r[0]) for r in rows)
    for name, desc, score in rows:
        print(f"{name:<{name_w}}  | {desc:<18} | {score}")
    print("=====================================================")
    print("주의: 로컬 임베딩(KURE/bge) 실험은 langchain-huggingface + "
          "sentence-transformers 설치가 필요하다.")


if __name__ == "__main__":
    main()
