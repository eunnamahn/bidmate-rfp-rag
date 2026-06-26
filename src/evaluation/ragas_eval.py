"""RAGAS + 커스텀 지표 기반 조합 평가.

새 골든셋 포맷(question / ground_truth / sourcefile)을 사용한다.
  - answerable  : 정답이 있는 항목(전부 sourcefile 보유)
  - unanswerable: ground_truth가 "...확인할 수 없습니다"인 항목(거절해야 정답)

지표
  [커스텀]
    - Hit@k, MRR        : 검색 정확도(sourcefile 기준, answerable)
    - 사실추출 정확도    : 정답 문자열이 답변에 포함되는지(substring, answerable)
    - 거절 정확도        : unanswerable을 올바르게 거절했는지
  [RAGAS] (answerable 대상, 고정 judge LLM/임베딩으로 채점해 조합 간 공정 비교)
    - faithfulness       : 답변이 검색 컨텍스트에 근거하는가
    - context_precision  : 검색 컨텍스트가 정답과 관련 있는가
    - context_recall     : 검색 컨텍스트가 정답을 포괄하는가
    - answer_correctness : 답변이 정답과 일치하는가
"""

from __future__ import annotations

import sys
import types

# ragas 0.4.3은 langchain v1에서 제거된 langchain_community.chat_models.vertexai를
# 모듈 로드 시점에 임포트한다. VertexAI는 사용하지 않으므로 스텁을 주입해 통과시킨다.
_VERTEX = "langchain_community.chat_models.vertexai"
if _VERTEX not in sys.modules:
    _stub = types.ModuleType(_VERTEX)
    class ChatVertexAI:  # noqa: D401,E701 - import 통과용 더미
        ...
    _stub.ChatVertexAI = ChatVertexAI
    sys.modules[_VERTEX] = _stub

import contextlib
import json
import math
import re
import time
from typing import Any, Dict, List, Optional

from src.generator import generate_answer

# OpenAI 토큰/비용 집계용. 없거나 비-OpenAI(예: Ollama)면 0으로 처리한다.
try:
    from langchain_community.callbacks.manager import get_openai_callback
except Exception:  # pragma: no cover
    get_openai_callback = None

REFUSAL_MARK = "확인할 수 없습니다"


def load_golden(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_unanswerable(item: Dict[str, Any]) -> bool:
    return REFUSAL_MARK in str(item.get("ground_truth", ""))


def _norm(text: Any) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def _avg(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else None


def generate_records(retriever, llm, golden: List[Dict[str, Any]],
                     mode: str = "qa") -> List[Dict[str, Any]]:
    """각 골든 항목에 대해 (답변, 검색 컨텍스트, 검색 파일목록)을 생성한다."""
    records: List[Dict[str, Any]] = []
    total = len(golden)
    for i, g in enumerate(golden, 1):
        # 실시간 진행 표시: 같은 줄을 덮어쓴다(\r). 직접 실행 시 답변 생성 진척을 본다.
        print(f"\r  답변 생성 {i}/{total}", end="", flush=True)
        q = g["question"]
        docs = retriever.invoke(q)  # Hit@k/컨텍스트용 top-k (결정적)
        # 답변 생성 구간만 지연시간으로 측정하고, OpenAI 토큰/비용을 집계한다.
        cb_ctx = get_openai_callback() if get_openai_callback else contextlib.nullcontext()
        t0 = time.perf_counter()
        with cb_ctx as cb:
            answer, _ = generate_answer(retriever, llm, q, mode=mode)
        latency = time.perf_counter() - t0
        records.append({
            "question": q,
            "answer": answer,
            "contexts": [d.page_content for d in docs],
            "retrieved_files": [d.metadata.get("파일명") for d in docs],
            "ground_truth": g.get("ground_truth", ""),
            "sourcefile": g.get("sourcefile"),
            "unanswerable": is_unanswerable(g),
            "latency_s": latency,
            "answer_len": len(answer or ""),
            "in_tokens": getattr(cb, "prompt_tokens", 0) if cb else 0,
            "out_tokens": getattr(cb, "completion_tokens", 0) if cb else 0,
            "cost_usd": getattr(cb, "total_cost", 0.0) if cb else 0.0,
        })
    print(f"\r  답변 생성 {total}/{total} 완료. RAGAS 채점 중...", flush=True)
    return records


def operational_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """운영 지표: 평균 답변 길이(자), 평균 지연(초), 토큰 총량, 총비용($)."""
    return {
        "avg_answer_len": _avg([r["answer_len"] for r in records]),
        "avg_latency_s": _avg([r["latency_s"] for r in records]),
        "total_in_tokens": sum(r["in_tokens"] for r in records),
        "total_out_tokens": sum(r["out_tokens"] for r in records),
        "total_cost_usd": sum(r["cost_usd"] for r in records),
    }


def custom_metrics(records: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    """검색/사실추출/거절 등 규칙 기반 지표."""
    answerable = [r for r in records if not r["unanswerable"]]
    unanswerable = [r for r in records if r["unanswerable"]]

    hits: List[float] = []
    rrs: List[float] = []
    for r in (r for r in answerable if r.get("sourcefile")):
        files = r["retrieved_files"][:k]
        src = r["sourcefile"]
        hits.append(1.0 if src in files else 0.0)
        rr = 0.0
        for i, f in enumerate(files):
            if f == src:
                rr = 1.0 / (i + 1)
                break
        rrs.append(rr)

    substr = [
        1.0 if _norm(r["ground_truth"]) and _norm(r["ground_truth"]) in _norm(r["answer"])
        else 0.0
        for r in answerable
    ]
    refusal = [1.0 if REFUSAL_MARK in r["answer"] else 0.0 for r in unanswerable]

    return {
        "hit@k": _avg(hits),
        "mrr": _avg(rrs),
        "substring_acc": _avg(substr),
        "refusal_acc": _avg(refusal),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "k": k,
    }


def _set_judge_temperature(judge_llm) -> None:
    """judge LLM의 채점 온도를 모델 제약에 맞춰 설정한다.

    GPT-5 계열은 temperature=1만 허용하므로 1로 둔다(RAGAS의 온도 덮어쓰기는
    bypass_temperature=True로 막는다). 그 외 모델(EXAONE 등 Ollama 포함)은
    채점 결정성을 위해 0으로 둔다.
    """
    model = str(
        getattr(judge_llm, "model_name", None)
        or getattr(judge_llm, "model", "")
        or ""
    )
    temp = 1 if "gpt-5" in model.lower() else 0
    try:
        judge_llm.temperature = temp
    except Exception:
        pass


def ragas_metrics(records: List[Dict[str, Any]], judge_llm, judge_emb,
                  metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """answerable 항목에 대해 RAGAS 지표를 고정 judge로 채점한다.

    metric_names로 일부 지표만 채점할 수 있다(예: 외부 시스템 답변 평가 시
    검색 컨텍스트가 없으면 answer_correctness만 선택). 기본은 4종 전부.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_correctness, context_precision,
                               context_recall, faithfulness)

    _ALL = {
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    names = metric_names or list(_ALL)

    items = [r for r in records if not r["unanswerable"]]
    if not items:
        return {}
    data = [{
        "user_input": r["question"],
        "retrieved_contexts": r["contexts"] or [""],
        "response": r["answer"] or "",
        "reference": r["ground_truth"] or "",
    } for r in items]

    # 채점 온도를 모델에 맞춰 설정한다(GPT-5는 1, EXAONE 등은 0). RAGAS는 채점 호출에
    # temperature=0.01을 강제하므로 bypass_temperature=True로 덮어쓰기를 막는다.
    _set_judge_temperature(judge_llm)

    # answer_relevancy는 사실추출 단답을 구조적으로 저평가해 제외한다(정답일수록 손해).
    metrics = [_ALL[n] for n in names if n in _ALL]
    result = evaluate(
        EvaluationDataset.from_list(data),
        metrics=metrics,
        llm=LangchainLLMWrapper(judge_llm, bypass_temperature=True),
        embeddings=LangchainEmbeddingsWrapper(judge_emb),
        raise_exceptions=False,
        show_progress=True,  # 실시간 진행 막대(직접 실행 시 채점 진척 확인)
    )
    df = result.to_pandas()
    out: Dict[str, Any] = {}
    for name in names:
        if name in df.columns:
            out[name] = _avg(list(df[name]))
    return out


def judge_descriptive(records: List[Dict[str, Any]], judge_llm) -> Optional[float]:
    """answerable 답변을 judge_llm으로 0~1 채점한 평균(서술형 사실일치).

    RAGAS와 별개로, 답변이 ground_truth와 사실적으로 맞는지 LLM 심판이 단일 점수로
    매긴다(옛 evaluate.py의 body_grounded avg_score를 새 골든셋 스키마에 이식).
    골든셋에 type 필드가 없으므로 answerable 항목 전체를 대상으로 한다.
    """
    items = [r for r in records if not r["unanswerable"]]
    if not items:
        return None
    # 채점 온도를 모델에 맞춰 보정한다(ragas_metrics와 동일: GPT-5는 1, 그 외는 0).
    _set_judge_temperature(judge_llm)
    scores: List[float] = []
    for r in items:
        prompt = (
            "다음 질문에 대한 모델 답변이 참고 정답과 사실적으로 일치하는지"
            " 0.0~1.0 사이 숫자 하나로만 평가하라.\n"
            f"질문: {r['question']}\n참고 정답: {r['ground_truth']}\n"
            f"모델 답변: {r['answer']}\n점수:"
        )
        raw = judge_llm.invoke(prompt)
        text = getattr(raw, "content", raw)
        m = re.search(r"[01](?:\.\d+)?", str(text))
        scores.append(float(m.group()) if m else 0.0)
    return _avg(scores)


def evaluate_config(retriever, llm, golden, judge_llm=None, judge_emb=None,
                    k: int = 4, mode: str = "qa") -> Dict[str, Any]:
    """한 조합(retriever+llm)의 답변을 생성하고 전체 지표를 계산한다."""
    records = generate_records(retriever, llm, golden, mode=mode)
    report = {"custom": custom_metrics(records, k),
              "operational": operational_metrics(records)}
    # judge가 없으면 RAGAS(LLM 채점) 단계는 건너뛴다(커스텀+운영 지표만).
    report["ragas"] = (ragas_metrics(records, judge_llm, judge_emb)
                       if judge_llm is not None and judge_emb is not None else {})
    # 서술형 LLM 심판 점수(avg_score). judge_llm만 있으면 RAGAS 없이도 계산한다.
    report["judge_score"] = (judge_descriptive(records, judge_llm)
                             if judge_llm is not None else None)
    report["records"] = records
    return report


def summary_row(report: Dict[str, Any]) -> Dict[str, Any]:
    """비교표 한 줄로 쓸 핵심 지표만 평탄화한다."""
    c = report.get("custom", {})
    r = report.get("ragas", {})
    row = {
        f"Hit@{c.get('k')}": c.get("hit@k"),
        "MRR": c.get("mrr"),
        "사실추출": c.get("substring_acc"),
        "거절": c.get("refusal_acc"),
    }
    row.update({
        "faith": r.get("faithfulness"),
        "ans_rel": r.get("answer_relevancy"),
        "ctx_prec": r.get("context_precision"),
        "ctx_rec": r.get("context_recall"),
        "ans_corr": r.get("answer_correctness"),
    })
    return row
