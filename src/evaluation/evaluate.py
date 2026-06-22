"""골든셋 기반 평가 모듈.

plan 7절 인터페이스:
  evaluate(chain, retriever, golden, k, judge_llm=None) -> report(dict)
  summary_score(report) -> str

골든셋 질문 유형:
  - metadata_extraction : 본문에서 금액/마감일 등 추출, 정답은 메타데이터
  - body_grounded       : 평가배점 등 서술형(정답은 참고용, judge_llm으로 채점)
  - unanswerable        : 근거 없는 질문, 거절해야 정답

검색 평가는 청크 metadata의 '파일명'으로 채점한다(Task 3에서 헤더 상속 보장).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# 거절 정답 판정 기준 문구
REFUSAL_MARK = "확인할 수 없습니다"


def load_golden(path: str) -> List[Dict[str, Any]]:
    """골든셋 JSON을 읽어 항목 리스트로 반환한다."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _retrieve(retriever, query: str):
    """검색기 버전에 상관없이 문서 리스트를 얻는다."""
    try:
        return retriever.invoke(query)
    except AttributeError:
        return retriever.get_relevant_documents(query)


def _norm(text: Any) -> str:
    """공백 제거 + 소문자화로 느슨한 문자열 비교를 준비한다."""
    return re.sub(r"\s+", "", str(text)).lower()


def _avg(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _retrieved_files(docs) -> List[Optional[str]]:
    return [d.metadata.get("파일명") for d in docs]


def _eval_retrieval(retriever, golden: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    """source_files가 있는 항목에 대해 Hit@k와 MRR을 계산한다."""
    items = [g for g in golden if g.get("source_files")]
    hits: List[float] = []
    rrs: List[float] = []
    for g in items:
        docs = _retrieve(retriever, g["question"])[:k]
        files = _retrieved_files(docs)
        sources = set(g["source_files"])
        hits.append(1.0 if any(f in sources for f in files) else 0.0)
        rr = 0.0
        for i, f in enumerate(files):
            if f in sources:
                rr = 1.0 / (i + 1)
                break
        rrs.append(rr)
    return {"n": len(items), "hit@k": _avg(hits), "mrr": _avg(rrs), "k": k}


def _eval_metadata(chain, golden: List[Dict[str, Any]]) -> Dict[str, Any]:
    """metadata_extraction 항목의 정답 문자열이 답변에 포함되는지로 정확도를 잰다."""
    items = [g for g in golden if g.get("type") == "metadata_extraction"]
    correct = 0
    for g in items:
        answer = chain.invoke(g["question"])
        expected = _norm(g.get("answer", ""))
        if expected and expected in _norm(answer):
            correct += 1
    return {"n": len(items), "accuracy": (correct / len(items)) if items else None}


def _eval_refusal(chain, golden: List[Dict[str, Any]]) -> Dict[str, Any]:
    """unanswerable 항목을 올바르게 거절했는지로 거절 정확도를 잰다."""
    items = [g for g in golden if g.get("type") == "unanswerable"]
    correct = 0
    for g in items:
        answer = chain.invoke(g["question"])
        if REFUSAL_MARK in answer:
            correct += 1
    return {
        "n": len(items),
        "refusal_accuracy": (correct / len(items)) if items else None,
    }


def _eval_body(chain, golden: List[Dict[str, Any]], judge_llm) -> Dict[str, Any]:
    """body_grounded 서술형을 judge_llm으로 0~1 점수 채점한다(judge 없으면 호출 안 함)."""
    items = [g for g in golden if g.get("type") == "body_grounded"]
    scores: List[float] = []
    for g in items:
        answer = chain.invoke(g["question"])
        prompt = (
            "다음 질문에 대한 모델 답변이 참고 정답과 사실적으로 일치하는지"
            " 0.0~1.0 사이 숫자 하나로만 평가하라.\n"
            f"질문: {g['question']}\n참고 정답: {g.get('answer', '')}\n"
            f"모델 답변: {answer}\n점수:"
        )
        raw = judge_llm.invoke(prompt)
        text = getattr(raw, "content", raw)
        match = re.search(r"[01](?:\.\d+)?", str(text))
        scores.append(float(match.group()) if match else 0.0)
    return {"n": len(items), "avg_score": _avg(scores)}


def evaluate(chain, retriever, golden: List[Dict[str, Any]], k: int = 4,
             judge_llm=None) -> Dict[str, Any]:
    """검색/메타추출/거절(/서술형) 평가를 수행해 리포트 dict를 반환한다."""
    report: Dict[str, Any] = {
        "retrieval": _eval_retrieval(retriever, golden, k),
        "metadata_extraction": _eval_metadata(chain, golden),
        "refusal": _eval_refusal(chain, golden),
        "body_grounded": None,
    }
    body_items = [g for g in golden if g.get("type") == "body_grounded"]
    if judge_llm is not None and body_items:
        report["body_grounded"] = _eval_body(chain, golden, judge_llm)
    return report


def summary_score(report: Dict[str, Any]) -> str:
    """구성 비교용 한 줄 지표를 만든다."""

    def fmt(x: Any) -> str:
        return f"{x:.3f}" if isinstance(x, (int, float)) else "N/A"

    r = report["retrieval"]
    m = report["metadata_extraction"]
    f = report["refusal"]
    parts = [
        f"Hit@{r['k']}={fmt(r['hit@k'])}",
        f"MRR={fmt(r['mrr'])}",
        f"메타추출={fmt(m['accuracy'])}",
        f"거절={fmt(f['refusal_accuracy'])}",
    ]
    body = report.get("body_grounded")
    if body:
        parts.append(f"서술형={fmt(body['avg_score'])}")
    return " | ".join(parts)
