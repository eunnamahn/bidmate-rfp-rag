"""LangGraph 수퍼바이저 라우터.

사용자 질문을 의도(intent)별로 분기하는 StateGraph를 조립한다.

  classify_intent ─┬─ qa       → 기존 generate_answer(mode="qa")
                   ├─ summary  → 기존 generate_answer(mode="summary")
                   ├─ metadata → retriever 검색 → metadata_agent.extract_metadata
                   └─ collect  → 나라장터 배치 실행 안내(여기서 수집을 실행하지 않음)

기존 자산을 재사용한다(중복 구현 금지):
  - src.generator.generate_answer  : qa/summary 노드 본체
  - src.agents.metadata_agent.extract_metadata : metadata 노드
  - retriever_fn(question, filename) : 질문별 검색기 선택을 주입(백엔드의 기관/파일 필터 재사용)

nara_agent(수집)는 그래프 상시 노드가 아니라 독립 배치/CLI로 유지한다.
collect 의도로 분류되면 실제 수집 대신 실행 방법을 안내한다.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.agents.metadata_agent import extract_metadata
from src.generator import generate_answer

# retriever_fn(question, filename) -> retriever
RetrieverFn = Callable[[str, Optional[str]], Any]

# 나라장터 수집은 그래프에서 실행하지 않고 배치로 안내한다(Q2: 독립 배치/CLI 유지).
_COLLECT_GUIDE = (
    "나라장터 신규 공고 수집은 배치 잡으로 실행하세요.\n"
    "  - 당일 수집:   python -m src.agents.nara_agent\n"
    "  - 특정 날짜:   python -m src.agents.nara_agent 2024-04-09\n"
    "  - 건수 제한:   python -m src.agents.nara_agent 2024-04-09 --max 5\n"
    "수집된 공고는 FAISS에 증분 추가되어 이후 질문에서 검색됩니다."
)


class GraphState(TypedDict, total=False):
    """수퍼바이저 그래프의 공유 상태."""
    question: str
    history: List[Tuple[str, str]]
    filename: Optional[str]
    intent: str                       # qa | summary | metadata | collect
    answer: str
    sources: List[Document]
    metadata: Optional[Dict[str, Any]]


# ══════════════════════════════════════════════════════
# 의도 분류 (키워드 우선, 모호 시 LLM 폴백)
# ══════════════════════════════════════════════════════

_COLLECT_RE = re.compile(r"(신규\s*공고|공고\s*수집|나라장터.*수집|최신\s*공고|공고.*가져)")
_METADATA_RE = re.compile(r"(메타\s*데이터|발주\s*기관.*마감|기본\s*정보\s*추출|핵심\s*정보\s*추출)")
_SUMMARY_RE = re.compile(r"(요약|정리해|핵심\s*정리|개요)")

_INTENTS = ("qa", "summary", "metadata", "collect")

_CLASSIFY_SYSTEM = """당신은 사용자 질문을 4개 의도 중 하나로 분류하는 라우터입니다.
아래 라벨 중 하나만 정확히 한 단어로 출력하세요. 다른 텍스트는 출력하지 마세요.

- qa: RFP 문서 내용에 대한 일반 질의응답(예: "예산이 얼마야?", "제출 방식 알려줘")
- summary: 문서 전체를 양식에 맞춰 요약/정리 요청(예: "이 사업 요약해줘")
- metadata: 발주기관/금액/마감일/사업명 같은 구조화 메타데이터 추출 요청
- collect: 나라장터에서 신규/최신 공고를 수집/가져오라는 요청

라벨: qa | summary | metadata | collect"""


def _keyword_intent(question: str) -> Optional[str]:
    """키워드 휴리스틱으로 의도를 빠르게 판정한다. 모호하면 None."""
    if _COLLECT_RE.search(question):
        return "collect"
    if _METADATA_RE.search(question):
        return "metadata"
    if _SUMMARY_RE.search(question):
        return "summary"
    return None


def classify_intent(question: str, llm) -> str:
    """질문 의도를 qa|summary|metadata|collect 중 하나로 분류한다.

    키워드로 먼저 판정하고, 모호하면 LLM으로 분류한다. 실패 시 기본값 qa.
    """
    hit = _keyword_intent(question)
    if hit:
        return hit
    try:
        raw = getattr(
            llm.invoke([
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": question},
            ]),
            "content", "",
        )
    except Exception:
        return "qa"
    label = str(raw).strip().lower()
    for intent in _INTENTS:
        if intent in label:
            return intent
    return "qa"


# ══════════════════════════════════════════════════════
# 노드 구현
# ══════════════════════════════════════════════════════

def _format_metadata(meta: Dict[str, Any]) -> str:
    """추출 메타데이터를 사람이 읽는 양식으로 변환한다."""
    def _v(x: Any) -> str:
        return str(x) if x not in (None, "") else "문서에 명시되지 않음"

    금액 = meta.get("사업금액")
    금액_str = f"{금액:,}원" if isinstance(금액, int) else _v(금액)
    return (
        f"- 발주기관: {_v(meta.get('발주기관'))}\n"
        f"- 사업금액: {금액_str}\n"
        f"- 마감일: {_v(meta.get('마감일'))}\n"
        f"- 사업명: {_v(meta.get('사업명'))}"
    )


def build_graph(cfg: Dict[str, Any], llm, retriever_fn: RetrieverFn):
    """수퍼바이저 그래프를 조립해 컴파일된 그래프를 반환한다.

    Args:
        cfg:          config.yaml 로드 dict (metadata 추출 등에 전달).
        llm:          make_llm으로 만든 LLM (분류 + 답변 생성에 공유).
        retriever_fn: (question, filename) -> retriever. 질문별 검색기 선택을 주입한다.
    """

    def _router(state: GraphState) -> GraphState:
        # 외부에서 intent를 강제 지정했으면(예: /summarize) 분류를 생략한다.
        intent = state.get("intent") or classify_intent(state["question"], llm)
        return {"intent": intent}

    def _route_edge(state: GraphState) -> str:
        return state.get("intent", "qa")

    def _qa(state: GraphState) -> GraphState:
        retriever = retriever_fn(state["question"], state.get("filename"))
        answer, sources = generate_answer(
            retriever, llm, state["question"],
            mode="qa", history=state.get("history") or [],
        )
        return {"answer": answer, "sources": sources}

    def _summary(state: GraphState) -> GraphState:
        retriever = retriever_fn(state["question"], state.get("filename"))
        answer, sources = generate_answer(
            retriever, llm, state["question"],
            mode="summary", history=state.get("history") or [],
        )
        return {"answer": answer, "sources": sources}

    def _metadata(state: GraphState) -> GraphState:
        retriever = retriever_fn(state["question"], state.get("filename"))
        docs = retriever.invoke(state["question"])
        if not docs:
            return {
                "answer": "해당 내용은 제공된 문서에서 확인할 수 없습니다.",
                "sources": [],
                "metadata": None,
            }
        text = "\n\n".join(d.page_content for d in docs)
        meta = extract_metadata(text, cfg)
        return {
            "answer": _format_metadata(meta),
            "sources": docs,
            "metadata": meta,
        }

    def _collect(state: GraphState) -> GraphState:
        # 그래프에서는 수집을 실행하지 않고 배치 실행 방법만 안내한다.
        return {"answer": _COLLECT_GUIDE, "sources": [], "metadata": None}

    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", _router)
    graph.add_node("qa", _qa)
    graph.add_node("summary", _summary)
    graph.add_node("metadata", _metadata)
    graph.add_node("collect", _collect)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_edge,
        {"qa": "qa", "summary": "summary", "metadata": "metadata", "collect": "collect"},
    )
    for node in ("qa", "summary", "metadata", "collect"):
        graph.add_edge(node, END)

    return graph.compile()
