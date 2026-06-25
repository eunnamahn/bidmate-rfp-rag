"""LangGraph 수퍼바이저 — 도구 호출(tool-calling) 에이전트.

수퍼바이저 LLM이 사용자 요청을 보고 **어떤 도구를 호출할지 스스로 결정**한다.
도구는 시스템의 능력(capability)이고, 에이전트는 그 도구를 오케스트레이션한다:

  에이전트(LLM) ─tool_call─┬─ answer_from_documents   → generator.generate_answer(qa)
                            ├─ summarize_rfp           → generator.generate_answer(summary)
                            ├─ extract_document_metadata → metadata_agent.extract_metadata
                            └─ search_nara_notices     → nara_agent.search_notices (메타데이터만)

설계 메모:
  - nara/metadata는 '에이전트'가 아니라 에이전트가 호출하는 '도구(tool)'다.
    (결정적 파이프라인/단발 추출 — 자율 추론 루프가 아님)
  - 라우팅은 도구 호출 1회로 끝낸다(전체 ReAct 루프 대신 단일 도구 선택).
    비용/지연은 기존 분류기와 비슷하면서, 인자(검색어 등)를 LLM이 채워 더 유연하다.
  - 도구 호출을 지원하지 않는 LLM(일부 Ollama 모델)은 키워드 휴리스틱으로 폴백한다.
  - 도구 본체는 모두 기존 자산을 재사용한다(중복 구현 금지).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.agents.metadata_agent import extract_metadata
from src.agents.nara_agent import notice_summary, search_notices
from src.generator import generate_answer

# retriever_fn(question, filename) -> retriever
RetrieverFn = Callable[[str, Optional[str]], Any]

# NARA_API_KEY가 없을 때 collect(나라장터 검색) 분기에서 보여줄 안내.
_NARA_KEY_GUIDE = (
    "나라장터 검색에는 NARA_API_KEY가 필요합니다(.env에 설정).\n"
    "공공데이터포털에서 발급 후 백엔드를 재기동하면 채팅에서 바로 공고를 검색할 수 있습니다."
)

# 최근 몇 턴의 대화를 라우팅 판단에 포함할지(후속 질문 맥락).
_MAX_HISTORY_TURNS = 4


class GraphState(TypedDict, total=False):
    """수퍼바이저 그래프의 공유 상태."""
    question: str
    history: List[Tuple[str, str]]
    filename: Optional[str]
    intent: str                        # qa | summary | metadata | collect
    tool_args: Dict[str, Any]          # LLM이 채운 도구 인자(검색어 등)
    answer: str
    sources: List[Document]
    metadata: Optional[Dict[str, Any]]


# ══════════════════════════════════════════════════════
# 도구 정의 (LLM에게 노출하는 스키마)
#   본체는 실행되지 않는다 — 노드가 기존 자산으로 처리한다.
#   docstring/인자가 라우팅 결정을 좌우하므로 설명을 명확히 둔다.
# ══════════════════════════════════════════════════════

@tool
def answer_from_documents(question: str) -> str:
    """인덱싱된 RFP 문서 내용에 대한 사실 질의응답.

    예산·마감일·제출 방식·요구사항·평가 배점 등 '문서에 적힌 내용'을 물을 때 사용한다.
    question에는 (필요하면 대화 맥락을 반영해) 독립적으로 검색 가능한 질문을 넣는다.
    """
    return ""


@tool
def summarize_rfp(query: str) -> str:
    """특정 RFP/사업을 대상 기관·예산·마감일·핵심 요구조건·제출 방식 양식으로 요약한다.

    "이 사업 요약해줘", "핵심만 정리해줘" 처럼 문서 전체 개요를 원할 때 사용한다.
    """
    return ""


@tool
def extract_document_metadata(query: str) -> str:
    """문서에서 발주기관·사업금액·마감일·사업명 같은 구조화 메타데이터만 추출한다.

    "이 사업 발주기관/금액/마감일 뽑아줘", "메타데이터 추출" 류 요청에 사용한다.
    """
    return ""


@tool
def search_nara_notices(keyword: str, days: int = 3) -> str:
    """나라장터에서 신규/최근 입찰공고를 키워드로 검색한다(메타데이터만, 인덱스 적재 없음).

    "나라장터에서 ○○ 공고 찾아줘", "최신 공고", "신규 공고 수집" 처럼 외부(나라장터)
    공고를 찾을 때 사용한다. keyword에는 공고명/사업 핵심어를 넣는다.
    """
    return ""


_TOOLS = [answer_from_documents, summarize_rfp, extract_document_metadata, search_nara_notices]

# 도구 이름 → 내부 intent(노드)
_TOOL_TO_INTENT = {
    "answer_from_documents": "qa",
    "summarize_rfp": "summary",
    "extract_document_metadata": "metadata",
    "search_nara_notices": "collect",
}

_AGENT_SYSTEM = """당신은 공공 RFP 분석 어시스턴트 '입찰메이트'의 라우터입니다.
사용자 요청을 처리하기 위해 가장 적합한 도구를 정확히 하나 호출하세요.

- 문서 내용 질의응답(예산·마감일·요구사항·평가 배점 등) → answer_from_documents
- 문서 전체 요약/정리 → summarize_rfp
- 발주기관·금액·마감일·사업명 등 구조화 메타데이터 추출 → extract_document_metadata
- 나라장터에서 신규/최근 공고 검색 → search_nara_notices

직접 장문으로 답하지 말고 반드시 도구를 호출하세요. 도구 인자에는 대화 맥락을 반영한
독립적인 검색어/질문을 넣으세요."""


# ══════════════════════════════════════════════════════
# 도구 미지원 LLM용 키워드 폴백
# ══════════════════════════════════════════════════════

_COLLECT_RE = re.compile(r"(나라장터|신규\s*공고|공고\s*수집|최신\s*공고|공고.*가져|공고.*수집)")
_METADATA_RE = re.compile(r"(메타\s*데이터|발주\s*기관.*마감|기본\s*정보\s*추출|핵심\s*정보\s*추출)")
_SUMMARY_RE = re.compile(r"(요약|정리해|핵심\s*정리|개요)")


def _keyword_intent(question: str) -> str:
    """키워드 휴리스틱으로 의도를 판정한다(도구 미지원 LLM 폴백). 기본값 qa."""
    if _COLLECT_RE.search(question):
        return "collect"
    if _METADATA_RE.search(question):
        return "metadata"
    if _SUMMARY_RE.search(question):
        return "summary"
    return "qa"


def _history_messages(history: List[Tuple[str, str]]) -> List[Any]:
    """최근 대화 히스토리를 라우팅용 메시지로 변환한다."""
    from langchain_core.messages import AIMessage

    msgs: List[Any] = []
    for user_msg, assistant_msg in (history or [])[-_MAX_HISTORY_TURNS:]:
        msgs.append(HumanMessage(content=user_msg))
        msgs.append(AIMessage(content=assistant_msg))
    return msgs


def route_with_tools(state: GraphState, llm) -> Dict[str, Any]:
    """LLM 도구 호출로 intent와 도구 인자를 정한다.

    도구 호출을 지원하지 않거나 호출이 없으면 키워드 휴리스틱으로 폴백한다.
    반환: {"intent": ..., "tool_args": {...}}
    """
    question = state["question"]
    try:
        bound = llm.bind_tools(_TOOLS)
        messages = (
            [SystemMessage(content=_AGENT_SYSTEM)]
            + _history_messages(state.get("history") or [])
            + [HumanMessage(content=question)]
        )
        response = bound.invoke(messages)
        calls = getattr(response, "tool_calls", None) or []
    except Exception:
        calls = []

    if calls:
        call = calls[0]
        intent = _TOOL_TO_INTENT.get(call.get("name", ""), "qa")
        return {"intent": intent, "tool_args": call.get("args", {}) or {}}

    # 도구 호출 없음 → 키워드 폴백
    return {"intent": _keyword_intent(question), "tool_args": {}}


# ══════════════════════════════════════════════════════
# 노드 구현 (도구 본체 — 기존 자산 재사용)
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
    """도구 호출 수퍼바이저 그래프를 조립해 컴파일된 그래프를 반환한다.

    Args:
        cfg:          config.yaml 로드 dict (메타데이터 추출·나라장터 검색에 전달).
        llm:          make_llm으로 만든 LLM (라우팅 도구 호출 + 답변 생성에 공유).
        retriever_fn: (question, filename) -> retriever. 질문별 검색기 선택을 주입한다.
    """

    def _agent(state: GraphState) -> GraphState:
        # 외부에서 intent를 강제 지정했으면(예: /summarize) 도구 선택을 생략한다.
        if state.get("intent"):
            return {"tool_args": state.get("tool_args") or {}}
        return route_with_tools(state, llm)

    def _route_edge(state: GraphState) -> str:
        return state.get("intent", "qa")

    def _search_query(state: GraphState, *keys: str) -> str:
        """LLM이 채운 도구 인자(검색어)를 우선 쓰고, 없으면 원 질문을 쓴다."""
        args = state.get("tool_args") or {}
        for k in keys:
            if args.get(k):
                return str(args[k])
        return state["question"]

    def _qa(state: GraphState) -> GraphState:
        q = _search_query(state, "question")
        retriever = retriever_fn(q, state.get("filename"))
        answer, sources = generate_answer(
            retriever, llm, q, mode="qa", history=state.get("history") or [],
        )
        return {"answer": answer, "sources": sources}

    def _summary(state: GraphState) -> GraphState:
        q = _search_query(state, "query")
        retriever = retriever_fn(q, state.get("filename"))
        answer, sources = generate_answer(
            retriever, llm, q, mode="summary", history=state.get("history") or [],
        )
        return {"answer": answer, "sources": sources}

    def _metadata(state: GraphState) -> GraphState:
        q = _search_query(state, "query")
        retriever = retriever_fn(q, state.get("filename"))
        docs = retriever.invoke(q)
        if not docs:
            return {
                "answer": "해당 내용은 제공된 문서에서 확인할 수 없습니다.",
                "sources": [],
                "metadata": None,
            }
        text = "\n\n".join(d.page_content for d in docs)
        meta = extract_metadata(text, cfg)
        return {"answer": _format_metadata(meta), "sources": docs, "metadata": meta}

    def _collect(state: GraphState) -> GraphState:
        # 타겟+지연 인덱싱: 목록 API로 메타데이터만 검색해 보여주고(인덱스 비대화 방지),
        # 실제 본문 적재는 사용자가 '나라장터' 탭에서 고른 공고 1건만 수행한다.
        keyword = _search_query(state, "keyword")
        days = int((state.get("tool_args") or {}).get("days", 3))
        try:
            items = search_notices(cfg, keyword=keyword, days=days, max_results=8)
        except RuntimeError:
            return {"answer": _NARA_KEY_GUIDE, "sources": [], "metadata": None}
        except Exception as e:
            return {"answer": f"나라장터 검색에 실패했습니다: {e}", "sources": [], "metadata": None}

        if not items:
            return {
                "answer": "최근 나라장터 공고에서 일치하는 건을 찾지 못했습니다. 키워드를 바꿔보세요.",
                "sources": [],
                "metadata": None,
            }

        summaries = [notice_summary(it) for it in items]
        lines = [
            f"- {s['사업명']} · {s['기관'] or '미상'} · 마감 {s['마감일'] or '미상'}"
            f" · [공고번호 {s['공고번호']}]"
            for s in summaries
        ]
        answer = (
            "나라장터에서 다음 공고를 찾았습니다(메타데이터만 조회 — 인덱스에는 적재하지 않음):\n"
            + "\n".join(lines)
            + "\n\n특정 공고의 전체 내용을 질문하려면 ‘나라장터’ 탭에서 그 공고를 선택해 인덱싱하세요."
        )
        return {"answer": answer, "sources": [], "metadata": {"nara_search": summaries}}

    graph = StateGraph(GraphState)
    graph.add_node("agent", _agent)
    graph.add_node("qa", _qa)
    graph.add_node("summary", _summary)
    graph.add_node("metadata", _metadata)
    graph.add_node("collect", _collect)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _route_edge,
        {"qa": "qa", "summary": "summary", "metadata": "metadata", "collect": "collect"},
    )
    for node in ("qa", "summary", "metadata", "collect"):
        graph.add_edge(node, END)

    return graph.compile()
