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

from src.agents.eligibility_agent import check_eligibility, check_eligibility_with_company_info
from src.agents.nara_agent import notice_summary, search_notices
from src.agents.risk_agent import detect_risks
from src.generator import generate_answer, generate_checklist, generate_diagram

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
    company_info: Optional[Dict[str, Any]]  # 프론트에서 전달한 회사 정보(적격성 판정용)


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
def search_nara_notices(keyword: str, days: int = 3) -> str:
    """나라장터에서 신규/최근 입찰공고를 키워드로 검색한다(메타데이터만, 인덱스 적재 없음).

    "나라장터에서 ○○ 공고 찾아줘", "최신 공고", "신규 공고 수집" 처럼 외부(나라장터)
    공고를 찾을 때 사용한다. keyword에는 공고명/사업 핵심어를 넣는다.
    """
    return ""


_TOOLS = [answer_from_documents, summarize_rfp, search_nara_notices]

# 도구 이름 → 내부 intent(노드)
_TOOL_TO_INTENT = {
    "answer_from_documents": "qa",
    "summarize_rfp": "summary",
    "search_nara_notices": "collect",
}

_AGENT_SYSTEM = """당신은 공공 RFP 분석 어시스턴트 '입찰메이트'의 라우터입니다.
사용자 요청을 처리하기 위해 가장 적합한 도구를 정확히 하나 호출하세요.

- 문서 내용 질의응답(예산·마감일·요구사항·평가 배점 등) → answer_from_documents
- 문서 전체 요약/정리 → summarize_rfp
- 나라장터에서 신규/최근 공고 검색 → search_nara_notices

직접 장문으로 답하지 말고 반드시 도구를 호출하세요. 도구 인자에는 대화 맥락을 반영한
독립적인 검색어/질문을 넣으세요."""


# ══════════════════════════════════════════════════════
# 도구 미지원 LLM용 키워드 폴백
# ══════════════════════════════════════════════════════

_COLLECT_RE = re.compile(r"(나라장터|신규\s*공고|공고\s*수집|최신\s*공고|공고.*가져|공고.*수집)")
_SUMMARY_RE = re.compile(r"(요약|정리해|핵심\s*정리|개요)")


def _keyword_intent(question: str) -> str:
    """키워드 휴리스틱으로 의도를 판정한다(도구 미지원 LLM 폴백). 기본값 qa."""
    if _COLLECT_RE.search(question):
        return "collect"
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

# 심각도 정렬/표시 순서(높은 위험을 위로).
_SEVERITY_ORDER = {"높음": 0, "중간": 1, "낮음": 2}


def _format_risks(risks: List[Dict[str, Any]]) -> str:
    """detect_risks 결과를 심각도순 markdown으로 변환한다."""
    if not risks:
        return "이 문서에서 특별히 주의할 리스크/실격 조항을 찾지 못했습니다."
    ordered = sorted(risks, key=lambda r: _SEVERITY_ORDER.get(r.get("심각도"), 3))
    lines = ["리스크 탐지 결과입니다.\n"]
    for r in ordered:
        lines.append(f"**[{r['심각도']}] {r['항목']}**  ·  {r['유형']}")
        if r.get("내용"):
            lines.append(f"- {r['내용']}")
        if r.get("심각도_이유"):
            lines.append(f"- 사유: {r['심각도_이유']}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_checklist(data: Dict[str, Any]) -> str:
    """generate_checklist 결과를 카테고리별 markdown 체크리스트로 변환한다."""
    if data.get("parse_error"):
        return "체크리스트 생성 결과를 해석하지 못했습니다. 다시 시도해 주세요."
    title = data.get("title") or "입찰 준비"
    lines = [f"**{title}** 입찰 준비 체크리스트입니다.\n"]
    has_item = False
    for cat in data.get("categories", []):
        items = cat.get("items", [])
        if not items:
            continue
        has_item = True
        lines.append(f"**{cat.get('name', '')}**")
        for it in items:
            note = f" — {it['note']}" if it.get("note") else ""
            lines.append(f"- [ ] {it.get('label', '')}{note}")
        lines.append("")
    if not has_item:
        return "문서에서 체크리스트 항목을 추출하지 못했습니다."
    return "\n".join(lines).strip()


def _frontend_to_backend_company(info: Dict[str, Any]) -> Dict[str, Any]:
    """프론트엔드 CompanyInfo → 백엔드 eligibility_agent 판정 포맷 변환.

    프론트 필드명(영문)을 백엔드 필드명(한글)으로 매핑하고,
    문자열 매출액("30억", "5억 미만" 등)을 원 단위 정수로 변환한다.
    """
    revenue_str = str(info.get("annualRevenue") or "")
    매출액: Optional[int] = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*억", revenue_str)
    if m:
        매출액 = int(float(m.group(1)) * 1_0000_0000)
    else:
        m2 = re.search(r"(\d+)", revenue_str)
        if m2:
            매출액 = int(m2.group(1))

    certs = info.get("certifications")
    return {
        "업종": info.get("industry") or None,
        "규모": info.get("size") or None,
        "창업연수": int(info["foundingYears"]) if info.get("foundingYears") else None,
        "매출액": 매출액,
        "직원수": int(info["headcount"]) if info.get("headcount") else None,
        "인증": certs if isinstance(certs, list) and certs else None,
    }


def _user_messages(history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """히스토리에서 사용자 발화만 role/content dict로 추린다.

    적격성 회사 정보 추출(extract_company_info)에는 사용자가 직접 말한 정보만 넘긴다.
    어시스턴트 답변(RFP 본문/검색 결과)이나 버튼 지시문까지 함께 넘기면 그 안의
    기관명·규모 표현을 회사 정보로 오추출(예: 입력하지 않은 '중견기업')할 수 있다.
    """
    return [{"role": "user", "content": u} for u, _ in (history or [])]


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
        filename = state.get("filename")
        retriever = retriever_fn(q, filename)
        answer, sources = generate_answer(
            retriever, llm, q, mode="qa", history=state.get("history") or [],
            active_filename=filename,
        )
        return {"answer": answer, "sources": sources}

    def _summary(state: GraphState) -> GraphState:
        q = _search_query(state, "query")
        filename = state.get("filename")
        retriever = retriever_fn(q, filename)
        answer, sources = generate_answer(
            retriever, llm, q, mode="summary", history=state.get("history") or [],
            active_filename=filename,
        )
        return {"answer": answer, "sources": sources}

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

    # ── 분석 에이전트 노드(적격성·리스크·체크리스트·평가 구조) ──
    # 모두 활성 문서(filename)로 한정 검색해 그 문서 1건을 분석한다.

    def _eligibility(state: GraphState) -> GraphState:
        # 활성 문서에서 참가자격 관련 본문을 모아 회사 정보와 대조 판정한다.
        q = state["question"] + " 입찰참가자격 요건 자격요건"
        retriever = retriever_fn(q, state.get("filename"))
        docs = retriever.invoke(q)
        rfp_text = "\n\n".join(d.page_content for d in docs)

        frontend_company = state.get("company_info")
        if frontend_company:
            # 프론트에서 company_info를 전달했으면 LLM 추출 없이 직접 판정한다.
            backend_company = _frontend_to_backend_company(frontend_company)
            result = check_eligibility_with_company_info(backend_company, rfp_text, cfg)
        else:
            # 전달이 없으면 기존 방식: 사용자 발화에서만 회사 정보를 추출한다.
            result = check_eligibility(_user_messages(state.get("history")), rfp_text, cfg)

        return {"answer": result["message"], "sources": docs, "metadata": result}

    def _risk(state: GraphState) -> GraphState:
        q = state["question"] + " 리스크 실격 조항 자격요건 제출"
        retriever = retriever_fn(q, state.get("filename"))
        docs = retriever.invoke(q)
        rfp_text = "\n\n".join(d.page_content for d in docs)
        risks = detect_risks(rfp_text, cfg)
        return {"answer": _format_risks(risks), "sources": docs, "metadata": {"risks": risks}}

    def _checklist(state: GraphState) -> GraphState:
        q = _search_query(state, "query")
        retriever = retriever_fn(q, state.get("filename"))
        data, docs = generate_checklist(retriever, llm, q)
        return {"answer": _format_checklist(data), "sources": docs, "metadata": data}

    def _evaluation(state: GraphState) -> GraphState:
        # 평가배점 구조를 Mermaid 다이어그램으로. 답변의 ```mermaid 블록을 UI가 렌더한다.
        q = _search_query(state, "query")
        retriever = retriever_fn(q, state.get("filename"))
        mermaid, docs = generate_diagram(retriever, llm, q, "evaluation")
        return {"answer": mermaid, "sources": docs, "metadata": None}

    graph = StateGraph(GraphState)
    graph.add_node("agent", _agent)
    graph.add_node("qa", _qa)
    graph.add_node("summary", _summary)
    graph.add_node("collect", _collect)
    graph.add_node("eligibility", _eligibility)
    graph.add_node("risk", _risk)
    graph.add_node("checklist", _checklist)
    graph.add_node("evaluation", _evaluation)

    _ROUTES = {
        "qa": "qa", "summary": "summary", "collect": "collect",
        "eligibility": "eligibility", "risk": "risk",
        "checklist": "checklist", "evaluation": "evaluation",
    }
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _route_edge, _ROUTES)
    for node in _ROUTES.values():
        graph.add_edge(node, END)

    return graph.compile()
