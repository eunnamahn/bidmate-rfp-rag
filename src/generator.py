"""프롬프트 + LLM 체인 구성 (LCEL).

거절 유도 지침을 반드시 포함한다(평가의 거절 정확도와 직결).
QA용/요약용 프롬프트를 각각 제공하며, 요약은 컨설턴트가 필요한 4대 정보축
(대상 기관/예산/마감일/요구조건/제출방식)을 고정 양식으로 채운다.
대화 히스토리 전략 2가지를 지원한다:
  - "full"   : 히스토리를 통째로 프롬프트에 포함 (베이스라인)
  - "rewrite": 후속 질문을 독립적인 완성형 질문으로 재작성 후 검색 (고도화)
config.yaml의 history.strategy로 전환한다.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 히스토리 전략/턴 수는 config.yaml에서 읽는다(없으면 기본값).
try:
    from src.config import load_config

    _cfg = load_config("config.yaml")
    HISTORY_STRATEGY = _cfg.get("history", {}).get("strategy", "full")
    MAX_HISTORY_TURNS = _cfg.get("history", {}).get("max_turns", 5)
except Exception:
    HISTORY_STRATEGY = "full"
    MAX_HISTORY_TURNS = 5


# ══════════════════════════════════════════════════════
# 프롬프트 정의
# ══════════════════════════════════════════════════════

_SYSTEM_QA = """당신은 공공기관 입찰 전문 컨설팅 스타트업 '입찰메이트'의 RFP 분석 어시스턴트입니다.

[역할]
사용자가 제공한 RFP(제안요청서) 문서를 바탕으로 정확하고 신뢰할 수 있는 정보를 제공합니다.

[행동 지침]
1. 반드시 제공된 문서 내용만을 근거로 답변하세요.
2. 문서에 없는 내용은 절대 추측하거나 지어내지 마세요. 대신 "해당 내용은 제공된 문서에서 확인할 수 없습니다."라고 명확히 답변하세요.
3. 문서에 없는 조언, 전략, 추천 의견은 절대 추가하지 마세요.
4. 내부 사고 과정이나 의도 파악 내용은 절대 출력하지 마세요. 바로 답변만 제공하세요.
5. 답변은 간결하고 명확한 한국어로 작성하세요.
6. 수치(예산, 날짜, 기간 등)는 문서에 있는 그대로 정확하게 인용하세요.
7. 여러 문서를 비교할 때는 각 문서의 내용을 명확히 구분해서 답변하세요.
8. 표 형식은 원문이 표 형태인 경우에만 사용하고, 그 외에는 간결한 문장으로 답변하세요.

[출력 형식]
- 핵심 정보를 먼저 답변하고, 필요시 부연 설명을 추가하세요.
- 문서에서 인용한 내용은 가능하면 출처 섹션명을 함께 표기하세요.
- 불필요하게 길게 늘어지지 않도록 하세요."""

# 요약은 컨설턴트가 필요한 4대 정보축(+마감일)을 고정 양식으로 항상 채운다.
# 문서 자체가 없으면 전체 거절('문서에서 확인할 수 없습니다' — 평가가 감지하는 표지),
# 개별 항목만 없으면 '문서에 명시되지 않음'으로 적어 요약이 통째로 거절로 오인되지 않게 한다.
_SYSTEM_SUMMARY = """당신은 공공기관 입찰 전문 컨설팅 스타트업 '입찰메이트'의 RFP 분석 어시스턴트입니다.

[역할]
컨설턴트가 RFP의 핵심을 빠르게 파악하도록, 제공된 문서를 아래 양식으로 정리합니다.

[행동 지침]
1. 반드시 제공된 문서 내용만을 근거로 정리하세요(Faithfulness). 추측하거나 지어내지 마세요.
2. 요청한 사업의 내용이 문맥에 전혀 없으면 "문서에서 확인할 수 없습니다."라고만 답하세요.
3. 문맥에 있으면 아래 양식을 빠짐없이 채우되, 개별 항목의 근거가 없으면 그 항목에 "문서에 명시되지 않음"이라고 적으세요.
4. 내부 사고 과정은 노출하지 말고 결론만 한국어로 간결히 작성하세요.
5. 수치(예산, 날짜)는 문서에 있는 그대로 정확하게 인용하세요.

[출력 양식]
- 대상 기관:
- 예산(사업금액):
- 입찰 마감일:
- 핵심 요구조건:
- 제출 방식:"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_QA),
        ("human", "문맥:\n{context}\n\n질문: {question}\n답변:"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_SUMMARY),
        ("human", "문맥:\n{context}\n\n요청: {question}\n정리:"),
    ]
)

# 후속 질문 재작성용 프롬프트 (rewrite 전략)
_REWRITE_PROMPT = """아래는 사용자와 어시스턴트의 대화 히스토리와 후속 질문입니다.
후속 질문을 대화 맥락을 반영해서 독립적으로 이해할 수 있는 완성된 질문으로 재작성하세요.
재작성된 질문만 출력하고 다른 설명은 하지 마세요.

[대화 히스토리]
{history}

[후속 질문]
{question}

[재작성된 질문]"""


# ══════════════════════════════════════════════════════
# 유틸 함수
# ══════════════════════════════════════════════════════

def _format_docs(docs: List[Document]) -> str:
    """검색된 문서들을 하나의 문맥 문자열로 합친다.

    각 청크 본문 앞에는 청킹 단계에서 붙인 [기관|금액|마감일] 헤더가 이미 들어있다.
    """
    return "\n\n---\n\n".join(d.page_content for d in docs)


def _format_history(history: List[Tuple[str, str]]) -> str:
    """대화 히스토리를 텍스트로 변환한다. 최근 MAX_HISTORY_TURNS 턴만 포함."""
    recent = history[-MAX_HISTORY_TURNS:]
    lines = []
    for user_msg, assistant_msg in recent:
        lines.append(f"사용자: {user_msg}")
        lines.append(f"어시스턴트: {assistant_msg}")
    return "\n".join(lines)


def rewrite_query(question: str, history: List[Tuple[str, str]], llm) -> str:
    """rewrite 전략: 후속 질문을 독립적인 완성형 질문으로 재작성한다.

    히스토리가 없으면 원래 질문을 그대로 반환한다. llm은 openai/ollama 모두 가능.
    """
    if not history:
        return question
    prompt = _REWRITE_PROMPT.format(
        history=_format_history(history),
        question=question,
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return getattr(response, "content", str(response)).strip()


# ══════════════════════════════════════════════════════
# 근거 인용(citation) 처리
# ══════════════════════════════════════════════════════

# 답변이 이 표지를 포함하면 전체 거절로 보고 출처를 표시하지 않는다.
_REFUSAL_MARK = "확인할 수 없습니다"
# "[사용근거: 1, 3]" 형태에서 번호를 뽑는다.
_CITATION_RE = re.compile(r"사용\s*근거\s*[:：]?\s*([0-9,，\s]+)")


def _number_sources(docs: List[Document]) -> Tuple[Dict[str, int], List[Document]]:
    """검색 문서를 파일명 기준으로 중복 제거하며 1부터 번호를 매긴다.

    반환: (파일명->번호 매핑, 번호순 대표 Document 리스트)
    """
    order: Dict[str, int] = {}
    unique: List[Document] = []
    for d in docs:
        fn = d.metadata.get("파일명")
        if fn not in order:
            order[fn] = len(unique) + 1
            unique.append(d)
    return order, unique


def _format_labeled(docs: List[Document], order: Dict[str, int]) -> str:
    """각 청크 앞에 [자료 N] 라벨을 붙여 문맥 문자열을 만든다(N=파일명별 번호)."""
    parts = []
    for d in docs:
        n = order.get(d.metadata.get("파일명"))
        parts.append(f"[자료 {n}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _parse_citation(text: str) -> List[int]:
    """답변에서 실제로 인용한 자료 번호를 뽑는다(없으면 빈 리스트).

    '[사용근거: 1, 2]'와 모델이 즐겨 쓰는 '[출처: 자료 1]'/'(문서: 자료 1)'
    같은 인라인 표기를 모두 인식한다.
    """
    nums: set = set()
    m = _CITATION_RE.search(text)
    if m:
        nums.update(int(x) for x in re.findall(r"\d+", m.group(1)))
    # "자료 N" 형태로 본문에서 참조한 번호도 사용한 것으로 본다.
    nums.update(int(x) for x in re.findall(r"자료\s*(\d+)", text))
    return sorted(nums)


def _strip_citation(text: str) -> str:
    """답변 본문에서 인용 표기를 제거한다('[사용근거: ...]', '[출처: 자료 N]' 등)."""
    text = re.sub(r"\[?\s*사용\s*근거\s*[:：][^\]\n]*\]?", "", text)
    # "[출처: 자료 1]", "(문서: 자료 1)" 같은 인라인 표기 제거
    text = re.sub(r"[\[(]\s*(?:출처|문서)\s*[:：]?\s*자료\s*\d+\s*[\])]", "", text)
    # 정제 후 남는 빈 괄호/연속 공백 정리
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


# ══════════════════════════════════════════════════════
# 핵심 함수
# ══════════════════════════════════════════════════════

def generate_answer(
    retriever,
    llm,
    question: str,
    mode: str = "qa",
    history: Optional[List[Tuple[str, str]]] = None,
    strategy: Optional[str] = None,
) -> Tuple[str, List[Document]]:
    """질문에 대해 (답변, 근거 문서 리스트)를 반환한다.

    history : [(user_msg, assistant_msg), ...] 대화 히스토리(없으면 단순 답변).
    strategy: "full" | "rewrite" (None이면 config의 history.strategy).
    검색을 한 번만 수행해 답변과 근거를 같이 얻는다(백엔드 sources 표시용).
    """
    history = history or []
    strategy = strategy or HISTORY_STRATEGY
    system_text = _SYSTEM_QA if mode == "qa" else _SYSTEM_SUMMARY
    ask_label, ans_label = ("질문", "답변") if mode == "qa" else ("요청", "정리")

    # rewrite 전략: 후속 질문을 완성형으로 재작성한 뒤 검색(검색 품질 향상)
    if strategy == "rewrite" and history:
        search_query = rewrite_query(question, history, llm)
    else:
        search_query = question

    docs = retriever.invoke(search_query)
    # 검색 문서에 [자료 N] 번호를 붙이고, 답변에서 실제 인용한 번호만 출처로 남긴다.
    order, unique_docs = _number_sources(docs)
    context = _format_labeled(docs, order)

    user_content = (
        f"문맥:\n{context}\n\n{ask_label}: {question}\n{ans_label}:\n"
        "(답변 맨 마지막 줄에, 실제로 근거로 사용한 자료 번호만 '[사용근거: 1, 2]' "
        "형식으로 적으세요. 사용하지 않은 자료 번호는 절대 넣지 마세요.)"
    )

    # full 전략: 히스토리를 메시지로 통째로 포함해 문맥 유지
    messages = [{"role": "system", "content": system_text}]
    if strategy == "full" and history:
        for user_msg, assistant_msg in history[-MAX_HISTORY_TURNS:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_content})

    raw = getattr(llm.invoke(messages), "content", "")
    cited = _parse_citation(raw)
    answer = _strip_citation(raw)

    # 출처 선정: 인용한 번호만 -> (인용 없으면) 거절이면 없음, 아니면 전체로 폴백
    if cited:
        used = [unique_docs[n - 1] for n in cited if 1 <= n <= len(unique_docs)]
        used = used or unique_docs
    elif _REFUSAL_MARK in answer:
        used = []
    else:
        used = unique_docs

    return answer, used


def build_chain(retriever, llm, mode: str = "qa"):
    """검색기→프롬프트→LLM→문자열 파서로 이어지는 LCEL 체인(히스토리 없는 단순 체인).

    체인 입력은 질문 문자열, 출력은 답변 문자열이다. mode="qa" | "summary".
    """
    if mode not in ("qa", "summary"):
        raise ValueError(f"지원하지 않는 mode: {mode}")

    prompt = QA_PROMPT if mode == "qa" else SUMMARY_PROMPT
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
