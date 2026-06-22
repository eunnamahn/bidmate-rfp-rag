"""프롬프트 + LLM 체인 구성 (LCEL).

거절 유도 지침을 반드시 포함한다(평가의 거절 정확도와 직결).
QA용/요약용 프롬프트를 각각 제공한다.
대화 히스토리 전략 2가지를 지원한다:
  - "full"   : 히스토리를 통째로 프롬프트에 포함 (베이스라인)
  - "rewrite": 후속 질문을 독립적인 완성형 질문으로 재작성 후 검색 (고도화)
config.yaml의 history.strategy로 전환한다.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

try:
    from settings import load_config
    _cfg = load_config()
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

[컨텍스트 활용]
아래 제공되는 RFP 문서 내용을 참고하여 답변하세요.

[출력 형식]
- 핵심 정보를 먼저 답변하고, 필요시 부연 설명을 추가하세요.
- 문서에서 인용한 내용은 출처 섹션명을 함께 표기하세요. (예: [과업내용] 섹션에 따르면...)
- 불필요하게 길게 늘어지지 않도록 하세요."""

_SYSTEM_SUMMARY = """당신은 공공기관 입찰 전문 컨설팅 스타트업 '입찰메이트'의 RFP 분석 어시스턴트입니다.

[역할]
사용자가 제공한 RFP(제안요청서) 문서를 바탕으로 정확하고 신뢰할 수 있는 정보를 제공합니다.

[행동 지침]
1. 반드시 제공된 문서 내용만을 근거로 정리하세요.
2. 문서에 없는 내용은 절대 추측하거나 지어내지 마세요. 대신 해당 항목에 "문서에서 확인할 수 없습니다."라고 명확히 표기하세요.
3. 내부 사고 과정이나 의도 파악 내용은 절대 출력하지 마세요. 바로 답변만 제공하세요.
4. 아래 양식에 맞춰 RFP를 정리하세요.
   - 발주기관:
   - 사업금액:
   - 입찰마감일:
   - 핵심 요구사항:"""

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

# 후속 질문 재작성용 프롬프트 (방법 2 — rewrite 전략)
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
    """검색된 문서들을 하나의 문맥 문자열로 합친다."""
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('section', '문서')}]\n{d.page_content}"
        for d in docs
    )


def _format_history(history: List[Tuple[str, str]]) -> str:
    """대화 히스토리를 텍스트로 변환한다. 최근 MAX_HISTORY_TURNS 턴만 포함."""
    recent = history[-MAX_HISTORY_TURNS:]
    lines = []
    for user_msg, assistant_msg in recent:
        lines.append(f"사용자: {user_msg}")
        lines.append(f"어시스턴트: {assistant_msg}")
    return "\n".join(lines)


def rewrite_query(
    question: str,
    history: List[Tuple[str, str]],
    llm: ChatOpenAI,
) -> str:
    """방법 2: 후속 질문을 독립적인 완성형 질문으로 재작성한다.
    히스토리가 없으면 원래 질문을 그대로 반환한다."""
    if not history:
        return question

    prompt = _REWRITE_PROMPT.format(
        history=_format_history(history),
        question=question,
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content.strip()


# ══════════════════════════════════════════════════════
# 핵심 함수
# ══════════════════════════════════════════════════════

def generate_answer(
    retriever,
    llm,
    question: str,
    mode: str = "qa",
    history: List[Tuple[str, str]] | None = None,
    strategy: str | None = None,
) -> Tuple[str, List[Document]]:
    """질문에 대해 (답변, 근거 문서 리스트)를 반환한다.

    Args:
        retriever : build_retriever()로 만든 검색기
        llm       : ChatOpenAI 또는 ChatOllama 인스턴스
        question  : 사용자 질문
        mode      : "qa" | "summary"
        history   : [(user_msg, assistant_msg), ...] 형태의 대화 히스토리
                    None이거나 빈 리스트면 히스토리 없이 단순 답변
        strategy  : "full" | "rewrite"
                    None이면 config.yaml의 history.strategy 값 사용

    Returns:
        (answer, source_docs)
    """
    history = history or []
    strategy = strategy or HISTORY_STRATEGY
    prompt = QA_PROMPT if mode == "qa" else SUMMARY_PROMPT

    # 검색 쿼리 결정
    # rewrite 전략: 후속 질문을 완성형으로 재작성한 뒤 검색 (검색 품질 향상)
    # full 전략 또는 히스토리 없음: 원래 질문 그대로 검색
    if strategy == "rewrite" and history:
        search_query = rewrite_query(question, history, llm)
    else:
        search_query = question

    docs = retriever.invoke(search_query)
    context = _format_docs(docs)

    # 답변 생성
    # full 전략: 히스토리를 메시지 리스트에 통째로 포함해서 문맥 유지
    # rewrite 전략 또는 히스토리 없음: 단순 체인으로 답변
    if strategy == "full" and history:
        messages = [{"role": "system", "content": prompt.messages[0].prompt.template}]
        for user_msg, assistant_msg in history[-MAX_HISTORY_TURNS:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({
            "role": "user",
            "content": f"문맥:\n{context}\n\n질문: {question}\n답변:"
        })
        answer = llm.invoke(messages).content
    else:
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

    return answer, docs


def build_chain(retriever, llm, mode: str = "qa"):
    """히스토리 없는 단순 체인 (기존 인터페이스 유지).

    체인 입력은 질문 문자열, 출력은 답변 문자열이다.
    mode="qa" | "summary"로 프롬프트를 전환한다.
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