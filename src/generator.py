"""프롬프트 + LLM 체인 구성 (LCEL).

'입찰메이트' 컨설턴트가 RFP의 핵심(대상 기관·예산·마감일·요구조건·제출방식)을
빠르게 파악하도록 돕는 것이 목표다. 거절 유도 지침을 반드시 포함한다
(문맥에 근거가 없으면 '확인할 수 없습니다'로 거절 — 평가의 거절 정확도와 직결).
QA용/요약용 프롬프트를 각각 제공한다.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 시스템 공통 원칙: 역할 정의 + 컨텍스트 충실(Faithfulness) + 거절 + 금지사항 + 출력형식.
# 거절 문구에는 평가가 감지하는 표지("확인할 수 없습니다")가 반드시 들어가야 한다.
_SYSTEM_BASE = (
    "너는 공공입찰 컨설팅 스타트업 '입찰메이트'의 RFP(제안요청서) 분석 어시스턴트다. "
    "컨설턴트가 수십 페이지 RFP를 직접 읽지 않고도 핵심을 빠르게 파악하도록 돕는 것이 목표다.\n"
    "원칙:\n"
    "1. 아래 제공된 문맥에만 근거해 답하라(Faithfulness). 문맥에 없는 내용은 추측하거나 지어내지 말라.\n"
    "2. 문맥에서 근거를 찾을 수 없으면 '문서에서 확인할 수 없습니다'라고만 답하라.\n"
    "3. 묻지 않은 조언이나 입찰 전략을 덧붙이지 말라. 요청한 정보만 제공하라.\n"
    "4. 추론 과정을 노출하지 말고 결론만 한국어로 간결히 제시하라.\n"
    "5. 핵심 정보를 먼저 제시하고, 가능하면 근거가 된 섹션명/항목을 함께 표기하라. "
    "원문이 표일 때만 표 형식을 사용하라."
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_BASE),
        ("human", "문맥:\n{context}\n\n질문: {question}\n답변:"),
    ]
)

# 요약은 컨설턴트가 필요한 4대 정보축(+마감일)을 고정 양식으로 항상 채운다.
# 문서 자체를 못 찾으면 전체를 거절하고, 개별 항목 근거가 없으면 '문서에 명시되지 않음'으로
# 적어 요약 전체가 거절로 오인되지 않게 한다.
_SUMMARY_FORMAT = (
    "요청한 사업의 내용이 문맥에 전혀 없으면 '문서에서 확인할 수 없습니다'라고만 답하라.\n"
    "문맥에 있으면 아래 양식을 빠짐없이 채워 정리하라. 개별 항목의 근거가 문맥에 없으면 "
    "그 항목에 '문서에 명시되지 않음'이라고 적어라:\n"
    "- 대상 기관:\n"
    "- 예산(사업금액):\n"
    "- 입찰 마감일:\n"
    "- 핵심 요구조건:\n"
    "- 제출 방식:"
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_BASE + "\n\n" + _SUMMARY_FORMAT),
        ("human", "문맥:\n{context}\n\n요청: {question}\n정리:"),
    ]
)


def _format_docs(docs: List[Document]) -> str:
    """검색된 문서들을 하나의 문맥 문자열로 합친다."""
    return "\n\n---\n\n".join(d.page_content for d in docs)


def generate_answer(retriever, llm, question: str, mode: str = "qa"):
    """질문에 대해 (답변, 근거 문서 리스트)를 함께 반환한다.

    검색을 한 번만 수행해 답변과 근거를 같이 얻는다(백엔드 sources 표시용).
    """
    prompt = QA_PROMPT if mode == "qa" else SUMMARY_PROMPT
    docs = retriever.invoke(question)
    context = _format_docs(docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs


def build_chain(retriever, llm, mode: str = "qa"):
    """검색기→프롬프트→LLM→문자열 파서로 이어지는 LCEL 체인을 만든다.

    체인 입력은 질문 문자열, 출력은 답변 문자열이다.
    mode="qa" | "summary"로 프롬프트를 전환한다.
    """
    if mode == "qa":
        prompt = QA_PROMPT
    elif mode == "summary":
        prompt = SUMMARY_PROMPT
    else:
        raise ValueError(f"지원하지 않는 mode: {mode}")

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
