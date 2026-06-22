"""프롬프트 + LLM 체인 구성 (LCEL).

거절 유도 지침을 반드시 포함한다(평가의 거절 정확도와 직결).
QA용/요약용 프롬프트를 각각 제공한다.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 거절 유도 지침: 문맥에 근거가 없으면 지어내지 말고 거절하게 한다.
_REFUSAL = (
    "아래 문맥만 근거로 답하라. 문맥에 없으면 '문서에서 확인할 수 없습니다'라고 답하라."
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _REFUSAL + " 한국어로 간결하게 답하라."),
        ("human", "문맥:\n{context}\n\n질문: {question}\n답변:"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            _REFUSAL
            + " 아래 양식에 맞춰 RFP를 정리하라:\n"
            + "- 발주기관:\n- 사업금액:\n- 입찰마감일:\n- 핵심 요구사항:",
        ),
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
