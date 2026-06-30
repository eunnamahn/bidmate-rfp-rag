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

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
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
9. "그 공고", "거기", "그것", "그럼" 등 대명사로 지칭되는 대상은 직전 대화에서 언급된 공고·기관을 기준으로 답변하세요. 임의로 다른 공고·기관으로 전환하지 마세요.
10. 사용자가 새 주제를 명시하지 않는 한, 이전 대화에서 다루던 공고·기관을 계속 유지해서 답변하세요.
11. 이전 답변과 모순되는 내용을 출력하지 마세요. 동일 문서에 대해 앞뒤가 다른 정보를 제공하면 안 됩니다.

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

[재작성 규칙]
- "그 공고", "거기", "그것", "그럼" 등 대명사나 생략된 주어가 있으면, 직전 대화에서 언급된 공고명 또는 기관명으로 반드시 대체하세요.
- 질문의 의도(무엇을 묻는지)는 절대 바꾸지 마세요.
- 재작성된 질문만 출력하고 다른 설명은 하지 마세요.

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


def _summarize_old_history(old_turns: List[Tuple[str, str]], llm) -> str:
    """MAX_HISTORY_TURNS를 초과한 오래된 대화를 LLM으로 한 줄 요약한다."""
    if not old_turns:
        return ""
    lines = []
    for user_msg, assistant_msg in old_turns:
        lines.append(f"사용자: {user_msg}")
        lines.append(f"어시스턴트: {assistant_msg}")
    history_text = "\n".join(lines)
    prompt = (
        "아래 대화를 한 문장으로 요약하세요. 핵심 주제(공고명·기관명·주요 질문)를 포함하세요. "
        "요약문만 출력하세요.\n\n" + history_text
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


def _norm_match(text: Any) -> str:
    """매칭용 정규화: 공백/괄호 제거 + 소문자화."""
    return re.sub(r"[\s()（）]", "", str(text)).lower()


def _docs_mentioned_in_answer(docs: List[Document], answer: str) -> List[Document]:
    """답변 본문에 기관명/사업명이 실제로 등장한 문서만 추린다.

    모델이 [사용근거] 인용을 빠뜨렸을 때 검색 문서 '전부'를 근거로 노출하면
    질문과 무관한 검색 결과(예: 다른 기관의 유사 시스템 RFP)까지 출처로 표시된다.
    답변이 실제로 언급한 기관/사업명을 가진 문서만 남겨 이 오노출을 막는다.
    """
    a = _norm_match(answer)
    picked: List[Document] = []
    for d in docs:
        for key in ("기관", "사업명"):
            v = _norm_match(d.metadata.get(key) or "")
            if len(v) >= 3 and v in a:
                picked.append(d)
                break
    return picked


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
    """답변 본문에서 인용 표기를 제거한다.

    '[사용근거: ...]', '[출처: 자료 N]'/'(문서: 자료 N)' 인라인 표기,
    그리고 '- 출처: 자료 1' 같은 줄 단위 표기까지 모두 제거한다.
    """
    # 1) "출처:"/"사용근거:" 로 시작하는 줄(불릿 포함) 제거
    text = re.sub(
        r"(?m)^[ \t]*[-*•]?[ \t]*(?:출처|사용\s*근거)[ \t]*[:：].*$", "", text
    )
    # 2) 라벨만 있는 헤더 줄("출처"/"사용근거") 제거
    text = re.sub(r"(?m)^[ \t]*#*[ \t]*[-*•]?[ \t]*(?:출처|사용\s*근거)[ \t]*$", "", text)
    # 3) "자료 N"만 있는 줄(불릿 포함) 제거
    text = re.sub(r"(?m)^[ \t]*[-*•]?[ \t]*자료\s*\d+[ \t]*$", "", text)
    # 4) 인라인 표기 제거: [사용근거: ...], [출처: 자료 N], (문서: 자료 N)
    text = re.sub(r"\[?\s*사용\s*근거\s*[:：][^\]\n]*\]?", "", text)
    text = re.sub(r"[\[(]\s*(?:출처|문서)\s*[:：]?\s*자료\s*\d+[^\])\n]*[\])]", "", text)
    # 공백/빈 줄 정리
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
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
    active_filename: Optional[str] = None,
) -> Tuple[str, List[Document]]:
    """질문에 대해 (답변, 근거 문서 리스트)를 반환한다.

    history         : [(user_msg, assistant_msg), ...] 대화 히스토리(없으면 단순 답변).
    strategy        : "full" | "rewrite" (None이면 config의 history.strategy).
    active_filename : 현재 분석 중인 공고 파일명(지정 시 시스템 프롬프트에 명시).
    검색을 한 번만 수행해 답변과 근거를 같이 얻는다(백엔드 sources 표시용).
    """
    history = history or []
    strategy = strategy or HISTORY_STRATEGY
    system_text = _SYSTEM_QA if mode == "qa" else _SYSTEM_SUMMARY

    # 현재 분석 중인 공고를 시스템 메시지에 명시
    if active_filename:
        system_text = system_text + f"\n\n[현재 분석 중인 공고]\n{active_filename}"

    # 히스토리 압축: MAX_HISTORY_TURNS 초과분은 LLM으로 한 줄 요약해 시스템 메시지에 포함
    if len(history) > MAX_HISTORY_TURNS:
        old_turns = history[:-MAX_HISTORY_TURNS]
        recent_turns = history[-MAX_HISTORY_TURNS:]
        summary = _summarize_old_history(old_turns, llm)
        if summary:
            system_text = system_text + f"\n\n[이전 대화 요약]\n{summary}"
    else:
        recent_turns = history

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

    # full 전략: 최근 MAX_HISTORY_TURNS 턴을 메시지로 포함해 문맥 유지
    messages = [{"role": "system", "content": system_text}]
    if strategy == "full" and recent_turns:
        for user_msg, assistant_msg in recent_turns:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_content})

    raw = getattr(llm.invoke(messages), "content", "")
    cited = _parse_citation(raw)
    answer = _strip_citation(raw)

    # 출처 선정 우선순위:
    #   1) 모델이 인용한 번호([사용근거: ...])가 있으면 그 문서만.
    #   2) 거절 답변이면 출처 없음.
    #   3) 인용이 없거나 무효면 답변에 실제로 언급된(기관/사업명) 문서만.
    #      그래도 못 찾으면 가장 관련도 높은 1건만(검색 전체를 근거로 띄우지 않는다).
    if cited:
        used = [unique_docs[n - 1] for n in cited if 1 <= n <= len(unique_docs)]
    elif _REFUSAL_MARK in answer:
        used = []
    else:
        used = []
    if not used and _REFUSAL_MARK not in answer:
        used = _docs_mentioned_in_answer(unique_docs, answer) or unique_docs[:1]

    return answer, used


# ══════════════════════════════════════════════════════
# 다이어그램 / 체크리스트 / 인포그래픽 생성 (구조화 출력)
#   검색 → 전용 프롬프트 → LLM. 결과는 챗 액션 버튼에서 사용한다.
# ══════════════════════════════════════════════════════

_SYSTEM_DIAGRAM_EVAL = """당신은 공공기관 RFP 문서를 분석하는 전문가입니다.
주어진 RFP 문서에서 평가배점 구조를 추출하여 Mermaid 다이어그램으로 출력하세요.

[출력 규칙]
1. 반드시 ```mermaid 코드 블록으로 시작하고 ``` 로 끝내세요.
2. mindmap 또는 graph TD 형식을 사용하세요.
3. 평가항목과 배점(점수)을 노드에 포함하세요.
4. 문서에 평가배점 정보가 없으면 "문서에서 평가배점 정보를 확인할 수 없습니다."라고만 답하세요.
5. 코드 블록 외에 설명을 추가하지 마세요.

[출력 예시]
```mermaid
graph TD
    A[평가 총점: 100점] --> B[기술평가 80점]
    A --> C[가격평가 20점]
    B --> D[수행능력 30점]
    B --> E[기술제안 50점]
```"""

_SYSTEM_DIAGRAM_GANTT = """당신은 공공기관 RFP 문서를 분석하는 전문가입니다.
주어진 RFP 문서에서 입찰 일정을 추출하여 Mermaid Gantt 다이어그램으로 출력하세요.

[출력 규칙]
1. 반드시 ```mermaid 코드 블록으로 시작하고 ``` 로 끝내세요.
2. gantt 형식을 사용하세요.
3. dateFormat YYYY-MM-DD 를 사용하세요.
4. 날짜가 불명확한 경우 after 구문으로 상대적 순서만 표현하세요.
5. 문서에 일정 정보가 없으면 "문서에서 일정 정보를 확인할 수 없습니다."라고만 답하세요.
6. 코드 블록 외에 설명을 추가하지 마세요.

[출력 예시]
```mermaid
gantt
    title 입찰 일정
    dateFormat YYYY-MM-DD
    section 공고
        입찰공고          :done, 2024-01-01, 2024-01-10
    section 질의
        질의서 접수       :active, 2024-01-10, 5d
    section 입찰
        입찰서 제출       :2024-01-15, 3d
        개찰              :milestone, 2024-01-18, 0d
```"""

_SYSTEM_DIAGRAM_ELIGIBILITY = """당신은 공공기관 RFP 문서를 분석하는 전문가입니다.
주어진 RFP 문서에서 입찰 참가자격 요건을 추출하여 적격성 판정 흐름도를 Mermaid 다이어그램으로 출력하세요.

[출력 규칙]
1. 반드시 ```mermaid 코드 블록으로 시작하고 ``` 로 끝내세요.
2. flowchart TD 형식을 사용하세요.
3. 각 자격 요건을 조건 분기(다이아몬드)로 표현하고, 충족/미충족 경로를 나누세요.
4. 최종 결론은 "입찰 참가 가능" 또는 "입찰 참가 불가"로 끝내세요.
5. 문서에 자격 요건 정보가 없으면 "문서에서 참가자격 정보를 확인할 수 없습니다."라고만 답하세요.
6. 코드 블록 외에 설명을 추가하지 마세요.

[출력 예시]
```mermaid
flowchart TD
    A[입찰 참가 신청] --> B{면허 보유?}
    B -- 예 --> C{실적 요건 충족?}
    B -- 아니오 --> Z[입찰 참가 불가]
    C -- 예 --> D[입찰 참가 가능]
    C -- 아니오 --> Z
```"""

# 시스템 프롬프트에 JSON/Mermaid 예시의 중괄호가 많아, f-string 템플릿으로 파싱되지 않도록
# 고정 SystemMessage로 넘긴다(변수 치환은 human 메시지의 {context}에만 적용).
DIAGRAM_PROMPTS = {
    "evaluation": ChatPromptTemplate.from_messages([
        SystemMessage(content=_SYSTEM_DIAGRAM_EVAL),
        ("human", "문맥:\n{context}\n\n위 RFP의 평가배점 구조를 Mermaid 다이어그램으로 출력하세요."),
    ]),
    "gantt": ChatPromptTemplate.from_messages([
        SystemMessage(content=_SYSTEM_DIAGRAM_GANTT),
        ("human", "문맥:\n{context}\n\n위 RFP의 입찰 일정을 Mermaid Gantt 다이어그램으로 출력하세요."),
    ]),
    "eligibility": ChatPromptTemplate.from_messages([
        SystemMessage(content=_SYSTEM_DIAGRAM_ELIGIBILITY),
        ("human", "문맥:\n{context}\n\n위 RFP의 입찰 참가자격 요건을 적격성 판정 흐름도로 출력하세요."),
    ]),
}


def generate_diagram(
    retriever,
    llm,
    question: str,
    diagram_type: str,
) -> Tuple[str, List[Document]]:
    """RFP 문서에서 다이어그램을 생성한다.

    Args:
        retriever    : build_retriever()로 만든 검색기
        llm          : ChatOpenAI 또는 ChatOllama 인스턴스
        question     : 사용자가 입력한 사업명 또는 검색 쿼리
        diagram_type : "evaluation" | "gantt" | "eligibility"

    Returns:
        (mermaid_code, source_docs)
    """
    if diagram_type not in DIAGRAM_PROMPTS:
        raise ValueError(f"지원하지 않는 diagram_type: {diagram_type}")

    docs = retriever.invoke(question)
    context = _format_docs(docs)
    prompt = DIAGRAM_PROMPTS[diagram_type]
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context})
    return result, docs


_SYSTEM_CHECKLIST = """당신은 공공기관 RFP 문서를 분석하는 전문 입찰 컨설턴트입니다.
주어진 RFP 문서에서 입찰 준비에 필요한 항목을 카테고리별로 추출하여 JSON으로 출력하세요.

[추출 대상 카테고리]
1. "제출서류" - 입찰 시 제출해야 하는 서류 목록
2. "참가자격" - 입찰 참가자격 요건 (면허, 실적, 자본금, 인력 등)
3. "평가준비" - 기술평가·제안서 작성을 위해 준비해야 할 항목
4. "일정확인" - 놓치면 안 되는 핵심 일정 및 마감

[출력 규칙]
1. 반드시 아래 JSON 형식만 출력하세요. 코드 블록(```)이나 설명을 추가하지 마세요.
2. 각 항목은 "label"(항목명)과 "note"(비고/조건, 없으면 빈 문자열)로 구성하세요.
3. 문서에서 확인되지 않는 카테고리는 빈 배열([])로 남기세요.
4. 항목은 중복 없이 구체적으로 작성하세요.

[출력 형식]
{
  "title": "사업명",
  "categories": [
    {
      "name": "제출서류",
      "items": [
        {"label": "입찰참가신청서", "note": "소정 양식 사용"},
        {"label": "사업자등록증 사본", "note": ""}
      ]
    },
    {
      "name": "참가자격",
      "items": [
        {"label": "소프트웨어사업자 신고 확인서", "note": ""},
        {"label": "동종 사업 수행실적 1건 이상", "note": "최근 5년 이내"}
      ]
    },
    {
      "name": "평가준비",
      "items": [
        {"label": "기술제안서 작성", "note": "OOO 방법론 포함"},
        {"label": "수행조직 구성안", "note": "PM 포함 최소 3인"}
      ]
    },
    {
      "name": "일정확인",
      "items": [
        {"label": "질의서 제출 마감", "note": "입찰공고일로부터 7일 이내"},
        {"label": "입찰서 제출 마감", "note": "2024-03-15 18:00"}
      ]
    }
  ]
}"""

_CHECKLIST_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=_SYSTEM_CHECKLIST),
    ("human", "문맥:\n{context}\n\n위 RFP의 입찰 준비 체크리스트를 JSON으로 출력하세요."),
])


_SYSTEM_INFOGRAPHIC = """당신은 공공기관 RFP 문서를 분석하는 전문 입찰 컨설턴트입니다.
주어진 RFP 문서에서 인포그래픽 카드에 표시할 핵심 정보를 추출하여 JSON으로 출력하세요.

[출력 규칙]
1. 반드시 아래 JSON 형식만 출력하세요. 코드 블록(```)이나 설명을 추가하지 마세요.
2. 문서에서 확인되지 않는 항목은 null로 표기하세요.
3. evaluation_scores는 평가항목과 배점을 최대 5개까지만 추출하세요.
4. key_requirements는 핵심 요구사항을 3~5개 간결한 문장으로 추출하세요.
5. 모든 문자열은 한국어로 작성하세요.

[출력 형식]
{
  "title": "사업명",
  "agency": "발주기관명",
  "budget": "50억원",
  "deadline": "2024.03.15",
  "contract_type": "협상에 의한 계약",
  "duration": "24개월",
  "evaluation_scores": [
    {"name": "기술평가", "score": 80, "max_score": 100},
    {"name": "가격평가", "score": 20, "max_score": 100}
  ],
  "key_requirements": [
    "AWS/Azure 기반 클라우드 전환",
    "PM 10년 이상 경력 필수",
    "보안 인증 (ISMS) 보유"
  ],
  "qualification_summary": "소프트웨어사업자 신고 업체, 동종 실적 1건 이상"
}"""

_INFOGRAPHIC_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=_SYSTEM_INFOGRAPHIC),
    ("human", "문맥:\n{context}\n\n위 RFP의 인포그래픽 카드용 핵심 정보를 JSON으로 출력하세요."),
])


def _parse_json_block(raw: str) -> Optional[Dict]:
    """LLM 응답에서 JSON을 파싱한다(```json 코드 블록도 처리). 실패 시 None."""
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# 평가배점 텍스트 패턴
# 기존 문자 클래스 방식([능력평가점수]*)은 "기술제안서", "기술평가항목" 등을 못 잡아냄.
# \S* 로 키워드 뒤 접미어를, [^\d\n]{0,20} 으로 ": " 또는 " 배점 " 같은 구분자를 허용.
# 점/% 접미어를 필수로 요구해 단순 숫자 언급과 구분한다.
_EVAL_SCORE_RE = re.compile(
    r'(기술|가격|제안|수행|발표|창의)'
    r'\S*'               # 제안서, 평가항목, 제안평가 등 임의 접미어
    r'(?:[^\d\n]{0,20})' # 콜론·공백·레이블 등 구분자 (최대 20자, 줄넘김 금지)
    r'(\d{1,3})\s*(?:점|%)',
)
_EVAL_LABEL_MAP = {"기술": "기술평가", "가격": "가격평가", "제안": "제안평가", "수행": "수행능력평가"}


def _fallback_eval_scores(text: str) -> List[Dict]:
    """LLM 추출 실패 시 RFP 텍스트에서 직접 배점을 추출한다."""
    found: Dict[str, int] = {}
    for m in _EVAL_SCORE_RE.finditer(text):
        raw_name, score = m.group(1), int(m.group(2))
        if not (1 <= score <= 100):   # 비율 범위 밖은 무시
            continue
        # 대표 레이블로 정규화
        key = next((v for k, v in _EVAL_LABEL_MAP.items() if k in raw_name), raw_name)
        if key not in found:
            found[key] = score
    return [{"name": k, "score": v, "max_score": 100} for k, v in found.items()]


def generate_infographic_data(
    retriever,
    llm,
    question: str,
) -> Tuple[Dict, List[Document]]:
    """RFP 문서에서 인포그래픽 카드에 표시할 핵심 정보를 JSON으로 추출한다.

    Returns:
        (info_dict, source_docs). 파싱 실패 시 info_dict={"parse_error": 원본}.
    """
    # 평가배점 관련 청크가 검색되도록 키워드를 쿼리에 추가
    eval_query = f"{question} 평가배점 기술평가 가격평가 평가기준"
    docs = retriever.invoke(eval_query)
    context = _format_docs(docs)
    chain = _INFOGRAPHIC_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"context": context})
    result = _parse_json_block(raw)
    if result is None:
        result = {"parse_error": raw}
        return result, docs

    # LLM이 배점을 모두 0 또는 null로 반환한 경우 텍스트 직접 파싱으로 보완
    scores = result.get("evaluation_scores") or []
    all_zero = not scores or all(not (s.get("score") or 0) for s in scores)
    if all_zero:
        fallback = _fallback_eval_scores(context)
        if fallback:
            result["evaluation_scores"] = fallback

    return result, docs


def generate_checklist(
    retriever,
    llm,
    question: str,
) -> Tuple[Dict, List[Document]]:
    """RFP 문서에서 입찰 준비 체크리스트를 카테고리별 JSON으로 추출한다.

    Returns:
        (checklist_dict, source_docs)
        checklist_dict 구조: {"title": str, "categories": [{"name": str, "items": [...]}]}
        파싱 실패 시 {"title": question, "categories": [], "parse_error": 원본}.
    """
    docs = retriever.invoke(question)
    context = _format_docs(docs)
    chain = _CHECKLIST_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"context": context})
    result = _parse_json_block(raw)
    if result is None:
        result = {"title": question, "categories": [], "parse_error": raw}
    return result, docs


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
