"""RFP 문서 텍스트에서 메타데이터를 LLM으로 추출하는 에이전트.

입력: 문서 텍스트 (앞 2000자 권장)
출력: {"발주기관": str|None, "사업금액": int|None, "마감일": str|None, "사업명": str|None}
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import make_llm

_EMPTY: Dict[str, Optional[Any]] = {
    "발주기관": None,
    "사업금액": None,
    "마감일": None,
    "사업명": None,
}

_SYSTEM = """\
당신은 공공 조달 RFP(제안요청서) 문서 분석 전문가입니다.
주어진 텍스트에서 아래 4개 항목을 추출해 JSON으로만 응답하세요.
추출할 수 없는 항목은 null로 설정하세요.
JSON 외 다른 텍스트는 절대 출력하지 마세요.

{
  "발주기관": "기관명 전체 (예: 벤처기업협회)" 또는 null,
  "사업금액": 원 단위 정수 (예: 3520000000) 또는 null,
  "마감일": "YYYY-MM-DD 형식 날짜" 또는 null,
  "사업명": "사업명 전체" 또는 null
}\
"""

_USER = "다음 RFP 문서 텍스트에서 메타데이터를 추출하세요:\n\n{text}"


def _to_int(value: Any) -> Optional[int]:
    """금액 값을 정수로 변환한다. 변환 불가 시 None 반환."""
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", str(value))
    if not digits:
        return None
    amount = int(digits)
    return None if amount == 0 else amount


def extract_metadata(text: str, cfg: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    """RFP 문서 텍스트에서 발주기관·사업금액·마감일·사업명을 추출한다.

    Args:
        text: 문서 본문 (앞 2000자 정도를 권장).
        cfg:  config.yaml 에서 로드한 설정 dict (make_llm에 그대로 전달).

    Returns:
        {"발주기관": str|None, "사업금액": int|None, "마감일": str|None, "사업명": str|None}
        추출 실패한 항목은 None.
    """
    snippet = text[:2000].strip()
    snippet = re.sub(r'(?<=\w) (?=\w)', '', snippet)
    if not snippet:
        return dict(_EMPTY)

    llm = make_llm(cfg)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=_USER.format(text=snippet)),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        print("LLM 응답:", raw)  # 추가
    except Exception as e:
        print("LLM 에러:", e)  # 추가
        return dict(_EMPTY)

    # ```json ... ``` 블록 또는 순수 {} 블록 추출
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return dict(_EMPTY)

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return dict(_EMPTY)

    return {
        "발주기관": data.get("발주기관") or None,
        "사업금액": _to_int(data.get("사업금액")),
        "마감일": data.get("마감일") or None,
        "사업명": data.get("사업명") or None,
    }
