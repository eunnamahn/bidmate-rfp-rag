"""입찰참가자격 적격 여부를 판정하는 에이전트.

대화 히스토리에서 회사 정보(업종·규모·창업연수 등)를 추출하고,
RFP 입찰참가자격 텍스트와 비교해 적격/부적격을 판정한다.
회사 정보가 충분하지 않으면 부족한 항목을 안내하는 메시지를 반환한다.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.config import make_llm

# ─── 프롬프트 ─────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
당신은 대화 내용에서 회사(기업) 정보를 추출하는 전문가입니다.
대화 히스토리를 분석해서 아래 항목들을 찾아 JSON으로만 응답하세요.
언급되지 않은 항목은 null로 설정하세요.
JSON 외 다른 텍스트는 절대 출력하지 마세요.

{
  "업종": "회사의 주요 사업 분야 (예: IT서비스, 소프트웨어 개발, 컨설팅 등)" 또는 null,
  "규모": "기업 규모 (중소기업/중견기업/대기업/소기업 중 하나)" 또는 null,
  "창업연수": 창업·설립 후 경과 년수 정수 또는 null,
  "매출액": 최근 연도 매출액 원 단위 정수 또는 null,
  "직원수": 임직원 수 정수 또는 null,
  "인증": ["보유 인증명 목록 (예: 벤처기업, ISO9001, SW개발보안)"] 또는 null
}\
"""

_EXTRACT_USER = "다음 대화 히스토리에서 회사 정보를 추출하세요:\n\n{history}"

_JUDGE_SYSTEM = """\
당신은 공공 조달 입찰참가자격 적격 여부를 판정하는 전문가입니다.
RFP의 입찰참가자격 요건과 회사 정보를 비교해서 적격/부적격을 판정하세요.
반드시 JSON으로만 응답하세요. JSON 외 다른 텍스트는 절대 출력하지 마세요.

{
  "판정": "적격" 또는 "부적격",
  "사유": "판정 근거를 2~3문장으로 설명한 문자열",
  "충족_요건": ["충족한 요건 항목 목록"],
  "미충족_요건": ["미충족한 요건 항목 목록"],
  "불확실_요건": ["회사 정보 부족으로 판단이 어려운 요건 목록"]
}\
"""

_JUDGE_USER = """\
[입찰참가자격 요건]
{rfp_text}

[회사 정보]
{company_info}

위 회사가 입찰참가자격 요건을 충족하는지 판정하세요.\
"""

# 판정에 최소 하나 이상 있어야 하는 필드
_REQUIRED_FIELDS = ["업종", "규모"]

_REQUEST_MESSAGE = (
    "입찰참가자격 적격 여부를 판정하려면 회사 정보가 필요합니다.\n"
    "아래 항목을 알려주세요:\n\n"
    "- **업종**: 주요 사업 분야 (예: IT서비스, 소프트웨어 개발)\n"
    "- **기업 규모**: 중소기업 / 중견기업 / 대기업 / 소기업\n"
    "- **창업연수**: 설립 후 경과 년수\n"
    "- **매출액**: 최근 연도 매출액\n"
    "- **직원수**: 임직원 수\n"
    "- **보유 인증**: 벤처기업, ISO, SW개발보안 등 (해당되는 경우)\n\n"
    "입력하신 내용을 바탕으로 적격 여부를 분석해 드리겠습니다."
)

# ─── 내부 유틸 ────────────────────────────────────────────────────────────────


def _history_to_text(
    history: List[Union[BaseMessage, Dict[str, str]]],
) -> str:
    """대화 히스토리를 평문 텍스트로 변환한다."""
    lines: List[str] = []
    for msg in history:
        if isinstance(msg, BaseMessage):
            role = getattr(msg, "type", "unknown")
            content = msg.content if hasattr(msg, "content") else str(msg)
        else:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """LLM 응답에서 JSON 블록을 추출·파싱한다."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _has_enough_info(company_info: Dict[str, Any]) -> bool:
    """판정에 충분한 회사 정보가 있는지 확인한다."""
    return any(company_info.get(f) is not None for f in _REQUIRED_FIELDS)


def _company_info_to_text(company_info: Dict[str, Any]) -> str:
    """회사 정보 dict를 사람이 읽기 쉬운 텍스트로 변환한다."""
    labels = {
        "업종": "업종",
        "규모": "기업 규모",
        "창업연수": "창업연수",
        "매출액": "매출액",
        "직원수": "직원수",
        "인증": "보유 인증",
    }
    lines = []
    for key, label in labels.items():
        val = company_info.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            lines.append(f"- {label}: {', '.join(val)}")
        elif key == "매출액":
            lines.append(f"- {label}: {val:,}원")
        else:
            lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "정보 없음"


# ─── 공개 API ─────────────────────────────────────────────────────────────────


def extract_company_info(
    history: List[Union[BaseMessage, Dict[str, str]]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """대화 히스토리에서 회사 정보를 추출한다.

    Args:
        history: LangChain BaseMessage 객체 또는 {"role": ..., "content": ...} dict 목록.
        cfg:     config.yaml에서 로드한 설정 dict.

    Returns:
        {"업종": str|None, "규모": str|None, "창업연수": int|None,
         "매출액": int|None, "직원수": int|None, "인증": list|None}
    """
    empty: Dict[str, Any] = {
        "업종": None, "규모": None, "창업연수": None,
        "매출액": None, "직원수": None, "인증": None,
    }
    if not history:
        return empty

    history_text = _history_to_text(history)
    llm = make_llm(cfg)
    messages = [
        SystemMessage(content=_EXTRACT_SYSTEM),
        HumanMessage(content=_EXTRACT_USER.format(history=history_text)),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception:
        return empty

    data = _parse_json(raw)
    if data is None:
        return empty

    return {
        "업종": data.get("업종") or None,
        "규모": data.get("규모") or None,
        "창업연수": int(data["창업연수"]) if data.get("창업연수") is not None else None,
        "매출액": int(re.sub(r"[^\d]", "", str(data["매출액"]))) if data.get("매출액") else None,
        "직원수": int(data["직원수"]) if data.get("직원수") is not None else None,
        "인증": data.get("인증") if isinstance(data.get("인증"), list) else None,
    }


def check_eligibility(
    history: List[Union[BaseMessage, Dict[str, str]]],
    rfp_text: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """대화 히스토리의 회사 정보와 RFP 입찰참가자격을 비교해 적격 여부를 판정한다.

    Args:
        history:  대화 히스토리 (LangChain 메시지 또는 role/content dict 목록).
        rfp_text: RFP 문서의 입찰참가자격 관련 텍스트.
        cfg:      config.yaml에서 로드한 설정 dict.

    Returns:
        status == "missing_info":
            {"status": "missing_info", "message": str, "company_info": dict}
        status == "eligible" | "ineligible":
            {"status": str, "message": str, "company_info": dict,
             "충족_요건": list, "미충족_요건": list, "불확실_요건": list}
        status == "error":
            {"status": "error", "message": str, "company_info": dict}
    """
    company_info = extract_company_info(history, cfg)

    if not _has_enough_info(company_info):
        return {
            "status": "missing_info",
            "message": _REQUEST_MESSAGE,
            "company_info": company_info,
        }

    company_text = _company_info_to_text(company_info)
    llm = make_llm(cfg)
    messages = [
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=_JUDGE_USER.format(
            rfp_text=rfp_text.strip(),
            company_info=company_text,
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception:
        return {
            "status": "error",
            "message": "판정 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "company_info": company_info,
        }

    data = _parse_json(raw)
    if data is None:
        return {
            "status": "error",
            "message": "판정 결과를 파싱할 수 없습니다.",
            "company_info": company_info,
        }

    판정 = data.get("판정", "")
    status = "eligible" if 판정 == "적격" else "ineligible"
    사유 = data.get("사유", "")

    if status == "eligible":
        message = f"적격입니다.\n\n{사유}"
    else:
        미충족 = data.get("미충족_요건", [])
        미충족_text = "\n".join(f"  - {r}" for r in 미충족) if 미충족 else "  없음"
        message = f"부적격입니다.\n\n{사유}\n\n미충족 요건:\n{미충족_text}"

    불확실 = data.get("불확실_요건", [])
    if 불확실:
        message += (
            "\n\n※ 아래 항목은 제공된 회사 정보가 부족해 확인이 필요합니다:\n"
            + "\n".join(f"  - {r}" for r in 불확실)
        )

    return {
        "status": status,
        "message": message,
        "company_info": company_info,
        "충족_요건": data.get("충족_요건", []),
        "미충족_요건": data.get("미충족_요건", []),
        "불확실_요건": 불확실,
    }
