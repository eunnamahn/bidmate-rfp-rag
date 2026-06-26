"""RFP 텍스트에서 리스크/실격 조항을 탐지하는 에이전트."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import make_llm

# ─── 프롬프트 ─────────────────────────────────────────────────────────────────

_SYSTEM = """\
당신은 공공 조달 RFP(제안요청서) 문서에서 리스크 및 실격 조항을 탐지하는 전문가입니다.
주어진 RFP 텍스트를 분석해서 아래 유형의 리스크 항목을 모두 찾아 JSON 배열로만 응답하세요.
JSON 외 다른 텍스트는 절대 출력하지 마세요.

탐지 대상 리스크 유형:
- 인증 요구: 특정 자격증, 인증, 허가 보유 의무
- 실적 요건: 유사 프로젝트 수행 실적 또는 납품 실적 요구
- 기술 요건: 특정 기술 스택, 솔루션, 장비 보유 의무
- 기업 규모: 중소기업, 중견기업 등 기업 규모 제한
- 실격 조항: 특정 조건 미충족 시 입찰 자격 박탈

응답 형식 (JSON 배열):
[
  {
    "항목": "리스크 항목 이름 (짧게)",
    "내용": "해당 조항의 원문 또는 요약",
    "유형": "인증 요구 | 실적 요건 | 기술 요건 | 기업 규모 | 실격 조항",
    "심각도": "높음 | 중간 | 낮음",
    "심각도_이유": "이 심각도로 분류한 이유"
  }
]

심각도 기준:
- 높음: 미충족 시 즉시 실격되거나 참여 자체가 불가능한 조건
- 중간: 충족이 어렵거나 별도 준비가 필요한 조건
- 낮음: 충족 가능성이 높거나 대응 방안이 있는 조건

탐지된 리스크가 없으면 빈 배열 []을 반환하세요.\
"""

_USER = """\
다음 RFP 텍스트에서 리스크 및 실격 조항을 탐지하세요:

{rfp_text}\
"""

# ─── 내부 유틸 ────────────────────────────────────────────────────────────────


def _parse_json_array(raw: str) -> Optional[List[Dict[str, Any]]]:
    """LLM 응답에서 JSON 배열을 추출·파싱한다."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group())
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None


_VALID_SEVERITIES = {"높음", "중간", "낮음"}
_VALID_TYPES = {"인증 요구", "실적 요건", "기술 요건", "기업 규모", "실격 조항"}


def _normalize_item(item: Any) -> Optional[Dict[str, Any]]:
    """파싱된 항목을 검증하고 정규화한다."""
    if not isinstance(item, dict):
        return None
    severity = item.get("심각도", "")
    if severity not in _VALID_SEVERITIES:
        severity = "낮음"
    risk_type = item.get("유형", "")
    if risk_type not in _VALID_TYPES:
        risk_type = "실격 조항"
    return {
        "항목": str(item.get("항목", "")).strip(),
        "내용": str(item.get("내용", "")).strip(),
        "유형": risk_type,
        "심각도": severity,
        "심각도_이유": str(item.get("심각도_이유", "")).strip(),
    }


# ─── 공개 API ─────────────────────────────────────────────────────────────────


def detect_risks(
    rfp_text: str,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """RFP 텍스트에서 리스크 및 실격 조항을 탐지한다.

    Args:
        rfp_text: 분석할 RFP 문서 텍스트.
        cfg:      config.yaml에서 로드한 설정 dict.

    Returns:
        리스크 항목 목록. 각 항목은 아래 키를 가진다:
        - "항목": str — 리스크 항목 이름
        - "내용": str — 해당 조항 원문 또는 요약
        - "유형": str — "인증 요구" | "실적 요건" | "기술 요건" | "기업 규모" | "실격 조항"
        - "심각도": str — "높음" | "중간" | "낮음"
        - "심각도_이유": str — 심각도 분류 이유
        탐지 실패 시 빈 리스트를 반환한다.
    """
    if not rfp_text or not rfp_text.strip():
        return []

    llm = make_llm(cfg)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=_USER.format(rfp_text=rfp_text.strip())),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception:
        return []

    items = _parse_json_array(raw)
    if items is None:
        return []

    result = []
    for item in items:
        normalized = _normalize_item(item)
        if normalized and normalized["항목"]:
            result.append(normalized)

    return result
