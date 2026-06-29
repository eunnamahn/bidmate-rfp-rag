"""RFP 문서 텍스트에서 메타데이터를 LLM으로 추출하는 에이전트.

입력: 문서 텍스트 (앞 2000자 권장)
출력: {"발주기관": str|None, "사업금액": int|None, "마감일": str|None, "사업명": str|None}
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
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
    except Exception:
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


# data_list.csv 기본 헤더(파일이 없을 때만 생성용). 실제 파일이 있으면 그 헤더를 따른다.
_CSV_HEADER = [
    "공고 번호", "공고 차수", "사업명", "사업 금액", "발주 기관", "공개 일자",
    "입찰 참여 시작일", "입찰 참여 마감일", "사업 요약", "파일형식", "파일명", "텍스트",
]


def append_metadata_row(
    csv_path: str,
    파일명: str,
    파일형식: str,
    meta: Dict[str, Optional[Any]],
    텍스트: str = "",
    공고번호: str = "",
) -> bool:
    """추출한 메타데이터를 data_list.csv 끝에 한 행으로 추가한다.

    문서가 추가될 때(업로드 등) 자동 호출해 메타데이터를 대장(csv)에 누적한다.
    이미 같은 파일명이 있으면 중복으로 보고 추가하지 않는다.
    헤더는 기존 파일을 따르고(없으면 기본 12컬럼 생성), 알 수 있는 항목만 채운다.
    컬럼명은 공백 유무에 관계없이 매칭한다(예: '사업 금액' == '사업금액').

    Returns: 추가했으면 True, 중복으로 건너뛰면 False.
    """
    path = Path(csv_path)

    header = list(_CSV_HEADER)
    existing_files: set = set()
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        if rows:
            header = rows[0]
            norm = [c.replace(" ", "") for c in header]
            if "파일명" in norm:
                idx = norm.index("파일명")
                existing_files = {r[idx] for r in rows[1:] if len(r) > idx}

    if 파일명 in existing_files:
        return False

    금액 = meta.get("사업금액")
    values = {
        "공고번호": 공고번호 or "",
        "사업명": meta.get("사업명") or "",
        "사업금액": str(금액) if 금액 is not None else "",
        "발주기관": meta.get("발주기관") or "",
        "입찰참여마감일": meta.get("마감일") or "",
        "파일형식": 파일형식 or "",
        "파일명": 파일명,
        "텍스트": (텍스트 or "")[:2000],   # 텍스트 컬럼은 앞부분 발췌만 기록
    }
    row = [values.get(c.replace(" ", ""), "") for c in header]

    new_file = not path.exists()
    # 기존 파일은 utf-8로 이어쓴다(utf-8-sig로 append하면 파일 중간에 BOM이 박힌다).
    enc = "utf-8-sig" if new_file else "utf-8"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding=enc, newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)
    return True
