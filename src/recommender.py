"""규칙 기반 공고 추천 엔진.

나라장터 API에서 가져온 공고 목록을 회사 프로필 기준으로 필터링·점수화한다.

점수 구성 (100점 만점):
  - 사업분류 일치      : 40점 (대분류 10 / 중분류 15 / 세분류 15)
  - 예산 범위 내       : 30점
  - 입찰방법 선호 일치 : 15점
  - 마감 여유 (3일+)   : 15점
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _parse_dt_v2(s: Optional[str]) -> Optional[datetime]:
    """날짜 문자열을 파싱한다. 여러 포맷을 시도한다.

    각 포맷이 실제로 만들어내는 문자열 길이(예: "%Y"는 4자리)로 잘라야 하는데,
    과거 코드는 len(fmt)로 포맷 "패턴 문자열" 자체의 글자 수(예: "%Y%m%d%H%M%S"는
    12자)로 잘라 나라장터가 실제로 주는 "2026-07-07 12:00:00" 같은 값이 항상
    파싱에 실패해 None이 되던 버그가 있었다(마감 여유 15점이 전 공고에서 늘 0점 처리됨).
    """
    if not s:
        return None
    for fmt, length in (
        ("%Y%m%d%H%M%S", 14),
        ("%Y%m%d%H%M", 12),
        ("%Y%m%d", 8),
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y-%m-%dT%H:%M:%S", 19),
        ("%Y-%m-%d", 10),
    ):
        try:
            return datetime.strptime(s[:length], fmt)
        except ValueError:
            continue
    return None


_CATEGORY_SPLIT = re.compile(r"[/,·\s]+")

# CompanySetup.tsx의 고정 업종 드롭다운(INDUSTRY_OPTIONS) → 나라장터 분류명에서
# 실제로 쓰이는 동의어/약어. "IT서비스/소프트웨어"를 토큰 분리해도 "SW", "ICT" 같은
# 약어와는 문자열이 달라 매칭이 걸리지 않아 별도로 확장한다.
_INDUSTRY_SYNONYMS: Dict[str, List[str]] = {
    "IT서비스/소프트웨어": ["소프트웨어", "SW", "ICT", "정보시스템", "정보처리", "IT서비스"],
    "시스템통합(SI)": ["시스템통합", "SI", "ICT", "정보시스템", "시스템"],
    "데이터/AI": ["데이터", "AI", "인공지능", "빅데이터", "정보시스템"],
    "건설/엔지니어링": ["건설", "토목", "건축", "엔지니어링", "시설공사"],
    "컨설팅": ["컨설팅", "자문", "연구조사", "경영"],
    "교육/훈련": ["교육", "훈련", "연수", "수련활동"],
    "의료/바이오": ["의료", "바이오", "제약", "헬스케어", "병원"],
    "제조": ["제조", "생산", "장비", "기자재"],
    "유통/물류": ["유통", "물류", "운송", "보관", "배송"],
}


def _category_matches(pref: str, bid_value: str) -> bool:
    """분류 일치 여부를 판단한다.

    회사 프로필의 업종은 CompanySetup의 고정 드롭다운 값(예: "IT서비스/소프트웨어")이라
    나라장터 공식 분류명(예: "소프트웨어개발", "SW 및 시스템 개발")과 완전 포함 관계가
    되는 경우가 드물다. "/" 등으로 분리한 토큰 단위 부분 일치에 더해, 약어(SW/ICT 등)
    까지 잡아내도록 고정 동의어 사전(_INDUSTRY_SYNONYMS)도 함께 확인한다.
    """
    if not pref or not bid_value:
        return False
    if pref in bid_value or bid_value in pref:
        return True
    tokens = [t for t in _CATEGORY_SPLIT.split(pref) if len(t) >= 2]
    if any(t in bid_value for t in tokens):
        return True
    synonyms = _INDUSTRY_SYNONYMS.get(pref, [])
    return any(s in bid_value for s in synonyms)


# score_breakdown()이 반환하는 근거 키 → 사유는 프론트(REASON_LABELS 등)에서 문구로 매핑한다.
REASON_CATEGORY = "category"
REASON_BUDGET = "budget"
REASON_METHOD = "method"
REASON_DEADLINE = "deadline"


def score_breakdown(bid: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[int, List[str]]:
    """단일 공고에 대해 회사 프로필 적합 점수(0~100)와, 만점을 받은 근거 태그를 함께 계산한다.

    근거 태그는 카드에 "왜 추천됐는지" 짧게 보여주기 위한 것이라, 프로필에 실제
    제약이 없어 항상 만족되는 항목(budget_min/max 둘 다 None, bid_methods 빈 배열)은
    태그를 붙이지 않는다 — 그렇지 않으면 거의 모든 카드에 의미 없이 뜨게 된다.

    Returns: (score, reasons) — reasons는 REASON_* 값들의 리스트.
    """
    score = 0
    reasons: List[str] = []

    # 1. 사업분류 일치 (40점)
    pref_lrg = (profile.get("large_category") or "").strip()
    pref_mid = (profile.get("mid_category") or "").strip()
    pref_sml = (profile.get("small_category") or "").strip()

    bid_lrg = (bid.get("pubPrcrmntLrgClsfcNm") or "").strip()
    bid_mid = (bid.get("pubPrcrmntMidClsfcNm") or "").strip()
    bid_sml = (bid.get("pubPrcrmntClsfcNm") or "").strip()

    category_hit = False
    if _category_matches(pref_lrg, bid_lrg):
        score += 10
        category_hit = True
    if _category_matches(pref_mid, bid_mid):
        score += 15
        category_hit = True
    if _category_matches(pref_sml, bid_sml):
        score += 15
        category_hit = True
    if category_hit:
        reasons.append(REASON_CATEGORY)

    # 2. 예산 범위 (30점)
    budget_min = profile.get("budget_min")  # 원 단위
    budget_max = profile.get("budget_max")
    bid_price = bid.get("presmptPrce") or bid.get("asignBdgtAmt")
    try:
        bid_price = int(bid_price)
    except (TypeError, ValueError):
        bid_price = None

    if bid_price is not None:
        in_min = (budget_min is None) or (bid_price >= int(budget_min))
        in_max = (budget_max is None) or (bid_price <= int(budget_max))
        if in_min and in_max:
            score += 30
            if budget_min is not None or budget_max is not None:
                reasons.append(REASON_BUDGET)
        elif in_min or in_max:
            score += 10  # 한쪽만 맞아도 부분 점수

    # 3. 입찰방법 선호 일치 (15점)
    pref_methods: List[str] = profile.get("bid_methods") or []
    bid_method = (bid.get("bidMethdNm") or "").strip()
    method_hit = bool(pref_methods) and any(m in bid_method for m in pref_methods)
    if not pref_methods or method_hit:
        score += 15
        if method_hit:
            reasons.append(REASON_METHOD)

    # 4. 마감 여유 (15점) — 오늘로부터 3일 이상 남은 공고
    close_dt = _parse_dt_v2(bid.get("bidClseDt"))
    if close_dt:
        days_left = (close_dt - datetime.now()).days
        if days_left >= 3:
            score += 15
            reasons.append(REASON_DEADLINE)
        elif days_left >= 0:
            score += 5

    return score, reasons


def score_bid(bid: Dict[str, Any], profile: Dict[str, Any]) -> int:
    """단일 공고에 대해 회사 프로필 적합 점수(0~100)만 계산한다(하위 호환용)."""
    return score_breakdown(bid, profile)[0]


def recommend(
    bids: List[Dict[str, Any]],
    profile: Dict[str, Any],
    min_score: int = 30,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """공고 목록에서 프로필에 적합한 공고를 점수순으로 반환한다.

    Args:
        bids: 나라장터 API 응답 공고 항목 리스트.
        profile: 회사 프로필 딕셔너리.
        min_score: 이 점수 미만은 제외한다.
        top_n: 최대 반환 건수.

    Returns:
        점수 내림차순으로 정렬된 공고 목록 (_score, _reasons 필드 추가).
        _reasons는 만점을 받은 근거 태그 목록(REASON_* 값들) — 카드 UI에서
        "업종 일치"/"예산 규모 적합"/"마감 여유 충분" 같은 짧은 태그로 표시한다.
    """
    scored = []
    for bid in bids:
        s, reasons = score_breakdown(bid, profile)
        if s >= min_score:
            scored.append({**bid, "_score": s, "_reasons": reasons})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_n]
