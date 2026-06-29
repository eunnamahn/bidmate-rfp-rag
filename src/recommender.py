"""규칙 기반 공고 추천 엔진.

나라장터 API에서 가져온 공고 목록을 회사 프로필 기준으로 필터링·점수화한다.

점수 구성 (100점 만점):
  - 사업분류 일치      : 40점 (대분류 10 / 중분류 15 / 세분류 15)
  - 예산 범위 내       : 30점
  - 입찰방법 선호 일치 : 15점
  - 마감 여유 (3일+)   : 15점
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def _parse_dt_v2(s: Optional[str]) -> Optional[datetime]:
    """날짜 문자열을 파싱한다. 여러 포맷을 시도한다."""
    if not s:
        return None
    for fmt in (
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s[:len(fmt)], fmt)
        except ValueError:
            continue
    return None


def score_bid(bid: Dict[str, Any], profile: Dict[str, Any]) -> int:
    """단일 공고에 대해 회사 프로필 적합 점수(0~100)를 계산한다."""
    score = 0

    # 1. 사업분류 일치 (40점)
    pref_lrg = (profile.get("large_category") or "").strip()
    pref_mid = (profile.get("mid_category") or "").strip()
    pref_sml = (profile.get("small_category") or "").strip()

    bid_lrg = (bid.get("pubPrcrmntLrgClsfcNm") or "").strip()
    bid_mid = (bid.get("pubPrcrmntMidClsfcNm") or "").strip()
    bid_sml = (bid.get("pubPrcrmntClsfcNm") or "").strip()

    if pref_lrg and bid_lrg and pref_lrg in bid_lrg:
        score += 10
    if pref_mid and bid_mid and pref_mid in bid_mid:
        score += 15
    if pref_sml and bid_sml and pref_sml in bid_sml:
        score += 15

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
        elif in_min or in_max:
            score += 10  # 한쪽만 맞아도 부분 점수

    # 3. 입찰방법 선호 일치 (15점)
    pref_methods: List[str] = profile.get("bid_methods") or []
    bid_method = (bid.get("bidMethdNm") or "").strip()
    if not pref_methods or any(m in bid_method for m in pref_methods):
        score += 15

    # 4. 마감 여유 (15점) — 오늘로부터 3일 이상 남은 공고
    close_dt = _parse_dt_v2(bid.get("bidClseDt"))
    if close_dt:
        days_left = (close_dt - datetime.now()).days
        if days_left >= 3:
            score += 15
        elif days_left >= 0:
            score += 5

    return score


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
        점수 내림차순으로 정렬된 공고 목록 (score 필드 추가).
    """
    scored = []
    for bid in bids:
        s = score_bid(bid, profile)
        if s >= min_score:
            scored.append({**bid, "_score": s})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_n]
