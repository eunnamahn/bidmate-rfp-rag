"""인포그래픽 카드 생성 모듈.

RFP 핵심 정보(dict)를 HTML/CSS 카드로 렌더링해 문자열로 반환한다.
챗 UI는 이 HTML을 st.components.html로 인라인 렌더한다(Playwright/PNG 미사용).
"""

from __future__ import annotations

import textwrap
from typing import Dict


def _esc(text: str) -> str:
    """HTML 특수문자를 이스케이프한다."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_html(data: Dict) -> str:
    title = data.get("title") or "사업명 미확인"
    agency = data.get("agency") or "발주기관 미확인"
    budget = data.get("budget") or "-"
    deadline = data.get("deadline") or "-"
    contract_type = data.get("contract_type") or "-"
    duration = data.get("duration") or "-"
    qualification = data.get("qualification_summary") or "-"
    requirements: list = data.get("key_requirements") or []
    eval_scores: list = data.get("evaluation_scores") or []

    req_items = "\n".join(
        f'<li>{_esc(r)}</li>' for r in requirements
    )

    score_bars = ""
    for es in eval_scores[:5]:
        name = _esc(str(es.get("name", "")))
        score = int(es.get("score") or 0)
        max_score = int(es.get("max_score") or 100)
        pct = round(score / max_score * 100) if max_score else 0
        score_bars += f"""
        <div class="score-row">
          <span class="score-label">{name}</span>
          <div class="bar-bg">
            <div class="bar-fill" style="width:{pct}%"></div>
          </div>
          <span class="score-val">{score}점</span>
        </div>"""

    return textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
    <meta charset="UTF-8"/>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap');
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        font-family: 'Noto Sans KR', sans-serif;
        background: #f0f4f8;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 32px;
      }}
      .card {{
        width: 640px;
        background: #ffffff;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 40px rgba(0,0,0,0.12);
      }}
      /* 헤더 */
      .header {{
        background: linear-gradient(135deg, #1a4fd6 0%, #0f2e8a 100%);
        padding: 28px 32px 24px;
        color: #fff;
      }}
      .header-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.2);
        color: #fff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 10px;
        text-transform: uppercase;
      }}
      .header-agency {{
        font-size: 13px;
        font-weight: 500;
        color: rgba(255,255,255,0.75);
        margin-bottom: 6px;
      }}
      .header-title {{
        font-size: 22px;
        font-weight: 900;
        line-height: 1.35;
        letter-spacing: -0.3px;
      }}
      /* 메타 정보 칩 */
      .meta-row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        padding: 18px 32px;
        background: #f8faff;
        border-bottom: 1px solid #e8edf5;
      }}
      .meta-chip {{
        display: flex;
        align-items: center;
        gap: 6px;
        background: #ffffff;
        border: 1px solid #dde3f0;
        border-radius: 10px;
        padding: 7px 13px;
        font-size: 12px;
        color: #333;
      }}
      .meta-chip .chip-icon {{ font-size: 14px; }}
      .meta-chip .chip-label {{ color: #888; font-size: 11px; }}
      .meta-chip .chip-value {{ font-weight: 700; color: #1a1a2e; }}
      /* 본문 섹션 */
      .section {{
        padding: 20px 32px;
        border-bottom: 1px solid #f0f2f8;
      }}
      .section:last-child {{ border-bottom: none; }}
      .section-title {{
        font-size: 11px;
        font-weight: 700;
        color: #1a4fd6;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 14px;
      }}
      /* 평가배점 바 */
      .score-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
      }}
      .score-label {{
        width: 90px;
        font-size: 12px;
        font-weight: 500;
        color: #444;
        flex-shrink: 0;
      }}
      .bar-bg {{
        flex: 1;
        height: 10px;
        background: #eef1f8;
        border-radius: 6px;
        overflow: hidden;
      }}
      .bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, #1a4fd6, #4f8ef7);
        border-radius: 6px;
      }}
      .score-val {{
        width: 36px;
        font-size: 12px;
        font-weight: 700;
        color: #1a4fd6;
        text-align: right;
        flex-shrink: 0;
      }}
      /* 핵심 요구사항 */
      .req-list {{
        list-style: none;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }}
      .req-list li {{
        display: flex;
        align-items: flex-start;
        gap: 8px;
        font-size: 12.5px;
        color: #333;
        line-height: 1.5;
      }}
      .req-list li::before {{
        content: "▸";
        color: #1a4fd6;
        font-size: 11px;
        margin-top: 2px;
        flex-shrink: 0;
      }}
      /* 참가자격 */
      .qual-text {{
        font-size: 12.5px;
        color: #444;
        line-height: 1.6;
        background: #f4f7ff;
        border-left: 3px solid #1a4fd6;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
      }}
      /* 푸터 */
      .footer {{
        background: #1a1a2e;
        padding: 12px 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }}
      .footer-brand {{
        font-size: 12px;
        font-weight: 700;
        color: #fff;
        letter-spacing: 0.5px;
      }}
      .footer-note {{
        font-size: 10px;
        color: rgba(255,255,255,0.4);
      }}
    </style>
    </head>
    <body>
    <div class="card">

      <div class="header">
        <div class="header-badge">RFP 분석 카드</div>
        <div class="header-agency">{_esc(agency)}</div>
        <div class="header-title">{_esc(title)}</div>
      </div>

      <div class="meta-row">
        <div class="meta-chip">
          <span class="chip-icon">💰</span>
          <div>
            <div class="chip-label">사업금액</div>
            <div class="chip-value">{_esc(budget)}</div>
          </div>
        </div>
        <div class="meta-chip">
          <span class="chip-icon">📅</span>
          <div>
            <div class="chip-label">입찰마감</div>
            <div class="chip-value">{_esc(deadline)}</div>
          </div>
        </div>
        <div class="meta-chip">
          <span class="chip-icon">📋</span>
          <div>
            <div class="chip-label">계약방식</div>
            <div class="chip-value">{_esc(contract_type)}</div>
          </div>
        </div>
        <div class="meta-chip">
          <span class="chip-icon">⏱</span>
          <div>
            <div class="chip-label">사업기간</div>
            <div class="chip-value">{_esc(duration)}</div>
          </div>
        </div>
      </div>

      {'<div class="section"><div class="section-title">평가배점</div>' + score_bars + '</div>' if score_bars else ''}

      {'<div class="section"><div class="section-title">핵심 요구사항</div><ul class="req-list">' + req_items + '</ul></div>' if req_items else ''}

      {'<div class="section"><div class="section-title">참가자격 요약</div><div class="qual-text">' + _esc(qualification) + '</div></div>' if qualification and qualification != '-' else ''}

      <div class="footer">
        <span class="footer-brand">입찰메이트</span>
        <span class="footer-note">AI 기반 RFP 분석 · 참고용 자료</span>
      </div>

    </div>
    </body>
    </html>
    """).strip()


def render_infographic_html(data: Dict) -> str:
    """인포그래픽 데이터 dict를 받아 HTML 문자열을 반환한다(Streamlit 인라인 렌더용)."""
    return _build_html(data)
