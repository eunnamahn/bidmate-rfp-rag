"""입찰메이트 RFP RAG - Streamlit UI.

중요: 이 UI는 RAG 파이프라인을 직접 import하지 않는다.
오직 백엔드 FastAPI(/chat, /summarize, /documents, /health)만 호출한다.

실행:
  streamlit run src/streamlit/app.py
백엔드 주소는 환경변수 BACKEND_URL로 바꿀 수 있다(기본 http://127.0.0.1:8000).
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

import json

st.set_page_config(page_title="입찰메이트 RFP RAG", layout="wide")
st.title("입찰메이트 RFP RAG")
st.caption("RFP 문서 기반 질의응답 / 요약 / 공고 추천 (백엔드 API 연동)")


def _won(amount):
    return f"{amount:,}원" if isinstance(amount, int) else "미기재"


def _render_sources(sources):
    """근거 출처(어느 RFP에서 나왔는지)를 표시한다."""
    if not sources:
        st.info("표시할 근거 출처가 없습니다.")
        return
    st.markdown("**근거 출처**")
    for i, s in enumerate(sources, 1):
        파일명 = s.get("파일명") or "(파일명 없음)"
        기관 = s.get("기관") or "미상"
        st.markdown(
            f"{i}. **{파일명}**  \n"
            f"   - 기관: {기관} · 사업금액: {_won(s.get('금액'))} · 마감일: {s.get('마감일') or '미상'}"
        )


def _render_answer(data):
    """답변을 표시하되 거절 응답은 구분해서 보여준다."""
    if data.get("refused"):
        st.warning("거절 응답: 문서에서 근거를 찾지 못했습니다.")
        st.markdown(f"> {data.get('answer', '')}")
    else:
        st.success("답변")
        st.markdown(data.get("answer", ""))
    st.divider()
    _render_sources(data.get("sources", []))


def _preset_selectbox(label, group, help_reindex=False):
    """카탈로그의 한 그룹(llm/embedding/chunk)을 드롭다운으로 그린다.

    선택된 프리셋의 key를 반환한다. 현재 적용된 프리셋이 기본 선택된다.
    """
    presets = group.get("presets", [])
    if not presets:
        return None
    keys = [p["key"] for p in presets]
    label_map = {p["key"]: p["label"] for p in presets}
    current = group.get("current")
    index = keys.index(current) if current in keys else 0
    help_txt = "변경하면 재인덱싱이 필요합니다(시간이 걸릴 수 있음)." if help_reindex else None
    return st.selectbox(
        label, keys, index=index,
        format_func=lambda k: label_map.get(k, k), help=help_txt,
    )


def _render_settings_panel():
    """카탈로그를 불러와 프리셋 드롭다운을 그리고, 적용 시 /config로 보낸다."""
    st.header("설정")
    try:
        cfg = requests.get(f"{BACKEND_URL}/config", timeout=10).json()
    except Exception as e:
        st.error(f"설정을 불러올 수 없습니다: {e}")
        return

    catalog = cfg.get("catalog", {})
    settings = cfg.get("settings", {})

    llm_key = _preset_selectbox("LLM 모델", catalog.get("llm", {}))
    emb_key = _preset_selectbox("임베딩", catalog.get("embedding", {}), help_reindex=True)
    chunk_key = _preset_selectbox("청킹", catalog.get("chunk", {}), help_reindex=True)

    with st.expander("세부 조정"):
        retr = settings.get("retriever", {})
        k = st.slider("검색 문서 수 (k)", 1, 10, int(retr.get("k", 4)))
        use_bm25 = st.checkbox("BM25 하이브리드 검색", bool(retr.get("use_bm25", False)))
        temperature = st.slider(
            "temperature", 0.0, 1.0, float(settings.get("llm", {}).get("temperature", 0)), 0.1
        )

    # 임베딩/청킹을 현재와 다르게 골랐으면 재인덱싱 경고
    reindex_will = (
        emb_key != catalog.get("embedding", {}).get("current")
        or chunk_key != catalog.get("chunk", {}).get("current")
    )
    if reindex_will:
        st.caption("임베딩/청킹이 바뀌었습니다. 적용 시 캐시가 없으면 재인덱싱(시간 소요)이 일어납니다.")

    if st.button("설정 적용", use_container_width=True):
        payload = {
            "llm_preset": llm_key,
            "embedding_preset": emb_key,
            "chunk_preset": chunk_key,
            "retriever_k": k,
            "retriever_use_bm25": use_bm25,
            "llm_temperature": temperature,
        }
        with st.spinner("설정 적용 중... (재인덱싱이 필요하면 시간이 걸립니다)"):
            try:
                resp = requests.post(f"{BACKEND_URL}/config", json=payload, timeout=1800)
            except Exception as e:
                st.error(f"설정 적용 실패: {e}")
                return
        if resp.status_code == 200:
            data = resp.json()
            if data.get("reindexed"):
                st.success("설정 적용 완료 (재인덱싱 수행됨).")
            else:
                st.success("설정 적용 완료 (기존 인덱스 재사용).")
            st.rerun()
        else:
            st.error(f"오류 {resp.status_code}: {resp.text}")


# 사이드바: 백엔드 상태 + 문서 목록
with st.sidebar:
    st.header("상태")
    try:
        h = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        if h.get("status") == "ok":
            st.success(f"백엔드 연결됨 · 벡터 {h.get('vectors', 0):,}개")
        else:
            st.warning(f"백엔드 준비 중: {h}")
    except Exception as e:
        st.error(f"백엔드에 연결할 수 없습니다.\n{BACKEND_URL}\n{e}")

    if st.button("문서 목록 불러오기"):
        try:
            d = requests.get(f"{BACKEND_URL}/documents", timeout=10).json()
            st.caption(f"인덱싱 문서 {d.get('count', 0)}건")
            st.dataframe(d.get("documents", []), use_container_width=True)
        except Exception as e:
            st.error(f"문서 목록 조회 실패: {e}")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()
    _render_settings_panel()

# 히스토리 초기화
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# 본문: 탭 구성
tab_qa, tab_recommend = st.tabs(["QA / 요약", "공고 추천"])

# ── QA / 요약 탭 ──────────────────────────────────────────────
with tab_qa:
    mode = st.radio("모드", ["QA (질문답변)", "요약"], horizontal=True)
    endpoint = "/chat" if mode.startswith("QA") else "/summarize"

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    placeholder = (
        "예: 이 사업의 평가 배점 기준은?"
        if endpoint == "/chat"
        else "예: 고려대학교 차세대 포털 학사 정보시스템 사업을 요약해줘"
    )
    question = st.text_area("질문 / 요청", placeholder=placeholder, height=100)

    if st.button("실행", type="primary"):
        if not question.strip():
            st.warning("질문을 입력하세요.")
        else:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("백엔드에 요청 중..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}{endpoint}",
                        json={"question": question, "history": st.session_state.history},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        with st.chat_message("assistant"):
                            _render_answer(data)
                        st.session_state.history.append((question, data.get("answer", "")))
                        st.session_state.messages.append(
                            {"role": "assistant", "content": data.get("answer", "")}
                        )
                    else:
                        st.error(f"오류 {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"요청 실패: {e}")

# ── 공고 추천 탭 ───────────────────────────────────────────────
with tab_recommend:
    st.subheader("회사 프로필 설정")

    col1, col2 = st.columns(2)
    with col1:
        large_cat = st.text_input("대분류", placeholder="예: 용역")
        mid_cat   = st.text_input("중분류", placeholder="예: 소프트웨어")
        small_cat = st.text_input("세분류", placeholder="예: 시스템 소프트웨어")
    with col2:
        budget_min = st.number_input("최소 예산 (원, 0=제한없음)", min_value=0, value=0, step=10_000_000)
        budget_max = st.number_input("최대 예산 (원, 0=제한없음)", min_value=0, value=0, step=10_000_000)
        bid_method_input = st.text_input("선호 입찰방법 (쉼표 구분)", placeholder="예: 전자입찰, 직찰")

    col3, col4 = st.columns(2)
    with col3:
        min_score = st.slider("최소 적합 점수", 0, 100, 30)
    with col4:
        top_n = st.number_input("최대 추천 건수", min_value=1, max_value=100, value=20)

    st.divider()
    st.subheader("나라장터 공고 데이터 입력")
    st.caption("나라장터 API 응답 JSON 배열을 붙여넣으세요. (items 배열 전체 또는 단일 항목 목록)")
    bids_json = st.text_area("공고 JSON", height=200, placeholder='[{"bidNtceNo": "...", "bidNtceNm": "...", ...}, ...]')

    if st.button("추천 실행", type="primary"):
        # JSON 파싱
        try:
            bids = json.loads(bids_json)
            if isinstance(bids, dict):
                # 나라장터 API 전체 응답인 경우 items 추출 시도
                bids = bids.get("items") or bids.get("response", {}).get("body", {}).get("items") or []
            if not isinstance(bids, list):
                st.error("공고 데이터는 JSON 배열이어야 합니다.")
                st.stop()
        except json.JSONDecodeError as e:
            st.error(f"JSON 파싱 오류: {e}")
            st.stop()

        bid_methods = [m.strip() for m in bid_method_input.split(",") if m.strip()]
        profile = {
            "large_category": large_cat or None,
            "mid_category": mid_cat or None,
            "small_category": small_cat or None,
            "budget_min": int(budget_min) if budget_min > 0 else None,
            "budget_max": int(budget_max) if budget_max > 0 else None,
            "bid_methods": bid_methods,
            "min_score": min_score,
            "top_n": int(top_n),
        }

        with st.spinner("추천 계산 중..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/recommend",
                    json={"profile": profile, "bids": bids},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    st.success(f"총 {data.get('count', 0)}건의 적합 공고를 찾았습니다.")
                    for rank, bid in enumerate(results, 1):
                        score = bid.get("_score", 0)
                        title = bid.get("bidNtceNm") or "(공고명 없음)"
                        inst  = bid.get("ntceInsttNm") or ""
                        price = bid.get("presmptPrce") or bid.get("asignBdgtAmt")
                        close = bid.get("bidClseDt") or ""
                        method = bid.get("bidMethdNm") or ""
                        url   = bid.get("bidNtceDtlUrl") or ""

                        with st.expander(f"#{rank}  [{score}점]  {title}"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"**기관:** {inst}")
                                st.markdown(f"**추정가격:** {int(price):,}원" if price else "**추정가격:** 미기재")
                                st.markdown(f"**입찰방법:** {method}")
                            with col_b:
                                st.markdown(f"**마감일:** {close}")
                                st.markdown(f"**대분류:** {bid.get('pubPrcrmntLrgClsfcNm','')}")
                                st.markdown(f"**중분류:** {bid.get('pubPrcrmntMidClsfcNm','')}")
                                st.markdown(f"**세분류:** {bid.get('pubPrcrmntClsfcNm','')}")
                            if url:
                                st.markdown(f"[공고 상세 보기]({url})")
                else:
                    st.error(f"오류 {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"요청 실패: {e}")
