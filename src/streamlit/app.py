"""입찰메이트 RFP RAG - Streamlit UI.

컨설턴트가 수백 건의 RFP를 빠르게 훑고(목록), 핵심을 요약하고(요약),
세부를 질문(QA)하는 흐름에 맞춰 탭으로 구성한다.
이 UI는 RAG 파이프라인을 직접 import하지 않고 백엔드 FastAPI만 호출한다.

실행:
  streamlit run src/streamlit/app.py
백엔드 주소는 환경변수 BACKEND_URL로 바꿀 수 있다(기본 http://127.0.0.1:8000).
"""

import os
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="입찰메이트 RFP RAG", layout="wide")
st.title("입찰메이트 RFP RAG")
st.caption("공공·기업 RFP를 추출·요약해 입찰 기회를 빠르게 찾는 컨설팅 어시스턴트")


# ──────────────────────────────────────────────
# 공통 유틸 / 렌더러
# ──────────────────────────────────────────────

def _won(amount):
    # DataFrame을 거치면 금액이 numpy float64/int64로 바뀌어 isinstance(int) 판별이
    # 실패한다(특히 None이 섞이면 컬럼 전체가 float화). 숫자형이면 모두 받아준다.
    if amount is None or pd.isna(amount):
        return "미기재"
    try:
        return f"{int(amount):,}원"
    except (TypeError, ValueError):
        return "미기재"


def _file_links(파일명: str) -> str:
    """원본 파일 보기/다운로드 HTML 링크를 만든다(새 탭)."""
    q = quote(파일명)
    return (
        f"<a href='{BACKEND_URL}/file?name={q}' target='_blank'>파일 보기</a> · "
        f"<a href='{BACKEND_URL}/file?name={q}&download=1' target='_blank'>다운로드</a>"
    )


def _render_sources(sources):
    """근거 출처(어느 RFP에서 나왔는지)를 표시한다."""
    if not sources:
        st.info("표시할 근거 출처가 없습니다.")
        return
    st.markdown("**근거 출처**")
    for i, s in enumerate(sources, 1):
        파일명 = s.get("파일명")
        표시명 = 파일명 or "(파일명 없음)"
        기관 = s.get("기관") or "미상"
        st.markdown(
            f"{i}. **{표시명}**  \n"
            f"   - 기관: {기관} · 사업금액: {_won(s.get('금액'))} · 마감일: {s.get('마감일') or '미상'}"
        )
        if 파일명:
            st.markdown("&nbsp;&nbsp;&nbsp;" + _file_links(파일명), unsafe_allow_html=True)


def _render_answer(data):
    """답변을 표시하되 거절 응답은 구분해서 보여준다."""
    if data.get("refused"):
        st.warning("거절 응답: 문서에서 근거를 찾지 못했습니다.")
        st.markdown(f"> {data.get('answer', '')}")
    else:
        st.markdown(data.get("answer", ""))
    st.divider()
    _render_sources(data.get("sources", []))


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_documents():
    """인덱싱된 문서 목록을 가져온다(120초 캐시)."""
    r = requests.get(f"{BACKEND_URL}/documents", timeout=20)
    r.raise_for_status()
    return r.json().get("documents", [])


def _post(endpoint: str, payload: dict, timeout: int = 180):
    """백엔드 POST 호출 공통 래퍼. (성공 시 dict, 실패 시 None+에러표시)"""
    try:
        resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=timeout)
    except Exception as e:
        st.error(f"요청 실패: {e}")
        return None
    if resp.status_code != 200:
        st.error(f"오류 {resp.status_code}: {resp.text}")
        return None
    return resp.json()


def _upload_file(uploaded, timeout: int = 600):
    """업로드 파일을 백엔드 /upload로 보낸다(멀티파트). 성공 시 dict, 실패 시 None."""
    files = {"file": (uploaded.name, uploaded.getvalue())}
    try:
        resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=timeout)
    except Exception as e:
        st.error(f"업로드 실패: {e}")
        return None
    if resp.status_code != 200:
        st.error(f"오류 {resp.status_code}: {resp.text}")
        return None
    return resp.json()


# ──────────────────────────────────────────────
# 설정 패널 (사이드바)
# ──────────────────────────────────────────────

def _preset_selectbox(label, group, help_reindex=False):
    """카탈로그 한 그룹(llm/embedding/chunk)을 드롭다운으로 그린다."""
    presets = group.get("presets", [])
    if not presets:
        return None
    keys = [p["key"] for p in presets]
    label_map = {p["key"]: p["label"] for p in presets}
    current = group.get("current")
    index = keys.index(current) if current in keys else 0
    help_txt = "변경하면 재인덱싱이 필요할 수 있습니다." if help_reindex else None
    return st.selectbox(
        label, keys, index=index,
        format_func=lambda k: label_map.get(k, k), help=help_txt,
    )


def _render_upload_panel():
    """문서(HWP/PDF) 업로드 → 백엔드 /upload → FAISS 영구 추가."""
    st.header("문서 업로드")
    uploaded = st.file_uploader(
        "HWP·HWPX·PDF 파일", type=["hwp", "hwpx", "pdf"],
        accept_multiple_files=False,
        help="업로드하면 파싱·인덱싱되어 바로 질문/요약에 사용됩니다.",
    )
    if uploaded is not None and st.button("업로드 & 인덱싱", use_container_width=True):
        with st.spinner(f"‘{uploaded.name}’ 파싱·인덱싱 중..."):
            data = _upload_file(uploaded)
        if data is not None:
            _fetch_documents.clear()  # 새 문서가 목록에 보이도록 캐시 무효화
            st.success(
                f"‘{data['파일명']}’ 추가됨 · 청크 {data['chunks']}개 · "
                f"전체 벡터 {data['vectors']:,}개"
            )
            기관 = data.get("기관") or "미상"
            st.caption(f"추출: {기관} · {_won(data.get('금액'))} · 마감 {data.get('마감일') or '미상'}")
            st.rerun()


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
    hwp_key = _preset_selectbox("HWP 파서", catalog.get("parser_hwp", {}), help_reindex=True)
    pdf_key = _preset_selectbox("PDF 파서", catalog.get("parser_pdf", {}), help_reindex=True)

    with st.expander("세부 조정"):
        retr = settings.get("retriever", {})
        k = st.slider("검색 문서 수 (k)", 1, 10, int(retr.get("k", 4)))
        use_bm25 = st.checkbox("BM25 하이브리드 검색", bool(retr.get("use_bm25", False)))
        temperature = st.slider(
            "temperature", 0.0, 1.0,
            float(settings.get("llm", {}).get("temperature", 0)), 0.1,
        )

    reindex_will = (
        emb_key != catalog.get("embedding", {}).get("current")
        or chunk_key != catalog.get("chunk", {}).get("current")
        or hwp_key != catalog.get("parser_hwp", {}).get("current")
        or pdf_key != catalog.get("parser_pdf", {}).get("current")
    )
    if reindex_will:
        st.caption("임베딩/청킹이 바뀌었습니다. 적용 시 캐시가 없으면 재인덱싱(시간 소요).")

    if st.button("설정 적용", use_container_width=True):
        payload = {
            "llm_preset": llm_key, "embedding_preset": emb_key, "chunk_preset": chunk_key,
            "parser_hwp_preset": hwp_key, "parser_pdf_preset": pdf_key,
            "retriever_k": k, "retriever_use_bm25": use_bm25, "llm_temperature": temperature,
        }
        with st.spinner("설정 적용 중... (재인덱싱이 필요하면 시간이 걸립니다)"):
            data = _post("/config", payload, timeout=1800)
        if data is not None:
            _fetch_documents.clear()  # 인덱스 바뀌었을 수 있으니 목록 캐시 무효화
            msg = "재인덱싱 수행됨" if data.get("reindexed") else "기존 인덱스 재사용"
            st.success(f"설정 적용 완료 ({msg}).")
            st.rerun()


# ──────────────────────────────────────────────
# 탭 1: RFP 목록 대시보드
# ──────────────────────────────────────────────

def _tab_documents():
    st.subheader("RFP 목록")
    try:
        docs = _fetch_documents()
    except Exception as e:
        st.error(f"문서 목록을 불러올 수 없습니다: {e}")
        return
    if not docs:
        st.info("인덱싱된 문서가 없습니다.")
        return

    df = pd.DataFrame(docs)
    # 필터 UI
    c1, c2, c3 = st.columns([2, 2, 1])
    orgs = sorted({d.get("기관") for d in docs if d.get("기관")})
    sel_orgs = c1.multiselect("기관 필터", orgs, placeholder="전체")
    keyword = c2.text_input("사업명 검색", placeholder="키워드")
    sort_by = c3.selectbox("정렬", ["마감일", "금액(높은순)", "사업명"])

    view = df.copy()
    if sel_orgs:
        view = view[view["기관"].isin(sel_orgs)]
    if keyword.strip():
        kw = keyword.strip()
        view = view[view["사업명"].fillna("").str.contains(kw)
                    | view["파일명"].fillna("").str.contains(kw)]

    if sort_by == "마감일":
        view = view.sort_values("마감일", na_position="last")
    elif sort_by == "금액(높은순)":
        view = view.sort_values("금액", ascending=False, na_position="last")
    else:
        view = view.sort_values("사업명", na_position="last")

    st.caption(f"표시 {len(view)}건 / 전체 {len(df)}건")
    show = view[["사업명", "기관", "금액", "마감일", "파일형식"]].copy()
    show["금액"] = show["금액"].apply(_won)
    st.dataframe(show, use_container_width=True, hide_index=True, height=380)

    # 문서 선택 -> 파일 링크 + 요약 바로가기
    st.divider()
    names = view["파일명"].tolist()
    if names:
        sel = st.selectbox("문서 선택(상세/요약)", names,
                           format_func=lambda fn: _label_of(docs, fn))
        st.markdown(_file_links(sel), unsafe_allow_html=True)
        if st.button("요약 대상으로 지정", type="primary"):
            st.session_state.selected_doc = sel
            st.success("‘요약’ 탭에서 이 문서가 선택됩니다. 요약 탭으로 이동하세요.")


def _label_of(docs, fn):
    for d in docs:
        if d.get("파일명") == fn:
            return f"{d.get('사업명') or fn}  ·  {d.get('기관') or '미상'}"
    return fn


# ──────────────────────────────────────────────
# 탭 2: 원클릭 요약 카드
# ──────────────────────────────────────────────

_SUMMARY_Q = "이 RFP의 핵심을 대상 기관·예산·마감일·핵심 요구조건·제출 방식으로 정리해줘."


def _tab_summary():
    st.subheader("RFP 요약")
    try:
        docs = _fetch_documents()
    except Exception as e:
        st.error(f"문서 목록을 불러올 수 없습니다: {e}")
        return
    if not docs:
        st.info("인덱싱된 문서가 없습니다.")
        return

    names = [d.get("파일명") for d in docs]
    default = st.session_state.get("selected_doc")
    idx = names.index(default) if default in names else 0
    sel = st.selectbox("문서 선택", names, index=idx,
                       format_func=lambda fn: _label_of(docs, fn))

    if "summaries" not in st.session_state:
        st.session_state.summaries = {}

    col1, col2 = st.columns([1, 3])
    if col1.button("요약 생성", type="primary"):
        with st.spinner("요약 생성 중..."):
            data = _post("/summarize", {"question": _SUMMARY_Q, "filename": sel})
        if data is not None:
            st.session_state.summaries[sel] = data
    col2.markdown(_file_links(sel), unsafe_allow_html=True)

    data = st.session_state.summaries.get(sel)
    if data:
        with st.container(border=True):
            _render_answer(data)
    else:
        st.caption("‘요약 생성’을 누르면 이 문서만 근거로 4대 항목 요약을 만듭니다.")


# ──────────────────────────────────────────────
# 탭 3: QA (대화)
# ──────────────────────────────────────────────

def _tab_qa():
    st.subheader("질의응답 (QA)")
    if "turns" not in st.session_state:
        st.session_state.turns = []

    if st.session_state.turns:
        cols = st.columns([3, 1])
        cols[0].caption(f"대화 {len(st.session_state.turns)}턴")
        if cols[1].button("대화 초기화", use_container_width=True):
            st.session_state.turns = []
            st.rerun()
        for turn in st.session_state.turns:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                _render_answer(turn["data"])

    question = st.chat_input("예: 한국대학스포츠협의회 사업의 평가 배점 기준은?")
    if question:
        history = [(t["question"], t["data"].get("answer", ""))
                   for t in st.session_state.turns]
        with st.spinner("답변 생성 중..."):
            data = _post("/chat", {"question": question, "history": history})
        if data is not None:
            st.session_state.turns.append({"question": question, "data": data})
            st.rerun()


# ──────────────────────────────────────────────
# 레이아웃
# ──────────────────────────────────────────────

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
    st.divider()
    _render_upload_panel()
    st.divider()
    _render_settings_panel()

tab_list, tab_summary, tab_qa = st.tabs(["RFP 목록", "요약", "QA"])
with tab_list:
    _tab_documents()
with tab_summary:
    _tab_summary()
with tab_qa:
    _tab_qa()
