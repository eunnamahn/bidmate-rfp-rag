"""입찰메이트 — 챗봇 UI (Streamlit).

단일 채팅 화면으로 RFP를 질문·요약·분석한다. AI 답변 아래에 에이전트 액션 버튼
(요약·메타데이터·적격성 판정·리스크 탐지·인포그래픽·체크리스트·평가 구조)을 노출하고,
사이드바에서 '현재 문서'를 골라 분석 범위를 한 건으로 한정한다.

이 UI는 RAG 파이프라인을 직접 import하지 않고 백엔드 FastAPI만 호출한다.
에이전트 버튼 목록은 백엔드 GET /agents(레지스트리)에서 받아 동적으로 렌더하므로,
팀원이 새 에이전트를 추가하면 코드 수정 없이 버튼이 늘어난다.

실행:
  streamlit run src/streamlit/app.py
백엔드 주소는 환경변수 BACKEND_URL로 바꿀 수 있다(기본 http://127.0.0.1:8000).
"""

import os
import re
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="입찰메이트", layout="wide")


# ──────────────────────────────────────────────
# 공통 유틸 / 백엔드 호출
# ──────────────────────────────────────────────

def _won(amount):
    # DataFrame을 거치면 금액이 numpy 타입으로 바뀌어 isinstance(int) 판별이 실패한다.
    # 숫자형이면 모두 받아준다.
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


def _post(endpoint: str, payload: dict, timeout: int = 180):
    """백엔드 POST 호출 공통 래퍼.(성공 시 dict, 실패 시 None+에러표시)"""
    try:
        resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=timeout)
    except Exception as e:
        st.error(f"요청 실패: {e}")
        return None
    if resp.status_code != 200:
        st.error(f"오류 {resp.status_code}: {resp.text}")
        return None
    return resp.json()


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_documents():
    """인덱싱된 문서 목록을 가져온다(120초 캐시)."""
    r = requests.get(f"{BACKEND_URL}/documents", timeout=20)
    r.raise_for_status()
    return r.json().get("documents", [])


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_agents():
    """에이전트 액션 버튼 목록을 가져온다(레지스트리, 300초 캐시)."""
    r = requests.get(f"{BACKEND_URL}/agents", timeout=10)
    r.raise_for_status()
    return r.json().get("agents", [])


def _nara_search(keyword: str, days: int = 3, timeout: int = 60):
    """나라장터 메타데이터 검색(GET /nara/search). 성공 시 notices, 실패 시 None."""
    try:
        resp = requests.get(
            f"{BACKEND_URL}/nara/search",
            params={"q": keyword, "days": days}, timeout=timeout,
        )
    except Exception as e:
        st.error(f"검색 실패: {e}")
        return None
    if resp.status_code != 200:
        st.error(f"오류 {resp.status_code}: {resp.text}")
        return None
    return resp.json().get("notices", [])


def _nara_ingest(item: dict, timeout: int = 600):
    """선택 공고 1건 지연 인덱싱(POST /nara/ingest)."""
    try:
        resp = requests.post(f"{BACKEND_URL}/nara/ingest", json={"item": item}, timeout=timeout)
    except Exception as e:
        st.error(f"인덱싱 실패: {e}")
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
# 렌더러
# ──────────────────────────────────────────────

def _render_sources(sources):
    """근거 출처(어느 RFP에서 나왔는지)를 표시한다."""
    if not sources:
        return
    with st.expander(f"근거 출처 {len(sources)}건"):
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


# 답변에 섞여 들어오는 ```mermaid ... ``` 코드 블록을 뽑아낸다(평가 구조 다이어그램).
_MERMAID_RE = re.compile(r"```mermaid\s*(.*?)```", re.DOTALL)


def _sanitize_mermaid(code: str) -> str:
    """Mermaid 노드 라벨의 특수문자(괄호·콜론·물음표 등)로 인한 파싱 오류를 막는다.

    [...] / {...} 안의 라벨을 큰따옴표로 감싼다(이미 따옴표면 유지). 괄호가 든 라벨
    (예: [전략 및 방법론 (20점)])이 'Syntax error in text'를 내는 문제를 해결한다.
    """
    def _make(open_ch: str, close_ch: str):
        pat = re.compile(
            re.escape(open_ch) + r"([^" + re.escape(open_ch + close_ch) + r"]*)" + re.escape(close_ch)
        )

        def repl(m):
            inner = m.group(1).strip()
            if not inner or (inner.startswith('"') and inner.endswith('"')):
                return m.group(0)
            return f'{open_ch}"{inner.replace(chr(34), chr(39))}"{close_ch}'

        return pat, repl

    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        pat, repl = _make(open_ch, close_ch)
        code = pat.sub(repl, code)
    return code


def _render_mermaid(code: str):
    """Mermaid 코드를 mermaid.js(CDN)로 렌더한다."""
    html = f"""
<html><head>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
</head><body style="margin:0;padding:8px;background:#fff;">
<div class="mermaid">
{code}
</div></body></html>"""
    components.html(html, height=480, scrolling=True)


def _render_answer(data):
    """답변을 표시하되 거절 응답·Mermaid 다이어그램을 구분해서 보여준다."""
    answer = data.get("answer", "")
    if data.get("refused"):
        st.warning("문서에서 근거를 찾지 못했습니다.")
        st.markdown(f"> {answer}")
        _render_sources(data.get("sources", []))
        return

    # ```mermaid 블록이 있으면 그 부분만 다이어그램으로 렌더하고 나머지는 마크다운으로.
    m = _MERMAID_RE.search(answer)
    if m:
        before = answer[:m.start()].strip()
        if before:
            st.markdown(before)
        code = _sanitize_mermaid(m.group(1).strip())
        _render_mermaid(code)
        with st.expander("Mermaid 코드 보기"):
            st.code(code, language="text")
    else:
        st.markdown(answer)
    _render_sources(data.get("sources", []))


def _render_infographic(data):
    """인포그래픽 HTML 카드를 인라인으로 렌더한다(/infographic 응답)."""
    if data.get("parse_error"):
        st.warning("인포그래픽 정보 추출에 실패했습니다. LLM 원본 응답:")
        st.code(data["parse_error"])
        _render_sources(data.get("sources", []))
        return
    html = data.get("html", "")
    if html:
        components.html(html, height=760, scrolling=True)
    else:
        st.warning("인포그래픽을 생성하지 못했습니다.")
    _render_sources(data.get("sources", []))


def _doc_label(docs, fn):
    """파일명을 '사업명 · 기관' 형태의 사람이 읽는 라벨로 바꾼다."""
    for d in docs:
        if d.get("파일명") == fn:
            return f"{d.get('사업명') or fn} · {d.get('기관') or '미상'}"
    return fn


# ──────────────────────────────────────────────
# 채팅 송신
# ──────────────────────────────────────────────

# 적격성 판정이 회사 정보를 요청했음을 알리는 표지(이 응답이 오면 다음 입력을 이어받는다).
_ELIGIBILITY_INFO_MARK = "회사 정보가 필요합니다"


def _context_filename():
    """현재 대화가 다루는 '특정 공고' 문서를 정한다(없으면 None).

    사이드바에서 고른 활성 문서가 있으면 그것을, 없으면 가장 최근 답변이 '단일 문서'로
    근거가 좁혀졌을 때 그 문서를 본다. 여러 문서가 섞여 나온 광범위한 질문(특정 공고를
    지목하지 않은 경우)에는 None을 반환해, 에이전트 버튼이 뜨지 않도록 한다.
    """
    active = st.session_state.get("active_filename")
    if active:
        return active
    for turn in reversed(st.session_state.get("turns", [])):
        srcs = (turn.get("data") or {}).get("sources") or []
        names = {s.get("파일명") for s in srcs if s.get("파일명")}
        if names:
            return next(iter(names)) if len(names) == 1 else None
    return None


def _send(question: str, agent=None, filename=None):
    """질문(또는 에이전트 액션)을 백엔드 /chat으로 보내고 대화에 추가한다.

    filename이 주어지면 그 문서로 검색을 한정하고, 없으면 활성 문서를 쓴다.
    agent가 주어지면 백엔드가 해당 에이전트로 강제 라우팅한다.
    """
    turns = st.session_state.turns
    history = [(t["question"], t["data"].get("answer", "")) for t in turns]
    payload = {"question": question, "history": history}
    if agent:
        payload["agent"] = agent
    fn = filename or st.session_state.get("active_filename")
    if fn:
        payload["filename"] = fn
    data = _post("/chat", payload)
    if data is not None:
        turns.append({"kind": "chat", "question": question, "data": data})
        # 적격성이 회사 정보를 요청했으면, 다음 사용자 입력을 이 흐름으로 이어받도록 표시.
        if agent == "eligibility":
            if _ELIGIBILITY_INFO_MARK in (data.get("answer") or ""):
                st.session_state.pending_agent = "eligibility"
                st.session_state.pending_prompt = question
            else:
                st.session_state.pending_agent = None


def _continue_pending(user_text: str):
    """대기 중인 에이전트(적격성)에 사용자의 추가 입력을 이어붙여 실행한다.

    사용자가 자유 입력한 회사 정보를 히스토리에 사용자 발화로 포함시켜, 일반 QA가 아니라
    같은 공고에 대한 적격성 판정으로 이어지게 한다(대화 맥락 유지).
    """
    agent_id = st.session_state.get("pending_agent")
    prompt = st.session_state.get("pending_prompt") or user_text
    st.session_state.pending_agent = None
    st.session_state.pending_prompt = ""

    turns = st.session_state.turns
    history = [(t["question"], t["data"].get("answer", "")) for t in turns]
    history.append((user_text, ""))   # 이번 입력을 회사 정보로 추출되게 사용자 발화로 추가
    payload = {"question": prompt, "agent": agent_id, "history": history}
    fn = _context_filename()
    if fn:
        payload["filename"] = fn
    data = _post("/chat", payload)
    if data is not None:
        turns.append({"kind": "chat", "question": user_text, "data": data})
        # 여전히 정보가 부족하면 다시 대기 상태로 둔다(재요청).
        if _ELIGIBILITY_INFO_MARK in (data.get("answer") or ""):
            st.session_state.pending_agent = agent_id
            st.session_state.pending_prompt = prompt


def _send_infographic(question: str, filename=None):
    """인포그래픽 에이전트: /infographic으로 HTML 카드를 받아 대화에 추가한다."""
    turns = st.session_state.turns
    payload = {"question": question}
    fn = filename or st.session_state.get("active_filename")
    if fn:
        payload["filename"] = fn
    data = _post("/infographic", payload)
    if data is not None:
        turns.append({"kind": "infographic", "question": question, "data": data})


def _dispatch_agent(agent: dict):
    """에이전트 스펙에 따라 적절한 엔드포인트로 보낸다(/chat 또는 전용 엔드포인트).

    분석 대상은 대화 맥락의 문서(_context_filename)로 한정한다.
    """
    # 새 버튼 클릭은 진행 중이던 대기(예: 적격성 회사정보 입력) 흐름을 초기화한다.
    st.session_state.pending_agent = None
    st.session_state.pending_prompt = ""
    question = agent.get("prompt") or agent["label"]
    fn = _context_filename()
    if agent.get("endpoint", "/chat") == "/infographic":
        _send_infographic(question, fn)
    else:
        _send(question, agent["id"], fn)


def _render_action_bar(agents):
    """에이전트 액션 버튼 바를 그린다. 클릭 시 해당 agent dict 반환, 아니면 None."""
    if not agents:
        return None
    target = _context_filename()
    if target:
        st.caption(f"에이전트 액션 · 분석 대상: {target}")
    else:
        st.caption("에이전트 액션 · 분석 대상 문서가 없습니다(공고를 먼저 찾거나 사이드바에서 선택)")
    cols = st.columns(len(agents))
    clicked = None
    for col, a in zip(cols, agents):
        help_txt = a["description"] + ("" if a.get("implemented") else "  (준비 중)")
        if col.button(a["label"], key=f"agent_{a['id']}", help=help_txt,
                      use_container_width=True):
            clicked = a
    return clicked


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────

def _sidebar_status():
    try:
        h = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        if h.get("status") == "ok":
            st.success(f"백엔드 연결됨 · 벡터 {h.get('vectors', 0):,}개")
        else:
            st.warning(f"백엔드 준비 중: {h}")
    except Exception as e:
        st.error(f"백엔드에 연결할 수 없습니다.\n{BACKEND_URL}\n{e}")


def _sidebar_active_document():
    """현재 분석 대상 문서를 고른다(없으면 전체 문서 대상)."""
    st.subheader("현재 문서")
    try:
        docs = _fetch_documents()
    except Exception as e:
        st.error(f"문서 목록을 불러올 수 없습니다: {e}")
        return
    if not docs:
        st.caption("인덱싱된 문서가 없습니다. + 로 업로드하거나 나라장터에서 가져오세요.")
        return

    names = [d.get("파일명") for d in docs]
    options = [None] + names
    active = st.session_state.get("active_filename")
    index = options.index(active) if active in options else 0
    sel = st.selectbox(
        "분석 대상", options, index=index,
        format_func=lambda fn: "(전체 문서 대상)" if fn is None else _doc_label(docs, fn),
        label_visibility="collapsed",
    )
    st.session_state.active_filename = sel
    if sel:
        st.markdown(_file_links(sel), unsafe_allow_html=True)


def _sidebar_upload():
    """문서(HWP/PDF) 업로드 → /upload → FAISS 추가 → 활성 문서로 지정."""
    with st.expander("문서 업로드 (+)"):
        uploaded = st.file_uploader(
            "HWP·HWPX·PDF", type=["hwp", "hwpx", "pdf"],
            accept_multiple_files=False, label_visibility="collapsed",
        )
        if uploaded is not None and st.button("업로드 & 인덱싱", use_container_width=True):
            with st.spinner(f"‘{uploaded.name}’ 파싱·인덱싱 중..."):
                data = _upload_file(uploaded)
            if data is not None:
                _fetch_documents.clear()
                st.session_state.active_filename = data["파일명"]  # 업로드 문서를 바로 활성화
                st.success(f"‘{data['파일명']}’ 추가됨 · 청크 {data['chunks']}개")
                st.rerun()


def _sidebar_nara():
    """나라장터 검색 → 선택 공고 지연 인덱싱 → 활성 문서로 지정."""
    with st.expander("나라장터 공고 검색"):
        keyword = st.text_input("공고명 키워드", placeholder="예: 테마식현장체험학습")
        days = st.number_input("최근 N일", min_value=1, max_value=30, value=3)
        if st.button("검색", use_container_width=True):
            if not keyword.strip():
                st.warning("검색 키워드를 입력하세요.")
            else:
                with st.spinner("나라장터 조회 중..."):
                    st.session_state.nara_results = _nara_search(keyword.strip(), int(days))

        notices = st.session_state.get("nara_results")
        if not notices:
            return
        st.caption(f"검색 결과 {len(notices)}건 — 인덱싱하면 활성 문서가 됩니다.")
        for i, n in enumerate(notices):
            st.markdown(
                f"**{n.get('사업명') or '(제목 없음)'}**  \n"
                f"기관: {n.get('기관') or '미상'} · 마감: {n.get('마감일') or '미상'}"
            )
            if st.button("인덱싱", key=f"nara_ing_{i}", use_container_width=True):
                with st.spinner("공고 본문 다운로드·임베딩 중..."):
                    data = _nara_ingest(n.get("item", {}))
                if data is not None:
                    _fetch_documents.clear()
                    st.session_state.active_filename = data["파일명"]
                    st.success(f"‘{data['파일명']}’ 인덱싱 완료 — 이제 질문할 수 있습니다.")
                    st.rerun()


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
    return st.selectbox(label, keys, index=index,
                        format_func=lambda k: label_map.get(k, k), help=help_txt)


def _sidebar_settings():
    """카탈로그 프리셋 드롭다운 → /config로 적용."""
    with st.expander("설정"):
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

        retr = settings.get("retriever", {})
        k = st.slider("검색 문서 수 (k)", 1, 10, int(retr.get("k", 4)))
        use_bm25 = st.checkbox("BM25 하이브리드 검색", bool(retr.get("use_bm25", False)))
        use_reranker = st.checkbox(
            "리랭커 (bge-reranker-v2-m3)", bool(retr.get("use_reranker", False)),
            help="후보를 다시 채점해 정확도를 높입니다. 처음 켤 때 모델(약 2.2GB) 다운로드로 느릴 수 있습니다.",
        )
        temperature = st.slider(
            "temperature", 0.0, 1.0,
            float(settings.get("llm", {}).get("temperature", 0)), 0.1,
        )

        if st.button("설정 적용", use_container_width=True):
            payload = {
                "llm_preset": llm_key, "embedding_preset": emb_key, "chunk_preset": chunk_key,
                "parser_hwp_preset": hwp_key, "parser_pdf_preset": pdf_key,
                "retriever_k": k, "retriever_use_bm25": use_bm25,
                "retriever_use_reranker": use_reranker, "llm_temperature": temperature,
            }
            with st.spinner("설정 적용 중... (재인덱싱이 필요하면 시간이 걸립니다)"):
                data = _post("/config", payload, timeout=1800)
            if data is not None:
                _fetch_documents.clear()
                msg = "재인덱싱 수행됨" if data.get("reindexed") else "기존 인덱스 재사용"
                st.success(f"설정 적용 완료 ({msg}).")
                st.rerun()


# ──────────────────────────────────────────────
# 레이아웃
# ──────────────────────────────────────────────

def _init_state():
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("active_filename", None)
    st.session_state.setdefault("pending_agent", None)   # 회사정보 대기 등 이어받을 에이전트
    st.session_state.setdefault("pending_prompt", "")


def main():
    _init_state()

    with st.sidebar:
        st.title("입찰메이트")
        st.caption("B2G 입찰지원 AI 컨설턴트")
        _sidebar_status()
        st.divider()
        _sidebar_active_document()
        _sidebar_upload()
        _sidebar_nara()
        _sidebar_settings()
        st.divider()
        if st.session_state.turns and st.button("대화 초기화", use_container_width=True):
            st.session_state.turns = []
            st.rerun()

    # 현재 분석 대상 배너
    active = st.session_state.get("active_filename")
    if active:
        try:
            docs = _fetch_documents()
        except Exception:
            docs = []
        st.info(f"현재 문서: {_doc_label(docs, active)}", icon="📄")
    else:
        st.info("전체 문서를 대상으로 검색합니다. 사이드바에서 특정 공고를 고르면 그 문서로 한정합니다.")

    # 대화 히스토리
    if not st.session_state.turns:
        with st.chat_message("assistant"):
            st.markdown(
                "안녕하세요, 입찰메이트입니다. RFP 공고에 대해 무엇이든 물어보세요.\n\n"
                "사이드바에서 문서를 업로드하거나 나라장터에서 공고를 가져온 뒤, "
                "아래 에이전트 액션 버튼으로 요약·메타데이터 추출 등을 실행할 수 있습니다."
            )
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            if turn.get("kind") == "infographic":
                _render_infographic(turn["data"])
            else:
                _render_answer(turn["data"])

    # 에이전트 액션 버튼 바: '특정 공고'가 대화 맥락에 잡혔을 때만 노출한다.
    clicked = None
    if _context_filename():
        try:
            agents = _fetch_agents()
        except Exception:
            agents = []
        clicked = _render_action_bar(agents)

    # 적격성이 회사 정보를 기다리는 중이면 입력 안내를 바꾼다.
    if st.session_state.get("pending_agent") == "eligibility":
        placeholder = "회사 정보를 입력하세요 (예: IT서비스, 중소기업, 창업 3년, 매출 10억, 30인, ISO)"
    else:
        placeholder = "입찰 공고에 대해 무엇이든 물어보세요..."

    # 입력
    prompt = st.chat_input(placeholder)
    st.caption("RFP 문서를 사이드바 + 로 업로드하거나, 공고명으로 바로 질문하세요.")

    if prompt:
        if st.session_state.get("pending_agent"):
            # 대기 중인 에이전트(적격성)에 이어붙여 같은 공고로 실행(일반 QA로 새지 않게).
            with st.spinner("적격성 판정 중..."):
                _continue_pending(prompt)
        else:
            with st.spinner("답변 생성 중..."):
                _send(prompt, None)
        st.rerun()
    elif clicked:
        with st.spinner(f"‘{clicked['label']}’ 실행 중..."):
            _dispatch_agent(clicked)
        st.rerun()


main()
