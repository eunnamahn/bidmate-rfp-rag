# 시스템 아키텍처

> 입찰메이트 — RFP RAG 질의응답 시스템 · 기준 브랜치 `limdongkyu`
> 💡 아래는 Mermaid 다이어그램 — GitHub·VS Code 미리보기에서 자동 렌더링됩니다.

## 전체 구조

```mermaid
flowchart TD
    catalog["catalog.py · 교체 프리셋"] --> cfg["config.yaml"]

    subgraph ING["📥 인덱싱 파이프라인 (배치)"]
        direction TB
        files["data/files<br/>HWP · PDF"] --> loader["loader.py<br/>파서 5종 · parse_file"]
        loader --> chunk["chunking.py<br/>recursive / section"]
        chunk --> embed["embedding.py<br/>build_vectorstore"]
        embed --> faiss[("FAISS 캐시<br/>data/faiss_rfp")]
    end

    subgraph NARA["🛰️ 나라장터 수집 에이전트 (독립 배치/CLI)"]
        direction TB
        napi["나라장터 OpenAPI<br/>입찰공고 목록"] --> dl["다운로드 · zip 해제"]
        dl --> pf["loader.parse_file"]
        pf --> meta1["metadata_agent<br/>발주기관·금액·마감일·사업명"]
        meta1 --> split2["chunking.split_documents"]
        split2 --> add["embedding.add_to_vectorstore<br/>(증분 추가)"]
    end

    subgraph GRAPH["🧭 LangGraph 수퍼바이저 (src/graph.py)"]
        direction TB
        cls{"classify_intent<br/>키워드 → LLM 폴백"}
        cls -->|qa| qa["qa 노드<br/>generate_answer(qa)"]
        cls -->|summary| sm["summary 노드<br/>generate_answer(summary)"]
        cls -->|metadata| md["metadata 노드<br/>검색 → extract_metadata"]
        cls -->|collect| col["collect 노드<br/>배치 실행 안내만"]
    end

    subgraph RT["💬 검색·생성 런타임"]
        direction TB
        retr["retriever.py<br/>top-k · 메타필터 · 기관 라우팅"]
        gen["generator.py<br/>답변 · 인용 · 후속질문 재작성"]
    end

    subgraph IF["🖥️ 인터페이스"]
        ui["streamlit/app.py"] --> backend["backend/app.py · FastAPI<br/>/chat · /summarize"]
    end

    subgraph EV["📊 평가"]
        golden["golden_set.json<br/>50문항"] --> ragas["ragas_eval.py<br/>run_ragas_experiments"]
        ragas --> report["results_summary.md"]
    end

    backend --> cls
    qa --> retr
    sm --> retr
    md --> retr
    qa --> gen
    sm --> gen
    retr --> faiss
    add --> faiss
    col -. "안내" .-> NARA

    cfg -. 제어 .-> ING
    cfg -. 제어 .-> GRAPH
    cfg -. 제어 .-> NARA
    ragas -. 재사용 .-> retr
    ragas -. 재사용 .-> gen
```

## 질의응답 흐름 (LangGraph 라우팅)

```mermaid
sequenceDiagram
    autonumber
    actor U as 사용자
    participant API as FastAPI (/chat)
    participant G as LangGraph 수퍼바이저
    participant R as retriever
    participant MA as metadata_agent
    participant GEN as generator + LLM

    U->>API: 질문 입력
    API->>G: graph.invoke(question, history, filename)
    Note over G: /summarize는 intent="summary" 강제 주입
    G->>G: classify_intent (키워드 → 모호 시 LLM)

    alt intent = qa / summary
        G->>R: 질문별 검색기 선택(_select_retriever)
        R-->>G: 관련 청크 top-k
        G->>GEN: generate_answer(context, question)
        GEN-->>G: 답변 + 근거 출처
    else intent = metadata
        G->>R: 검색
        R-->>G: 관련 청크
        G->>MA: extract_metadata(text)
        MA-->>G: {발주기관·금액·마감일·사업명}
    else intent = collect
        G-->>G: 배치 실행 안내(수집 미실행)
    end

    G-->>API: answer + sources
    API-->>U: 응답 표시
```

## 컴포넌트 요약

| 레이어 | 파일 | 역할 |
|---|---|---|
| 설정 | `config.yaml`, `src/catalog.py` | 교체 축(임베딩·LLM·청킹·검색·파서) 스위치 + `nara` 섹션 |
| 파싱 | `src/loader.py` | HWP(pyhwp·hwp-hwpx-parser·olefile) / PDF(pymupdf·pdfplumber) · `parse_file`(단일 파일) |
| 청킹 | `src/chunking.py` | recursive / section(키워드 섹션) |
| 임베딩 | `src/embedding.py` | FAISS 캐시(`build_vectorstore`) + 증분 추가(`add_to_vectorstore`) |
| 검색 | `src/retriever.py` | top-k + 메타데이터 필터 + 기관 자동 라우팅 |
| 생성 | `src/generator.py` | 답변 + 인용(citation) 정제 + 후속질문 재작성 |
| **오케스트레이션** | `src/graph.py` | **LangGraph 수퍼바이저: 의도 분류 → qa/summary/metadata/collect 라우팅** |
| **에이전트** | `src/agents/metadata_agent.py`, `src/agents/nara_agent.py` | 메타데이터 LLM 추출 · 나라장터 공고 수집(독립 배치/CLI) |
| 인터페이스 | `backend/app.py`, `src/streamlit/app.py` | FastAPI API · Streamlit UI |
| 평가 | `src/evaluation/ragas_eval.py`, `run_ragas_experiments.py` | 골든셋 18조합 RAGAS + 커스텀 지표 |

## 의도 라우팅 (수퍼바이저)

| 의도 | 트리거(키워드 우선·모호 시 LLM) | 동작 |
|---|---|---|
| `qa` | 기본값 | 검색 → `generate_answer(qa)` |
| `summary` | "요약/정리/개요" · `/summarize`는 강제 | 검색 → `generate_answer(summary)` 고정 양식 |
| `metadata` | "메타데이터/기본 정보 추출" | 검색 → `metadata_agent.extract_metadata` |
| `collect` | "신규 공고/공고 수집/나라장터…수집" | 수집 미실행, 배치 실행 방법 안내 |

## 나라장터 수집 (독립 배치/CLI)

```bash
python -m src.agents.nara_agent              # 당일 수집
python -m src.agents.nara_agent 2024-04-09   # 특정 날짜
python -m src.agents.nara_agent 2024-04-09 --max 5   # 건수 제한
```

- 환경변수 `NARA_API_KEY`(공공데이터포털 발급) 필요 — `.env`에 저장
- 수집 산출물: `data/nara_seen.json`(중복 방지), `data/nara_downloads/`(첨부)
- 수집 문서는 `add_to_vectorstore`로 기존 FAISS에 증분 추가되어 이후 질문에서 검색됨

## 교체 가능 부품 (catalog 프리셋)

| 축 | 옵션 |
|---|---|
| LLM | gpt-5-nano · gpt-5-mini · exaone3.5:7.8b(Ollama) · qwen2.5:14b(Ollama) |
| 임베딩 | OpenAI 3-small · KURE-v1(로컬) · BGE-M3(로컬) |
| 청킹 | recursive 300/500/1000 · section 500/1000 |
| 파서 | HWP: pyhwp·hwp-hwpx-parser·olefile / PDF: pymupdf·pdfplumber |
| 검색 | k, 메타데이터 필터, (옵션) BM25·리랭커 |
