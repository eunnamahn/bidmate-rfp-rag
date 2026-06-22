# 입찰메이트 RFP RAG 베이스라인

가상 B2G 입찰컨설팅 스타트업 "입찰메이트"의 사내 RAG 시스템 베이스라인.
RFP 문서를 대상으로 QA와 요약 두 태스크를 수행한다. 상세 설계는 `plan.md` 참고.

## 설계 원칙

- 베이스라인은 OpenAI API로 빠르게 구축한다.
- 교체 축(임베딩/LLM/청킹/검색)은 전부 `config.yaml`과 팩토리 함수로 분리해
  config 한두 줄 수정으로 갈아끼울 수 있다.
- 본문 소스로 메타데이터의 `텍스트` 컬럼을 쓰지 않고 원본 파일을 직접 재파싱한다.
- API 키는 코드에 하드코딩하지 않고 `.env`에서만 읽는다.

## 설치

패키지는 한 줄씩 설치한다. (예시)

```
pip install langchain
pip install langchain-core
pip install langchain-community
pip install langchain-openai
pip install langchain-text-splitters
pip install faiss-cpu
pip install pandas
pip install pyyaml
pip install python-dotenv
pip install pdfplumber
pip install hwp-hwpx-parser
pip install rank-bm25
```

로컬 모델 실험 시에만 추가로:

```
pip install langchain-huggingface
pip install langchain-ollama
pip install sentence-transformers
```

## 환경 변수

프로젝트 루트에 `.env` 파일을 만들고 키를 넣는다(`.env`는 깃에 올라가지 않음):

```
OPENAI_API_KEY=sk-...
```

## 디렉토리 구조

```
.
├── config.yaml              # 모든 교체 지점 설정
├── .env                     # OPENAI_API_KEY (직접 생성, gitignore)
├── data/
│   ├── data_list.csv        # 메타데이터 (직접 배치)
│   ├── files/               # 원본 HWP/PDF (직접 배치)
│   └── faiss_rfp/           # 벡터스토어 캐시 (자동 생성)
├── src/
│   ├── config.py            # config 로드 + 부품 팩토리
│   ├── loader.py            # 원본 파싱 + 전처리 + Document 변환
│   ├── chunking.py          # 청킹 전략
│   ├── embedding.py         # 임베딩 + FAISS 빌드/캐시
│   ├── retriever.py         # 검색기 (dense / hybrid / rerank)
│   ├── generator.py         # 프롬프트 + LLM 체인
│   ├── pipeline.py          # 전체 조립
│   ├── streamlit/
│   │   └── app.py           # Streamlit UI (백엔드 API만 호출)
│   └── evaluation/          # 골든셋 채점
│       ├── evaluate.py
│       ├── golden_set.json
│       └── build_golden.py
├── backend/
│   └── app.py               # FastAPI 서버 (/health /documents /chat /summarize)
├── main.py                  # 인덱싱 엔트리포인트 (FAISS 저장)
├── run_baseline.py          # 베이스라인 실행 + 채점
└── run.sh                   # 백엔드 + 프론트 동시 실행
```

## 실행

### 1) 인덱싱 (최초 1회)

```
python main.py
```

원본 100건을 파싱/임베딩해 FAISS 캐시(`data/faiss_rfp/<모델명>`)를 만든다.

### 2) 베이스라인 평가

```
python run_baseline.py
```

검색(Hit@k, MRR), 메타추출 정확도, 거절 정확도를 출력한다.

### 3) 풀스택 실행 (백엔드 + UI)

```
./run.sh
```

FastAPI(기본 8000)와 Streamlit(기본 8501)을 함께 띄운다. 브라우저에서
`http://localhost:8501`을 열어 질문 -> 답변 -> 근거 출처를 확인한다.
백엔드만 따로 띄우려면 `python -m uvicorn backend.app:app --port 8000`.

> 참고: 사용 중인 OpenAI 키가 gpt-4o-mini에 접근 권한(RPM=0)이 없어
> 생성 LLM은 `gpt-5-nano`로 설정되어 있다(`config.yaml`의 `llm.model`).

## 데이터 준비 (실행 전 필요)

이 저장소에는 코드만 있고 데이터는 포함되어 있지 않다. 실행하려면:

1. `data/data_list.csv` (100건, 12컬럼)와 `data/files/`의 원본 HWP/PDF를 배치한다.
2. 골든셋을 실제 데이터에 맞게 생성한다:
   ```
   python -m src.evaluation.build_golden --n 20
   ```
   metadata_extraction(금액/마감일)과 unanswerable은 자동 생성된다.
   `body_grounded`(서술형, 평가배점 등) 항목의 `answer`는 본문을 보고 직접 채워 넣고,
   채점하려면 `evaluate(..., judge_llm=...)`로 judge 모델을 넘긴다.

## 프론트엔드에서 모델/임베딩/청킹 전환

Streamlit 사이드바의 "설정" 패널에서 LLM·임베딩·청킹 프리셋을 드롭다운으로
골라 "설정 적용"을 누르면 백엔드가 즉시 재조립한다. 임베딩/청킹을 바꾸면
재인덱싱이 필요할 수 있고(캐시가 있으면 즉시 전환), LLM 교체는 즉시 적용된다.

선택지를 추가하려면 `src/catalog.py`의 `LLM_PRESETS`/`EMBEDDING_PRESETS`/
`CHUNK_PRESETS`에 항목 하나만 추가하면 된다(백엔드 `/config`와 프론트 드롭다운에
자동 반영). provider(openai/huggingface/ollama)가 아닌 완전히 새로운 종류를
추가할 때만 `src/config.py`의 팩토리도 함께 손본다.

## 실험 전환 예시 (config.yaml 직접 수정)

`config.yaml`만 수정해도 된다.

- HWP 파서: 기본 `parser.hwp: pyhwp`(빠름) / 표 마크다운 보존이 필요하면 `hwp-hwpx-parser`
- PDF 파서: 기본 `parser.pdf: pymupdf`(빠름/안정) / 표 정밀도가 필요하면 `pdfplumber`
- BM25 하이브리드: `retriever.use_bm25: true`
- 청크 크기: `chunk.size`, `chunk.overlap`
- 로컬 임베딩: `embedding.provider: huggingface`, `embedding.model: nlpai-lab/KURE-v1`
- 로컬 LLM: `llm.provider: ollama`, `llm.model: qwen3:8b`
