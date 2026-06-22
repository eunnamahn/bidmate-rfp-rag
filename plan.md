# 입찰메이트 RFP RAG — 베이스라인 구축 계획 (plan.md)

## 0. 이 문서의 목적

OpenAI API로 동작하는 RAG 베이스라인을 먼저 완성하고, 이후 임베딩·청킹·검색·생성 전략을 부분적으로 교체하며 실험할 수 있는 구조를 만든다. 핵심 원칙은 "베이스라인은 API로 빠르게, 교체 지점은 전부 설정/인자로 분리"다. 코드를 짤 때 이 원칙을 위반하지 말 것.

작업 언어/스타일 규칙:
- 모든 주석·문서·로그는 한국어, 이모지 사용 금지.
- pip 설치는 패키지별로 한 줄씩 분리해서 안내(한 줄에 여러 패키지 묶지 말 것).
- API 키는 코드에 하드코딩 금지. `.env`(OPENAI_API_KEY)에서 읽고 `.env`는 `.gitignore`에 추가.

## 1. 프로젝트 개요

가상 B2G 입찰컨설팅 스타트업 "입찰메이트"의 사내 RAG 시스템. RFP 문서를 대상으로 두 태스크를 수행한다.
- QA: "이 사업 평가배점 기준은?" 같은 질문에 검색 후 답변.
- 요약: RFP 한 건을 받아 예산·기관·마감일·핵심 요구사항을 양식에 맞춰 정리.

두 태스크는 같은 인덱싱 파이프라인을 공유하고 런타임 프롬프트만 다르다.

## 2. 데이터 상황 (EDA 확정 결론)

- 메타데이터: `data/data_list.csv` (100건, 12컬럼). 컬럼: 공고번호, 공고차수, 사업명, 사업금액, 발주기관, 공개일자, 입찰참여시작일, 입찰참여마감일, 사업요약, 파일형식, 파일명, 텍스트.
- 원본 파일: `data/files/` (HWP 96, PDF 4), 메타와 파일명으로 1:1 매칭됨(정합성 검증 완료).
- 가장 중요한 결론: 메타데이터의 `텍스트` 컬럼은 원본의 앞부분 발췌본(원본의 약 1/28, 단어 99.6% 일치하지만 분량 부족)이다. 따라서 본문 소스로 `텍스트` 컬럼을 쓰지 말고 원본 파일을 직접 재파싱한다.
- 원본 HWP는 문서당 평균 124개의 표를 가지며, 평가배점·요구사항·제출서류 등 핵심 정보가 표에 들어있다. 표를 마크다운으로 보존해야 한다.
- 본문 길이 편차가 크고(원본 평균 8만 자), 발주기관명이 본문에 없는 문서가 약 32건이다. 청크에 메타데이터 헤더를 붙여야 한다.
- "추출 성공"이 "본문 확보"를 보장하지 않는다(스캔본 HWP 1건, 추출 실패 1건 존재). 품질 플래그가 필요하다.
- 사업금액에 0원으로 입력된 6건은 미기재이므로 결측처럼 취급한다.

## 3. 확정 설계 결정

| 단계 | 베이스라인(우선 구현) | 교체 축(실험, 인터페이스만 열어둘 것) |
| --- | --- | --- |
| 본문 소스 | 원본 재파싱 (HWP=hwp-hwpx-parser 표 마크다운, PDF=pdfplumber) | (고정) |
| 전처리 | 목차 제거 + 제어문자/깨진 특수문자 정제 | 정제 강도 |
| 청킹 | RecursiveCharacterTextSplitter (size=500, overlap=50) | 섹션 인식 청킹, 크기/오버랩 |
| 임베딩 | OpenAI text-embedding-3-small | nlpai-lab/KURE-v1, BAAI/bge-m3 |
| 벡터DB | FAISS (save_local로 캐시) | Chroma |
| 검색 | dense top-k (k=4) | BM25 하이브리드(EnsembleRetriever), reranker(bge-reranker-v2-m3), 메타데이터 필터 |
| 생성 | OpenAI gpt-4o-mini (temperature=0) | Ollama Qwen3-8B |
| 라우팅 | (베이스라인 제외, 인터페이스만 준비) | 본문 RAG vs 메타데이터 집계 분기 |

교체 축은 베이스라인에서 당장 구현하지 않아도 되지만, 나중에 한 줄로 갈아끼울 수 있도록 팩토리 함수와 config로 분리해 둘 것.

## 4. 디렉토리 구조

```
.
├── config.yaml              # 모든 교체 지점 설정
├── .env                     # OPENAI_API_KEY (gitignore)
├── .gitignore
├── data/
│   ├── data_list.csv
│   ├── files/               # 원본 HWP/PDF
│   └── faiss_rfp/           # 저장된 벡터스토어 (자동 생성)
├── src/
│   ├── config.py            # config 로드 + 부품 팩토리
│   ├── loader.py            # 원본 파싱 + 전처리 + Document 변환
│   ├── chunking.py          # 청킹 전략
│   ├── embedding.py         # 임베딩 팩토리 + FAISS 빌드/캐시
│   ├── retriever.py         # 검색기 구성 (dense / hybrid / rerank)
│   ├── generator.py         # 프롬프트 + LLM 팩토리 + 체인
│   ├── pipeline.py          # build_pipeline: 전체 조립
│   └── evaluation/          # 이미 작성됨, 재사용
│       ├── golden_set.json
│       ├── evaluate.py
│       └── build_golden.py
└── run_baseline.py          # 베이스라인 실행 + 골든셋 채점
```

`src/evaluation/`은 이미 구현되어 있으니 새로 만들지 말고 그대로 import해서 쓴다.

## 5. 구현 태스크 (순서대로)

### Task 1. 환경 및 설정
- `config.yaml` 작성(아래 6절 스키마).
- `src/config.py`: yaml 로드, `OPENAI_API_KEY`를 `.env`에서 로드(python-dotenv), `make_embedding(cfg)`·`make_llm(cfg)` 팩토리 제공.
- `make_embedding`: provider가 openai면 OpenAIEmbeddings, huggingface면 HuggingFaceEmbeddings 반환.
- `make_llm`: provider가 openai면 ChatOpenAI, ollama면 ChatOllama 반환.
- `.gitignore`에 `.env`, `data/faiss_rfp/`, `__pycache__/` 추가.

### Task 2. 로더 (`src/loader.py`)
- `load_documents(data_dir) -> List[Document]`.
- `data_list.csv`를 읽고, 각 행의 `파일명`으로 `data/files/`의 원본을 파싱한다.
- HWP는 hwp-hwpx-parser로 텍스트+표(마크다운) 추출, PDF는 pdfplumber.
- 파싱 실패/타임아웃 대비: 파일당 timeout, 실패 시 해당 문서는 `fallback_used=True`로 표시하되 진행 중단하지 말 것.
- 품질 플래그: 추출 글자수가 임계(예: 1000자) 미만이면 `likely_scanned=True`.
- 전처리: 목차 블록 제거, 제어문자·깨진 특수문자(예: 茼ȃ류) 제거, 연속 공백 정리.
- 각 Document의 metadata에 다음을 반드시 넣는다: `파일명`, `기관`(발주기관), `금액`(사업금액, 0원은 None 처리), `마감일`(datetime 변환), `파일형식`, `likely_scanned`, `fallback_used`.
- 본문 소스는 원본 파싱 결과만 사용. `텍스트` 컬럼은 쓰지 않는다.

### Task 3. 청킹 (`src/chunking.py`)
- `split_documents(docs, cfg) -> List[Document]`.
- 베이스라인: RecursiveCharacterTextSplitter(chunk_size, overlap은 config에서).
- 각 청크 metadata는 원본 Document의 metadata를 상속하고, 청크 본문 앞에 "[기관 | 사업금액 | 마감일]" 헤더 문자열을 덧붙인다(기관명이 본문에 없는 문서 대응).
- 섹션 인식 청킹은 함수 분기로 자리만 만들어 두고 베이스라인은 recursive 사용.

### Task 4. 임베딩 + 벡터스토어 (`src/embedding.py`)
- `build_vectorstore(chunks, embedding, cache_dir) -> FAISS`.
- cache_dir에 저장본이 있으면 `FAISS.load_local`로 로드, 없으면 `FAISS.from_documents` 후 `save_local`.
- 임베딩 모델이 바뀌면 캐시 키도 달라야 하므로 cache_dir 이름에 모델명을 포함.

### Task 5. 검색기 (`src/retriever.py`)
- `build_retriever(vectorstore, chunks, cfg) -> Retriever`.
- 베이스라인: `vectorstore.as_retriever(search_kwargs={"k": k})`.
- config의 `use_bm25`가 true면 BM25Retriever와 EnsembleRetriever로 하이브리드 구성.
- reranker는 자리만 만들어 두고 베이스라인은 미적용.

### Task 6. 생성 (`src/generator.py`)
- 프롬프트: "아래 문맥만 근거로 답하라. 문맥에 없으면 '문서에서 확인할 수 없습니다'라고 답하라." (거절 유도 — 평가의 거절 정확도와 직결되므로 반드시 포함)
- QA용 프롬프트와 요약용 프롬프트를 각각 제공.
- `build_chain(retriever, llm, mode="qa"|"summary") -> Runnable` (LCEL로 context→prompt→llm→StrOutputParser).

### Task 7. 파이프라인 조립 (`src/pipeline.py`)
- `build_pipeline(cfg) -> (chain, retriever)`:
  로더→청킹→임베딩→벡터스토어→검색기→생성체인을 config 기준으로 조립.
- 이미 만든 docs/chunks/vectorstore를 재사용할 수 있도록 인자로 받는 옵션도 둘 것(매 실험마다 재파싱·재임베딩 방지).

### Task 8. 베이스라인 실행 + 평가 (`run_baseline.py`)
- config.yaml 로드 → build_pipeline → `src/evaluation/evaluate.py`의 `evaluate(chain, retriever, golden, k)` 호출.
- 골든셋은 `src/evaluation/golden_set.json` 사용.
- 출력: retrieval(Hit@k, MRR), metadata_extraction(accuracy), refusal(refusal_accuracy)를 보기 좋게 print. (서술형 LLM-judge는 judge_llm 넘기면 동작, 베이스라인에선 옵션)
- 이 점수가 이후 모든 실험의 기준선이 된다.

## 6. config.yaml 스키마

```yaml
data:
  csv: data/data_list.csv
  files: data/files
  faiss_cache: data/faiss_rfp

embedding:
  provider: openai            # openai | huggingface
  model: text-embedding-3-small

llm:
  provider: openai            # openai | ollama
  model: gpt-4o-mini
  temperature: 0

chunk:
  strategy: recursive         # recursive | section
  size: 500
  overlap: 50

retriever:
  k: 4
  use_bm25: false
  use_reranker: false

eval:
  golden_set: src/evaluation/golden_set.json
  k: 4
```

provider만 바꾸면 OpenAI ↔ 로컬 모델이 스위치되도록 구현할 것.

## 7. 평가 연동 (이미 구현된 모듈)

`src/evaluation/evaluate.py`의 인터페이스를 그대로 사용한다.
- 골든셋은 4종 질문 유형: metadata_extraction(본문에서 금액·마감일 추출, 정답은 메타), body_grounded(평가배점 등 서술형), unanswerable(근거 없는 질문 → 거절해야 정답). (집계형 aggregation은 추후 추가 예정)
- `evaluate(chain, retriever, golden, k, judge_llm=None)` → retrieval/metadata_extraction/refusal/(body_grounded) 리포트.
- 검색 평가가 동작하려면 청크 metadata에 `파일명`이 있어야 한다(Task 3에서 보장).
- `summary_score(report)`로 구성 비교용 한 줄 지표를 뽑는다.

## 8. 실험 단계 (베이스라인 완성 후, 이번 작업 범위 아님)

베이스라인이 돌고 기준 점수가 나오면, OpenAI를 고정한 채 우리가 실제로 검증하려는 축(청킹 전략, BM25 하이브리드, reranker)을 하나씩 켜서 골든셋 점수를 비교한다. 마지막에 임베딩을 KURE로, LLM을 Qwen3-8B로 바꿔 API 대비 로컬 성능을 비교한다. 각 실험은 config 한두 줄 수정으로 가능해야 한다.

## 9. 완료 기준 (Definition of Done)

- `python run_baseline.py`가 에러 없이 끝나고 골든셋 점수(Hit@k, MRR, 메타추출 정확도, 거절 정확도)를 출력한다.
- config.yaml의 embedding.provider / llm.provider를 huggingface / ollama로 바꿔도 코드 수정 없이 동작한다(로컬 모델이 설치되어 있다는 전제).
- 본문 소스로 `텍스트` 컬럼을 쓰지 않고 원본 파싱 결과만 사용한다.
- 청크 metadata에 파일명·기관·금액·마감일·품질플래그가 들어있다.
- API 키가 코드/깃에 노출되지 않는다.
