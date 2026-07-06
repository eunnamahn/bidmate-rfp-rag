# 오토리서치 결과 데이터 (v1 → v3, 약 700런)

결과보고서 §10.3 "RAG 최적화 3단계 검증"의 원본 데이터.
평가셋·채점기는 `src/evaluation/`에 있다:
- `golden_set_autoresearch_v1_q50.json` — v1 그리드용 (KMeans 대표 19문서, 50문항)
- `golden_set_autoresearch_v2_q95.json` — v2·v3 평가셋 (5유형 × 19문항 = 95)
- `scorer_v2.py` — 유형별 하이브리드 채점기 (모든 점수의 산출 코드)

## 파일

| 파일 | 단계 | 내용 |
|---|---|---|
| `v1_grid_results.json` | v1 | 검색 그리드 484런 (청킹×임베딩×k×BM25, 검색 지표 기준) |
| `v2_finalists_qwen.json` | v2 | 상위 후보 답변 평가 12런 (qwen2.5, 청크 곡선 2700/3500/4000 포함) |
| `v2_finalists_exaone.json` | v2 | 상위 후보 답변 평가 9런 (exaone3.5) |
| `v3_matrix_summary.json` | v3 | 공정 조건 재검증 32런 — **최종 근거** (임베딩 8종 × 청킹 2 × LLM 2) |

각 런 공통 필드: `통합점수`(5유형 macro×100) · `유형별`(단답/숫자/계약조건/요약/거절) ·
`숫자환각률` · `근거포함` · `hit@k` · `mrr` · `latency` · `config`

## 단계 요약

1. **v1 그리드(484런)**: 전 조합을 검색 지표로 넓게 탐색. 이때의 골든셋(q50)·채점의
   한계를 확인하고 v2 평가셋(q95)과 유형별 채점기를 재구축했다 → v1 수치는 v2 이후와
   직접 비교하지 말 것.
2. **v2 깔때기(192+21런)**: 검색 그리드로 거른 상위 조합만 95문항 답변 평가.
   챔피언 후보 확정 + BM25 off 확정. ⚠️ 이 단계의 k=8 행들은 컨텍스트 창(8192) 초과로
   점수가 눌린 조건 — 임베딩 간 비교는 v3를 볼 것.
3. **v3 공정 매트릭스(32런)**: 모든 후보를 동일 조건(k=4 · chunk 2000 · overlap 200 ·
   num_ctx 8192 · BM25 off)에서 재검증. **최종 채택: recursive · 2000/200 · KURE-v1 ·
   EXAONE · dense k=4 · num_ctx 8192** (통합 82.8) — 현 config.yaml 기본값의 근거.

해석 주의: 동일 조합 반복 실행 시 ±1점 수준의 재현 편차(GPU 비결정성) → 1~2점 차이는 동급.

재현: `scorer_v2.evaluate_combo(overrides, load_golden_v2(), Embedder(), k=4)`
(Ollama 터널 필요. 실험 러너 원본은 실험 작업공간에 보존되어 있음)
