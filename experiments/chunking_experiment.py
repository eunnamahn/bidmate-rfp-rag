"""
전처리 파이프라인 실험 코드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실험 목적:
    어떤 파싱/청킹 조합이 가장 좋은 청크를 만드는지 비교한다.
    RAGAS 점수 나오면 이 결과와 함께 최적 조합을 선정한다.

실험 항목 (총 48가지):
    [HWP 파서]   pyhwp / hwp-hwpx-parser / olefile  (3가지)
    [PDF 파서]   pymupdf / pdfplumber                (2가지)
    [청킹 전략]  recursive / section                 (2가지)
    [청킹 크기]  500 / 1000 / 1500 / 2000            (4가지)
    → 3 × 2 × 2 × 4 = 48가지

사용 방법:
    1. 구글 드라이브 마운트 (아래 셀 참고)
    2. 패키지 설치 (아래 셀 참고)
    3. 이 셀 실행

[사전 준비 - 먼저 실행할 셀들]

셀 1) 드라이브 마운트
    from google.colab import drive
    drive.mount('/content/drive')

셀 2) 패키지 설치
    !pip install pymupdf pyhwp langchain-text-splitters pdfplumber hwp-hwpx-parser olefile -q

셀 3) 이 파일 실행
    exec(open('/content/drive/MyDrive/codeit_mission/중급프로젝트/experiments/chunking_experiment.py').read())
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import importlib.util
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# ── 경로 설정 ────────────────────────────────────────
# ⚠️ 본인 드라이브 경로 다르면 여기만 수정
BASE      = '/content/drive/MyDrive/codeit_mission/중급프로젝트'
CSV_PATH  = '/content/drive/MyDrive/codeit_mission/원본 데이터/data_list.csv'
FILES_DIR = '/content/drive/MyDrive/codeit_mission/원본 데이터/files'

if BASE not in sys.path:
    sys.path.insert(0, BASE)

# ── 실험 조합 자동 생성 (32가지) ─────────────────────
HWP_PARSERS    = ["pyhwp", "hwp-hwpx-parser", "olefile"]
PDF_PARSERS    = ["pymupdf", "pdfplumber"]
CHUNK_STRATEGIES = ["recursive", "section"]
CHUNK_SIZES    = [
    (500,  50),
    (1000, 100),
    (1500, 150),
    (2000, 200),
]

experiments = []
for hwp, pdf, strategy, (size, overlap) in itertools.product(
    HWP_PARSERS, PDF_PARSERS, CHUNK_STRATEGIES, CHUNK_SIZES
):
    experiments.append({
        "label"         : f"HWP={hwp} / PDF={pdf} / 전략={strategy} / 크기={size}",
        "CHUNK_SIZE"    : size,
        "CHUNK_OVERLAP" : overlap,
        "CHUNK_STRATEGY": strategy,
        "HWP_PARSER"    : hwp,
        "PDF_PARSER"    : pdf,
    })

# ── 실험 실행 ────────────────────────────────────────
print(f"실험 시작 — 총 {len(experiments)}개")
print("=" * 60)

results = []

for i, exp in enumerate(experiments, 1):
    print(f"\n[{i:02d}/{len(experiments)}] {exp['label']}")
    try:
        # config 로드 후 실험값 주입
        spec = importlib.util.spec_from_file_location("data.config", f"{BASE}/data/config.py")
        config = importlib.util.module_from_spec(spec)
        sys.modules["data.config"] = config
        spec.loader.exec_module(config)
        config.CHUNK_SIZE      = exp["CHUNK_SIZE"]
        config.CHUNK_OVERLAP   = exp["CHUNK_OVERLAP"]
        config.CHUNK_STRATEGY  = exp["CHUNK_STRATEGY"]
        config.HWP_PARSER      = exp["HWP_PARSER"]
        config.PDF_PARSER      = exp["PDF_PARSER"]
        config.REMOVE_TOC      = True
        config.REMOVE_HWP_TAGS = True

        # preprocessor 로드
        spec2 = importlib.util.spec_from_file_location(
            "preprocessor", f"{BASE}/experiments/preprocessor.py"
        )
        preprocessor = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(preprocessor)

        # 파이프라인 실행
        chunks  = preprocessor.run_pipeline(CSV_PATH, FILES_DIR)
        lengths = [len(c['text']) for c in chunks]

        results.append({
            "HWP 파서"          : exp["HWP_PARSER"],
            "PDF 파서"          : exp["PDF_PARSER"],
            "청킹 전략"         : exp["CHUNK_STRATEGY"],
            "청킹 크기"         : exp["CHUNK_SIZE"],
            "총 청크 수"        : len(chunks),
            "평균 길이(자)"     : round(sum(lengths) / len(lengths)),
            "최소 길이(자)"     : min(lengths),
            "최대 길이(자)"     : max(lengths),
            "짧은 청크(200자↓)" : sum(1 for l in lengths if l < 200),
            "상태"              : "성공",
        })
        print(f"    → {len(chunks)}청크 / 평균 {round(sum(lengths)/len(lengths))}자")

    except Exception as e:
        results.append({
            "HWP 파서"          : exp["HWP_PARSER"],
            "PDF 파서"          : exp["PDF_PARSER"],
            "청킹 전략"         : exp["CHUNK_STRATEGY"],
            "청킹 크기"         : exp["CHUNK_SIZE"],
            "총 청크 수"        : 0,
            "평균 길이(자)"     : 0,
            "최소 길이(자)"     : 0,
            "최대 길이(자)"     : 0,
            "짧은 청크(200자↓)" : 0,
            "상태"              : f"실패: {str(e)[:40]}",
        })
        print(f"    → 실패: {e}")

# ── 결과 테이블 ──────────────────────────────────────
print("\n" + "=" * 60)
print("실험 결과 비교표 (32가지)")
print("=" * 60)
df = pd.DataFrame(results)
display(df)

# ── CSV 저장 ─────────────────────────────────────────
save_path = f"{BASE}/experiments/실험결과_{pd.Timestamp.now().strftime('%m%d_%H%M')}.csv"
df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"\n결과 저장 완료: {save_path}")

# ── 시각화 ───────────────────────────────────────────
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

success_df = df[df["상태"] == "성공"].copy()
success_df["실험"] = (
    success_df["HWP 파서"].str.replace("hwp-hwpx-parser", "hwpx") + "\n"
    + success_df["PDF 파서"] + "\n"
    + success_df["청킹 전략"] + "\n"
    + success_df["청킹 크기"].astype(str)
)

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('전처리 실험 결과 비교 (32가지)', fontsize=14, fontweight='bold')

axes[0].bar(range(len(success_df)), success_df["총 청크 수"], color='#2196F3')
axes[0].set_title('총 청크 수')
axes[0].set_ylabel('청크 수')
axes[0].set_xticks(range(len(success_df)))
axes[0].set_xticklabels(success_df["실험"], rotation=90, fontsize=6)

axes[1].bar(range(len(success_df)), success_df["평균 길이(자)"], color='#4CAF50')
axes[1].set_title('평균 청크 길이 (자)')
axes[1].set_ylabel('글자 수')
axes[1].axhline(y=1000, color='red', linestyle='--', label='목표 1000자')
axes[1].legend()
axes[1].set_xticks(range(len(success_df)))
axes[1].set_xticklabels(success_df["실험"], rotation=90, fontsize=6)

plt.tight_layout()
plt.show()

print("\n다음 단계: RAGAS 점수와 함께 최적 조합 선정")
