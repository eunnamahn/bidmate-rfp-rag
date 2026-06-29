"""골든셋 생성 스캐폴드.

컨설턴트가 필요한 '4대 정보축'을 골고루 평가하도록 골든셋을 만든다.
  - 대상 기관 / 예산(사업금액) / (입찰)마감일 : 메타데이터로 정답을 자동 생성(자동 채점)
  - 핵심 요구조건 / 제출 방식 / 평가 배점     : 본문 서술형(judge_llm 채점, answer는 직접 채움)
  - unanswerable                              : 근거 없는 질문, 거절해야 정답

metadata_extraction은 정답이 메타에서 자동으로 나오므로 정량 백본이 되고,
body_grounded(서술형)는 본문을 보고 answer를 채운 뒤 judge_llm으로 채점한다.

사용:
  python -m src.evaluation.build_golden --n 12 --body-n 6
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

OUT_PATH = "src/evaluation/golden_set.json"

# 근거 문서가 없어 거절해야 하는 질문(어느 RFP에도 답이 없도록 구성)
UNANSWERABLE_TEMPLATES = [
    "입찰메이트 사옥의 주차 가능 대수는 몇 대인가?",
    "이 사업 담당자의 개인 휴대폰 번호는?",
    "전 직원의 점심 메뉴 선호도 조사 결과는?",
    "이 사업의 작년 동일 사업 낙찰 기업은 어디인가?",
    "이 RFP 작성자의 이메일 비밀번호는 무엇인가?",
    "발주기관 청사의 준공 연도는?",
]

# 본문 서술형 질문 템플릿: (정보축 라벨, 질문 형식). answer는 본문 검토 후 직접 채운다.
BODY_AXES = [
    ("평가 배점", "'{사업명}' 사업의 평가 배점 기준은 어떻게 되는가?"),
    ("핵심 요구조건", "'{사업명}' 사업에서 요구하는 핵심 과업/요구사항은 무엇인가?"),
    ("제출 방식", "'{사업명}' 사업의 제안서 제출 방법과 제출 서류는 무엇인가?"),
]


def _amount_to_text(value: Any) -> str:
    digits = re.sub(r"[^\d]", "", str(value))
    return f"{int(digits):,}" if digits and int(digits) != 0 else ""


def build(n: int, body_n: int) -> List[Dict[str, Any]]:
    df = pd.read_csv("data/data_list.csv")
    # 실제 CSV 컬럼명에 공백이 있다(예: '사업 금액'). 공백을 제거해 정규화한다.
    df.columns = [str(c).replace(" ", "") for c in df.columns]
    golden: List[Dict[str, Any]] = []

    # 1) 메타데이터 추출(자동 정답): 대상 기관 / 예산 / 마감일
    for _, row in df.head(n).iterrows():
        filename = str(row["파일명"])
        사업명 = str(row.get("사업명", "")).strip()

        기관 = str(row.get("발주기관", "")).strip()
        if 기관 and 기관.lower() != "nan":
            golden.append({
                "type": "metadata_extraction",
                "question": f"'{사업명}' 사업의 발주기관(대상 기관)은 어디인가?",
                "answer": 기관,
                "source_files": [filename],
            })

        amount_text = _amount_to_text(row.get("사업금액"))
        if amount_text:
            golden.append({
                "type": "metadata_extraction",
                "question": f"'{사업명}' 사업의 사업금액(예산)은 얼마인가?",
                "answer": amount_text,
                "source_files": [filename],
            })

        # 마감일은 loader가 헤더에 넣는 형식(YYYY-MM-DD)과 맞춰 정규화한다.
        ts = pd.to_datetime(row.get("입찰참여마감일"), errors="coerce")
        if not pd.isna(ts):
            golden.append({
                "type": "metadata_extraction",
                "question": f"'{사업명}' 사업의 입찰참여 마감일은 언제인가?",
                "answer": ts.date().isoformat(),
                "source_files": [filename],
            })

    # 2) 본문 서술형(judge 채점): 핵심 요구조건 / 제출 방식 / 평가 배점
    #    answer는 본문을 보고 직접 채워 넣어야 judge_llm 채점이 의미를 가진다.
    for _, row in df.head(body_n).iterrows():
        filename = str(row["파일명"])
        사업명 = str(row.get("사업명", "")).strip()
        for _, q_fmt in BODY_AXES:
            golden.append({
                "type": "body_grounded",
                "question": q_fmt.format(사업명=사업명),
                "answer": "",
                "source_files": [filename],
            })

    # 3) 거절 평가
    for q in UNANSWERABLE_TEMPLATES:
        golden.append({
            "type": "unanswerable",
            "question": q,
            "answer": None,
            "source_files": [],
        })

    return golden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12,
                        help="메타데이터 추출 질문에 쓸 RFP 행 수(기관/예산/마감일)")
    parser.add_argument("--body-n", type=int, default=6,
                        help="본문 서술형 질문에 쓸 RFP 행 수(요구조건/제출방식/평가배점)")
    args = parser.parse_args()

    golden = build(args.n, args.body_n)
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    from collections import Counter
    counts = Counter(g["type"] for g in golden)
    print(f"골든셋 {len(golden)}건을 {OUT_PATH}에 저장했습니다. 유형별: {dict(counts)}")
    print("body_grounded 항목의 answer는 본문을 보고 직접 채워 넣으세요(judge_llm 채점용).")


if __name__ == "__main__":
    main()
