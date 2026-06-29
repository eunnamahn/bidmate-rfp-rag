"""RAG 답변 수동 테스트용 스크립트 (골든셋 없이).

파이프라인을 한 번 빌드한 뒤 질문을 받아 답변과 검색된 출처를 보여준다.
벡터스토어는 캐시되므로 첫 실행만 느리고 이후엔 빠르다.

사용:
  python scripts/dev/ask.py                  # 대화형(REPL). 빈 줄 또는 'q' 입력 시 종료
  python scripts/dev/ask.py "평가 배점 기준은?"   # 일회성 질문
  python scripts/dev/ask.py --mode summary "이 사업 요약해줘"
"""

from __future__ import annotations

import sys
from pathlib import Path

# 저장소 루트를 import 경로에 추가한다(scripts 하위에서 from src... 동작).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from typing import Any, List

from src.config import load_config, make_llm
from src.generator import build_chain
from src.pipeline import build_components
from src.retriever import build_retriever


def _show_sources(retriever, question: str, k: int) -> None:
    """검색된 청크의 출처(파일명/기관)와 앞부분을 보여준다."""
    try:
        docs = retriever.invoke(question)
    except AttributeError:
        docs = retriever.get_relevant_documents(question)
    print(f"\n  [검색된 출처 {len(docs[:k])}건]")
    for i, d in enumerate(docs[:k], 1):
        파일명 = d.metadata.get("파일명", "?")
        기관 = d.metadata.get("기관") or "미상"
        snippet = d.page_content.replace("\n", " ")[:120]
        print(f"   {i}. {파일명} ({기관})")
        print(f"      {snippet}...")


def _answer(chain, retriever, question: str, k: int) -> None:
    answer = chain.invoke(question)
    print("\n  [답변]")
    print("  " + answer.replace("\n", "\n  "))
    _show_sources(retriever, question, k)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="질문(생략 시 대화형)")
    parser.add_argument("--mode", choices=["qa", "summary"], default="qa")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    k = cfg["retriever"]["k"]

    print("파이프라인 빌드 중 (첫 실행은 원본 파싱/임베딩으로 시간이 걸립니다)...",
          flush=True)
    docs, chunks, vectorstore = build_components(cfg)
    parsed = sum(1 for d in docs if d.page_content.strip())
    print(f"문서 {len(docs)}건 중 본문 확보 {parsed}건, 청크 {len(chunks)}개 준비 완료.",
          flush=True)

    retriever = build_retriever(vectorstore, chunks, cfg)
    llm = make_llm(cfg)
    chain = build_chain(retriever, llm, mode=args.mode)

    if args.question:
        question = " ".join(args.question)
        print(f"\n질문: {question}")
        _answer(chain, retriever, question, k)
        return

    print("\n대화형 모드. 질문을 입력하세요. (종료: 빈 줄 또는 q)\n")
    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() == "q":
            break
        _answer(chain, retriever, question, k)


if __name__ == "__main__":
    main()
