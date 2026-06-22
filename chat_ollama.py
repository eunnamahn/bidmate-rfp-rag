"""VM Ollama 모델과 직접 대화하는 테스트 스크립트 (RAG 없이 LLM만).

터널(.env의 OLLAMA_BASE_URL)을 통해 VM의 Ollama 모델에 바로 질문하고 응답을 본다.
RAG 파이프라인을 거치지 않으므로 모델 자체의 응답/속도를 빠르게 확인할 때 쓴다.

사용:
  python chat_ollama.py                         # 대화형(REPL), 기본 exaone3.5:7.8b
  python chat_ollama.py --model qwen2.5:14b     # 모델 지정
  python chat_ollama.py "RFP가 뭐야?"            # 일회성 질문
  python chat_ollama.py --list                  # VM에 있는 모델 목록
"""

from __future__ import annotations

import argparse
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"


def list_models() -> None:
    """VM Ollama에 받아져 있는 모델 목록을 출력한다."""
    try:
        data = requests.get(f"{BASE_URL}/api/tags", timeout=10).json()
    except Exception as e:
        print(f"연결 실패({BASE_URL}): {e}")
        print("SSH 터널이 열려 있는지, OLLAMA_BASE_URL이 맞는지 확인하세요.")
        return
    print(f"[{BASE_URL}] 사용 가능 모델:")
    for m in data.get("models", []):
        size_gb = m.get("size", 0) / 1e9
        print(f"  - {m['name']} ({size_gb:.1f}GB)")


def ask(model: str, question: str) -> None:
    """모델에 한 번 질문하고 응답과 소요 시간을 출력한다."""
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=model, temperature=0, base_url=BASE_URL)
    t0 = time.time()
    resp = llm.invoke(question)
    print(getattr(resp, "content", resp))
    print(f"\n(모델 {model} · {time.time() - t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="질문(생략 시 대화형)")
    parser.add_argument("--model", default="exaone3.5:7.8b", help="Ollama 모델 태그")
    parser.add_argument("--list", action="store_true", help="VM 모델 목록만 출력")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    print(f"연결: {BASE_URL} | 모델: {args.model}")

    if args.question:
        ask(args.model, " ".join(args.question))
        return

    print("대화형 모드. 질문을 입력하세요. (종료: 빈 줄 또는 q)\n")
    while True:
        try:
            q = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() == "q":
            break
        ask(args.model, q)
        print()


if __name__ == "__main__":
    main()
