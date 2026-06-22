"""설정 로드 및 부품 팩토리.

config.yaml과 .env를 읽어 임베딩/LLM 객체를 생성한다.
교체 축(provider)을 config 한 줄 수정으로 스위치할 수 있도록 팩토리로 분리한다.
API 키는 코드에 하드코딩하지 않고 .env(OPENAI_API_KEY)에서만 읽는다.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# .env에서 환경 변수 로드 (OPENAI_API_KEY 등)
load_dotenv()


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """config.yaml을 읽어 dict로 반환한다."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_openai_key() -> str:
    """OPENAI_API_KEY가 .env에 있는지 확인하고 반환한다."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            ".env에 OPENAI_API_KEY가 없습니다. .env 파일을 만들고 키를 설정하세요."
        )
    return key


def make_embedding(cfg: Dict[str, Any]):
    """config의 embedding.provider에 따라 임베딩 객체를 생성한다.

    openai      -> OpenAIEmbeddings
    huggingface -> HuggingFaceEmbeddings (예: nlpai-lab/KURE-v1, BAAI/bge-m3)
    """
    emb = cfg["embedding"]
    provider = emb["provider"]
    if provider == "openai":
        _require_openai_key()
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=emb["model"])
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=emb["model"])
    raise ValueError(f"지원하지 않는 embedding provider: {provider}")


def make_llm(cfg: Dict[str, Any]):
    """config의 llm.provider에 따라 LLM 객체를 생성한다.

    openai -> ChatOpenAI (예: gpt-4o-mini)
    ollama -> ChatOllama (예: qwen3:8b)
    """
    llm = cfg["llm"]
    provider = llm["provider"]
    temperature = llm.get("temperature", 0)
    base_url = llm.get("base_url") or None
    if provider == "openai":
        _require_openai_key()
        from langchain_openai import ChatOpenAI

        kwargs = {"model": llm["model"], "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # base_url로 원격 Ollama(예: GCP VM의 http://<ip>:11434)를 가리킬 수 있다.
        kwargs = {"model": llm["model"], "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)
    raise ValueError(f"지원하지 않는 llm provider: {provider}")
