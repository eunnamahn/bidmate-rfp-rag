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
    huggingface -> HuggingFaceEmbeddings (로컬 in-process, 예: nlpai-lab/KURE-v1, BAAI/bge-m3)
    ollama      -> OllamaEmbeddings (원격/로컬 Ollama 서버, 예: bge-m3). LLM과 동일하게
                   base_url(config) 또는 OLLAMA_BASE_URL(.env)로 VM 주소를 지정한다.
    """
    emb = cfg["embedding"]
    provider = emb["provider"]
    if provider == "openai":
        _require_openai_key()
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=emb["model"])
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        # make_llm과 동일한 규칙: config.embedding.base_url 우선, 없으면 .env의 OLLAMA_BASE_URL,
        # 둘 다 없으면 로컬(11434). 같은 VM의 Ollama가 LLM과 임베딩을 함께 서빙할 수 있다.
        base_url = emb.get("base_url") or os.getenv("OLLAMA_BASE_URL")
        kwargs = {"model": emb["model"]}
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaEmbeddings(**kwargs)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        # CUDA GPU가 있으면 GPU로 임베딩한다(없으면 CPU). RTX 등 로컬 GPU 활용.
        model_kwargs = {}
        try:
            import torch

            if torch.cuda.is_available():
                model_kwargs["device"] = "cuda"
        except Exception:
            pass
        # 특정 리비전 고정(예: bge-m3는 .bin만 있는 main 대신 safetensors 리비전 사용
        # — torch<2.6 + transformers의 CVE 체크로 .bin 로딩이 막히는 문제 회피).
        # revision은 모델 가중치를 바꾸지 않으므로 캐시 키에는 넣지 않는다.
        if emb.get("revision"):
            model_kwargs["revision"] = emb["revision"]
        return HuggingFaceEmbeddings(model_name=emb["model"], model_kwargs=model_kwargs)
    raise ValueError(f"지원하지 않는 embedding provider: {provider}")


def make_llm(cfg: Dict[str, Any]):
    """config의 llm.provider에 따라 LLM 객체를 생성한다.

    openai -> ChatOpenAI (예: gpt-5-mini)
    ollama -> ChatOllama (예: qwen3:8b). 원격 Ollama(GCP VM) 주소는 base_url로 지정한다.

    base_url은 ollama 전용이다. 원격 Ollama 주소는 config의 llm.base_url을 우선 쓰고,
    비어 있으면 .env의 OLLAMA_BASE_URL을 쓴다(둘 다 없으면 로컬 11434). openai에는
    base_url을 넘기지 않는다(VM 주소가 남아 있어도 openai 호출이 깨지지 않게).
    """
    llm = cfg["llm"]
    provider = llm["provider"]
    temperature = llm.get("temperature", 0)
    if provider == "openai":
        _require_openai_key()
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=llm["model"], temperature=temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = llm.get("base_url") or os.getenv("OLLAMA_BASE_URL")
        kwargs = {"model": llm["model"], "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)
    raise ValueError(f"지원하지 않는 llm provider: {provider}")
