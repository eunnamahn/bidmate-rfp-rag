"""교체 가능한 부품(LLM/임베딩/청킹)의 프리셋 카탈로그.

새 모델이나 전략을 추가할 때는 이 파일의 해당 딕셔너리에 항목 하나만 추가하면
백엔드 /config와 Streamlit 드롭다운에 자동으로 나타난다. 추가할 곳은 여기 한 곳뿐이다.

각 프리셋 규칙:
  - key(딕셔너리 키): 프론트와 주고받는 내부 식별자. 영문/숫자/하이픈 권장.
  - label : 사용자에게 보이는 이름.
  - reindex: 이 프리셋으로 바꾸면 FAISS 재인덱싱(임베딩 재계산)이 필요한가.
             LLM 교체는 False(즉시), 임베딩/청킹 교체는 True(캐시 없으면 시간 소요).
  - 나머지 키: cfg의 해당 섹션(llm/embedding/chunk)에 그대로 덮어쓸 설정값.

주의: provider 값(openai/huggingface/ollama)은 src/config.py의 팩토리가 아는 값이어야
한다. 완전히 새로운 provider를 추가할 때만 config.py도 함께 손본다.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


LLM_PRESETS: Dict[str, Dict[str, Any]] = {
    "openai-gpt-5-nano": {
        "label": "GPT-5 nano (OpenAI)",
        "provider": "openai",
        "model": "gpt-5-nano",
        "reindex": False,
    },
    "openai-gpt-5-mini": {
        "label": "GPT-5 mini (OpenAI)",
        "provider": "openai",
        "model": "gpt-5-mini",
        "reindex": False,
    },
    "ollama-exaone3.5-7.8b": {
        "label": "EXAONE 3.5 7.8B (Ollama VM) (추천)",
        "provider": "ollama",
        "model": "exaone3.5:7.8b",
        "num_ctx": 8192,
        "reindex": False,
    },
    "ollama-qwen2.5-14b": {
        "label": "Qwen2.5 14B (Ollama VM)",
        "provider": "ollama",
        "model": "qwen2.5:14b",
        "reindex": False,
    },
}

EMBEDDING_PRESETS: Dict[str, Dict[str, Any]] = {
    "openai-3-small": {
        "label": "OpenAI text-embedding-3-small",
        "provider": "openai",
        "model": "text-embedding-3-small",
        "reindex": True,
    },
    "hf-kure-v1": {
        "label": "KURE-v1 (HuggingFace, 로컬) (추천)",
        "provider": "huggingface",
        "model": "nlpai-lab/KURE-v1",
        "reindex": True,
    },
    "hf-bge-m3": {
        "label": "BGE-M3 (HuggingFace, 로컬)",
        "provider": "huggingface",
        "model": "BAAI/bge-m3",
        # main 리비전은 .bin뿐이라 torch<2.6에서 로딩이 막힌다. safetensors가 있는
        # 리비전으로 고정한다(가중치 동일 -> 캐시 키/인덱스 호환).
        "revision": "9a0624b896d81da7492a910ffa53731274b6cf3d",
        "reindex": True,
    },
}

CHUNK_PRESETS: Dict[str, Dict[str, Any]] = {
    "recursive-500-50": {
        "label": "Recursive 500/50",
        "strategy": "recursive",
        "size": 500,
        "overlap": 50,
        "reindex": True,
    },
    "recursive-1000-100": {
        "label": "Recursive 1000/100 (큰 청크)",
        "strategy": "recursive",
        "size": 1000,
        "overlap": 100,
        "reindex": True,
    },
    "recursive-2000-200": {
        "label": "Recursive 2000/200 (추천)",
        "strategy": "recursive",
        "size": 2000,
        "overlap": 200,
        "reindex": True,
    },
    "recursive-300-30": {
        "label": "Recursive 300/30 (작은 청크)",
        "strategy": "recursive",
        "size": 300,
        "overlap": 30,
        "reindex": True,
    },
    "section-500-50": {
        "label": "Section 키워드 청킹 500/50 (EDA 섹션 단위)",
        "strategy": "section",
        "size": 500,
        "overlap": 50,
        "reindex": True,
    },
    "section-1000-100": {
        "label": "Section 키워드 청킹 1000/100 (긴 상한)",
        "strategy": "section",
        "size": 1000,
        "overlap": 100,
        "reindex": True,
    },
}

# HWP/PDF 파서는 cfg["parser"]의 hwp/pdf 키 하나씩만 덮어쓴다(서로 독립).
# key를 파서 이름과 동일하게 두어 cfg 값과 일치시킨다(src/loader.py 레지스트리와 동일).
PARSER_HWP_PRESETS: Dict[str, Dict[str, Any]] = {
    "pyhwp": {"label": "pyhwp (텍스트, 빠름)", "hwp": "pyhwp", "reindex": True},
    "hwp-hwpx-parser": {
        "label": "hwp-hwpx-parser (표 마크다운 보존)",
        "hwp": "hwp-hwpx-parser", "reindex": True,
    },
    "olefile": {"label": "olefile (OLE 직접파싱, 텍스트) (추천)", "hwp": "olefile", "reindex": True},
}

PARSER_PDF_PRESETS: Dict[str, Dict[str, Any]] = {
    "pymupdf": {"label": "PyMuPDF (빠름/안정)", "pdf": "pymupdf", "reindex": True},
    "pdfplumber": {"label": "pdfplumber (표 정밀) (추천)", "pdf": "pdfplumber", "reindex": True},
}

# 프리셋 종류 -> (cfg 섹션 이름, 카탈로그, 현재 프리셋 판별에 쓸 설정 키들)
_SECTIONS: Dict[str, tuple] = {
    "llm": ("llm", LLM_PRESETS, ("provider", "model")),
    "embedding": ("embedding", EMBEDDING_PRESETS, ("provider", "model")),
    "chunk": ("chunk", CHUNK_PRESETS, ("strategy", "size", "overlap")),
    "parser_hwp": ("parser", PARSER_HWP_PRESETS, ("hwp",)),
    "parser_pdf": ("parser", PARSER_PDF_PRESETS, ("pdf",)),
}

# cfg에 덮어쓸 때 제외할 메타 키
_META_KEYS = {"label", "reindex"}


def _settings_of(preset: Dict[str, Any]) -> Dict[str, Any]:
    """프리셋에서 메타 키를 뺀 실제 설정값만 추린다."""
    return {k: v for k, v in preset.items() if k not in _META_KEYS}


def apply_preset(cfg: Dict[str, Any], kind: str, key: str) -> Dict[str, Any]:
    """cfg 사본에 kind(llm/embedding/chunk) 프리셋을 적용해 반환한다."""
    if kind not in _SECTIONS:
        raise ValueError(f"알 수 없는 프리셋 종류: {kind}")
    section, presets, _ = _SECTIONS[kind]
    if key not in presets:
        raise ValueError(f"알 수 없는 {kind} 프리셋: {key}")
    cfg = copy.deepcopy(cfg)
    cfg.setdefault(section, {})
    cfg[section].update(_settings_of(presets[key]))
    return cfg


def current_key(cfg: Dict[str, Any], kind: str) -> Optional[str]:
    """현재 cfg가 어떤 프리셋과 일치하는지 key를 찾는다(없으면 None)."""
    section, presets, fields = _SECTIONS[kind]
    cur = cfg.get(section, {}) or {}
    for key, preset in presets.items():
        if all(cur.get(f) == preset.get(f) for f in fields):
            return key
    return None


def needs_reindex(kind: str, key: str) -> bool:
    """해당 프리셋이 재인덱싱을 요구하는지 반환한다."""
    _, presets, _ = _SECTIONS[kind]
    return bool(presets.get(key, {}).get("reindex", False))


def _entries(presets: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"key": k, "label": p["label"], "reindex": p.get("reindex", False)}
        for k, p in presets.items()
    ]


def frontend_catalog(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """프론트가 드롭다운을 그릴 때 쓸 프리셋 목록 + 현재 선택값."""
    return {
        kind: {
            "presets": _entries(_SECTIONS[kind][1]),
            "current": current_key(cfg, kind),
        }
        for kind in _SECTIONS
    }
