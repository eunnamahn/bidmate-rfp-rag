"""청킹 전략.

베이스라인은 RecursiveCharacterTextSplitter를 사용한다.
각 청크 앞에 메타데이터 헤더("[기관 | 사업금액 | 마감일]")를 붙여,
발주기관명이 본문에 없는 문서(약 32건)에 대응한다.
섹션 인식 청킹은 EDA에서 발견한 공통 섹션 키워드(평가기준, 입찰참가자격 등) 기반으로
의미 단위 분할을 수행한다. 섹션을 못 찾으면 recursive로 자동 폴백한다.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ══════════════════════════════════════════════════════
# 섹션 인식 청킹용 상수
# EDA 결과 기반 — 100개 문서에 공통으로 등장하는 섹션 키워드
# (평가기준 91%, 입찰참가자격 90%, 사업기간 91% 등)
# ══════════════════════════════════════════════════════

_SECTION_PATTERNS = [
    ("사업개요",     re.compile(r"사업\s*(의\s*)?개요")),
    ("추진배경",     re.compile(r"추진\s*배경")),
    ("기대효과",     re.compile(r"기대\s*효과")),
    ("사업기간",     re.compile(r"사업\s*기간")),
    ("사업예산",     re.compile(r"사업\s*예산")),
    ("과업내용",     re.compile(r"과업\s*내용")),
    ("추진일정",     re.compile(r"추진\s*일정")),
    ("제출기한",     re.compile(r"제출\s*기한")),
    ("제출방법",     re.compile(r"제출\s*방법")),
    ("평가기준",     re.compile(r"평가\s*기준")),
    ("입찰참가자격", re.compile(r"입찰\s*참가\s*자격")),
    ("계약기간",     re.compile(r"계약\s*기간")),
    ("납품",         re.compile(r"납품")),
]

# 목차 줄 패턴 (쪽번호 달린 줄은 헤더에서 제외)
_TOC_TRAILER = re.compile(r"[·.\s]{2,}\d+\s*$")
_MAX_HEADER_LINE_LEN = 30


def _header(meta: Dict[str, Any]) -> str:
    """청크 본문 앞에 붙일 메타데이터 헤더 문자열을 만든다."""
    기관 = meta.get("기관") or "미상"
    금액 = meta.get("금액")
    금액_str = f"{금액:,}원" if isinstance(금액, int) else "미기재"
    마감일 = meta.get("마감일") or "미상"
    return f"[기관: {기관} | 사업금액: {금액_str} | 마감일: {마감일}]"


def _find_section_boundaries(text: str) -> list:
    """EDA 섹션 키워드가 있는 헤더 줄 위치를 찾는다.

    목차 줄(쪽번호 달린 줄)과 너무 긴 줄(본문 문장)은 제외한다.
    같은 키워드가 여러 번 나오면 첫 번째 등장 위치를 채택한다.
    """
    lines_with_pos = []
    pos = 0
    for line in text.split("\n"):
        lines_with_pos.append((pos, line))
        pos += len(line) + 1

    candidates = {}
    for line_start, line in lines_with_pos:
        stripped = line.strip()
        if not stripped or _TOC_TRAILER.search(stripped):
            continue
        if len(stripped) > _MAX_HEADER_LINE_LEN:
            continue
        for name, pattern in _SECTION_PATTERNS:
            if pattern.search(line):
                candidates.setdefault(name, []).append((line_start, 0))

    boundaries = []
    for name, _ in _SECTION_PATTERNS:
        cands = candidates.get(name)
        if cands:
            boundaries.append((cands[0][0], name))

    boundaries.sort(key=lambda x: x[0])
    return boundaries


def _recursive_split(docs: List[Document], cfg: Dict[str, Any]) -> List[Document]:
    """RecursiveCharacterTextSplitter로 청킹하고 헤더를 덧붙인다."""
    chunk_cfg = cfg["chunk"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg["size"],
        chunk_overlap=chunk_cfg["overlap"],
    )
    out: List[Document] = []
    for doc in docs:
        if not doc.page_content.strip():
            continue
        for piece in splitter.split_documents([doc]):
            header = _header(piece.metadata)
            piece.page_content = f"{header}\n{piece.page_content}"
            out.append(piece)
    return out


def _section_split(docs: List[Document], cfg: Dict[str, Any]) -> List[Document]:
    """EDA 기반 키워드 섹션 청킹.

    섹션 키워드를 못 찾으면 recursive로 폴백.
    섹션이 chunk_size보다 길면 RecursiveCharacterTextSplitter로 추가 분할.
    각 청크에 섹션명을 metadata로 추가해 나중에 섹션 필터링에 활용 가능.
    """
    chunk_cfg = cfg["chunk"]
    chunk_size = chunk_cfg["size"]
    chunk_overlap = chunk_cfg["overlap"]
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    out: List[Document] = []

    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue

        boundaries = _find_section_boundaries(text)
        header = _header(doc.metadata)

        if not boundaries:
            # 섹션 키워드를 하나도 못 찾은 문서 → recursive 폴백
            for piece in fallback.split_documents([doc]):
                piece.page_content = f"{header}\n{piece.page_content}"
                out.append(piece)
            continue

        # 첫 섹션 이전 서두 처리
        if boundaries[0][0] > 0:
            head = text[:boundaries[0][0]].strip()
            if head:
                out.append(Document(
                    page_content=f"{header}\n{head}",
                    metadata={**doc.metadata, "section": "서두"},
                ))

        # 섹션별 처리 — 200자 미만이면 다음 섹션과 병합
        MIN_SECTION_LEN = 200
        merged_sections = []
        buffer_text = ""
        buffer_name = ""

        for idx, (start, name) in enumerate(boundaries):
            end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue

            if buffer_text:
                buffer_text += "\n" + section_text
            else:
                buffer_text = section_text
                buffer_name = name

            if len(buffer_text) >= MIN_SECTION_LEN:
                merged_sections.append((buffer_text, buffer_name))
                buffer_text = ""
                buffer_name = ""

        # 마지막 버퍼에 남은 것 처리
        if buffer_text:
            if merged_sections:
                # 앞 섹션에 붙이기
                prev_text, prev_name = merged_sections[-1]
                merged_sections[-1] = (prev_text + "\n" + buffer_text, prev_name)
            else:
                merged_sections.append((buffer_text, buffer_name))

        # 병합된 섹션 청킹
        for section_text, name in merged_sections:
            if len(section_text) <= chunk_size:
                out.append(Document(
                    page_content=f"{header}\n{section_text}",
                    metadata={**doc.metadata, "section": name},
                ))
            else:
                tmp = Document(page_content=section_text, metadata=doc.metadata)
                for piece in fallback.split_documents([tmp]):
                    piece.page_content = f"{header}\n{piece.page_content}"
                    piece.metadata["section"] = name
                    out.append(piece)

    return out


def split_documents(docs: List[Document], cfg: Dict[str, Any]) -> List[Document]:
    """config의 chunk.strategy에 따라 청킹한다."""
    strategy = cfg["chunk"].get("strategy", "recursive")
    if strategy == "recursive":
        return _recursive_split(docs, cfg)
    if strategy == "section":
        return _section_split(docs, cfg)
    raise ValueError(f"지원하지 않는 chunk strategy: {strategy}")
