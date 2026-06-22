"""청킹 전략.

베이스라인은 RecursiveCharacterTextSplitter를 사용한다.
각 청크 앞에 메타데이터 헤더("[기관 | 사업금액 | 마감일]")를 붙여,
발주기관명이 본문에 없는 문서(약 32건)에 대응한다.
섹션 인식 청킹은 함수 분기로 자리만 만들어 둔다.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _header(meta: Dict[str, Any]) -> str:
    """청크 본문 앞에 붙일 메타데이터 헤더 문자열을 만든다."""
    기관 = meta.get("기관") or "미상"
    금액 = meta.get("금액")
    금액_str = f"{금액:,}원" if isinstance(금액, int) else "미기재"
    마감일 = meta.get("마감일") or "미상"
    return f"[기관: {기관} | 사업금액: {금액_str} | 마감일: {마감일}]"


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
            # 본문이 비어 있는 문서(파싱 실패/스캔본)는 청크를 만들지 않는다.
            continue
        # split_documents는 원본 metadata(파일명 포함)를 그대로 상속한다.
        for piece in splitter.split_documents([doc]):
            header = _header(piece.metadata)
            piece.page_content = f"{header}\n{piece.page_content}"
            out.append(piece)
    return out


def _section_split(docs: List[Document], cfg: Dict[str, Any]) -> List[Document]:
    """섹션 인식 청킹 자리. 베이스라인에서는 recursive로 위임한다."""
    # TODO: 헤딩/번호 패턴 기반 섹션 분할 구현 예정.
    return _recursive_split(docs, cfg)


def split_documents(docs: List[Document], cfg: Dict[str, Any]) -> List[Document]:
    """config의 chunk.strategy에 따라 청킹한다."""
    strategy = cfg["chunk"].get("strategy", "recursive")
    if strategy == "recursive":
        return _recursive_split(docs, cfg)
    if strategy == "section":
        return _section_split(docs, cfg)
    raise ValueError(f"지원하지 않는 chunk strategy: {strategy}")
