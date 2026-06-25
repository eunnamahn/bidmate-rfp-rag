"""원본 파일 파싱 + 전처리 + Document 변환.

핵심 원칙: 본문 소스로 메타데이터의 '텍스트' 컬럼을 쓰지 않고 원본 파일을 직접 재파싱한다.
('텍스트' 컬럼은 원본의 앞부분 발췌본이라 분량이 부족하다.)
파서는 config.yaml의 parser.hwp / parser.pdf로 교체할 수 있다(교체 축).
기본값은 안정/속도 우선으로 HWP=pyhwp, PDF=pymupdf이며,
표 마크다운 정밀 보존이 필요하면 hwp-hwpx-parser / pdfplumber로 바꾼다.
"""

from __future__ import annotations

import numbers
import re
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.documents import Document

# 파일당 파싱 타임아웃(초). 초과 시 해당 문서는 fallback 처리한다.
PARSE_TIMEOUT = 60
# 추출 글자수가 이 값 미만이면 스캔본/추출실패 가능성으로 표시한다.
MIN_CHARS = 1000

# 제어문자 제거용 (탭/개행은 보존)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 깨진 특수문자: 외톨이 서로게이트(U+D800~U+DFFF), 유니코드 치환문자(U+FFFD),
# 사용자정의영역(U+E000~U+F8FF). 서로게이트는 HF 임베딩 토크나이저를 깨뜨리므로 제거.
_BROKEN = re.compile("[\ud800-\udfff\ufffd\ue000-\uf8ff]")
# HWP 파싱 아티팩트: <표>, <그림>, <표 ...>, </그림> 등 태그(EDA: 문서당 평균 2,436자)
_HWP_TAG = re.compile(r"</?(?:표|그림)[^>]*>")
# 목차로 보이는 줄: '항목 ........ 12' 형태(점/공백 다수 + 끝 페이지번호)
_TOC_LINE = re.compile(r"^.{0,80}?[.·…‧\s]{4,}\d{1,4}\s*$", re.MULTILINE)


# =====================================================================
# 파서 구현부 (교체 축)
# 새 파서를 추가하려면 함수를 만들고 아래 HWP_PARSERS / PDF_PARSERS에 등록한다.
# config.yaml의 parser.hwp / parser.pdf 값으로 선택된다.
# =====================================================================

def _parse_hwp_hwpx_parser(path: Path) -> str:
    """HWP/HWPX 파서: hwp-hwpx-parser. 본문 + 표(마크다운)를 보존한다."""
    from hwp_hwpx_parser import Reader

    with Reader(str(path)) as r:
        body = r.text or ""
        # get_tables_as_markdown()는 표별 마크다운 문자열의 리스트를 반환한다.
        tables = r.get_tables_as_markdown() or []
    tables_md = "\n\n".join(tables)
    # 평가배점·요구사항·제출서류 등 핵심 정보가 표에 있으므로 마크다운으로 보존한다.
    return (body + "\n\n" + tables_md).strip()


def _parse_hwp_pyhwp(path: Path) -> str:
    """HWP 파서(대안): pyhwp의 hwp5txt. 텍스트만 추출(표 구조 보존 약함, 빠름).

    설치: pip install pyhwp
    """
    import subprocess

    out = subprocess.run(
        ["hwp5txt", str(path)], capture_output=True, timeout=PARSE_TIMEOUT
    )
    return out.stdout.decode("utf-8", "replace").strip()


def _parse_hwp_olefile(path: Path) -> str:
    """HWP 5.0 파서: olefile로 OLE 스트림을 직접 읽어 본문을 추출한다.

    BodyText/SectionN 스트림을 (FileHeader 플래그가 압축이면 zlib raw deflate로)
    해제한 뒤, HWPTAG_PARA_TEXT 레코드의 UTF-16LE 텍스트만 모은다. 인라인
    제어문자는 폭(1 또는 8 wchar)에 맞춰 건너뛴다. HWPX(zip)는 대상이 아니다.
    의존성 없이 가벼우나 표 구조는 보존하지 못한다(텍스트만).
    설치: pip install olefile
    """
    import olefile

    ole = olefile.OleFileIO(str(path))
    try:
        if not ole.exists("FileHeader"):
            raise ValueError("HWP 5.0 형식이 아님(FileHeader 스트림 없음)")
        is_compressed = bool(ole.openstream("FileHeader").read()[36] & 0x01)

        # BodyText/SectionN 스트림을 번호순으로 모은다.
        nums = [
            int(e[1][len("Section"):])
            for e in ole.listdir()
            if len(e) == 2 and e[0] == "BodyText" and e[1].startswith("Section")
        ]
        parts: List[str] = []
        for n in sorted(nums):
            raw = ole.openstream(f"BodyText/Section{n}").read()
            data = zlib.decompress(raw, -15) if is_compressed else raw
            parts.append(_hwp_section_text(data))
        return "\n".join(parts).strip()
    finally:
        ole.close()


# HWPTAG_PARA_TEXT = HWPTAG_BEGIN(0x010) + 50 = 67
_HWPTAG_PARA_TEXT = 67
# 인라인 제어문자 중 추가 파라미터까지 총 8 wchar(16바이트)를 차지하는 코드들
_HWP_EXT_CTRL = frozenset({1, 2, 3, 4, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23})


def _hwp_section_text(data: bytes) -> str:
    """해제된 BodyText 섹션을 레코드 단위로 훑어 문단 텍스트만 추출한다."""
    out: List[str] = []
    i, size = 0, len(data)
    while i + 4 <= size:
        rec = struct.unpack_from("<I", data, i)[0]
        tag = rec & 0x3ff
        length = (rec >> 20) & 0xfff
        i += 4
        if length == 0xfff:  # 확장 크기: 다음 4바이트가 실제 길이
            length = struct.unpack_from("<I", data, i)[0]
            i += 4
        if tag == _HWPTAG_PARA_TEXT:
            out.append(_hwp_para_text(data[i:i + length]))
        i += length
    return "\n".join(out)


def _hwp_para_text(buf: bytes) -> str:
    """PARA_TEXT 레코드(UTF-16LE + 인라인 제어문자)에서 글자만 뽑는다.

    텍스트 코드유닛은 원본 2바이트를 모아 마지막에 UTF-16LE로 한 번에 디코드한다.
    이렇게 해야 BMP 밖 문자의 서로게이트 쌍이 올바른 코드포인트로 합쳐진다.
    (코드유닛마다 chr()를 호출하면 외톨이 서로게이트가 남아, HF 임베딩 토크나이저가
     'TextEncodeInput' 오류로 깨진다.)
    """
    out = bytearray()
    j, n = 0, len(buf)
    while j + 2 <= n:
        code = struct.unpack_from("<H", buf, j)[0]
        if code in _HWP_EXT_CTRL:
            j += 16  # 제어문자 + 파라미터(총 8 wchar) 건너뛰기
        elif code < 32:
            if code in (10, 13):  # 줄/문단 바꿈만 개행으로 보존
                out += b"\n\x00"
            j += 2
        else:
            out += buf[j:j + 2]  # 원본 2바이트 유지(서로게이트 쌍 보존)
            j += 2
    return out.decode("utf-16-le", errors="ignore")


def _parse_pdf_pdfplumber(path: Path) -> str:
    """PDF 파서: pdfplumber. 페이지 텍스트 + 표(마크다운)를 추출한다."""
    import pdfplumber

    parts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
            for table in page.extract_tables():
                md = _table_to_markdown(table)
                if md:
                    parts.append(md)
    return "\n".join(parts).strip()


def _parse_pdf_pymupdf(path: Path) -> str:
    """PDF 파서: PyMuPDF(fitz). 텍스트가 매우 빠르고 대형/복잡 PDF에도 안정적.

    표는 find_tables().to_markdown()으로 마크다운 보존을 시도한다.
    설치: pip install pymupdf
    """
    import fitz

    parts: List[str] = []
    doc = fitz.open(str(path))
    try:
        for page in doc:
            parts.append(page.get_text() or "")
            try:
                for table in page.find_tables().tables:
                    md = table.to_markdown()
                    if md:
                        parts.append(md)
            except Exception:
                # 특정 페이지 표 탐지 실패는 무시하고 본문은 유지한다.
                pass
    finally:
        doc.close()
    return "\n".join(parts).strip()


# 파서 레지스트리: 이름 -> 추출 함수
HWP_PARSERS = {
    "hwp-hwpx-parser": _parse_hwp_hwpx_parser,
    "pyhwp": _parse_hwp_pyhwp,
    "olefile": _parse_hwp_olefile,
}
PDF_PARSERS = {
    "pdfplumber": _parse_pdf_pdfplumber,
    "pymupdf": _parse_pdf_pymupdf,
}


def make_hwp_parser(name: str):
    """이름으로 HWP 파서 함수를 반환한다."""
    if name not in HWP_PARSERS:
        raise ValueError(
            f"지원하지 않는 hwp 파서: {name} (가능: {list(HWP_PARSERS)})"
        )
    return HWP_PARSERS[name]


def make_pdf_parser(name: str):
    """이름으로 PDF 파서 함수를 반환한다."""
    if name not in PDF_PARSERS:
        raise ValueError(
            f"지원하지 않는 pdf 파서: {name} (가능: {list(PDF_PARSERS)})"
        )
    return PDF_PARSERS[name]


def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """pdfplumber가 추출한 2차원 표를 마크다운 표로 변환한다."""
    rows = [["" if c is None else str(c).replace("\n", " ") for c in row]
            for row in table if row]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    header = rows[0] + [""] * (width - len(rows[0]))
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in range(width)) + " |"]
    for row in rows[1:]:
        cells = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _parse_with_timeout(func, path: Path, timeout: int) -> str:
    """파일 파싱을 타임아웃과 함께 실행한다.

    윈도우에서는 signal 기반 타임아웃을 쓸 수 없어 스레드로 처리한다.
    타임아웃 시 작업 스레드는 백그라운드에 남지만 호출자는 즉시 예외를 받는다.
    (context manager를 쓰면 종료 시 멈춘 스레드를 기다려 타임아웃이 무력화되므로
     shutdown(wait=False)로 즉시 반환한다.)
    """
    ex = ThreadPoolExecutor(max_workers=1)
    future = ex.submit(func, path)
    try:
        return future.result(timeout=timeout)
    finally:
        ex.shutdown(wait=False)


def _clean(text: str) -> str:
    """줄바꿈 통일 + 제어문자·깨진 특수문자·HWP 태그·목차 블록 제거 + 공백 정리."""
    # \r\n, \r를 \n으로 통일(HWP 줄바꿈 정규화). 이후 정규식이 \n 기준으로 동작한다.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub(" ", text)
    text = _BROKEN.sub("", text)
    text = _HWP_TAG.sub("", text)
    text = _TOC_LINE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_field(value: Any) -> Optional[str]:
    """메타데이터 문자열 필드를 정리한다(결측은 None)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s or None


def _parse_amount(value: Any) -> Optional[int]:
    """사업금액을 정수로 변환한다. 0원/결측은 미기재로 보고 None 처리한다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    # 숫자형(예: pandas가 130000000.0 float로 읽음)은 바로 정수화한다.
    # str() 후 숫자만 뽑으면 '.0'의 0이 붙어 10배로 부풀려지는 버그가 있었다.
    if isinstance(value, numbers.Number):
        amount = int(value)
    else:
        digits = re.sub(r"[^\d]", "", str(value))
        if not digits:
            return None
        amount = int(digits)
    return None if amount == 0 else amount


def _parse_date(value: Any) -> Optional[str]:
    """마감일을 datetime으로 파싱해 ISO 날짜 문자열(YYYY-MM-DD)로 반환한다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def _build_metadata(row: pd.Series, filename: str, fmt: str,
                    likely_scanned: bool, fallback_used: bool) -> Dict[str, Any]:
    """청크가 상속할 Document 메타데이터를 구성한다."""
    return {
        "파일명": filename,
        "기관": _clean_field(row.get("발주기관")),
        "금액": _parse_amount(row.get("사업금액")),
        "마감일": _parse_date(row.get("입찰참여마감일")),
        "파일형식": fmt,
        "likely_scanned": likely_scanned,
        "fallback_used": fallback_used,
        # 참고용 보조 필드
        "사업명": _clean_field(row.get("사업명")),
        "공고번호": _clean_field(row.get("공고번호")),
    }


def _is_hwp(filename: str, fmt: str) -> bool:
    return filename.lower().endswith((".hwp", ".hwpx")) or "hwp" in fmt


def _is_pdf(filename: str, fmt: str) -> bool:
    return filename.lower().endswith(".pdf") or "pdf" in fmt


def _extract_hwp(path: Path, primary, fallback) -> tuple:
    """HWP 본문을 추출하되, 1차 파서가 실패/빈 결과면 대체 파서로 재시도한다.

    pyhwp(hwp5txt)가 일부 HWP에서 0자를 반환하는 경우가 있어, 그런 파일만
    hwp-hwpx-parser 등 대체 파서로 재시도해 본문을 복구한다.
    반환: (text, used_fallback)
    """
    try:
        text = _parse_with_timeout(primary, path, PARSE_TIMEOUT)
    except (FuturesTimeout, Exception):
        text = ""
    if text and text.strip():
        return text, False
    if fallback is None:
        return text, False
    # 1차가 빈 결과/실패 -> 대체 파서로 재시도
    try:
        return _parse_with_timeout(fallback, path, PARSE_TIMEOUT), True
    except (FuturesTimeout, Exception):
        return "", True


def parse_file(path: Path, cfg: Dict[str, Any]) -> str:
    """단일 파일을 설정된 파서로 파싱해 정제 텍스트를 반환한다.

    nara_agent 등 외부에서 개별 파일을 파싱할 때 쓰는 공개 API.
    HWP는 load_documents와 동일하게 1차 파서 실패/빈 결과 시 대체 파서로 폴백한다.
    파싱 실패나 미지원 형식이면 빈 문자열을 반환한다.
    """
    parser_cfg = cfg.get("parser", {})
    filename = path.name
    fmt = path.suffix.lower().lstrip(".")
    try:
        if _is_hwp(filename, fmt):
            hwp_name = parser_cfg.get("hwp", "pyhwp")
            primary = make_hwp_parser(hwp_name)
            fallback = make_hwp_parser(
                "hwp-hwpx-parser" if hwp_name != "hwp-hwpx-parser" else "pyhwp"
            )
            text, _ = _extract_hwp(path, primary, fallback)
        elif _is_pdf(filename, fmt):
            parser = make_pdf_parser(parser_cfg.get("pdf", "pymupdf"))
            text = _parse_with_timeout(parser, path, PARSE_TIMEOUT)
        else:
            return ""
    except (FuturesTimeout, Exception):
        return ""
    return _clean(text) if text and text.strip() else ""


def load_documents(cfg: Dict[str, Any]) -> List[Document]:
    """data_list.csv를 읽고 각 행의 원본 파일을 파싱해 Document 리스트로 반환한다.

    파싱 실패/타임아웃이 나도 진행을 멈추지 않고 fallback_used=True로 표시한다.
    """
    csv_path = Path(cfg["data"]["csv"])
    files_dir = Path(cfg["data"]["files"])
    df = pd.read_csv(csv_path)
    # 실제 CSV 컬럼명에 공백이 있다(예: '사업 금액'). 공백을 제거해 정규화한다.
    df.columns = [str(c).replace(" ", "") for c in df.columns]

    # config로 파서 선택(교체 축). 기본값은 베이스라인 파서.
    parser_cfg = cfg.get("parser", {})
    hwp_name = parser_cfg.get("hwp", "hwp-hwpx-parser")
    hwp_parser = make_hwp_parser(hwp_name)
    pdf_parser = make_pdf_parser(parser_cfg.get("pdf", "pdfplumber"))
    # 1차 HWP 파서가 실패/빈 결과일 때 쓸 대체 파서(다른 HWP 파서).
    # 예: 기본 pyhwp가 0자를 반환하는 일부 HWP를 hwp-hwpx-parser로 복구.
    hwp_fallback = make_hwp_parser(
        "hwp-hwpx-parser" if hwp_name != "hwp-hwpx-parser" else "pyhwp"
    )

    docs: List[Document] = []
    for _, row in df.iterrows():
        filename = str(row["파일명"])
        path = files_dir / filename
        fmt = str(row.get("파일형식", "")).lower()

        text = ""
        fallback_used = False
        if path.exists():
            if _is_hwp(filename, fmt):
                # HWP는 1차 실패/빈 결과 시 대체 파서로 재시도(본문 복구).
                text, fallback_used = _extract_hwp(path, hwp_parser, hwp_fallback)
            else:
                # PDF 또는 형식 불명: PDF 파서로 시도.
                try:
                    text = _parse_with_timeout(pdf_parser, path, PARSE_TIMEOUT)
                except (FuturesTimeout, Exception):
                    # 실패해도 중단하지 않는다. 본문은 비우고 플래그만 남긴다.
                    # ('텍스트' 컬럼으로 대체하지 않는다 — 발췌본이라 본문 소스 부적합)
                    text = ""
                    fallback_used = True
        else:
            fallback_used = True

        cleaned = _clean(text) if text else ""
        likely_scanned = len(cleaned) < MIN_CHARS

        docs.append(
            Document(
                page_content=cleaned,
                metadata=_build_metadata(
                    row, filename, fmt, likely_scanned, fallback_used
                ),
            )
        )
    return docs
