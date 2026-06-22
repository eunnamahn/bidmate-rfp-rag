"""
전처리 파이프라인 — 입찰메이트 RAG 시스템
파싱 → 정제 → 메타데이터 태깅 → 청킹
"""

import os
import re
import subprocess
import pandas as pd
import fitz  # pymupdf
from pathlib import Path
from datetime import datetime
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 실험 설정은 data/config.py 에서 관리
try:
    from data.config import (
        CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHAR,
        REMOVE_TOC, REMOVE_HWP_TAGS,
        HWP_PARSER, PDF_PARSER,
        CHUNK_STRATEGY,
    )
except ImportError:
    CHUNK_SIZE      = 1000
    CHUNK_OVERLAP   = 100
    MIN_CHAR        = 500
    REMOVE_TOC      = True
    REMOVE_HWP_TAGS = True
    HWP_PARSER      = "pyhwp"
    PDF_PARSER      = "pymupdf"
    CHUNK_STRATEGY  = "recursive"


# ══════════════════════════════════════════════════════
# 1. 파싱 (로더)
# ══════════════════════════════════════════════════════

def load_hwp(path: str) -> tuple[str, bool]:
    """
    pyhwp(hwp5txt) CLI로 HWP 텍스트 추출
    Returns: (텍스트, partial_extract 여부)
    - 정상 추출: (본문 전체, False)
    - 실패 시 PrvText 폴백: (미리보기 일부, True)
    - 완전 실패: ("", False)
    """
    try:
        result = subprocess.run(
            ["hwp5txt", path],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            return result.stdout, False
    except Exception:
        pass

    # 폴백: PrvText(미리보기 텍스트 스트림) 우회 추출
    try:
        result = subprocess.run(
            ["hwp5txt", "--loglevel", "CRITICAL", "--format", "text", path],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            print(f"  ⚠️  폴백 추출(PrvText): {path}")
            return result.stdout, True
    except Exception:
        pass

    return "", False


def load_pdf(path: str) -> str:
    """PyMuPDF로 PDF 텍스트 추출"""
    try:
        import fitz
        fitz.TOOLS.mute_warnings = True
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        return ""


def load_document(path: str) -> tuple[str, bool]:
    """
    확장자 기반 통합 로더 — data/config.py의 HWP_PARSER, PDF_PARSER 반영
    Returns: (텍스트, partial_extract 여부)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if PDF_PARSER == "pdfplumber":
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages), False
            except Exception:
                return "", False
        return load_pdf(path), False
    elif ext in (".hwp", ".hwpx"):
        if HWP_PARSER == "hwp-hwpx-parser":
            try:
                from hwp_hwpx_parser import Reader
                reader = Reader(path)
                return reader.get_text(), False
            except Exception:
                return "", False
        if HWP_PARSER == "libhwp":
            try:
                from libhwp import HWPReader
                reader = HWPReader(path)
                return "\n".join(p.text for p in reader.paragraphs if p.text), False
            except Exception:
                return "", False
        if HWP_PARSER == "olefile":
            try:
                import olefile
                import zlib
                ole = olefile.OleFileIO(path)
                if ole.exists("PrvText"):
                    data = ole.openstream("PrvText").read()
                    text = data.decode("utf-16-le", errors="ignore")
                    ole.close()
                    return text, False
                ole.close()
                return "", False
            except Exception:
                return "", False
        return load_hwp(path)  # (text, partial) 그대로 전달
    return "", False


# ══════════════════════════════════════════════════════
# 2. 전처리 (정제)
# ══════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    전처리 기법 적용:
    1. <표><그림> 태그 제거 (HWP 파싱 아티팩트)
    2. \r\n → \n 통일
    3. 연속 공백 정규화
    4. 연속 개행 3개 이상 → 2개로
    5. 목차 블록 제거 (····· 패턴)
    6. 깨진 특수문자 제거 (HWP 점 리더 파싱 아티팩트)
    """
    # 1. HWP 아티팩트 태그 제거 — data/config.py의 REMOVE_HWP_TAGS 값으로 ON/OFF
    if REMOVE_HWP_TAGS:
        text = re.sub(r"<표>|<그림>", "", text)

    # 2. 줄바꿈 통일
    text = re.sub(r"\r\n|\r", "\n", text)

    # 3. 연속 공백 → 단일 공백
    text = re.sub(r" {2,}", " ", text)

    # 4. 연속 개행 정규화
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. 목차 블록 제거 — data/config.py의 REMOVE_TOC 값으로 ON/OFF
    if REMOVE_TOC:
        text = re.sub(r"^.+[·\.]{5,}.+$", "", text, flags=re.MULTILINE)

    # 6. HWP 점 리더가 깨진 특수문자(한자 영역 오염) 제거
    text = re.sub(r"[㐀-䶿一-鿿]{1,2}[Ā-ɏ]", "", text)

    return text.strip()


def is_valid_document(text: str) -> bool:
    """빈 문서 또는 부실 문서 판별"""
    return len(text.strip()) >= MIN_CHAR


# ══════════════════════════════════════════════════════
# 3. 메타데이터 처리
# ══════════════════════════════════════════════════════

def classify_org(name: str) -> str:
    """발주기관 유형 자동 분류"""
    name = str(name)
    if any(k in name for k in ["대학교", "대학", "학교", "과학기술원"]):
        return "교육기관"
    elif any(k in name for k in ["시청", "군청", "구청", "광역시", "특별시", "도청", "시", "군", "구"]):
        return "지자체"
    elif any(k in name for k in ["부", "처", "청", "위원회"]):
        return "중앙부처"
    elif any(k in name for k in ["공사", "공단", "한국", "진흥원", "연구원", "재단", "연구소", "평가원", "진흥회"]):
        return "공공기관"
    elif any(k in name for k in ["협회", "조합", "중앙회", "센터"]):
        return "공기업·협회"
    return "기타"


def process_metadata(row: pd.Series) -> dict:
    """
    CSV 한 행 → RAG용 메타데이터 딕셔너리
    - 날짜 타입 변환
    - 금액 0원 → None 처리
    - 기관유형 자동 분류
    """
    amount = row.get("사업 금액")
    if pd.isna(amount) or amount == 0:
        amount = None

    공개일자 = row.get("공개 일자")
    try:
        공개일자 = pd.to_datetime(공개일자).strftime("%Y-%m-%d") if 공개일자 else None
    except:
        공개일자 = None

    org = str(row.get("발주 기관", ""))

    요약 = row.get("사업 요약")
    요약 = str(요약).strip() if 요약 and not pd.isna(요약) else None

    return {
        "사업명"    : str(row.get("사업명", "")),
        "발주기관"  : org,
        "기관유형"  : classify_org(org),
        "사업금액"  : amount,
        "사업금액_억": round(amount / 1e8, 2) if amount else None,
        "공개일자"  : 공개일자,
        "파일형식"  : str(row.get("파일형식", "")),
        "파일명"    : str(row.get("파일명", "")),
        "사업요약"  : 요약,
    }


# ══════════════════════════════════════════════════════
# 4. 청킹
# ══════════════════════════════════════════════════════

def chunk_document(text: str, metadata: dict) -> list[dict]:
    """
    텍스트 → 청크 리스트 — data/config.py의 CHUNK_STRATEGY 반영
    "recursive" : 글자 수 기준 (기본)
    "section"   : RFP 섹션 구조 기준 (□ / 제N조 / 가.나.다.)
    "semantic"  : 의미 경계 기준 (⚠️ 임베딩 모델 셋업 후 활성화)
    """
    if CHUNK_STRATEGY == "section":
        return _chunk_by_section(text, metadata)
    if CHUNK_STRATEGY == "semantic":
        return _chunk_by_semantic(text, metadata)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    return [
        {
            "text"    : _attach_header(chunk, metadata),
            "metadata": {
                **metadata,
                "chunk_index": i,
                "chunk_total": len(chunks),
            }
        }
        for i, chunk in enumerate(chunks)
    ]


def _attach_header(text: str, metadata: dict) -> str:
    """청크 텍스트 앞에 기관명/사업명 한 줄 부착 — 기관명 없는 문서(32건) 검색 보완"""
    parts = []
    if metadata.get("발주기관"):
        parts.append(metadata["발주기관"])
    if metadata.get("사업명"):
        parts.append(metadata["사업명"])
    if not parts:
        return text
    header = " | ".join(parts)
    return f"[{header}]\n{text}"


def _chunk_by_section(text: str, metadata: dict) -> list[dict]:
    """섹션 구조 기반 청킹 — RFP 문서의 □ / 제N조 / 가.나.다. 패턴 활용"""
    section_pattern = re.compile(
        r'(?=^□|^제\d+조|^\d+\.\s|^[가나다라마바사아자차]\.\s|^[Ⅰ-Ⅶ])',
        re.MULTILINE
    )
    sections = section_pattern.split(text)
    sections = [s.strip() for s in sections if s.strip()]

    # 너무 짧은 섹션은 다음 섹션과 합치기
    merged = []
    buffer = ""
    for s in sections:
        buffer += s + "\n"
        if len(buffer) >= CHUNK_SIZE:
            merged.append(buffer.strip())
            buffer = ""
    if buffer:
        merged.append(buffer.strip())

    return [
        {
            "text"    : _attach_header(chunk, metadata),
            "metadata": {
                **metadata,
                "chunk_index": i,
                "chunk_total": len(merged),
            }
        }
        for i, chunk in enumerate(merged)
    ]


def _chunk_by_semantic(text: str, metadata: dict) -> list[dict]:
    """
    ⚠️  임베딩 모델 셋업 후 활성화
    사용법:
        1. 임베딩 담당자에게 모델 확정 요청
        2. pip install langchain-experimental langchain-huggingface sentence-transformers
        3. data/config.py → CHUNK_STRATEGY = "semantic" 으로 변경
        4. 아래 model_name을 확정된 모델로 교체
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",  # ← 임베딩 담당자 확정 모델로 교체
            model_kwargs={"device": "cuda"},
        )
        splitter = SemanticChunker(embeddings)
        chunks = splitter.split_text(text)

        return [
            {
                "text"    : _attach_header(chunk, metadata),
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    except ImportError:
        print("⚠️  Semantic chunking 패키지 없음 → recursive로 대체")
        return chunk_document(text, metadata)


# ══════════════════════════════════════════════════════
# 5. 전체 파이프라인
# ══════════════════════════════════════════════════════

def run_pipeline(csv_path: str, files_dir: str) -> list[dict]:
    """
    전체 전처리 파이프라인 실행
    CSV + 파일 폴더 → 청크 리스트 반환

    Returns:
        [{"text": ..., "metadata": {...}}, ...]
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    files_dir = Path(files_dir)

    all_chunks = []
    stats = {"성공": 0, "빈문서": 0, "파일없음": 0, "오류": 0, "폴백추출": 0}

    for _, row in df.iterrows():
        fname = str(row.get("파일명", ""))
        fpath = files_dir / fname

        # 파일 존재 확인
        if not fpath.exists():
            stats["파일없음"] += 1
            continue

        # 파싱
        try:
            raw_text, is_partial = load_document(str(fpath))
        except Exception:
            stats["오류"] += 1
            continue

        # 정제
        clean = clean_text(raw_text)

        # 빈/부실 문서 필터링
        if not is_valid_document(clean):
            stats["빈문서"] += 1
            continue

        # 메타데이터
        meta = process_metadata(row)
        if is_partial:
            meta["partial_extract"] = True  # RAG 신뢰도 낮게 취급
            stats["폴백추출"] += 1

        # 청킹
        chunks = chunk_document(clean, meta)
        all_chunks.extend(chunks)
        stats["성공"] += 1

    # 결과 요약
    print("=" * 45)
    print("전처리 파이프라인 완료")
    print("=" * 45)
    print(f"성공       : {stats['성공']}개")
    print(f"폴백추출   : {stats['폴백추출']}개 (부분 추출, 신뢰도 낮음)")
    print(f"빈/부실    : {stats['빈문서']}개")
    print(f"파일없음   : {stats['파일없음']}개")
    print(f"오류       : {stats['오류']}개")
    print(f"총 청크 수 : {len(all_chunks)}개")
    print(f"평균 청크  : {len(all_chunks)/max(stats['성공'],1):.1f}개/문서")

    return all_chunks


# ══════════════════════════════════════════════════════
# 실행 예시
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    CSV_PATH  = "/content/drive/MyDrive/codeit_mission/원본 데이터/data_list.csv"
    FILES_DIR = "/content/drive/MyDrive/codeit_mission/원본 데이터/files"

    chunks = run_pipeline(CSV_PATH, FILES_DIR)

    # 샘플 확인
    print("\n=== 샘플 청크 ===")
    print(f"텍스트: {chunks[0]['text'][:200]}")
    print(f"메타데이터: {chunks[0]['metadata']}")
