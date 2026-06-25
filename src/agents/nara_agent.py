"""나라장터 입찰공고 자동 수집 에이전트.

나라장터 입찰공고정보서비스 OpenAPI를 통해 당일 신규 공고를 수집하고
파싱 → 청킹 → 임베딩 → FAISS에 점진적으로 추가한다.

공공데이터포털 API:
  https://www.data.go.kr/data/15129394/openapi.do
  endpoint: GET /1230000/BidPublicInfoService04/getBidPblancListInfoServc

환경 변수:
  NARA_API_KEY — .env에 저장 (공공데이터포털에서 발급받은 서비스키, URL 인코딩 전 원문)

사용법(라이브러리):
  from src.config import load_config
  from src.agents.nara_agent import collect

  cfg = load_config("config.yaml")
  n = collect(cfg)                    # 당일 수집
  n = collect(cfg, date(2024, 4, 9))  # 특정 날짜 수집

사용법(배치 CLI):
  python -m src.agents.nara_agent              # 당일 수집
  python -m src.agents.nara_agent 2024-04-09   # 특정 날짜
  python -m src.agents.nara_agent 2024-04-09 --max 5   # 최대 5건만
"""

from __future__ import annotations

import json
import os
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

import requests
from langchain_core.documents import Document

from src.agents.metadata_agent import extract_metadata
from src.chunking import split_documents
from src.config import make_embedding
from src.embedding import add_to_vectorstore, make_cache_key
from src.loader import _parse_amount, _parse_date, parse_file

_BASE_URL = "https://apis.data.go.kr/1230000/ad/BidPublicInfoService"
_LIST_URL = _BASE_URL + "/getBidPblancListInfoServc"

# 다운로드 대상 확장자
_TARGET_EXTS = {".hwp", ".hwpx", ".pdf"}


# ──────────────────────────────────────────────
# 중복 방지: seen 파일 관리
# ──────────────────────────────────────────────

def _load_seen(seen_path: Path) -> Set[str]:
    """이미 수집한 공고번호 집합을 파일에서 로드한다."""
    if not seen_path.exists():
        return set()
    return set(json.loads(seen_path.read_text(encoding="utf-8")))


def _save_seen(seen_path: Path, ids: Set[str]) -> None:
    seen_path.parent.mkdir(parents=True, exist_ok=True)
    seen_path.write_text(
        json.dumps(sorted(ids), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ──────────────────────────────────────────────
# API 호출
# ──────────────────────────────────────────────

def _fetch_page(
    api_key: str, inqry_bgn: str, inqry_end: str,
    page_no: int, page_size: int,
) -> Tuple[List[Dict], int]:
    """API 한 페이지를 호출해 (items, totalCount)를 반환한다."""
    params = {
        "serviceKey": api_key,
        "numOfRows": page_size,
        "pageNo": page_no,
        "inqryDiv": "1",        # 1=전체 공고 조회
        "inqryBgnDt": inqry_bgn,
        "inqryEndDt": inqry_end,
        "type": "json",
    }
    resp = requests.get(_LIST_URL, params=params, timeout=30)
    resp.raise_for_status()

    body = resp.json().get("response", {}).get("body", {})
    total = int(body.get("totalCount", 0))
    items_raw = body.get("items") or {}

    # API 응답: 단건이면 {"item": {...}}, 복수면 {"item": [...]}
    if isinstance(items_raw, dict):
        inner = items_raw.get("item", [])
        items = inner if isinstance(inner, list) else ([inner] if inner else [])
    elif isinstance(items_raw, list):
        items = items_raw
    else:
        items = []

    return items, total


def _fetch_notices(
    api_key: str, inqry_bgn: str, inqry_end: str, page_size: int = 100,
) -> List[Dict]:
    """지정 기간의 공고 목록을 페이지네이션으로 전부 수집해 반환한다."""
    all_items: List[Dict] = []
    page_no = 1
    while True:
        items, total = _fetch_page(api_key, inqry_bgn, inqry_end, page_no, page_size)
        all_items.extend(items)
        if not items or len(all_items) >= total:
            break
        page_no += 1
    return all_items


# ──────────────────────────────────────────────
# 파일 다운로드 / 압축 해제
# ──────────────────────────────────────────────

def _download_file(url: str, dest_dir: Path) -> Optional[Path]:
    """URL에서 파일을 다운로드해 dest_dir에 저장하고 경로를 반환한다."""
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()

        # Content-Disposition → RFC 5987(filename*=) 우선, 아니면 filename=
        # 각 값은 "; charset=..." 같은 후속 파라미터를 포함할 수 있으므로
        # ";" 기준으로 첫 토큰만 사용한다.
        cd = resp.headers.get("Content-Disposition", "")
        filename = ""
        if "filename*=" in cd:
            raw = cd.split("filename*=")[-1].split(";")[0].strip()
            filename = unquote(raw.split("''", 1)[-1])
        elif "filename=" in cd:
            filename = unquote(cd.split("filename=")[-1].split(";")[0].strip().strip('"\''))

        if not filename:
            filename = unquote(url.split("/")[-1].split("?")[0]) or "attachment"

        dest = dest_dir / filename
        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest
    except Exception as e:
        print(f"    [경고] 다운로드 실패 ({url[:60]}...): {e}")
        return None


def _extract_zip(zip_path: Path) -> List[Path]:
    """zip을 같은 디렉토리에 풀고 hwp/pdf 파일 경로 목록을 반환한다."""
    dest = zip_path.parent
    extracted: List[Path] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if Path(member).suffix.lower() in _TARGET_EXTS:
                    zf.extract(member, dest)
                    extracted.append(dest / member)
    except zipfile.BadZipFile:
        print(f"    [경고] 손상된 zip: {zip_path.name}")
    return extracted


def _collect_attachments(item: Dict, download_dir: Path) -> List[Path]:
    """item의 ntceSpecDocUrl1~10에서 URL을 읽어 다운로드하고 hwp/pdf 경로를 반환한다.

    - zip이면 압축 해제 후 내부 hwp/pdf만 추린다.
    - 공고별로 하위 디렉토리(bidNtceNo/)를 만들어 파일을 격리한다.
    """
    bid_id = item.get("bidNtceNo", "unknown")
    item_dir = download_dir / bid_id

    urls = [
        url for i in range(1, 11)
        if (url := (item.get(f"ntceSpecDocUrl{i}") or "").strip())
    ]
    result: List[Path] = []
    for url in urls:
        path = _download_file(url, item_dir)
        if path is None:
            continue

        ext = path.suffix.lower()
        if ext == ".zip":
            result.extend(_extract_zip(path))
        elif ext in _TARGET_EXTS:
            result.append(path)

    return result


# ──────────────────────────────────────────────
# Document 생성
# ──────────────────────────────────────────────

def _make_document(
    text: str,
    fpath: Path,
    bid_id: str,
    item: Dict,
    agent_meta: Dict,
) -> Document:
    """파싱 텍스트 + API 공고 데이터 + LLM 추출 메타를 합쳐 Document를 생성한다.

    메타데이터 우선순위: LLM 추출(agent_meta) > API 응답(item).
    기관명만 API 응답을 수요기관(dmndInsttNm) → 공고기관(ntceInsttNm) 순으로 사용한다.
    """
    openg_raw = item.get("opengDt", "") or ""
    api_deadline = _parse_date(openg_raw[:8]) if len(openg_raw) >= 8 else None

    기관 = (
        agent_meta.get("발주기관")
        or item.get("dmndInsttNm")
        or item.get("ntceInsttNm")
    )
    금액 = (
        agent_meta.get("사업금액")
        or _parse_amount(item.get("presmptPrce"))
        or _parse_amount(item.get("asignBdgtAmt"))
    )
    마감일 = agent_meta.get("마감일") or api_deadline
    사업명 = agent_meta.get("사업명") or item.get("bidNtceNm")

    return Document(
        page_content=text,
        metadata={
            "파일명": fpath.name,
            "기관": 기관,
            "금액": 금액,
            "마감일": 마감일,
            "파일형식": fpath.suffix.lower().lstrip("."),
            "likely_scanned": len(text) < 1000,
            "fallback_used": False,
            "사업명": 사업명,
            "공고번호": bid_id,
            "nara_source": True,
        },
    )


# ──────────────────────────────────────────────
# 공개 진입점
# ──────────────────────────────────────────────

def collect(cfg: Dict[str, Any], target_date: Optional[date] = None, max_collect: Optional[int] = None) -> int:
    """나라장터 신규 공고를 수집해 기존 FAISS에 점진적으로 추가한다.

    Args:
        cfg:         config.yaml을 로드한 설정 dict.
        target_date: 수집 대상 날짜. None이면 당일(today).
        max_collect: 수집할 최대 공고 수(None이면 전체).

    Returns:
        새로 처리된 공고 수. (첨부파일 없는 공고도 seen에 기록되므로 포함.)

    Raises:
        RuntimeError: NARA_API_KEY가 .env에 없을 때.
    """
    api_key = os.getenv("NARA_API_KEY")
    if not api_key:
        raise RuntimeError(
            ".env에 NARA_API_KEY가 없습니다. 공공데이터포털에서 발급 후 설정하세요."
        )

    nara_cfg = cfg.get("nara", {})
    seen_path = Path(nara_cfg.get("seen_file", "data/nara_seen.json"))
    download_dir = Path(nara_cfg.get("download_dir", "data/nara_downloads"))
    page_size = int(nara_cfg.get("page_size", 100))

    if target_date is None:
        target_date = date.today()

    inqry_bgn = target_date.strftime("%Y%m%d") + "0000"
    inqry_end = target_date.strftime("%Y%m%d") + "2359"

    seen = _load_seen(seen_path)

    print(f"[나라장터] {target_date} 공고 조회 중...", flush=True)
    try:
        notices = _fetch_notices(api_key, inqry_bgn, inqry_end, page_size)
    except Exception as e:
        print(f"[나라장터] API 호출 실패: {e}")
        return 0
    print(f"[나라장터] {len(notices)}건 조회 완료", flush=True)

    new_docs: List[Document] = []
    new_ids: Set[str] = set()

    for item in notices:
        bid_id = item.get("bidNtceNo", "")
        if not bid_id or bid_id in seen:
            continue
        if max_collect is not None and len(new_ids) >= max_collect:
            break

        ntce_nm = (item.get("bidNtceNm") or "")[:50]
        print(f"  [수집] {bid_id}  {ntce_nm}", flush=True)

        files = _collect_attachments(item, download_dir)
        new_ids.add(bid_id)  # 첨부파일 없어도 본 공고로 기록해 재시도 방지

        for fpath in files:
            text = parse_file(fpath, cfg)
            if not text:
                continue

            agent_meta = extract_metadata(text, cfg)
            doc = _make_document(text, fpath, bid_id, item, agent_meta)
            new_docs.append(doc)
            print(f"    [완료] {fpath.name} ({len(text):,}자)", flush=True)

    if new_docs:
        print(f"[나라장터] {len(new_docs)}개 문서 청킹 및 인덱싱 중...", flush=True)
        chunks = split_documents(new_docs, cfg)
        embedding = make_embedding(cfg)
        add_to_vectorstore(
            chunks, embedding,
            cfg["data"]["faiss_cache"],
            make_cache_key(cfg),
        )
        print(f"[나라장터] {len(chunks)}청크 FAISS 저장 완료", flush=True)

    seen.update(new_ids)
    _save_seen(seen_path, seen)

    print(
        f"[나라장터] 완료 — 신규 {len(new_ids)}건 처리 (문서 {len(new_docs)}개)",
        flush=True,
    )
    return len(new_ids)


# ──────────────────────────────────────────────
# 배치 CLI 진입점
# ──────────────────────────────────────────────

def _main() -> None:
    import argparse
    from datetime import datetime

    from src.config import load_config

    parser = argparse.ArgumentParser(
        description="나라장터 신규 공고를 수집해 FAISS에 증분 추가한다.",
    )
    parser.add_argument(
        "date", nargs="?", default=None,
        help="수집 날짜 YYYY-MM-DD (생략 시 당일)",
    )
    parser.add_argument(
        "--max", type=int, default=None, dest="max_collect",
        help="수집할 최대 공고 수 (생략 시 전체)",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="설정 파일 경로",
    )
    args = parser.parse_args()

    target_date = (
        datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    )
    cfg = load_config(args.config)
    collect(cfg, target_date, args.max_collect)


if __name__ == "__main__":
    _main()
