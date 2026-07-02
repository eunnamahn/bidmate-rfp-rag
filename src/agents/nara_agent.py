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
import re
import shutil
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

import requests
from langchain_core.documents import Document

from src.agents.metadata_agent import append_metadata_row, extract_metadata
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


# 첨부 중 RFP 본문에 해당하는 파일만 수집하기 위한 파일명 키워드.
# (공고서·붙임 양식 등 부수 문서까지 전수 적재하면 벡터DB가 비대해지므로 본문류만 받는다.)
# "제안요청서" 계열은 용역 입찰 기준 키워드라 물품구매·공사 입찰(규격서/시방서 계열)에는
# 하나도 안 걸려 실제로는 첨부가 있는데도 "첨부 없음"으로 이어졌다(AI추천 카드가 뜨고 나서
# '이 공고로 대화하기'를 누르면 인덱싱할 내용이 없다는 에러가 나던 원인).
_RFP_FILENAME_KEYWORDS = [
    "제안요청서", "rfp", "과업지시서", "과업내용서", "기술제안서",
    "규격서", "시방서",
]


def _is_rfp_attachment(filename: str) -> bool:
    """파일명이 RFP 본문류(제안요청서/과업지시서 등)인지 판별한다."""
    return any(kw in filename.lower() for kw in _RFP_FILENAME_KEYWORDS)


def _collect_attachments(item: Dict, download_dir: Path) -> List[Path]:
    """item의 ntceSpecDocUrl1~10에서 URL을 읽어 다운로드하고 hwp/pdf 경로를 반환한다.

    - 파일명이 RFP 본문류(제안요청서·과업지시서 등)인 첨부만 다운로드한다(부수 문서 제외).
    - zip이면 압축 해제 후 내부 hwp/pdf만 추린다.
    - 공고별로 하위 디렉토리(bidNtceNo/)를 만들어 파일을 격리한다.
    """
    bid_id = item.get("bidNtceNo", "unknown")
    item_dir = download_dir / bid_id

    result: List[Path] = []
    for i in range(1, 11):
        url = (item.get(f"ntceSpecDocUrl{i}") or "").strip()
        if not url:
            continue
        # 파일명은 메타(ntceSpecFileNm)를 우선 쓰고, 없으면 URL 끝에서 추출한다.
        filename = (item.get(f"ntceSpecFileNm{i}") or "").strip()
        check_name = filename or unquote(url.split("/")[-1].split("?")[0])
        if not _is_rfp_attachment(check_name):
            continue

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


# 두 첨부의 내용 토큰 자카드 유사도가 이 값 이상이면 같은 문서로 간주한다.
# (같은 공고의 hwp/pdf는 파일명이 달라도 본문이 거의 같다.)
_DUP_SIM_THRESHOLD = 0.6


def _content_tokens(text: str) -> set:
    """내용 비교용 토큰 집합(2자 이상 한글/영숫자). 공백·표기 차이에 견고."""
    return set(re.findall(r"[가-힣A-Za-z0-9]{2,}", text))


def _is_hwp_path(path: Path) -> bool:
    return path.suffix.lower() in (".hwp", ".hwpx")


def _dedup_attachments(paths: List[Path], cfg: Dict[str, Any]) -> List[Tuple[Path, str]]:
    """같은 공고의 첨부 중 내용이 같은 hwp/pdf를 묶어 1개만 남긴다.

    파일명이 아니라 **본문 내용 유사도**로 묶으므로, 같은 공고서가 서로 다른
    파일명의 hwp/pdf로 첨부돼도 하나로 합쳐진다. 그룹에서는 **HWP를 우선** 선택해
    (원본 + 사용자가 원하는 메타데이터 소스), HWP가 없거나 파싱 실패면 PDF를 쓴다.
    내용이 다른 문서(공고서·과업지시서 등)는 유사도가 낮아 각각 유지된다.

    반환: [(선택된 파일 경로, 파싱 텍스트)] — 텍스트 추출 실패 첨부는 제외.
    """
    parsed: List[Tuple[Path, str, set]] = []
    for p in paths:
        text = parse_file(p, cfg)
        if text:
            parsed.append((p, text, _content_tokens(text)))

    groups: List[List[Tuple[Path, str, set]]] = []
    for cand in parsed:
        for group in groups:
            rep_tokens = group[0][2]
            union = cand[2] | rep_tokens
            if union and len(cand[2] & rep_tokens) / len(union) >= _DUP_SIM_THRESHOLD:
                group.append(cand)
                break
        else:
            groups.append([cand])

    # 그룹마다 HWP 우선, 그다음 텍스트가 많은 것 선택
    chosen: List[Tuple[Path, str]] = []
    for group in groups:
        best = max(group, key=lambda c: (_is_hwp_path(c[0]), len(c[1])))
        chosen.append((best[0], best[1]))
    return chosen


def _copy_into_files(src: Path, files_dir: Path, bid_id: str) -> Path:
    """나라장터 첨부를 원본 코퍼스 디렉토리(data/files)로 복사하고 그 경로를 반환한다.

    전체 재빌드 시 load_documents는 data/files/<파일명>을 재파싱하므로, 지연 인덱싱한
    공고도 재빌드와 일관되려면 파일이 data/files에 있어야 한다. 이름이 충돌하면 공고번호를
    붙여 회피한다(같은 크기 파일이 이미 있으면 동일 파일로 보고 그대로 재사용).
    """
    files_dir.mkdir(parents=True, exist_ok=True)
    dest = files_dir / src.name
    if dest.exists():
        if dest.stat().st_size == src.stat().st_size:
            return dest
        dest = files_dir / f"{src.stem}__{bid_id}{src.suffix}"
        n = 1
        while dest.exists() and dest.stat().st_size != src.stat().st_size:
            dest = files_dir / f"{src.stem}__{bid_id}_{n}{src.suffix}"
            n += 1
        if dest.exists():
            return dest
    shutil.copy2(src, dest)
    return dest


def _docs_for_item(item: Dict, cfg: Dict[str, Any], download_dir: Path) -> List[Document]:
    """공고 1건의 첨부를 받아 파싱·메타추출해 Document 리스트로 만든다.

    collect(배치)와 ingest_notice(지연 인덱싱)가 공유하는 단건 처리 로직.
    같은 문서의 hwp/pdf 중복은 _dedup_attachments로 1개만 남긴다.
    첨부가 없거나 텍스트 추출 실패면 빈 리스트.

    추가된 문서는 (1) 원본 코퍼스(data/files)로 복사해 전체 재빌드와 일관되게 하고,
    (2) 메타데이터를 data_list.csv 대장에 자동 기록한다.
    """
    bid_id = item.get("bidNtceNo", "unknown")
    files_dir = Path(cfg["data"]["files"])
    csv_path = cfg["data"]["csv"]
    docs: List[Document] = []
    for fpath, text in _dedup_attachments(_collect_attachments(item, download_dir), cfg):
        agent_meta = extract_metadata(text, cfg)
        # 재빌드 일관성: 첨부를 data/files로 복사하고 그 경로 기준으로 Document를 만든다.
        dest = _copy_into_files(fpath, files_dir, bid_id)
        doc = _make_document(text, dest, bid_id, item, agent_meta)
        docs.append(doc)
        # 메타데이터 대장(csv) 자동 기록(중복 파일명은 내부에서 건너뜀, 실패는 무시).
        try:
            append_metadata_row(
                csv_path, dest.name, dest.suffix.lower().lstrip("."),
                {
                    "발주기관": doc.metadata.get("기관"),
                    "사업금액": doc.metadata.get("금액"),
                    "마감일": doc.metadata.get("마감일"),
                    "사업명": doc.metadata.get("사업명"),
                },
                텍스트=text, 공고번호=bid_id,
            )
        except Exception:
            pass
    return docs


# ──────────────────────────────────────────────
# 검색 (메타데이터만 · 인덱스에 적재하지 않음)
# ──────────────────────────────────────────────

# 검색어에서 의미 없는 호출/조사성 단어를 제거하기 위한 불용어/조사 처리.
_STOPWORDS = {
    "나라장터", "에서", "찾아서", "찾아", "설명", "해줘", "공고", "입찰", "수집",
    "최신", "신규", "관련", "알려", "정보", "내용", "위탁", "용역", "사업",
}
_PARTICLE = re.compile(r"(을|를|이|가|은|는|의|로|으로|에서|에|와|과|도|만)$")


def _significant_tokens(text: str) -> List[str]:
    """검색어에서 매칭에 쓸 의미 토큰만 추출한다(조사 제거 + 불용어 제외)."""
    tokens: List[str] = []
    for raw in re.findall(r"[가-힣A-Za-z0-9]+", text or ""):
        word = _PARTICLE.sub("", raw)
        if len(word) < 2 or word in _STOPWORDS:
            continue
        tokens.append(word)
    return tokens


def notice_summary(item: Dict) -> Dict[str, Any]:
    """API 공고 item에서 표시용 메타데이터만 뽑는다(다운로드/임베딩 없음)."""
    openg_raw = item.get("opengDt", "") or ""
    마감일 = _parse_date(openg_raw[:8]) if len(openg_raw) >= 8 else None
    return {
        "공고번호": item.get("bidNtceNo"),
        "사업명": item.get("bidNtceNm"),
        "기관": item.get("dmndInsttNm") or item.get("ntceInsttNm"),
        "금액": _parse_amount(item.get("presmptPrce")) or _parse_amount(item.get("asignBdgtAmt")),
        "마감일": 마감일,
    }


def search_notices(
    cfg: Dict[str, Any],
    keyword: Optional[str] = None,
    target_date: Optional[date] = None,
    days: int = 3,
    max_results: int = 10,
    max_pages: int = 5,
) -> List[Dict]:
    """나라장터 목록 API로 공고를 검색해 원본 item 리스트를 반환한다(메타데이터만).

    첨부 다운로드나 임베딩을 하지 않으므로 가볍다. keyword가 있으면 사업명에
    의미 토큰이 포함된 공고만 추려 매칭 수가 많은 순으로 정렬한다.

    target_date가 있으면 그날만, 없으면 최근 days일을 조회한다.
    max_pages로 조회량을 제한해(메타데이터라도) 과도한 페이지네이션을 막는다.

    Raises:
        RuntimeError: NARA_API_KEY가 없을 때.
    """
    api_key = os.getenv("NARA_API_KEY")
    if not api_key:
        raise RuntimeError(
            ".env에 NARA_API_KEY가 없습니다. 공공데이터포털에서 발급 후 설정하세요."
        )

    page_size = int(cfg.get("nara", {}).get("page_size", 100))
    end = target_date or date.today()
    start = end if target_date else end - timedelta(days=max(0, days - 1))
    inqry_bgn = start.strftime("%Y%m%d") + "0000"
    inqry_end = end.strftime("%Y%m%d") + "2359"

    items: List[Dict] = []
    for page_no in range(1, max_pages + 1):
        page_items, total = _fetch_page(api_key, inqry_bgn, inqry_end, page_no, page_size)
        items.extend(page_items)
        if not page_items or len(items) >= total:
            break

    tokens = _significant_tokens(keyword) if keyword else []
    if tokens:
        scored: List[Tuple[int, Dict]] = []
        for it in items:
            name = it.get("bidNtceNm") or ""
            hits = sum(1 for t in tokens if t in name)
            if hits:
                scored.append((hits, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        items = [it for _, it in scored]

    return items[:max_results]


def ingest_notice(
    cfg: Dict[str, Any],
    item: Dict,
    embedding=None,
    cache_key: Optional[str] = None,
) -> Dict[str, Any]:
    """공고 1건만 다운로드·파싱·임베딩해 FAISS에 추가한다(지연 인덱싱).

    Args:
        cfg:       설정 dict.
        item:      search_notices가 돌려준 원본 공고 item.
        embedding: 임베딩 객체(없으면 make_embedding(cfg)). 백엔드는 _STATE 것 재사용.
        cache_key: 대상 캐시 키(없으면 make_cache_key(cfg)).

    Returns:
        {공고번호, 사업명, 기관, 금액, 마감일, chunks, vectorstore}.
        chunks=0이면 첨부 없음/추출 실패.
    """
    download_dir = Path(cfg.get("nara", {}).get("download_dir", "data/nara_downloads"))
    docs = _docs_for_item(item, cfg, download_dir)
    summary = notice_summary(item)

    if not docs:
        return {**summary, "chunks": 0, "vectorstore": None}

    chunks = split_documents(docs, cfg)
    embedding = embedding or make_embedding(cfg)
    vs = add_to_vectorstore(
        chunks, embedding, cfg["data"]["faiss_cache"], cache_key or make_cache_key(cfg)
    )
    # 대표 문서 파일명(data/files 기준)도 함께 돌려준다. UI가 이걸 '활성 문서'로 지정해야
    # 에이전트 검색이 이 공고로 정확히 한정된다(사업명 라벨이 아니라 실제 파일명 사용).
    return {
        **summary,
        "chunks": len(chunks),
        "vectorstore": vs,
        "파일명": docs[0].metadata.get("파일명"),
    }


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

        new_ids.add(bid_id)  # 첨부파일 없어도 본 공고로 기록해 재시도 방지
        item_docs = _docs_for_item(item, cfg, download_dir)
        new_docs.extend(item_docs)
        for d in item_docs:
            print(f"    [완료] {d.metadata['파일명']} ({len(d.page_content):,}자)", flush=True)

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
