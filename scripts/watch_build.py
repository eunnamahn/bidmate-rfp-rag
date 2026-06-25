"""청킹×임베딩 그리드 빌드 진행 상황 라이브 모니터.

실행 중인 빌드를 건드리지 않고, 각 인덱스 칸의 완료 여부(index.faiss 존재),
진행 중인 칸, GPU 사용량, 경과 시간을 몇 초마다 갱신해 보여준다.

사용(자기 터미널에서):  python scripts/watch_build.py
종료: Ctrl-C
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import catalog
from src.config import load_config
from src.embedding import _safe_name, make_cache_key

PARSER = {"hwp": "olefile", "pdf": "pdfplumber"}
CHUNK_PRESETS = ["recursive-1000-100", "section-500-50", "section-1000-100"]
EMB_PRESETS = ["openai-3-small", "hf-kure-v1", "hf-bge-m3"]
REFRESH_SEC = 5
BUILD_LOG = Path(__file__).resolve().parent / "chunk_grid_build.log"


def latest_batch_line() -> str:
    """빌드 로그에서 가장 최근 tqdm 배치 진행줄(예: 'Batches 45%| 1200/2600')을 뽑는다."""
    try:
        raw = BUILD_LOG.read_bytes().decode("utf-8", "replace")
    except Exception:
        return ""
    # tqdm은 \r로 같은 줄을 갱신한다 -> \r/\n 모두 분리해 마지막 진행줄을 찾는다.
    segs = raw.replace("\r", "\n").split("\n")
    for seg in reversed(segs):
        if "Batches" in seg or "it/s" in seg or "it]" in seg:
            return seg.strip()[:70]
    return ""


def target_cells():
    base = load_config("config.yaml")
    cache_dir = base["data"]["faiss_cache"]
    cells = []
    for chunk in CHUNK_PRESETS:
        for emb in EMB_PRESETS:
            cfg = catalog.apply_preset(base, "embedding", emb)
            cfg = catalog.apply_preset(cfg, "chunk", chunk)
            cfg["parser"] = dict(PARSER)
            key = make_cache_key(cfg)
            path = Path(cache_dir) / _safe_name(key) / "index.faiss"
            cells.append((f"{emb:<14} × {chunk}", path))
    return cells


def gpu_status() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
        util, used, total = [x.strip() for x in out.split(",")]
        return f"GPU {util}% · VRAM {used}/{total} MiB"
    except Exception:
        return "GPU 상태 조회 불가"


def main() -> None:
    cells = target_cells()
    start = time.time()
    try:
        while True:
            done = [p.exists() for _, p in cells]
            building_idx = next((i for i, d in enumerate(done) if not d), None)
            lines = []
            lines.append("\033[2J\033[H")  # 화면 지우고 커서 맨 위로
            el = int(time.time() - start)
            lines.append(f"청킹×임베딩 그리드 빌드 모니터  "
                         f"(경과 {el//60}분 {el%60}초, {REFRESH_SEC}초마다 갱신, Ctrl-C 종료)")
            lines.append(f"{gpu_status()}")
            lines.append("-" * 56)
            for i, (label, _) in enumerate(cells):
                if done[i]:
                    mark = "✅ 완료"
                elif i == building_idx:
                    mark = "▶  빌드 중"
                else:
                    mark = "·  대기"
                lines.append(f"  {mark:<10} {label}")
            lines.append("-" * 56)
            batch = latest_batch_line()
            if batch:
                lines.append(f"  현재 배치: {batch}")
            n_done = sum(done)
            lines.append(f"  완료 {n_done}/{len(cells)}칸")
            if n_done == len(cells):
                lines.append("  🎉 모든 칸 빌드 완료!")
                print("\n".join(lines), flush=True)
                break
            print("\n".join(lines), flush=True)
            time.sleep(REFRESH_SEC)
    except KeyboardInterrupt:
        print("\n모니터 종료(빌드는 계속 진행됩니다).")


if __name__ == "__main__":
    main()
