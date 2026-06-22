#!/usr/bin/env bash
# 백엔드(FastAPI)와 프론트(Streamlit)를 한 번에 실행한다.
# 사용: ./run.sh   (또는 bash run.sh)
set -e

BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
export BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"

# 1) 백엔드 기동(백그라운드)
echo "[run.sh] 백엔드 기동: ${BACKEND_URL}"
python -m uvicorn backend.app:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
BACKEND_PID=$!

# 종료(Ctrl+C) 시 백엔드도 함께 정리
trap 'echo "[run.sh] 종료, 백엔드 정리"; kill ${BACKEND_PID} 2>/dev/null' EXIT

# 2) 백엔드 헬스 대기
echo "[run.sh] 백엔드 준비 대기..."
for i in $(seq 1 40); do
  if curl -s "${BACKEND_URL}/health" >/dev/null 2>&1; then
    echo "[run.sh] 백엔드 준비 완료"
    break
  fi
  sleep 1
done

# 3) 프론트 실행(포그라운드). 종료하면 trap이 백엔드도 정리한다.
echo "[run.sh] Streamlit 실행 (기본 http://localhost:8501)"
streamlit run src/streamlit/app.py
