import os
os.environ["OPENAI_API_KEY"] = "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"

import olefile
import sys
sys.path.insert(0, ".")

from src.config import load_config
from src.agents.metadata_agent import extract_metadata

# 파일 경로
file_path = "data/files/경기도 안양시_호계체육관 배드민턴장 및 탁구장 예약시스템 구축 용역.hwp"

# HWP 파일 읽기
ole = olefile.OleFileIO(file_path)
data = ole.openstream("PrvText").read()
text = data.decode("utf-16-le", errors="ignore")
ole.close()

# 메타데이터 추출 테스트
cfg = load_config("config.yaml")
meta = extract_metadata(text, cfg)
print(meta)

print("텍스트 앞 200자:", text[:200])