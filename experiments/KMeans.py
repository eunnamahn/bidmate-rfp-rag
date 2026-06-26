import sys, pickle, importlib.util
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

BASE      = '/content/drive/MyDrive/codeit_mission/중급프로젝트'
CSV_PATH  = '/content/drive/MyDrive/codeit_mission/원본 데이터/data_list.csv'
FILES_DIR = '/content/drive/MyDrive/codeit_mission/원본 데이터/files'

if BASE not in sys.path:
    sys.path.insert(0, BASE)

spec = importlib.util.spec_from_file_location("data.config", f"{BASE}/data/config.py")
config = importlib.util.module_from_spec(spec)
sys.modules["data.config"] = config
spec.loader.exec_module(config)
config.CHUNK_SIZE=1000; config.CHUNK_OVERLAP=100; config.CHUNK_STRATEGY="recursive"
config.HWP_PARSER="olefile"; config.PDF_PARSER="pdfplumber"
config.REMOVE_TOC=True; config.REMOVE_HWP_TAGS=True

spec2 = importlib.util.spec_from_file_location("preprocessor", f"{BASE}/experiments/preprocessor.py")
preprocessor = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(preprocessor)

chunks = preprocessor.run_pipeline(CSV_PATH, FILES_DIR)
print(f"총 청크: {len(chunks)}개")

doc_texts = defaultdict(list)
for c in chunks:
    doc_texts[c['metadata']['파일명']].append(c['text'])

doc_list   = list(doc_texts.keys())
doc_corpus = [" ".join(doc_texts[d]) for d in doc_list]

tfidf  = TfidfVectorizer(max_features=3000)
X      = tfidf.fit_transform(doc_corpus)
km     = KMeans(n_clusters=20, random_state=42, n_init=10)
labels = km.fit_predict(X)

selected_docs = []
for i in range(20):
    idxs  = np.where(labels == i)[0]
    dists = [np.linalg.norm(X[j].toarray() - km.cluster_centers_[i]) for j in idxs]
    selected_docs.append(doc_list[idxs[np.argmin(dists)]])

print(f"\n선정 문서 {len(selected_docs)}개:")
for d in selected_docs:
    print(f"  - {d}")