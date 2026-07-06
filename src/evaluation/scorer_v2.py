# -*- coding: utf-8 -*-
"""v2 하이브리드 채점기 — 유형별 정답률(macro 100점) + 보조 지표.

[채점 설계]
  단답    : 정규화 substring
  숫자    : 값 파싱 비교 (47,000천원 = 47,000,000, 날짜/비율 포맷 무시)
  계약조건 : ① 값/코드 매칭  ② 정규화 substring  우선  +  ③ 의미유사도 보조(높은 임계 0.80)
            → 법령·업종코드·금액·계약방식·비율은 정확히, 애매한 것만 의미 보조 (과대평가 방지)
  요약    : required_facts 의미매칭(fact ↔ 답변 '문장' 최대 유사도) 70% + 전체 의미유사도 30%
  거절    : 거절 표현 O  +  구체정보(숫자·금액) 비추측 O   (환각 방지)

메인 점수 = 5유형 정확도의 macro average × 100
보조 지표 = hit@k · MRR · 근거포함(context recall) · 숫자환각률 · latency  (탈락/동점 처리용, 점수엔 미포함)

* 의미매칭은 '고정 임베딩'(bge-m3)으로 계산 → 조합마다 임베딩이 달라도 채점 기준은 동일(공정).
* Embedder=None 이면 의미매칭 부분은 건너뜀(=substring/값 매칭만) → 오프라인 로직 점검용.
"""
import re

# ────────────────────────────────────────────────────────────
# 0. 유틸 : 정규화 / 숫자 파싱 / 문장 분리 / 거절 표현
# ────────────────────────────────────────────────────────────

def norm(s):
    """소문자화 + 공백·기호 제거 (한글·영문·숫자만). substring 비교용."""
    return re.sub(r"[^0-9a-z가-힣]", "", str(s).lower())


def nums(s):
    """금액/비율/코드/일수/날짜 숫자를 '값 집합'으로 정규화. 포맷 차이를 흡수한다.
    예) '47,000천원'→47000000, '20억'→2000000000, '90%'→'90%', '2025.2월'→{2025,2}
    """
    t = str(s).replace(",", "").replace(" ", "")
    vals, masked = set(), t
    for pat, mul in [(r"(\d+(?:\.\d+)?)억", 1e8),
                     (r"(\d+(?:\.\d+)?)천원", 1e3),
                     (r"(\d+(?:\.\d+)?)만원?", 1e4)]:
        for m in re.finditer(pat, t):
            vals.add(int(round(float(m.group(1)) * mul)))
        masked = re.sub(pat, " ", masked)
    for m in re.finditer(r"(\d+)%", masked):
        vals.add(m.group(1) + "%")
    masked = re.sub(r"\d+%", " ", masked)
    for m in re.finditer(r"\d+", masked):          # 남은 일반 숫자(원/일수/점수/코드/연월일)
        vals.add(int(m.group(0)))
    return vals


def sentences(s):
    """답변을 문장 단위로 분리 (요약 fact↔문장 최대유사도용)."""
    parts = re.split(r"[.\n!?·•]|(?<=다)\s|(?<=음)\s|(?<=함)\s", str(s))
    return [p.strip() for p in parts if len(p.strip()) >= 2]


# '모른다/없다' 계열 거절 표현
REFUSAL_PHRASES = [
    "확인할 수 없", "확인이 어렵", "찾을 수 없", "정보가 없", "내용이 없",
    "명시되어 있지 않", "명시되지 않", "나와 있지 않", "나와있지 않",
    "언급되지 않", "언급이 없", "제공되지 않", "제시되어 있지 않",
    "알 수 없", "문서에 없", "확인되지 않", "포함되어 있지 않",
]


# ────────────────────────────────────────────────────────────
# 1. 고정 임베더 (의미매칭용) — 조합과 무관하게 동일 기준
# ────────────────────────────────────────────────────────────

class Embedder:
    """bge-m3 고정 임베딩으로 코사인 유사도. normalize=True 라 dot=cos. 인코딩 캐시."""

    def __init__(self, model="BAAI/bge-m3", device=None):
        import torch
        # 채점 임베더는 소량이라 CPU 권장(GPU VRAM은 retrieval 임베딩에 양보 → double-load 회피)
        from langchain_huggingface import HuggingFaceEmbeddings
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": dev, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )
        self._cache = {}

    def _vec(self, text):
        text = str(text)
        if text not in self._cache:
            self._cache[text] = self.hf.embed_query(text)
        return self._cache[text]

    def cos(self, a, b):
        import numpy as np
        va, vb = np.array(self._vec(a)), np.array(self._vec(b))
        return float(va @ vb)  # 이미 정규화됨

    def cos_max(self, a, sents):
        """fact a 와 답변 문장들 중 '최대' 유사도 (답변 전체와 비교 시 뻥튀기 방지)."""
        return max((self.cos(a, s) for s in sents), default=0.0)


# ────────────────────────────────────────────────────────────
# 2. required_fact 1개가 답변에 '포함'됐는지 (하이브리드)
# ────────────────────────────────────────────────────────────

def fact_hit(fact, ans, ans_sents, emb, cos_thresh):
    """① 값/코드 일치 → ② 정규화 substring → ③ 의미유사도 보조(임계) 순으로 판정."""
    fn = nums(fact)
    if fn and fn <= nums(ans):                       # ① 숫자·코드·비율이 답변에 다 있음
        return True
    if norm(fact) and norm(fact) in norm(ans):        # ② 핵심 문구가 그대로
        return True
    if emb is not None and ans_sents:                 # ③ 의미유사도(문장 최대) — backup
        if emb.cos_max(fact, ans_sents) >= cos_thresh:
            return True
    return False


# ────────────────────────────────────────────────────────────
# 3. 유형별 채점 (각 0~1, 채점 불가면 None)
# ────────────────────────────────────────────────────────────

def score_fact(gt, ans):
    """단답: 정규화 substring (숫자성 답이면 값 매칭도 허용)."""
    if not gt:
        return None
    if norm(gt) and norm(gt) in norm(ans):
        return 1.0
    g = nums(gt)
    return 1.0 if (g and g <= nums(ans)) else 0.0


def score_number(gt, ans):
    """숫자/금액/날짜/기간: 값 파싱 비교 (포맷 차이 무시). substring도 보조."""
    if not gt:
        return None
    g = nums(gt)
    if g and g <= nums(ans):
        return 1.0
    return 1.0 if (norm(gt) and norm(gt) in norm(ans)) else 0.0


def score_condition(facts, ans, emb):
    """계약조건: required_facts 포함률. exact/값 우선 + 의미유사도는 '높은 임계(0.80)' 보조."""
    if not facts:
        return None
    sents = sentences(ans)
    hit = sum(1 for f in facts if fact_hit(f, ans, sents, emb, cos_thresh=0.80))
    return hit / len(facts)


def score_summary(facts, gt, ans, emb):
    """요약: 핵심 사실 포함률 70% + 전체 의미유사도 30%. 포함률은 의미매칭(임계 낮게 0.62)."""
    sents = sentences(ans)
    cov = None
    if facts:
        hit = sum(1 for f in facts if fact_hit(f, ans, sents, emb, cos_thresh=0.62))
        cov = hit / len(facts)
    sim = emb.cos(gt, ans) if (emb is not None and gt) else None
    if cov is None and sim is None:
        return None
    if cov is None:
        return max(0.0, sim)
    if sim is None:
        return cov
    return 0.7 * cov + 0.3 * max(0.0, sim)


def score_refusal(ans, contexts):
    """거절: 거절 표현 O + 구체 숫자를 지어내지 않음 O. (둘 다여야 정답)"""
    refused = any(p in str(ans) for p in REFUSAL_PHRASES)
    ctx_nums = nums(" ".join(contexts)) if contexts else set()
    invented = nums(ans) - ctx_nums          # 답변 숫자 중 문맥에 없는 것 = 지어낸 것
    return 1.0 if (refused and not invented) else 0.0


# ────────────────────────────────────────────────────────────
# 4. 보조 지표 (탈락/동점용)
# ────────────────────────────────────────────────────────────

def hallucinated(ans, contexts):
    """답변의 숫자/금액이 검색 문맥에 없으면 환각으로 본다(True)."""
    if not contexts:
        return False
    return bool(nums(ans) - nums(" ".join(contexts)))


def evidence_in_ctx(gt, facts, contexts):
    """정답 근거가 검색된 문맥에 '얼마나' 포함됐는지 비율로 (진짜 context recall에 근접).
    요약/계약조건은 required_facts 각각을, 단답/숫자는 gt를 대상으로 본다.
    """
    if not contexts:
        return None
    nctx = norm(" ".join(contexts))
    cnums = nums(" ".join(contexts))
    targets = [t for t in (list(facts) if facts else [gt]) if t]
    if not targets:
        return None
    hit = sum(1 for t in targets
              if (norm(t) and norm(t) in nctx) or (nums(t) and (nums(t) & cnums)))
    return hit / len(targets)


# ────────────────────────────────────────────────────────────
# 5. 레코드 채점 라우팅 + 조합 집계
# ────────────────────────────────────────────────────────────

def score_record(rec, emb):
    """레코드 1개를 유형에 맞는 채점기로 → 0~1 (또는 None)."""
    t = rec.get("type")
    ans = rec.get("answer") or ""
    gt = rec.get("ground_truth") or ""
    facts = rec.get("required_facts") or []
    ctx = rec.get("contexts") or []
    if t == "단답":
        return score_fact(gt, ans)
    if t == "숫자":
        return score_number(gt, ans)
    if t == "계약조건":
        return score_condition(facts, ans, emb)
    if t == "요약":
        return score_summary(facts, gt, ans, emb)
    if t == "거절":
        return score_refusal(ans, ctx)
    return None


def aggregate(records, emb, extra=None):
    """레코드 리스트 → 스코어카드(유형별 정확도 + 통합 100점 + 보조 지표)."""
    from collections import defaultdict
    per = defaultdict(list)
    halluc, recall = [], []
    for r in records:
        s = score_record(r, emb)
        if s is not None:
            per[r.get("type")].append(s)
        ctx = r.get("contexts") or []
        if r.get("type") != "거절":            # 숫자환각률·근거포함은 '답변형'만(거절은 근거가 문맥에 없는 게 정상)
            if ctx:
                halluc.append(1.0 if hallucinated(r.get("answer", ""), ctx) else 0.0)
            ev = evidence_in_ctx(r.get("ground_truth"), r.get("required_facts"), ctx)
            if ev is not None:
                recall.append(ev)

    type_acc = {t: round(sum(v) / len(v), 3) for t, v in per.items() if v}
    composite = round(100 * sum(type_acc.values()) / len(type_acc), 1) if type_acc else 0.0
    card = {
        "통합점수": composite,                              # ← 메인 (macro average ×100)
        "유형별": type_acc,
        "숫자환각률": round(sum(halluc) / len(halluc), 3) if halluc else None,
        "근거포함": round(sum(recall) / len(recall), 3) if recall else None,
        "n": len(records),
    }
    if extra:
        card.update(extra)                                 # hit@k, mrr, latency 등
    return card


# ────────────────────────────────────────────────────────────
# 6. 러너 : 한 조합을 v2 골든셋으로 평가 → 스코어카드  (※ VM 터널 필요)
# ────────────────────────────────────────────────────────────

import os

# 골든셋은 이 파일과 같은 폴더, 레포 루트는 두 단계 위 — 경로를 레포 기준으로 고정(컴퓨터 무관)
_HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_V2 = os.path.join(_HERE, "golden_set_autoresearch_v2_q95.json")
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def load_golden_v2(path=GOLDEN_V2):
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _set(cfg, path, val):
    d = cfg
    ks = path.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = val


def _debug_dump(recs, emb):
    """문항별 진단: fact별 [답변포함 | 문맥포함] + 검색된 청크 미리보기.
    문맥X = 근거가 애초에 검색 안 됨(검색/청킹 문제) · 문맥O·답변X = 검색됐는데 LLM이 못 씀(모델 문제).
    """
    print("\n" + "=" * 62)
    print("[DEBUG] 문항별 진단  ([답변|문맥] O=포함 X=없음)")
    for r in recs:
        s = score_record(r, emb)
        ans = r["answer"]
        ctxs = r["contexts"]
        nctx = norm(" ".join(str(c) for c in ctxs))
        cnums = nums(" ".join(str(c) for c in ctxs))
        print("\n" + "-" * 62)
        print("유형=%s  점수=%s" % (r["type"], s))
        print("질문:", r["question"])
        if r["ground_truth"]:
            print("정답:", r["ground_truth"])
        if r["required_facts"]:
            sents = sentences(ans)
            thr = 0.80 if r["type"] == "계약조건" else 0.62
            for f in r["required_facts"]:
                by_val = bool(nums(f) and nums(f) <= nums(ans))
                by_sub = bool(norm(f) and norm(f) in norm(ans))
                sem = emb.cos_max(f, sents) if (emb and sents) else 0.0
                ans_ok = by_val or by_sub or sem >= thr
                ctx_ok = bool((norm(f) and norm(f) in nctx)
                              or (nums(f) and (nums(f) & cnums)))
                print("   [답변%s|문맥%s] %s (의미%.2f/임계%.2f)"
                      % ("O" if ans_ok else "X", "O" if ctx_ok else "X", f, sem, thr))
        print("모델답변:", ans)
        for i, c in enumerate(ctxs):
            print("  ctx%d: %s" % (i, str(c)[:180].replace("\n", " ")))
    print("=" * 62)


def evaluate_combo(overrides, golden, emb, k=4, debug=False, save_records=None):
    """조합(overrides)을 v2 골든셋으로 평가 → 스코어카드.
    팀 파이프라인으로 답변을 생성(evaluate_config)하고, 그 레코드에 골든의 type/required_facts를
    질문 기준으로 붙인 뒤, 위 하이브리드 채점기로 점수를 낸다. (답변 생성이라 VM 터널 필요)
    """
    import os, sys, copy, time
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    from src.config import load_config, make_llm
    from src.pipeline import build_pipeline
    from src.evaluation.ragas_eval import evaluate_config
    import src.evaluation.ragas_eval as _RE

    cfg = load_config("config.yaml")
    cfg["parser"] = {"hwp": "olefile", "pdf": "pdfplumber"}
    for kk, vv in overrides.items():
        _set(cfg, kk, vv)

    gmap = {norm(g["question"]): g for g in golden}         # 질문 → 골든 메타

    cap = {}
    _orig = _RE.custom_metrics
    def _capture(records, kk):
        cap["records"] = list(records)
        return _orig(records, kk)
    _RE.custom_metrics = _capture
    try:
        _, retr = build_pipeline(cfg, mode="qa")
        llm = make_llm(cfg)
        t0 = time.time()
        rep = evaluate_config(retr, llm, golden, judge_llm=None, judge_emb=None, k=k)
    finally:
        _RE.custom_metrics = _orig

    recs = []
    for r in cap.get("records", []):
        g = gmap.get(norm(r.get("question", "")), {})
        recs.append({
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "ground_truth": r.get("ground_truth") or g.get("ground_truth", ""),
            "contexts": r.get("contexts") or r.get("retrieved_contexts") or [],
            "type": g.get("type"),
            "required_facts": g.get("required_facts") or [],
        })
    extra = {"hit@k": rep["custom"].get("hit@k"),
             "mrr": rep["custom"].get("mrr"),
             "latency": rep["operational"].get("avg_latency_s")}
    if debug:
        _debug_dump(recs, emb)
    if save_records:
        import json
        dump = [{"질문": r["question"], "유형": r["type"], "점수": score_record(r, emb),
                 "답변": r["answer"], "정답": r["ground_truth"],
                 "required_facts": r["required_facts"], "n_ctx": len(r["contexts"])}
                for r in recs]
        with open(save_records, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
    return aggregate(recs, emb, extra)


# ────────────────────────────────────────────────────────────
# 7. 오프라인 로직 점검 (터널·모델 없이 substring/값/거절 로직만 확인)
#    python scorer_v2.py  →  임베딩 없이 예시 채점 출력
# ────────────────────────────────────────────────────────────

def _demo():
    demo = [
        {"type": "단답", "ground_truth": "안양시", "answer": "발주기관은 안양시입니다.",
         "required_facts": [], "contexts": ["...안양시 발주..."]},
        {"type": "숫자", "ground_truth": "397,000,000원", "answer": "사업예산은 397,000,000원 입니다",
         "required_facts": [], "contexts": ["예산 397,000,000원"]},
        {"type": "숫자", "ground_truth": "계약체결일로부터 2025. 2월까지",
         "answer": "계약체결일로부터 2025년 2월까지입니다", "required_facts": [], "contexts": []},
        {"type": "계약조건", "ground_truth": "",
         "answer": "협상에 의한 계약이며 제한경쟁입찰입니다.",
         "required_facts": ["협상에 의한 계약", "제한경쟁입찰"], "contexts": []},
        {"type": "거절", "ground_truth": "문서에서 확인할 수 없습니다",
         "answer": "문서에서 확인할 수 없습니다.", "required_facts": [], "contexts": []},
        {"type": "거절", "ground_truth": "문서에서 확인할 수 없습니다",
         "answer": "확인할 수 없지만 추가 예산은 1억 원으로 보입니다.",  # 거절+지어냄 → 오답이어야
         "required_facts": [], "contexts": ["예산 관련 내용 없음"]},
    ]
    labels = ["danap", "num", "num_date", "cond", "refusal_ok", "refusal_halluc"]
    expected = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    actual = [score_record(r, emb=None) for r in demo]
    print("offline demo (substring/value/refusal only, no embedding)")
    for lb, ex, ac in zip(labels, expected, actual):
        print(f"  {lb:16} expected={ex}  actual={ac}  {'OK' if ex==ac else 'X'}")
    print("RESULT:", "PASS" if actual == expected else "CHECK NEEDED")
    print("(cond/summary semantic matching needs Embedder() - checked at real run)")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _demo()
