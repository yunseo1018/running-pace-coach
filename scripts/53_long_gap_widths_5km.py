# -*- coding: utf-8 -*-
"""
P2-4 보간 간격이 30초를 넘는 세션(평가 대상의 기준 기록·정답 세션 합집합 3,174개 중 16개): T5가 가질 수 있는 범위(앞 점 시각 ~ 뒤 점 시각)의 폭 분포만 집계한다.
재분석은 하지 않는다(T5는 그대로, 폭 = 보간에 쓴 두 점의 시간 간격). ID·좌표·점 값·T5 값은 싣지 않고 폭의 분포만 싣는다.
출력: docs/sample_reduction_5km.md의 <!-- LONGGAP_START -->~<!-- LONGGAP_END --> 블록. 사용법(저장소 루트): python scripts/53_long_gap_widths_5km.py (run_all에는 넣지 않음, 원본 JSON 필요, 약 4분)
"""
import ast
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import config  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m50 = _load("scripts/50_interp_gap_5km.py", "m50")
m43, m38, feas = m50.m43, m50.m38, m50.feas
DOC = Path("docs") / "sample_reduction_5km.md"
START, END = "<!-- LONGGAP_START -->", "<!-- LONGGAP_END -->"
THRESH = 30.0


def main():
    d = config.load_main_sessions().sort_values(["user_id", "start_time"]).reset_index(drop=True)
    d["cand"] = (d["merged_count"] == 1) & d["actual_distance_km"].between(m38.LO, m38.HI)
    tr = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    te = {int(x) for x in Path(config.TEST_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    grp = {int(u): ("train" if int(u) in tr else ("test" if int(u) in te else "other")) for u in d.user_id.unique()}
    cache = pd.read_pickle(Path("data") / "t5_main_all.pkl")
    ref_ids, ans_ids, n_targets = m50.collect_targets(d, grp, cache)
    assert n_targets == 2072 and (len(ref_ids), len(ans_ids), len(ref_ids | ans_ids)) == (2070, 2072, 3174)
    need = ref_ids | ans_ids
    start_of = {int(s): int(t) for s, t in zip(d.session_id, d.start_time)}
    parse = feas.import_parser()
    widths, n_scanned = {}, 0
    with open("endomondoHR.json", "rb") as f:
        for line in f:
            m = feas.ID_RE.search(line)
            if not m or int(m.group(1)) not in need:
                continue
            sid = int(m.group(1))
            r = ast.literal_eval(line.decode("utf-8"))
            if r.get("sport") != "run" or r["id"] != sid:
                continue
            ts = r["timestamp"]
            assert ts[0] == start_of[sid]
            cum, _sk = feas.cum_with_skip(parse, r["latitude"], r["longitude"], ts)
            j = int(np.searchsorted(cum, 5.0, "left"))
            assert 1 <= j < len(cum)
            n_scanned += 1
            dt = float(ts[j] - ts[j - 1])
            if dt > THRESH:
                widths[sid] = (dt, dt / float(cache["t5"][sid]))     # 폭(초), 폭 / T5(비율)
            del r, ts, cum
    assert n_scanned == len(need) == 3174
    w = np.array([v[0] for v in widths.values()])
    rel = np.array([v[1] for v in widths.values()])
    assert len(w) == 16, f"30초 초과 세션 수가 문서(16개)와 다름: {len(w)}"
    bins = [(30, 60), (60, 120), (120, 300), (300, 600), (600, 10**9)]
    labels = ["30초 초과 ~ 60초", "60 ~ 120초", "120 ~ 300초", "300 ~ 600초", "600초 초과"]
    cnts = [int(((w > lo) & (w <= hi)).sum()) for lo, hi in bins]
    assert sum(cnts) == len(w)
    block = [START, "",
             f"- 보간 간격(앞 점 시각 ~ 뒤 점 시각의 폭)이 30초를 넘는 세션은 평가 대상의 기준 기록·정답 세션 합집합 3,174개 중 **{len(w)}개**다(3-1절 표와 같음). "
             f"T5가 가질 수 있는 범위의 폭은 중앙값 {np.median(w):.0f}초, 최소 {w.min():.0f}초, 최대 {w.max():.0f}초이고(25%·75% 분위수 {np.percentile(w, 25):.0f}·{np.percentile(w, 75):.0f}초), "
             f"구간별 개수는 " + ", ".join(f"{lb} {c}개" for lb, c in zip(labels, cnts)) + "다. "
             f"폭이 그 세션 T5에서 차지하는 비율은 중앙값 {100 * np.median(rel):.1f}%, 최대 {100 * rel.max():.1f}%다.",
             "- 재분석은 하지 않았다(T5·기록 변화는 그대로이고 폭의 분포만 집계). 이 세션들이 평가 결과(MAE 등)에 얼마나 영향을 주는지는 계산하지 않았다.", "",
             END]
    text = "\n".join(block)
    doc = DOC.read_text(encoding="utf-8")
    assert START in doc and END in doc
    doc = doc[:doc.index(START)] + text + doc[doc.index(END) + len(END):]
    DOC.write_text(doc, encoding="utf-8", newline="\n")
    print("saved", DOC, "| n", len(w), "width med %.0f min %.0f max %.0f" % (np.median(w), w.min(), w.max()), "| bins", cnts, "| rel med %.1f%% max %.1f%%" % (100 * np.median(rel), 100 * rel.max()))


if __name__ == "__main__":
    main()
