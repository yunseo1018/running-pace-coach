# -*- coding: utf-8 -*-
"""
P1-1 표본 감소 표: 사전 선언 초안의 평가 대상(4,362개 / 470명)에서 확정 정의의 평가 대상(2,072개 / 349명)까지 **정의를 한 단계씩** 바꾸며 개수만 센다.
기록 변화(초)·T5 값의 분포·평균은 계산하지 않는다(개수만). 과거 자격 탈락과 미래 관측 탈락을 구분해 표시한다.
단계:
  S0 초안: 5km 후보 = 길이 4.5~5.5km 세션(병합 세션 포함), 극단 처리 없음           (docs/week4_data_audit.md 4절, week5_target_check.md B-1)
  S1 거리 하한 5.0km: 5.0~5.5km(정확한 5.000km 지점이 세션 안에 있는 세션만)         (week5_target_check.md B-2)
  S2 병합 세션 제외: merged_count == 1                                              (predeclaration_5km_final.md 4항)
  S3 극단 제외: 기준 기록은 T5 <= 5,000초(극단 아님), 정답 세션은 극단도 허용           (predeclaration_5km_final.md 6·7·8항, = 확정 정의)
공통: 기준 시점 = 월요일 00:00 격자(start_time 원값), 입력 창 = [t-28일, t)에 끝난 달리기 4회 이상, 정답 후보 = 시작 시각 [t, t+7일).
극단 여부는 data/t5_main_all.pkl 캐시(43번이 만듦; ID 포함, git 제외)의 값을 쓰고 개수에만 쓴다.
검산: S0 = 4,362/470(전체)·3,541/379(train)·821/91(test), S1 = 2,127/355, S3 = 2,072/349 (docs의 값과 assert로 일치 확인).
출력: docs/sample_reduction_5km.md (표 부분만; 나머지 절은 수기로 보탠 문서의 마커 사이를 갱신).
사용법(저장소 루트): python scripts/48_sample_reduction_5km.py
"""
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


m42 = _load("42_final_eval_5km.py", "m42")
m38, feas = m42.m38, m42.feas
WEEK = feas.WEEK
DOC = Path("docs") / "sample_reduction_5km.md"
START, END = "<!-- TABLE_START -->", "<!-- TABLE_END -->"


def md_table(rows):
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main():
    d = config.load_main_sessions().sort_values(["user_id", "start_time"]).reset_index(drop=True)
    cache = pd.read_pickle(Path("data") / "t5_main_all.pkl")
    ext_by_sid = cache["ext"]
    tr = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    te = {int(x) for x in Path(config.TEST_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    steps = [("S0", "초안: 4.5~5.5km 후보(병합 포함), 극단 처리 없음", 4.5, 5.5, False, False),
             ("S1", "후보 길이 하한 4.5 → 5.0km", 5.0, 5.5, False, False),
             ("S2", "병합 세션을 후보에서 제외", 5.0, 5.5, True, False),
             ("S3", "극단(T5 > 5,000초) 후보를 기준 기록에서 제외 (= 확정 정의)", 5.0, 5.5, True, True)]
    # 사용자별 월요일 격자, 4회↑ 기준 시점은 단계와 무관 → 한 번 계산
    users = {}
    for uid, g in d.groupby("user_id", sort=True):
        uid = int(uid)
        s = g["start_time"].to_numpy(np.int64)
        e = s + g["duration_sec"].to_numpy(np.int64)
        km = g["actual_distance_km"].to_numpy()
        mc = g["merged_count"].to_numpy()
        sids = g["session_id"].to_numpy()
        ext = np.array([ext_by_sid.get(int(x), False) for x in sids])
        users[uid] = dict(s=s, e=e, km=km, mc=mc, ext=ext, grp="train" if uid in tr else ("test" if uid in te else "other"))
    res = {}
    n_total = {"전체": [0, set()], "train": [0, set()], "test": [0, set()]}
    n_win4 = {"전체": [0, set()], "train": [0, set()], "test": [0, set()]}
    for name, label, lo, hi, nonmerged, noext in steps:
        cnt = {g: {"p": [0, set()], "pf": [0, set()]} for g in ("전체", "train", "test")}
        for uid, u in users.items():
            s, e, km, mc, ext = u["s"], u["e"], u["km"], u["mc"], u["ext"]
            cand = (km >= lo) & (km <= hi)
            if nonmerged:
                cand &= mc == 1
            for k in feas.span_ks({"s": s}, m38.MONDAY0):
                t = m38.MONDAY0 + k * WEEK
                if name == "S0":
                    groups = ["전체"] + ([u["grp"]] if u["grp"] in ("train", "test") else [])
                    for gname in groups:
                        n_total[gname][0] += 1
                        n_total[gname][1].add(uid)
                inwin = (e >= t - m38.WIN) & (e < t)
                if int(inwin.sum()) < 4:
                    continue
                groups = ["전체"] + ([u["grp"]] if u["grp"] in ("train", "test") else [])
                if name == "S0":
                    for gname in groups:
                        n_win4[gname][0] += 1
                        n_win4[gname][1].add(uid)
                cw = np.flatnonzero(inwin & cand)
                if noext:
                    cw = cw[~ext[cw]]
                if len(cw) == 0:
                    continue
                for gname in groups:
                    cnt[gname]["p"][0] += 1
                    cnt[gname]["p"][1].add(uid)
                fut = np.flatnonzero(cand & (s >= t) & (s < t + WEEK))
                if len(fut) == 0:
                    continue
                for gname in groups:
                    cnt[gname]["pf"][0] += 1
                    cnt[gname]["pf"][1].add(uid)
        res[name] = cnt
    # 검산
    exp = {("S0", "전체"): ((17164, 620), (4362, 470)), ("S0", "train"): ((13730, 496), (3541, 379)), ("S0", "test"): ((3434, 124), (821, 91)),
           ("S1", "전체"): ((11683, 558), (2127, 355)), ("S1", "train"): ((9282, 443), (1754, 283)), ("S1", "test"): ((2401, 115), (373, 72)),
           ("S3", "전체"): ((11363, 551), (2072, 349)), ("S3", "train"): ((9075, 438), (1720, 278)), ("S3", "test"): ((2288, 113), (352, 71))}
    for (nm, g), (p, pf) in exp.items():
        got_p = (res[nm][g]["p"][0], len(res[nm][g]["p"][1]))
        got_pf = (res[nm][g]["pf"][0], len(res[nm][g]["pf"][1]))
        assert got_p == p and got_pf == pf, (nm, g, got_p, p, got_pf, pf)
    assert (n_win4["전체"][0], len(n_win4["전체"][1])) == (38005, 710) and (n_total["전체"][0], len(n_total["전체"][1])) == (84875, 811)

    def cell(x):
        return f"{x[0]:,} / {len(x[1]):,}명"

    out = [START, "", "기준 시점 수 / 사용자 수. ② = 월요일 격자 중 직전 28일 안에 끝난 달리기 4회 이상(모든 단계 공통), ③ 과거 자격 = ②이면서 직전 28일 안에 기준 기록이 될 5km 후보가 1개 이상, ④ 평가 대상 = ③이면서 다음 7일 안 5km 후보가 1개 이상.", ""]
    for g in ("전체", "train", "test"):
        out += [f"### {g}", ""]
        rows = []
        prev = None
        for name, label, *_ in steps:
            c = res[name][g]
            p, pf = c["p"], c["pf"]
            n2 = n_win4[g]
            row = {"단계": f"{name} {label}", "② 4회↑": cell(n2), "③ 과거 자격 충족": cell(p), "④ 평가 대상": cell(pf),
                   "과거 자격 탈락(②→③)": f"{n2[0] - p[0]:,}", "미래 관측 탈락(③→④)": f"{p[0] - pf[0]:,}",
                   "직전 단계 대비 ④ 변화": "-" if prev is None else f"{pf[0] - prev[0]:+,}개 / {len(pf[1]) - len(prev[1]):+,}명"}
            rows.append(row)
            prev = pf
        out += [md_table(rows), ""]
    out += [END]
    block = "\n".join(out)
    if DOC.exists():
        doc = DOC.read_text(encoding="utf-8")
        if START in doc and END in doc:
            doc = doc[:doc.index(START)] + block + doc[doc.index(END) + len(END):]
        else:
            doc = doc.rstrip("\n") + "\n\n" + block + "\n"
    else:
        doc = block + "\n"
    DOC.write_text(doc, encoding="utf-8", newline="\n")
    print("saved", DOC)
    for g in ("전체",):
        for name, *_ in steps:
            c = res[name][g]
            print(name, g, "③", cell(c["p"]), "④", cell(c["pf"]))


if __name__ == "__main__":
    main()
