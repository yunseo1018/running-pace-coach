# -*- coding: utf-8 -*-
"""
P1-3 3번: T5(누적 거리 5.000km 지점을 선형 보간한 경과 시간)의 **남는 오차 원천** = 평가 대상의 기준 기록·정답 세션에서 보간에 쓴 두 점 사이의 시간 간격과 누적 거리 범위.
기록 변화(초)·T5 값의 분포는 계산하지 않는다. 집계값만 출력·저장하고 ID·좌표·점 값은 싣지 않는다.
- 평가 대상(2,072개)의 기준 기록·정답 세션을 42번 build_dataset와 같은 논리로 모은다(고유 세션 수 2,070 / 2,072 / 합집합 3,174 assert).
- 원본 JSON에서 해당 세션의 점 기록을 읽어 38번 compute_t5와 같은 규칙(37번 cum_with_skip)으로 누적 거리를 만들고, 5.000km에 처음 도달하는 두 점(j-1, j)을 찾는다.
  T5가 data/t5_main_all.pkl 캐시(49c02f8 이후 43번이 다시 계산해 저장, 같은 코드)와 같음을 assert한다.
출력: docs/sample_reduction_5km.md의 <!-- INTERP_START -->~<!-- INTERP_END --> 블록만 갱신.
사용법(저장소 루트): python scripts/50_interp_gap_5km.py   (run_all에는 넣지 않음; 원본 JSON 필요, 약 4분)
"""
import ast
import importlib.util
import os
import re
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


m43 = _load("43_within_person_5km.py", "m43")
m42, m38, feas = m43.m42, m43.m38, m43.feas
md_table = m42.md_table
DOC = Path("docs") / "sample_reduction_5km.md"
START, END = "<!-- INTERP_START -->", "<!-- INTERP_END -->"
WEEK = feas.WEEK


def collect_targets(d, grp, cache):
    ext_by_sid = cache["ext"]
    ref_ids, ans_ids = set(), set()
    n = 0
    for uid, g in d.groupby("user_id", sort=True):
        if int(uid) not in grp:
            continue
        s = g["start_time"].to_numpy(np.int64)
        e = s + g["duration_sec"].to_numpy(np.int64)
        iscand = g["cand"].to_numpy()
        sids = g["session_id"].to_numpy()
        ext = np.array([ext_by_sid.get(int(x), False) for x in sids])
        for k in feas.span_ks({"s": s}, m38.MONDAY0):
            t = m38.MONDAY0 + k * WEEK
            inwin = (e >= t - m38.WIN) & (e < t)
            if int(inwin.sum()) < 4:
                continue
            cw = np.flatnonzero(inwin & iscand)
            valid = cw[~ext[cw]]
            if len(valid) == 0:
                continue
            fut = np.flatnonzero(iscand & (s >= t) & (s < t + WEEK))
            if len(fut) == 0:
                continue
            ref_ids.add(int(sids[int(valid[-1])]))
            ans_ids.add(int(sids[int(fut[0])]))
            n += 1
    return ref_ids, ans_ids, n


def stats(name, ids, rec):
    x = pd.DataFrame([rec[i] for i in sorted(ids)])
    dt, gap_m = x.dt.to_numpy(float), (x.d_after - x.d_before).to_numpy(float) * 1000
    before_m, after_m = (5.0 - x.d_before).to_numpy(float) * 1000, (x.d_after - 5.0).to_numpy(float) * 1000
    return {"대상": name, "세션 수": f"{len(x):,}",
            "점 사이 시간 간격(초) 중앙값 / 90% / 최대": f"{np.median(dt):.0f} / {np.percentile(dt, 90):.0f} / {dt.max():.0f}",
            "간격 10초 이하 비율": f"{100 * (dt <= 10).mean():.1f}%", "간격 30초 초과 세션": f"{int((dt > 30).sum())}",
            "점 사이 거리 간격(m) 중앙값 / 90% / 최대": f"{np.median(gap_m):.1f} / {np.percentile(gap_m, 90):.1f} / {gap_m.max():.1f}",
            "앞 점 누적 거리(km) 최소 ~ 최대": f"{x.d_before.min():.4f} ~ {x.d_before.max():.4f}",
            "뒤 점 누적 거리(km) 최소 ~ 최대": f"{x.d_after.min():.4f} ~ {x.d_after.max():.4f}",
            "5.000km − 앞 점(m) 90% / 최대": f"{np.percentile(before_m, 90):.1f} / {before_m.max():.1f}",
            "뒤 점 − 5.000km(m) 90% / 최대": f"{np.percentile(after_m, 90):.1f} / {after_m.max():.1f}"}


def main():
    d = config.load_main_sessions().sort_values(["user_id", "start_time"]).reset_index(drop=True)
    d["cand"] = (d["merged_count"] == 1) & d["actual_distance_km"].between(m38.LO, m38.HI)
    tr = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    te = {int(x) for x in Path(config.TEST_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    grp = {int(u): ("train" if int(u) in tr else ("test" if int(u) in te else "other")) for u in d.user_id.unique()}
    cache = pd.read_pickle(Path("data") / "t5_main_all.pkl")
    ref_ids, ans_ids, n_targets = collect_targets(d, grp, cache)
    assert n_targets == 2072 and (len(ref_ids), len(ans_ids), len(ref_ids | ans_ids)) == (2070, 2072, 3174), (n_targets, len(ref_ids), len(ans_ids))
    need = ref_ids | ans_ids
    start_of = {int(s): int(t) for s, t in zip(d.session_id, d.start_time)}
    parse = feas.import_parser()
    rec = {}
    with open("endomondoHR.json", "rb") as f:
        for line in f:
            m = feas.ID_RE.search(line)
            if not m or int(m.group(1)) not in need:
                continue
            sid = int(m.group(1))
            r = ast.literal_eval(line.decode("utf-8"))
            if r.get("sport") != "run" or r["id"] != sid:
                continue
            lat, lon, ts = r["latitude"], r["longitude"], r["timestamp"]
            assert ts[0] == start_of[sid]
            cum, skipped = feas.cum_with_skip(parse, lat, lon, ts)
            j = int(np.searchsorted(cum, 5.0, "left"))
            assert 1 <= j < len(cum) and cum[j - 1] < 5.0 <= cum[j]
            frac = (5.0 - cum[j - 1]) / (cum[j] - cum[j - 1])
            t5 = (ts[j - 1] + frac * (ts[j] - ts[j - 1])) - ts[0]
            assert abs(t5 - cache["t5"][sid]) < 1e-9, "T5가 캐시와 다름"
            assert not skipped[j], "5.000km 도달 지점이 스킵 구간 안"
            rec[sid] = dict(dt=float(ts[j] - ts[j - 1]), d_before=float(cum[j - 1]), d_after=float(cum[j]))
            del lat, lon, ts, cum, r
    assert set(rec) == need, (len(rec), len(need))

    rows = [stats("기준 기록 세션", ref_ids, rec), stats("정답 세션", ans_ids, rec), stats("합집합(기준 기록 ∪ 정답)", need, rec)]
    doc5 = Path("docs/week5_target_check.md").read_text(encoding="utf-8")
    mm = re.search(r"\| 5\.000km를 가로지르는 구간\(병합되지 않은 5\.0km 이상 세션\) \| ([\d,]+) \| (\d+) \| (\d+) \|", doc5)
    n5, med5, p90_5 = mm.groups()
    allr = pd.DataFrame(list(rec.values()))
    pooled_gap = (allr.d_after - allr.d_before).to_numpy(float) * 1000
    block = [START, "",
             "평가 대상 2,072개의 기준 기록 세션(고유 2,070개)·정답 세션(고유 2,072개)의 5.000km 도달 구간(보간에 쓴 두 점)을 원본 점 기록에서 다시 찾아 집계했다(`scripts/50_interp_gap_5km.py`; "
             "T5가 `data/t5_main_all.pkl`과 같음, 도달 지점이 스킵 구간 안이 아님을 모든 세션에서 assert로 확인). 기록 변화(초)·T5 값의 분포는 계산하지 않았다.", "",
             md_table(rows), "",
             f"**기존 week5 수치와 비교**(`docs/week5_target_check.md` A-3, 병합되지 않은 5.0km 이상 후보 {n5}개): 5.000km를 가로지르는 구간의 시간 간격 중앙값 {med5}초, 90% 분위 {p90_5}초. "
             f"평가 대상 세션의 합집합({len(need):,}개)은 중앙값 {np.median(allr.dt):.0f}초, 90% 분위 {np.percentile(allr.dt, 90):.0f}초, 최대 {allr.dt.max():.0f}초로 같은 수준이다(분위수는 `np.percentile` 선형 보간). "
             "week5 문서에는 최대 값이 없어 최대는 비교하지 않았다.", "",
             "해석: T5의 선형 보간 오차는 도달 지점이 위 표의 두 점 사이라는 사실로 상한이 정해진다. 점 사이를 직선·등속으로 가정하므로 간격이 길수록 도달 시각이 덜 정밀하다. "
             "이 오차는 정답 세션과 기준 기록 세션 모두에 같은 방식으로 있으며, 위 표는 세션별 간격의 분포일 뿐 이 오차가 기록 변화(y)에 얼마나 반영되는지는 계산하지 않았다.", "",
             END]
    text = "\n".join(block)
    doc = DOC.read_text(encoding="utf-8")
    assert START in doc and END in doc, "문서에 INTERP 마커가 없음(먼저 3절 자리를 만들 것)"
    doc = doc[:doc.index(START)] + text + doc[doc.index(END) + len(END):]
    DOC.write_text(doc, encoding="utf-8", newline="\n")
    print("saved", DOC)
    for r in rows:
        print({k: v for k, v in r.items() if k in ("대상", "세션 수", "점 사이 시간 간격(초) 중앙값 / 90% / 최대", "점 사이 거리 간격(m) 중앙값 / 90% / 최대")})
    print("week5:", n5, med5, p90_5)


if __name__ == "__main__":
    main()
