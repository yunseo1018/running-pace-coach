# -*- coding: utf-8 -*-
"""
P2-3 극단 정답 2개의 원자료 점검(집계와 서술만): cold-start test 평가 대상 352개 중 정답 세션이 극단(T5 > 5,000초)인 2개(S2에서 제거되는 행).
사례마다: 거리 단위 이상 여부, 같은 사용자의 시간이 겹치는 세션(중복) 여부, 정지 구간 총 길이(속도 1km/h 미만이 이어지는 구간), 점 간격 최대값.
판정은 "명백한 오류" 또는 "원인 미확인" 중 하나로만(아래 RULE). 값·ID·좌표는 출력·저장하지 않는다(파생 길이·개수만).
정의(실행 전 고정):
- 점 사이 구간 i: 거리 seg_m(haversine), 시간 dt. dt <= 0 이거나 암시 속도 > 25/3.6 m/s 구간은 01번과 같이 거리를 0으로 누적(스킵).
- 느린 구간: dt > 0이고 암시 속도 < 1 km/h인 구간. 정지 구간 = 연속한 느린 구간의 묶음. "이어지는" 기준: 묶음의 시간 합이 60초 이상인 것만 정지 구간으로 센다.
  보고: (i) 정지 구간 총 길이(60초 이상 묶음의 시간 합), (ii) 가장 긴 정지 구간, (iii) 느린 구간 시간 합 전체(60초 미만 묶음 포함). 세션 전체와 5.000km 도달 지점까지(T5 구간)를 따로 낸다.
- 점 간격 최대값: 연속 점 사이 dt의 최대(세션 전체, T5 구간).
- 거리 단위 이상: (재계산 총 거리 != 세션 표 거리(1e-9km 초과) 또는 점 사이 거리 최대 > 1km) 중 하나라도 해당하면 "이상 있음". 스킵 구간(시간 역전·GPS 점프) 개수는 참고 값으로만 싣는다.
  [정정 이력] 첫 실행에서는 스킵 구간 존재도 "이상"으로 셌다. 결과(두 사례 모두 스킵 3~4개, 재계산 거리 일치, 점 사이 거리 최대 33~81m)를 본 뒤, 시간 역전·점프는 정상 기록에도 있을 수 있어 거리 단위 판정 기준으로 부적절하다고 보고 분리했다. 이 변경은 결과를 본 뒤의 변경이므로 문서에 이력을 적는다.
- 중복: 같은 사용자의 다른 세션과 시간 구간이 겹침(정제 전 세션 표 data/run_sessions.pkl과 메인 세션 표 각각).
RULE: 판정 = "명백한 오류"는 (거리 단위 이상 있음) 또는 (메인 세션 표에서 같은 사용자의 다른 세션과 시간이 겹침)일 때만. 그 밖에는 "원인 미확인"(긴 정지·큰 점 간격은 휴식·일시정지·기록 중단일 수 있어 오류로 단정하지 않는다).
출력: docs/extreme_answers_5km.md. 사용법(저장소 루트): python scripts/52_extreme_answers_5km.py (run_all에는 넣지 않음, 원본 JSON 필요)
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
DOC = Path("docs") / "extreme_answers_5km.md"
WEEK = feas.WEEK
SLOW_KMH, STOP_MIN_SEC = 1.0, 60.0


def find_extreme_answer_sessions(d, grp, cache, te):
    ext_by_sid = cache["ext"]
    out = []                       # (uid, ans_sid)
    for uid, g in d.groupby("user_id", sort=True):
        uid = int(uid)
        if uid not in te:
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
            if len(cw[~ext[cw]]) == 0:
                continue
            fut = np.flatnonzero(iscand & (s >= t) & (s < t + WEEK))
            if len(fut) == 0:
                continue
            a = int(fut[0])
            if ext[a]:
                out.append((uid, int(sids[a])))
    return out


def overlap_count(df, uid, sid, start, end):
    u = df[(df.user_id == uid) & (df.session_id != sid)]
    s2 = u.start_time.to_numpy(np.int64)
    e2 = s2 + u.duration_sec.to_numpy(np.int64)
    return int(((s2 < end) & (e2 > start)).sum())


def analyze(parse, rec, table_km, d_raw, d_main, uid, sid):
    lat, lon, ts = rec["latitude"], rec["longitude"], rec["timestamp"]
    n = len(lat)
    seg_m, dts, skipped = [], [], []
    cum = [0.0]
    for i in range(1, n):
        sm = parse.haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i]) * 1000.0
        dt = ts[i] - ts[i - 1]
        bad = dt <= 0 or sm / dt > parse.MAX_PLAUSIBLE_SPEED_MPS
        seg_m.append(sm)
        dts.append(dt)
        skipped.append(bool(bad))
        cum.append(cum[-1] + (0.0 if bad else sm / 1000.0))
    cum = np.array(cum)
    seg_m, dts, skipped = np.array(seg_m), np.array(dts, float), np.array(skipped)
    j = int(np.searchsorted(cum, 5.0, "left"))
    assert 1 <= j < len(cum)
    kmh = np.where(dts > 0, seg_m / np.where(dts > 0, dts, 1) * 3.6, np.inf)
    slow = (kmh < SLOW_KMH) & (dts > 0)

    def runs(upto):                 # upto = 구간 개수(포함 범위: 0..upto-1)
        total_all, total_stop, longest = 0.0, 0.0, 0.0
        i = 0
        while i < upto:
            if slow[i]:
                k = i
                tot = 0.0
                while k < upto and slow[k]:
                    tot += dts[k]
                    k += 1
                total_all += tot
                if tot >= STOP_MIN_SEC:
                    total_stop += tot
                    longest = max(longest, tot)
                i = k
            else:
                i += 1
        return total_all, total_stop, longest
    full = runs(len(dts))
    part = runs(j)                  # 5.000km 도달 지점(점 j)까지의 구간 0..j-1
    start, end = int(ts[0]), int(ts[-1])
    anomaly = (abs(cum[-1] - table_km) > 1e-9) or (seg_m.max() > 1000.0)
    return dict(n_points=n, table_km=table_km, recomputed_equal=abs(cum[-1] - table_km) <= 1e-9, max_step_m=float(seg_m.max()), n_skipped=int(skipped.sum()),
                anomaly=anomaly, gap_max_full=float(dts.max()), gap_max_t5=float(dts[:j].max()), stop_full=full, stop_t5=part,
                t5_elapsed_over_extreme=True, overlap_raw=overlap_count(d_raw, uid, sid, start, end), overlap_main=overlap_count(d_main, uid, sid, start, end))


def main():
    d = config.load_main_sessions().sort_values(["user_id", "start_time"]).reset_index(drop=True)
    d["cand"] = (d["merged_count"] == 1) & d["actual_distance_km"].between(m38.LO, m38.HI)
    tr = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    te = {int(x) for x in Path(config.TEST_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    grp = {int(u): ("train" if int(u) in tr else ("test" if int(u) in te else "other")) for u in d.user_id.unique()}
    cache = pd.read_pickle(Path("data") / "t5_main_all.pkl")
    found = find_extreme_answer_sessions(d, grp, cache, te)
    assert len(found) == 2, f"cold-start test에서 정답 극단 평가 대상은 2개여야 함: {len(found)}"
    d_raw = config.load_raw_reference_sessions()
    d_main = d
    want = {sid: uid for uid, sid in found}
    parse = feas.import_parser()
    res = {}
    with open("endomondoHR.json", "rb") as f:
        for line in f:
            m = feas.ID_RE.search(line)
            if not m or int(m.group(1)) not in want:
                continue
            sid = int(m.group(1))
            rec = ast.literal_eval(line.decode("utf-8"))
            if rec.get("sport") != "run" or rec["id"] != sid:
                continue
            tab = float(d_main.loc[d_main.session_id == sid, "actual_distance_km"].iloc[0])
            res[sid] = analyze(parse, rec, tab, d_raw, d_main, want[sid], sid)
            assert abs(float(cache["t5"][sid])) > 5000.0 and cache["ext"][sid]
            del rec
    assert set(res) == set(want)
    # personalized 평가의 극단 정답 1개가 이 2개 중 하나인지(개수만)
    ds = m43.main_dataset(d, grp, cache)
    _tr, p_ev, _u = m42.personalized_split(ds)
    n_pers_ext = int(p_ev.ans_extreme.sum())
    cold_ev = ds[ds.uid.isin(te)]
    assert int(cold_ev.ans_extreme.sum()) == 2
    names = ["사례 A", "사례 B"]
    order = sorted(res)
    rows = []
    for nm, sid in zip(names, order):
        r = res[sid]
        verdict = "명백한 오류" if (r["anomaly"] or r["overlap_main"] > 0) else "원인 미확인"
        rows.append((nm, r, verdict))

    def fmt_stop(x):
        return f"{x[1]:.0f}초 (가장 긴 {x[2]:.0f}초; 60초 미만 묶음 포함 느린 구간 합 {x[0]:.0f}초)"

    table = md_table([{"항목": "점 수", **{nm: f"{r['n_points']:,}" for nm, r, _v in rows}},
                      {"항목": "세션 표 거리와 재계산 거리(01번 규칙) 일치", **{nm: ("일치" if r["recomputed_equal"] else "불일치") for nm, r, _v in rows}},
                      {"항목": "점 사이 거리 최대(m) / 스킵 구간(점프·시간 역전) 개수(참고, 판정 기준 아님)", **{nm: f"{r['max_step_m']:.1f} / {r['n_skipped']}" for nm, r, _v in rows}},
                      {"항목": "거리 단위 이상 여부", **{nm: ("이상 있음" if r["anomaly"] else "이상 없음") for nm, r, _v in rows}},
                      {"항목": "시간이 겹치는 같은 사용자의 다른 세션: 정제 전 표 / 메인 표", **{nm: f"{r['overlap_raw']} / {r['overlap_main']}" for nm, r, _v in rows}},
                      {"항목": "정지 구간 총 길이, T5 구간(5.000km 지점까지)", **{nm: fmt_stop(r["stop_t5"]) for nm, r, _v in rows}},
                      {"항목": "정지 구간 총 길이, 세션 전체", **{nm: fmt_stop(r["stop_full"]) for nm, r, _v in rows}},
                      {"항목": "점 간격 최대값(초), T5 구간 / 세션 전체", **{nm: f"{r['gap_max_t5']:.0f} / {r['gap_max_full']:.0f}" for nm, r, _v in rows}},
                      {"항목": "**판정**", **{nm: f"**{v}**" for nm, _r, v in rows}}])
    sens = Path("docs/sensitivity_results_5km.md").read_text(encoding="utf-8")
    mm = re.search(r"\| 주 분석\(49c02f8, 참고\) \| cold-start test \| [^|]+\| ([\d.]+) \|", sens)
    s2 = re.search(r"\| S2 [^|]+\| cold-start test \| [^|]+\| ([\d.]+) \|", sens)
    out = ["# 극단 정답 2개의 원자료 점검 (P2-3, 집계와 서술만)", "",
           "생성: `scripts/52_extreme_answers_5km.py`. 대상은 cold-start test 평가 대상 352개 중 **정답 세션이 극단(T5 > 5,000초)인 2개**(민감도 S2에서 제거되는 행)다. "
           f"personalized 평가에는 극단 정답이 {n_pers_ext}개 있다(`docs/sensitivity_results_5km.md`). 값·ID·좌표는 싣지 않았고 파생 길이·개수만 적었다. 결과 숫자·정의·모형은 바꾸지 않았다. 시각은 start_time 원값(기준 미확인)이다.", "",
           "정의(실행 전 고정, 스크립트 docstring): 점 사이 구간의 속도 < 1km/h가 이어지는 구간을 정지 구간으로 보고(시간 합이 60초 이상인 묶음만 \"정지 구간\"으로 세고, 60초 미만 묶음까지 합한 값을 괄호에 병기), "
           "점 간격 최대값은 연속 점 사이 시간의 최대다. 판정은 \"명백한 오류\"(거리 단위 이상이거나 메인 세션 표에서 같은 사용자의 다른 세션과 시간이 겹칠 때) 또는 \"원인 미확인\" 중 하나로만 한다. "
           "긴 정지나 큰 점 간격은 휴식·일시정지·기록 중단일 수 있어 오류로 단정하지 않는다. "
           "**규칙 정정 이력**: 첫 실행에서는 스킵 구간(시간 역전·GPS 점프) 존재도 \"거리 단위 이상\"으로 세어 두 사례 모두 \"명백한 오류\"로 나왔다. 두 사례의 재계산 거리가 세션 표와 일치하고 점 사이 거리 최대가 작아(아래 표) 스킵 구간은 정상 기록에도 있을 수 있으므로 판정 기준에서 빼고 참고 값으로 옮겼다. 이는 결과를 본 뒤의 변경이다.", "",
           "## 1. 사례별 점검", "", table, "",
           "## 2. 판정과 해석", "",
           "- 위 판정은 규칙 적용 결과일 뿐 원인을 단정하지 않는다. T5는 경과 시간 기준(정지·휴식 포함)이며 정지 구간이 길수록 T5가 커진다. 정지 구간이 \"관측된 러닝 기록의 공백\"인지 신호 대기·일시정지인지는 데이터에서 구분할 수 없다.",
           f"- **235.1초는 주 결과로 유지한다**(cold-start 사용자 평균 MAE, B2 {mm.group(1) if mm else '235.117'}초, `docs/final_results_5km.md`). **205.1초(극단 정답 2개 제외 시 B2 {s2.group(1) if s2 else '205.102'}초)는 민감도 결과로만 쓰며** 235.1초를 대체하지 않는다(`docs/sensitivity_results_5km.md` S2).", "",
           "## 3. 시연에서 이상 기록을 거르는 규칙 제안 (구현하지 않음)", "",
           "- 원칙: **예측 시점에 관측 가능한 값, 곧 기준 기록 쪽 값만** 쓴다. 정답(미래) 값으로 거르지 않는다. 임계값은 train 안에서만 정하고(교차검증), test 결과는 \"탐색 결과\"로 표기한다.",
           "- 이미 구현된 것: 기준 기록 T5 > 5,000초(극단)이면 유효 5km에서 제외한다(`demo/coach_demo.py`의 `is_valid_5km`).",
           "- 제안(미구현): 기기에서 기준 기록 세션의 점 기록으로 (i) T5 구간 안 정지 구간 총 길이, (ii) T5 구간 안 점 간격 최대값을 계산해 입력 열로 받고, 둘 중 하나가 train 분포에서 정한 임계값을 넘으면 예측 대신 \"기준 기록에 긴 정지나 기록 공백이 있어 예측이 불안정할 수 있음\"이라고 경고한다(제외가 아니라 경고부터). 현재 입력 CSV에는 이 열이 없다.",
           "- 위 규칙의 효과(예: 평가 대상 중 몇 %가 걸러지는지, MAE가 어떻게 달라지는지)는 계산하지 않았다.", ""]
    DOC.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC)
    for nm, r, v in rows:
        print(nm, "| anomaly", r["anomaly"], "| overlap raw/main", r["overlap_raw"], r["overlap_main"], "| stop_t5 %.0f longest %.0f all %.0f" % (r["stop_t5"][1], r["stop_t5"][2], r["stop_t5"][0]),
              "| gap_max_t5 %.0f full %.0f" % (r["gap_max_t5"], r["gap_max_full"]), "|", v)


if __name__ == "__main__":
    main()
