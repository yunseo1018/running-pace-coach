# -*- coding: utf-8 -*-
"""
사후 분석: B2가 흔들리는 지점(층별 점검). 사양: docs/posthoc_spec_b2_failure_5km.md (커밋 f5ba594, 분석 실행 전 고정).
- 규칙을 정하는 자료 = train: 사용자 단위 GroupKFold(5, seed 42) 폴드 밖 B2 예측·잔차. test cold-start(352개)는 같은 표에 "탐색 결과(이미 열람)"로 병기.
- 층(예측 시점에 관측 가능한 값만): (a) 과거 유효 5km 개수, (b) |T5(기준) - m|, (c) 기준 기록 종료 후 며칠, (d) 직전 28일 달리기 횟수, (e) 기준 기록 세션 안 정지 시간(속도 1km/h 미만 구간 합).
- 층마다: 평가 대상·사용자 수, B2·B0 사용자 평균 MAE, 차이(B2-B0), 명목 80% 범위 포함률, 차이·포함률의 사용자 단위 bootstrap 95% CI(2,000회, seed 42, 층 안 사용자 복원추출).
- 평가 대상 30개 미만 층은 "표본 부족"으로 표시만 한다. 시연 규칙(B2 식·파라미터·범위)은 바꾸지 않는다. ID·행 값은 싣지 않는다(집계만).
사용법(저장소 루트): python scripts/56_b2_failure_5km.py   (run_all에는 넣지 않음; 원본 JSON 필요, 약 4분)
출력: docs/b2_failure_5km.md
"""
import ast
import importlib.util
import json
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


m43 = _load("43_within_person_5km.py", "m43")
m55 = _load("scripts/55_interval_calibration_5km.py", "m55")
m42, m38, feas = m43.m42, m43.m38, m43.feas
md_table = m42.md_table
SPEC_COMMIT = "f5ba594"
DOC = Path("docs") / "b2_failure_5km.md"
N_BOOT, SEED, MIN_N = 2000, 42, 30
WEEK, DAY = feas.WEEK, feas.DAY
SLOW_KMH = 1.0


def collect_targets(d, grp, cache):
    """평가 대상(uid, t)별 기준 기록 세션 id와 종료 시각. 42번 build_dataset와 같은 논리."""
    ext_by_sid = cache["ext"]
    rows = []
    for uid, g in d.groupby("user_id", sort=True):
        uid = int(uid)
        if uid not in grp:
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
            if len(np.flatnonzero(iscand & (s >= t) & (s < t + WEEK))) == 0:
                continue
            ref = int(valid[-1])
            rows.append(dict(uid=uid, t=t, ref_sid=int(sids[ref]), ref_end=int(e[ref])))
    return pd.DataFrame(rows)


def stop_seconds(parse, sids):
    """기준 기록 세션별 정지 시간(초): 시간차 > 0이고 속도 < 1km/h인 점 사이 구간의 시간 합(세션 전체)."""
    want = set(sids)
    out = {}
    with open("endomondoHR.json", "rb") as f:
        for line in f:
            m = feas.ID_RE.search(line)
            if not m or int(m.group(1)) not in want:
                continue
            sid = int(m.group(1))
            r = ast.literal_eval(line.decode("utf-8"))
            if r.get("sport") != "run" or r["id"] != sid:
                continue
            lat, lon, ts = r["latitude"], r["longitude"], r["timestamp"]
            tot = 0.0
            for i in range(1, len(lat)):
                dt = ts[i] - ts[i - 1]
                if dt <= 0:
                    continue
                seg_m = parse.haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i]) * 1000.0
                if seg_m / dt * 3.6 < SLOW_KMH:
                    tot += dt
            out[sid] = tot
            del r, lat, lon, ts
    assert set(out) == want, (len(out), len(want))
    return out


DIMS = [
    ("(a) 과거 유효 5km 개수", lambda df: pd.cut(df.n_past, [0, 1, 4, 10 ** 6], labels=["1개(x = 0)", "2~4개", "5개 이상"]), ["1개(x = 0)", "2~4개", "5개 이상"]),
    ("(b) |T5(기준) − m| (초)", lambda df: pd.cut(df.abs_x, [-1, 0, 30, 120, 10 ** 9], labels=["0초", "0초 초과 ~ 30초", "30초 초과 ~ 120초", "120초 초과"]),
     ["0초", "0초 초과 ~ 30초", "30초 초과 ~ 120초", "120초 초과"]),
    ("(c) 기준 기록이 t보다 며칠 전", lambda df: pd.cut(df.days_ref, [-1e-9, 7, 14, 28 + 1e-9], right=False, labels=["0~7일", "7~14일", "14~28일"]), ["0~7일", "7~14일", "14~28일"]),
    ("(d) 직전 28일 달리기 횟수", lambda df: pd.cut(df.n_runs, [0, 7, 15, 10 ** 6], labels=["4~7회", "8~15회", "16회 이상"]), ["4~7회", "8~15회", "16회 이상"]),
    ("(e) 기준 기록 세션 안 정지 시간 (초)", lambda df: pd.cut(df.stop_sec, [-1, 60, 300, 10 ** 9], labels=["0~60초", "60초 초과 ~ 300초", "300초 초과"]), ["0~60초", "60초 초과 ~ 300초", "300초 초과"]),
]


def stratum_stats(sub):
    """층 안 평가 대상(sub: uid, e2, e0, inside) -> 지표와 CI. 층의 사용자를 복원추출(2,000회, seed 42)."""
    users, inv = np.unique(sub.uid.to_numpy(), return_inverse=True)
    U = len(users)
    n = np.bincount(inv).astype(float)
    s2 = np.bincount(inv, weights=sub.e2.to_numpy(float))
    s0 = np.bincount(inv, weights=sub.e0.to_numpy(float))
    k = np.bincount(inv, weights=sub.inside.to_numpy(float))
    idx = np.random.default_rng(SEED).integers(0, U, size=(N_BOOT, U))
    dif_u = (s2 - s0) / n
    bd = dif_u[idx].mean(1)
    bc = k[idx].sum(1) / n[idx].sum(1)
    return dict(n=len(sub), users=U, mae2=float((s2 / n).mean()), mae0=float((s0 / n).mean()), diff=float(dif_u.mean()), diff_ci=tuple(np.percentile(bd, [2.5, 97.5])),
                cov=float(k.sum() / n.sum()), cov_ci=tuple(np.percentile(bc, [2.5, 97.5])))


def read_cell(s, kind):
    if kind == "diff":
        return f"{s['diff']:+.1f} [{s['diff_ci'][0]:+.1f}, {s['diff_ci'][1]:+.1f}]"
    return f"{100 * s['cov']:.1f}% [{100 * s['cov_ci'][0]:.1f}%, {100 * s['cov_ci'][1]:.1f}%]"


def main():
    d, trs, te, grp, cache = m43.load_base()
    ds = m43.main_dataset(d, grp, cache)
    m43.verify_against_final(ds, trs, te)
    tg = collect_targets(d, grp, cache)
    j = ds.merge(tg, on=["uid", "t"], how="left", validate="one_to_one")
    assert len(j) == 2072 and j.ref_sid.notna().all()
    ref_t5 = np.array([cache["t5"][int(s)] for s in j.ref_sid])
    assert np.array_equal(ref_t5, j.t5_ref.to_numpy()), "기준 기록 정합 실패"
    j["abs_x"] = j.t5_ref_minus_m.abs()
    j["days_ref"] = (j.t - j.ref_end) / DAY
    assert (j.days_ref >= 0).all() and (j.days_ref <= 28 + 1e-9).all()
    parse = feas.import_parser()
    stops = stop_seconds(parse, sorted(set(j.ref_sid.astype(int))))
    j["stop_sec"] = [stops[int(s)] for s in j.ref_sid]

    # ---- train 폴드 밖 B2
    tr = j[j.uid.isin(trs)].sort_values(["uid", "t"]).reset_index(drop=True)
    tr_file = pd.read_pickle(Path("data") / "5km_dataset_train.pkl").sort_values(["uid", "t"]).reset_index(drop=True)
    assert len(tr) == 1720 and np.array_equal(tr.uid.to_numpy(), tr_file.uid.to_numpy()) and np.array_equal(tr.t.to_numpy(), tr_file.t.to_numpy()) and np.array_equal(tr.y.to_numpy(), tr_file.y.to_numpy())
    q10, q90, _coefs, res = m55.oof_quantiles(tr_file)               # tr_file은 uid, t 순(55번과 같은 순서)
    p = json.loads((Path("demo") / "b2_params.json").read_text(encoding="utf-8"))
    assert abs(q10 - p["resid_q10"]) < 1e-9 and abs(q90 - p["resid_q90"]) < 1e-9, "시연 범위와 다름"
    y_tr = tr.y.to_numpy(float)
    yhat_oof = y_tr - res
    tr["e2"] = np.abs(y_tr - yhat_oof)
    tr["e0"] = np.abs(y_tr)
    tr["inside"] = (res >= q10) & (res <= q90)
    # ---- test cold-start (탐색 결과)
    ts_ = j[j.uid.isin(te)].sort_values(["uid", "t"]).reset_index(drop=True)
    assert len(ts_) == 352 and ts_.uid.nunique() == 71
    yhat = p["a"] + p["b"] * ts_.t5_ref_minus_m.to_numpy(float)
    y_te = ts_.y.to_numpy(float)
    ts_["e2"] = np.abs(y_te - yhat)
    ts_["e0"] = np.abs(y_te)
    ts_["inside"] = (y_te >= yhat + q10) & (y_te <= yhat + q90)
    sets = [("train 폴드 밖", tr), ("test cold-start (탐색 결과, 이미 열람)", ts_)]

    out = ["# 사후 분석 결과: B2가 흔들리는 지점 (층별 점검, 5km 과제)", "",
           f"생성: `scripts/56_b2_failure_5km.py`. 사양: `docs/posthoc_spec_b2_failure_5km.md`(커밋 {SPEC_COMMIT}, 분석 실행 전 고정). **사후 분석**이며 상관관계까지만 주장한다. "
           "규칙을 정하는 자료는 train 폴드 밖 B2 예측이고 test cold-start는 **탐색 결과(이미 열람)**다. 층 경계·정의는 결과를 본 뒤 바꾸지 않았고 **시연 규칙(B2 식·파라미터·범위·문구)을 바꾸지 않는다.** "
           "모든 평가는 \"다음 7일 안에 적격 5km 기록이 관측된 경우\"에 조건부다. ID·행 값은 싣지 않았다.", "",
           f"- 단위: 초. B2 − B0 차이가 음수이면 B2 쪽 사용자 평균 MAE가 작다. 포함률 = 명목 80% 범위(ŷ + [q10_oof, q90_oof] = ŷ + [{q10:.1f}, {q90:.1f}]초) 안 비율(층 안 풀링).",
           f"- 95% CI: 사용자 단위 bootstrap({N_BOOT:,}회, seed {SEED}), 층마다 그 층의 사용자를 복원추출(층·자료마다 `default_rng({SEED})` 새로 시작). 평가 대상이 {MIN_N}개 미만인 층은 **표본 부족**으로 표시만 하고 해석하지 않는다(수치는 싣는다).",
           "- 시연의 q10·q90 재계산값이 `demo/b2_params.json`과 같음, train 평가 행이 `data/5km_dataset_train.pkl`과 같음, 기준 기록의 T5가 데이터셋과 같음을 assert로 확인했다.", ""]
    # 전체 행
    rows = []
    for nm, df in sets:
        s = stratum_stats(df)
        rows.append({"자료": nm, "평가 대상": f"{s['n']:,}", "사용자": f"{s['users']:,}", "B2 사용자 평균 MAE": f"{s['mae2']:.1f}", "B0 사용자 평균 MAE": f"{s['mae0']:.1f}",
                     "B2 − B0 [95% CI]": read_cell(s, "diff"), "포함률 [95% CI]": read_cell(s, "cov")})
    out += ["## 0. 전체(참고)", "", md_table(rows), ""]
    flagged = {nm: [] for nm, _ in sets}
    all_strata = 0
    for di, (dname, fn, labels) in enumerate(DIMS, start=1):
        rows = []
        for nm, df in sets:
            cat = fn(df)
            assert cat.notna().all(), (dname, nm)
            assert sum(int((cat == lb).sum()) for lb in labels) == len(df)
            for lb in labels:
                sub = df[cat == lb]
                if len(sub) == 0:
                    rows.append({"자료": nm, "층": lb, "평가 대상": "0", "사용자": "0", "B2 사용자 평균 MAE": "-", "B0 사용자 평균 MAE": "-", "B2 − B0 [95% CI]": "-", "포함률 [95% CI]": "-", "비고": "층에 평가 대상 없음"})
                    continue
                s = stratum_stats(sub)
                all_strata += 1
                low = s["n"] < MIN_N
                rows.append({"자료": nm, "층": lb, "평가 대상": f"{s['n']:,}", "사용자": f"{s['users']:,}", "B2 사용자 평균 MAE": f"{s['mae2']:.1f}", "B0 사용자 평균 MAE": f"{s['mae0']:.1f}",
                             "B2 − B0 [95% CI]": read_cell(s, "diff"), "포함률 [95% CI]": read_cell(s, "cov"), "비고": "표본 부족" if low else ""})
                if not low:
                    if s["diff_ci"][0] > 0:
                        flagged[nm].append(f"{dname} {lb}: B2 − B0 {s['diff']:+.1f} [{s['diff_ci'][0]:+.1f}, {s['diff_ci'][1]:+.1f}] (CI 하한이 0보다 큼)")
                    if s["cov_ci"][1] < 0.8:
                        flagged[nm].append(f"{dname} {lb}: 포함률 {100 * s['cov']:.1f}% [{100 * s['cov_ci'][0]:.1f}%, {100 * s['cov_ci'][1]:.1f}%] (CI 상한이 80%보다 작음)")
                    elif s["cov_ci"][0] > 0.8:
                        flagged[nm].append(f"{dname} {lb}: 포함률 {100 * s['cov']:.1f}% [{100 * s['cov_ci'][0]:.1f}%, {100 * s['cov_ci'][1]:.1f}%] (CI 하한이 80%보다 큼)")
        out += [f"## {di}. {dname}", "", md_table(rows), ""]
    out += ["## 6. 읽는 법 (사양 4절의 해석 규칙)", "",
            "- 층 사이 차이는 **상관이며 원인이 아니다.** 층 변수들은 서로 관련될 수 있다(예: (a)와 (b)는 과거 5km가 1개뿐이면 x = 0).",
            f"- 층을 여러 개(5개 차원, 16개 층, 두 자료) 보므로 **우연히 CI가 0을 벗어나거나 포함률이 80%에서 벗어나는 층이 있을 수 있다.** 다중 비교 보정은 하지 않았다. CI가 0을 포함하면 \"점추정은 컸으나(작았으나) 우열을 확정할 수 없음\" 또는 \"현재 자료로 뚜렷한 연관성을 확인하지 못함\"이다. CI 하한이 0보다 클 때만 \"B2가 B0보다 오차가 큼\"이라고 쓴다.",
            "- 포함률의 CI가 80%를 포함하면 \"명목 80%에서 벗어났다고 확정할 수 없음\"이다. 표본 부족 층은 해석하지 않는다.",
            "- test cold-start는 이미 열람한 자료의 탐색 결과이며 train 결과와 방향이 같은지만 본다.", "",
            "### 사양의 기준을 넘는 층 (표본 30개 이상, 위 해석 규칙 적용; 우연 가능성 있음)", ""]
    for nm, lst in flagged.items():
        out.append(f"**{nm}**" + ("" if lst else ": 없음"))
        out += [f"- {x}" for x in lst]
        out.append("")
    out += ["- 이 결과로 주 결론, 모델, 정의, 분할, 시연 규칙(B2 식·파라미터·범위·문구)을 바꾸지 않는다.", ""]
    DOC.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC, "| strata", all_strata)
    for nm, lst in flagged.items():
        print(nm, len(lst))


if __name__ == "__main__":
    main()
