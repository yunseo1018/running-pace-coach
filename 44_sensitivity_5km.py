# -*- coding: utf-8 -*-
"""
사전 지정 민감도 S1~S5 (docs/predeclaration_5km_final.md 18항, 3fad402). 사양: docs/posthoc_spec_5km.md (커밋 49b5f2b).
주 결론(49c02f8)을 바꾸지 않는다. 하이퍼파라미터는 84670d6의 기본(11) 입력 선택값을 그대로 쓰고 다시 고르지 않는다(E-MLP 제외).
각 민감도마다 해당 정의로 데이터셋을 다시 만들고 B2의 a·b·전처리·MLP 에폭 수는 그 정의의 train(또는 personalized 학습분)에서만 적합한다.
S2만 재학습 없이 49c02f8 저장 예측값에 필터를 적용한다.
상관관계까지만. ID·행 값은 출력·저장하지 않는다(예측 중간 산출물 없음).
출력: docs/sensitivity_results_5km.md
"""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ridge_lib


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m43 = _load("43_within_person_5km.py", "m43")
m42, m38, m40, feas = m43.m42, m43.m38, m43.m42.m40, m43.feas
m25 = m40.m25
md_table = m42.md_table
DOC_PATH = Path("docs") / "sensitivity_results_5km.md"
RAW_PATH = Path("endomondoHR.json")
BASE_SET = "기본(11)"
BASE_COLS = m40.SETS[BASE_SET]
WEEK, WIN, DAY = feas.WEEK, feas.WIN, feas.DAY
EXTREME = 5000.0
SHIFTS = [-12, -6, 6, 12]


# ------------------------------------------------------------------ 데이터셋 (42번 build_dataset의 일반화: 기준 시점 이동·타깃 방식)
def build_variant(d, ext_by_sid, t5_by_sid, base, mode="first"):
    rows = []
    for uid, g in d.groupby("user_id", sort=True):
        uid = int(uid)
        s = g["start_time"].to_numpy(np.int64)
        e = s + g["duration_sec"].to_numpy(np.int64)
        km = g["actual_distance_km"].to_numpy()
        hr = g["avg_heart_rate"].to_numpy()
        pace = g["avg_pace_sec_per_km"].to_numpy()
        iscand = g["cand"].to_numpy()
        sids = g["session_id"].to_numpy()
        ext = np.array([ext_by_sid.get(int(x), False) for x in sids])
        t5 = np.array([t5_by_sid.get(int(x), np.nan) for x in sids])
        gender = g["gender"].iloc[0]
        for k in feas.span_ks({"s": s}, base):
            t = base + k * WEEK
            inwin = (e >= t - WIN) & (e < t)
            n_in = int(inwin.sum())
            if n_in < 4:
                continue
            cw = np.flatnonzero(inwin & iscand)
            valid = cw[~ext[cw]]
            if len(valid) == 0:
                continue
            ref = int(valid[-1])
            fut = np.flatnonzero(iscand & (s >= t) & (s < t + WEEK))
            if len(fut) == 0:
                continue
            tgt = {"first": t5[fut[0]], "median": float(np.median(t5[fut])), "min": float(np.min(t5[fut]))}[mode]
            past = np.flatnonzero(iscand & (e < t) & ~ext)
            m = float(np.median(t5[past]))
            if len(past) == 1:
                m = float(t5[ref])
            w = np.flatnonzero(inwin)
            wk = [float(km[w][(e[w] >= t - WIN + q * WEEK) & (e[w] < t - WIN + (q + 1) * WEEK)].sum()) for q in range(4)]
            rows.append(dict(
                uid=uid, k=k, t=t, n_runs=n_in, total_km=float(km[w].sum()), wk1_km=wk[0], wk2_km=wk[1], wk3_km=wk[2], wk4_km=wk[3],
                longest_km=float(km[w].max()), days_since_last=float((t - e[w].max()) / DAY), mean_hr=float(hr[w].mean()),
                t5_ref=float(t5[ref]), t5_ref_minus_m=float(t5[ref] - m), m=m, n_past=int(len(past)),
                mean_pace=float(pace[w].mean()), gender=gender, y=float(tgt - t5[ref]), ans_extreme=bool(ext[fut[0]])))
    return pd.DataFrame(rows).sort_values(["uid", "t"]).reset_index(drop=True)


# ------------------------------------------------------------------ S5: 병합 세션 T5 (조각 사이 공백 포함)
def merged_t5(parse, d, pieces):
    """km ∈ [5.0, 5.5]인 병합 세션의 T5. 조각을 시작 시각순으로 잇고 조각 거리 합이 5.000km에 처음 도달하는 조각에서 보간."""
    mg = d[(d.merged_count > 1) & d.actual_distance_km.between(m38.LO, m38.HI)]
    pc = pieces.set_index("session_id")
    need = {}
    for _, r in mg.iterrows():
        for sid in r["merged_session_ids"]:
            need[int(sid)] = int(r["session_id"])
    arrays = {}
    with open(RAW_PATH, "rb") as f:
        for line in f:
            m = feas.ID_RE.search(line)
            if not m or int(m.group(1)) not in need:
                continue
            sid = int(m.group(1))
            rec = ast.literal_eval(line.decode("utf-8"))
            if rec.get("sport") != "run" or rec["id"] != sid:
                continue
            cum, _ = feas.cum_with_skip(parse, rec["latitude"], rec["longitude"], rec["timestamp"])
            arrays[sid] = (cum, np.array(rec["timestamp"], dtype=np.int64))
    assert len(arrays) == len(need)
    out = {}
    for _, r in mg.iterrows():
        ids = sorted([int(x) for x in r["merged_session_ids"]], key=lambda x: pc.loc[x, "start_time"])
        before = 0.0
        t5 = None
        for sid in ids:
            cum, ts = arrays[sid]
            if before + cum[-1] >= 5.0:
                rem = 5.0 - before
                j = int(np.searchsorted(cum, rem, "left"))
                j = max(j, 1)
                frac = (rem - cum[j - 1]) / (cum[j] - cum[j - 1])
                t5 = (ts[j - 1] + frac * (ts[j] - ts[j - 1])) - int(r["start_time"])
                break
            before += cum[-1]
        assert t5 is not None
        out[int(r["session_id"])] = float(t5)
    return out


# ------------------------------------------------------------------ 평가
def predict_basic(tr, ev, hp):
    p = {}
    ytr, Xtr, Xev, uid_tr = tr.y.to_numpy(), tr[BASE_COLS].to_numpy(float), ev[BASE_COLS].to_numpy(float), tr.uid.to_numpy()
    a, b = m40.lad_fit(tr.t5_ref_minus_m.to_numpy(), ytr)
    p["B2"] = a + b * ev.t5_ref_minus_m.to_numpy()
    p["Ridge"] = ridge_lib.RidgePipeline(hp[("Ridge", BASE_SET)]["alpha"]).fit(Xtr, ytr).predict(Xev)
    p["GBM"] = m25.build_gbt(hp[("GBM", BASE_SET)], "absolute_error").fit(Xtr, ytr).predict(Xev)
    p["MLP"], _ = m40.fit_predict_mlp(hp[("MLP", BASE_SET)], Xtr, ytr, Xev, uid_tr)
    return p


def summarize(ev, preds):
    R, U = m42.analyze(ev, preds)
    rows, npass = [], 0
    for m in ["Ridge", "GBM", "MLP"]:
        d0 = R[m]["mae"][0][0] - R["B2"]["mae"][0][0]
        bs = R[m]["mae"][1][0] - R["B2"]["mae"][1][0]
        hi = np.percentile(bs, 97.5)
        npass += int(hi < 0)
        rows.append({"모델(기본)": m, "사용자 평균 MAE 차이(모델 − B2)": f"{d0:+.3f} {m42.ci(bs)}", "통계적으로 B2를 넘음(CI 상한 < 0)": "예" if hi < 0 else "아니오",
                     "실용적(개선 ≥ 10초)": "예" if d0 <= -10 else "아니오"})
    return dict(n=len(ev), U=U, b2=R["B2"]["mae"][0][0], npass=npass, rows=rows)


def run_variant(ds, tr_users, te_users, hp):
    cold_tr = ds[ds.uid.isin(tr_users)].reset_index(drop=True)
    cold_ev = ds[ds.uid.isin(te_users)].reset_index(drop=True)
    p_tr, p_ev, _ = m42.personalized_split(ds)
    res = {}
    for name, tr, ev in (("cold-start test", cold_tr, cold_ev), ("personalized", p_tr, p_ev)):
        res[name] = summarize(ev, predict_basic(tr, ev, hp)) | dict(n_train=len(tr))
    return res


def main():
    torch.set_num_threads(4)
    hp = m42.read_selected()
    d, tr, te, grp, cache = m43.load_base()
    parse = feas.import_parser()
    pieces = pd.read_pickle("data/run_sessions_deduped.pkl")
    results = {}   # (이름) -> {조건: dict}

    # ---- 주 분석 데이터셋 (S2, S3 기준 확인)
    main_ds = m43.main_dataset(d, grp, cache)
    assert len(main_ds) == 2072

    # ---- S2: 49c02f8 저장 예측값에서 정답 극단 행 제거 (재학습 없음)
    saved = pd.read_pickle(m42.PRED_OUT)
    ext_map = main_ds.set_index(["uid", "t"])["ans_extreme"]
    s2, ref_main = {}, {}
    for nm, cname in (("cold", "cold-start test"), ("pers", "personalized")):
        sv = saved[nm]
        ev = pd.DataFrame({"uid": sv["uid"], "t": sv["t"], "y": sv["y"]})
        ev["ext"] = [bool(ext_map.loc[(u, t)]) for u, t in zip(ev.uid, ev.t)]
        pr = {"B2": sv["preds"]["B2"]}
        for m in ["Ridge", "GBM", "MLP"]:
            pr[m] = sv["preds"][str((m, BASE_SET))]
        ref_main[cname] = summarize(ev, pr)
        keep = ~ev.ext.to_numpy()
        s2[cname] = summarize(ev[keep].reset_index(drop=True), {k: v[keep] for k, v in pr.items()}) | dict(n_removed=int((~keep).sum()), n_train=None)
    results["S2 정답 극단 제외(49c02f8 예측값, 재학습 없음)"] = s2

    # ---- S1: 비병합 4.5~5.5km, 환산 T5 = 경과 시간 ÷ 거리 × 5
    d1 = d.copy()
    d1["cand"] = (d1.merged_count == 1) & d1.actual_distance_km.between(4.5, 5.5)
    c1 = d1[d1.cand]
    t5_1 = {int(s): float(du / k * 5.0) for s, du, k in zip(c1.session_id, c1.duration_sec, c1.actual_distance_km)}
    ext_1 = {s: v > EXTREME for s, v in t5_1.items()}
    ds1 = build_variant(d1, ext_1, t5_1, m38.MONDAY0)
    results["S1 5km 환산 시간(비병합 4.5~5.5km)"] = run_variant(ds1, tr, te, hp)
    print("S1 done", len(ds1), flush=True)

    # ---- S3: 기준 시점 이동
    for sh in SHIFTS:
        ds3 = build_variant(d, cache["ext"], cache["t5"], m38.MONDAY0 + sh * 3600)
        results[f"S3 기준 시점 {sh:+d}h 이동"] = run_variant(ds3, tr, te, hp)
        print("S3", sh, len(ds3), flush=True)

    # ---- S4: 다음 7일 5km 후보 T5의 중앙값 / 최솟값
    for mode, lab in (("median", "S4a 정답 = 다음 7일 5km 후보 T5 중앙값"), ("min", "S4b 정답 = 다음 7일 5km 후보 T5 최솟값")):
        ds4 = build_variant(d, cache["ext"], cache["t5"], m38.MONDAY0, mode)
        results[lab] = run_variant(ds4, tr, te, hp)
        print(lab, len(ds4), flush=True)

    # ---- S5: 병합 세션도 후보
    t5_m = merged_t5(parse, d, pieces)
    d5 = d.copy()
    d5["cand"] = d5.actual_distance_km.between(m38.LO, m38.HI)
    t5_5 = dict(cache["t5"])
    t5_5.update(t5_m)
    ext_5 = {s: v > EXTREME for s, v in t5_5.items()}
    ds5 = build_variant(d5, ext_5, t5_5, m38.MONDAY0)
    results["S5 병합 세션도 5km 후보(조각 사이 공백 포함)"] = run_variant(ds5, tr, te, hp)
    n_c5 = int(d5.cand.sum())
    print("S5 done", len(ds5), flush=True)

    # ------------------------------------------------------------- 문서
    out = ["# 사전 지정 민감도 S1~S5 (5km 과제)", "",
           "생성: `44_sensitivity_5km.py`. **사전 지정 민감도(3fad402 18항)**이며 사양은 `docs/posthoc_spec_5km.md`(커밋 49b5f2b)에 실행 전 고정했다. "
           "**주 결론(49c02f8, `docs/final_results_5km.md`)을 바꾸지 않는다.** 하이퍼파라미터는 84670d6의 기본(11) 입력 선택값을 그대로 쓰고 다시 고르지 않았으며 E-MLP는 제외했다. "
           "민감도마다 해당 정의로 데이터셋을 다시 만들고 B2의 a·b와 전처리는 그 정의의 train(cold-start) 또는 personalized 학습분에서만 적합했다(S2는 재학습 없이 49c02f8 저장 예측값에 필터만). "
           "시각은 start_time 원값(기준 미확인). 상관관계까지만 주장한다. 비교가 많아 우연히 CI가 0을 벗어나는 경우가 있을 수 있고, CI가 0을 포함하는 차이는 동등성의 증명이 아니다. ID·행 값은 싣지 않았다.", "",
           "## 요약", "",
           "\"통계적으로 B2를 넘은 모델 수\" = 기본 입력 Ridge·GBM·MLP 3개 중 (모델 − B2)의 사용자 평균 MAE 차이 95% CI 상한이 0 미만인 모델 수. 사용자 단위 bootstrap 2,000회(seed 42), 같은 추출.", ""]
    rows = []

    def add(name, cname, r):
        rows.append({"민감도": name, "평가": cname, "평가 대상(기준 시점/사용자)": f"{r['n']:,} / {r['U']}명", "B2 사용자 평균 MAE(초)": f"{r['b2']:.3f}",
                     "통계적으로 B2를 넘은 모델 수(3개 중)": r["npass"]})
    for cname in ["cold-start test", "personalized"]:
        add("주 분석(49c02f8, 참고)", cname, ref_main[cname])
    for name, res in results.items():
        for cname in ["cold-start test", "personalized"]:
            add(name, cname, res[cname])
    out += [md_table(rows), "", "## 민감도별 상세 (주 결론 표 형식: 기본 입력 모델 − B2)", ""]
    for name, res in results.items():
        out += [f"### {name}", ""]
        if name.startswith("S5"):
            out += [f"5km 후보 {n_c5:,}개(병합 세션 포함; 주 분석 5,406개 + 병합 세션 {len(t5_m)}개)." + "", ""]
        for cname in ["cold-start test", "personalized"]:
            r = res[cname]
            extra = f", 정답 극단으로 제거된 평가 행 {r['n_removed']}개" if r.get("n_removed") is not None else ""
            tr_txt = f", 학습 기준 시점 {r['n_train']:,}개" if r.get("n_train") else ""
            out += [f"**{cname}** — 평가 대상 {r['n']:,}개·{r['U']}명{tr_txt}{extra}, B2 사용자 평균 MAE {r['b2']:.3f}초", "", md_table(r["rows"]), ""]
    DOC_PATH.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC_PATH)


if __name__ == "__main__":
    main()
