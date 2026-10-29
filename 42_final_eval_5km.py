# -*- coding: utf-8 -*-
"""
5km 과제 최종 평가: cold-start test + personalized, **각 1회** (docs/evaluation_freeze_5km.md, 커밋 f1857fb, 태그 freeze-5km-v1 그대로).

  python 42_final_eval_5km.py --dry-run   # 코드 점검용: test 사용자 정답을 만들지 않고 train 사용자만으로 같은 함수를 실행(문서 저장 안 함)
  python 42_final_eval_5km.py             # 실제 평가: data/final_eval_done.flag가 있으면 중단(재실행 금지), 성공 시 표식 생성

- 평가 목록: B0/B1/B2, Ridge·GBM·MLP × {기본, +평균 페이스, +성별}, E-MLP × 3 입력 세트(탐색적).
  하이퍼파라미터는 docs/model_cv_results_5km.md(84670d6 및 E-MLP 절)에 기록된 선택값을 읽어 그대로 쓴다(다시 고르지 않음).
- cold-start: 학습 = train 평가 대상 전체(1,720개), 평가 = test 평가 대상(352개·71명). 개수가 다르면 중단.
- personalized: 학습 = personalized 학습 기준 시점(1,531개), 평가 = 442개(250명). 개수가 다르면 중단. B2 a·b·전처리·MLP 에폭 수는 학습분에서만.
- MLP·E-MLP 에폭 수는 학습 자료 안 사용자 20% 내부 검증(seed 42)으로 정한 뒤 학습 자료 전체로 재학습(CV와 같은 방식, m40/m41 함수 재사용).
- 지표/CI: 사용자 평균 MAE(주), 풀링 MAE, 방향 일치율(±10초), 사용자 단위 bootstrap 2,000회(seed 42), 같은 추출, 백분위 95% CI.
- 예측값(사용자 ID 포함)은 data/final_preds_5km.pkl에만 저장(git 제외). 문서에는 ID·행 값을 싣지 않는다.
출력: docs/final_results_5km.md. sklearn 의존 없음.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import ridge_lib


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m38 = _load("38_build_5km_dataset.py", "m38")
m40 = _load("40_models_cv_5km.py", "m40")
m41 = _load("41_emlp_cv_5km.py", "m41")
feas, m25 = m38.feas, m40.m25
md_table, direction, lad_fit = m40.md_table, m40.direction, m40.lad_fit

DOC_PATH = Path("docs") / "final_results_5km.md"
CV_DOC = Path("docs") / "model_cv_results_5km.md"
FLAG = Path("data") / "final_eval_done.flag"
PRED_OUT = Path("data") / "final_preds_5km.pkl"
TRAIN_DATASET = Path("data") / "5km_dataset_train.pkl"
BAND, N_BOOT, SEED = 10.0, 2000, 42
WEEK = feas.WEEK
SETS = m40.SETS
MODELS = ["Ridge", "GBM", "MLP"]
EXPECT = dict(cold_train=(1720, 278), cold_test=(352, 71), pers_train=1531, pers_eval=442, pers_users=250)


# ------------------------------------------------------------------ 데이터셋 (38번과 같은 계산, 대상 사용자만)
def build_dataset(include, grp, d, ext_by_sid, t5_by_sid):
    rows = []
    for uid, g in d.groupby("user_id", sort=True):
        uid = int(uid)
        if uid not in include:
            continue
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
        for k in feas.span_ks({"s": s}, m38.MONDAY0):
            t = m38.MONDAY0 + k * WEEK
            inwin = (e >= t - m38.WIN) & (e < t)
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
            ans = int(fut[0])
            past = np.flatnonzero(iscand & (e < t) & ~ext)
            m = float(np.median(t5[past]))
            if len(past) == 1:
                m = float(t5[ref])
            w = np.flatnonzero(inwin)
            wk = [float(km[w][(e[w] >= t - m38.WIN + q * WEEK) & (e[w] < t - m38.WIN + (q + 1) * WEEK)].sum()) for q in range(4)]
            rows.append(dict(
                uid=uid, k=k, t=t, n_runs=n_in, total_km=float(km[w].sum()), wk1_km=wk[0], wk2_km=wk[1], wk3_km=wk[2], wk4_km=wk[3],
                longest_km=float(km[w].max()), days_since_last=float((t - e[w].max()) / m38.DAY), mean_hr=float(hr[w].mean()),
                t5_ref=float(t5[ref]), t5_ref_minus_m=float(t5[ref] - m), m=m, n_past=int(len(past)),
                mean_pace=float(pace[w].mean()), gender=gender, y=float(t5[ans] - t5[ref]), ans_extreme=bool(ext[ans])))
    ds = pd.DataFrame(rows)
    for gname in ["male", "female", "unknown"]:
        ds["g_" + gname] = (ds.gender == gname).astype(float)
    return ds.sort_values(["uid", "t"]).reset_index(drop=True)


def personalized_split(ds):
    tr_idx, ev_idx, users = [], [], 0
    for uid, g in ds.groupby("uid", sort=True):
        g = g.sort_values("t")
        nn = len(g)
        if nn < 2:
            continue
        users += 1
        ne = max(1, int(round(0.2 * nn)))
        first_eval = g.t.iloc[nn - ne]
        pre = g.iloc[: nn - ne]
        tr_idx += list(pre.index[(pre.t + WEEK) <= first_eval])
        ev_idx += list(g.index[nn - ne:])
    return ds.loc[tr_idx].reset_index(drop=True), ds.loc[ev_idx].reset_index(drop=True), users


# ------------------------------------------------------------------ 선택된 하이퍼파라미터 읽기
def read_selected():
    txt = CV_DOC.read_text(encoding="utf-8")
    sec = txt[txt.index("## 1. 선택된 하이퍼파라미터"): txt.index("## 2. 지표")]
    hp = {}
    for line in sec.splitlines():
        m = re.match(r"\| (Ridge|GBM|MLP) \| ([^|]+) \| (\{.*?\}) \|", line)
        if m:
            hp[(m.group(1), m.group(2).strip())] = json.loads(m.group(3))
    e = txt[txt.index(m41.START): txt.index(m41.END)]
    e = e[e.index("### 7-1"): e.index("### 7-2")]
    for line in e.splitlines():
        m = re.match(r"\| ([^|]+) \| (\{.*?\}) \|", line)
        if m and m.group(1).strip() in SETS:
            hp[("E-MLP", m.group(1).strip())] = json.loads(m.group(2))
    assert len(hp) == 12, sorted(hp)
    return hp


# ------------------------------------------------------------------ 학습·예측
def predict_all(tr, ev, hp):
    """tr/ev: 데이터프레임. 반환 {방법: ev 예측}. 모든 학습·전처리는 tr에서만."""
    out = {"B0": np.zeros(len(ev)), "B1": -ev.t5_ref_minus_m.to_numpy()}
    a, b = lad_fit(tr.t5_ref_minus_m.to_numpy(), tr.y.to_numpy())
    out["B2"] = a + b * ev.t5_ref_minus_m.to_numpy()
    out["_B2_coef"] = (a, b)
    ytr, uid_tr = tr.y.to_numpy(), tr.uid.to_numpy()
    for sname, cols in SETS.items():
        Xtr, Xev = tr[cols].to_numpy(float), ev[cols].to_numpy(float)
        p = hp[("Ridge", sname)]
        out[("Ridge", sname)] = ridge_lib.RidgePipeline(p["alpha"]).fit(Xtr, ytr).predict(Xev)
        p = hp[("GBM", sname)]
        out[("GBM", sname)] = m25.build_gbt(p, "absolute_error").fit(Xtr, ytr).predict(Xev)
        out[("MLP", sname)], ep = m40.fit_predict_mlp(hp[("MLP", sname)], Xtr, ytr, Xev, uid_tr)
        out[("_ep", "MLP", sname)] = ep
        out[("E-MLP", sname)], ep = m41.emlp_fit_predict(hp[("E-MLP", sname)], Xtr, ytr, Xev, uid_tr)
        out[("_ep", "E-MLP", sname)] = ep
    return out


# ------------------------------------------------------------------ 지표·표
def analyze(ev, preds):
    uid, y = ev.uid.to_numpy(), ev.y.to_numpy()
    users, inv = np.unique(uid, return_inverse=True)
    U = len(users)
    cnt = np.bincount(inv, minlength=U).astype(float)
    dy = direction(y)
    idx = np.random.default_rng(SEED).integers(0, U, size=(N_BOOT, U))
    R = {}
    for k, p in preds.items():
        if isinstance(k, tuple) and k[0] == "_ep" or k == "_B2_coef":
            continue
        err = np.abs(y - p)
        agree = (direction(p) == dy).astype(float)
        o = {}
        for key, v in (("mae", err), ("agree", agree)):
            s = np.bincount(inv, weights=v, minlength=U)
            um = s / cnt
            o[key] = ((um.mean(), s.sum() / cnt.sum()), (um[idx].mean(1), s[idx].sum(1) / cnt[idx].sum(1)))
        R[k] = o
    return R, U


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def label(k):
    return {"B0": "B0 (ŷ=0)", "B1": "B1 (m−T5(기준))", "B2": "B2 (LAD)"}.get(k) if isinstance(k, str) else f"{k[0]} / {k[1]}"


def tables(title, ev, preds):
    R, U = analyze(ev, preds)
    y = ev.y.to_numpy()
    dy = direction(y)
    cls = pd.Series(dy).map({-1: "개선", 0: "변화 없음", 1: "악화"}).value_counts()

    def diff(a, b, k=0):
        return R[a]["mae"][0][k] - R[b]["mae"][0][k], R[a]["mae"][1][k] - R[b]["mae"][1][k]

    def drow(nm, a, b):
        d0, bs = diff(a, b)
        dp, bp = diff(a, b, 1)
        return {"비교(앞 − 뒤)": nm, "사용자 평균 MAE 차이": f"{d0:+.3f} {ci(bs)}", "풀링 MAE 차이": f"{dp:+.3f} {ci(bp)}"}
    out = [f"## {title}", "",
           f"평가 기준 시점 {len(ev):,}개, 사용자 {U}명. 단위는 초. 사용자 평균 MAE가 주 지표.", "",
           "### (a) 정답 y 요약", "",
           md_table([{"개수": f"{len(y):,}", "사용자": U, "중앙값(초)": f"{np.median(y):.1f}", "25% 분위(초)": f"{np.percentile(y, 25):.1f}", "75% 분위(초)": f"{np.percentile(y, 75):.1f}"}]), "",
           md_table([{"방향": k, "개수": f"{int(cls.get(k, 0)):,}", "비율": f"{100 * cls.get(k, 0) / len(y):.1f}%"} for k in ["개선", "변화 없음", "악화"]]), "",
           "### (b) 모든 평가 대상의 지표 (95% CI)", ""]
    keys = ["B0", "B1", "B2"] + [(m, s) for m in MODELS for s in SETS] + [("E-MLP", s) for s in SETS]
    out += [md_table([{"방법": label(k) + (" — 탐색적" if isinstance(k, tuple) and k[0] == "E-MLP" else ""),
                       "사용자 평균 MAE(주)": f"{R[k]['mae'][0][0]:.3f} {ci(R[k]['mae'][1][0])}", "풀링 MAE": f"{R[k]['mae'][0][1]:.3f} {ci(R[k]['mae'][1][1])}",
                       "방향 일치율 풀링": f"{R[k]['agree'][0][1]:.3f} {ci(R[k]['agree'][1][1])}", "방향 일치율 사용자 평균": f"{R[k]['agree'][0][0]:.3f} {ci(R[k]['agree'][1][0])}"} for k in keys]), "",
            "### (c) 주 결론: 기본 입력 Ridge·GBM·MLP − B2", ""]
    rows = []
    for m in MODELS:
        d0, bs = diff((m, "기본(11)"), "B2")
        dp, bp = diff((m, "기본(11)"), "B2", 1)
        hi = np.percentile(bs, 97.5)
        rows.append({"모델(기본)": m, "사용자 평균 MAE 차이(모델 − B2)": f"{d0:+.3f} {ci(bs)}", "풀링 MAE 차이": f"{dp:+.3f} {ci(bp)}",
                     "통계적 통과(CI 상한 < 0)": "예" if hi < 0 else "아니오", "실용적 통과(개선 ≥ 10초)": "예" if d0 <= -10 else "아니오"})
    out += [md_table(rows), "",
            "주 결론은 이 3개 비교만으로 내린다. 비교가 여러 개라 우연히 통과할 가능성이 있다(아래 보조·탐색적 표의 통과 여부는 요약에 쓰지 않는다). CI가 0을 포함하는 차이는 동등성의 증명이 아니다.", "",
            "### (d) 보조: 추가 비교와 기준치 간 비교", ""]
    rows = []
    for m in MODELS:
        rows.append(drow(f"{m}: 기본+평균 페이스(12) − 기본", (m, "기본+평균 페이스(12)"), (m, "기본(11)")))
        rows.append(drow(f"{m}: 기본+성별(14) − 기본", (m, "기본+성별(14)"), (m, "기본(11)")))
    rows += [drow("B1 − B0", "B1", "B0"), drow("B2 − B0", "B2", "B0"), drow("B2 − B1", "B2", "B1")]
    out += [md_table(rows), "", "### (e) 탐색적: E-MLP − B2 (주 결론에 쓰지 않음)", ""]
    rows = []
    for s in SETS:
        d0, bs = diff(("E-MLP", s), "B2")
        dp, bp = diff(("E-MLP", s), "B2", 1)
        rows.append({"비교": f"E-MLP / {s} − B2", "사용자 평균 MAE 차이": f"{d0:+.3f} {ci(bs)}", "풀링 MAE 차이": f"{dp:+.3f} {ci(bp)}",
                     "통계적 통과(참고)": "예" if np.percentile(bs, 97.5) < 0 else "아니오", "실용적 통과(참고)": "예" if d0 <= -10 else "아니오"})
    out += [md_table(rows), ""]
    return out


def run_condition(tr, ev, hp):
    preds = predict_all(tr, ev, hp)
    a, b = preds["_B2_coef"]
    eps = {f"{k[1]}/{k[2]}": int(v) for k, v in preds.items() if isinstance(k, tuple) and k[0] == "_ep"}
    return preds, (a, b), eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and FLAG.exists():
        sys.exit("중단: 최종 평가는 이미 실행되었다(data/final_eval_done.flag). 재실행하지 않는다.")
    torch.set_num_threads(4)
    hp = read_selected()
    d = config.load_main_sessions().sort_values(["user_id", "start_time"]).reset_index(drop=True)
    tr_users = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    te_users = {int(x) for x in Path(config.TEST_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    grp = {int(u): ("train" if int(u) in tr_users else ("test" if int(u) in te_users else "other")) for u in d.user_id.unique()}
    d["cand"] = (d["merged_count"] == 1) & d["actual_distance_km"].between(m38.LO, m38.HI)
    c = d[d.cand]
    cand_start = {int(s): int(t) for s, t in zip(c.session_id, c.start_time)}
    parse = feas.import_parser()
    if args.dry_run:
        store = {int(s): grp[int(u)] == "train" for s, u in zip(c.session_id, c.user_id)}   # test T5는 저장하지 않음(38번과 같음)
        include = tr_users & set(grp)
    else:
        store = {int(s): True for s in c.session_id}                                       # 최종 평가: 모든 사용자의 T5·y 계산
        include = set(grp)
    ext_by_sid, t5_by_sid = m38.scan_raw(parse, cand_start, store)
    ds = build_dataset(include, grp, d, ext_by_sid, t5_by_sid)
    # 38번 데이터셋과 train 행 일치 확인
    ref = pd.read_pickle(TRAIN_DATASET).sort_values(["uid", "t"]).reset_index(drop=True)
    mine = ds[ds.uid.isin(tr_users)].reset_index(drop=True)
    assert len(ref) == len(mine) == EXPECT["cold_train"][0]
    for col in ["uid", "t", "y", "t5_ref", "t5_ref_minus_m", "n_runs", "total_km", "mean_hr", "mean_pace", "days_since_last"]:
        assert np.array_equal(ref[col].to_numpy(), mine[col].to_numpy()), col
    print("38번 train 데이터셋과 일치 확인", flush=True)

    if args.dry_run:
        trm, vam = next(ridge_lib.group_kfold_indices(mine.uid.to_numpy(), 5, SEED))
        cold_tr, cold_ev = mine[trm].reset_index(drop=True), mine[vam].reset_index(drop=True)
        p_tr, p_ev, p_users = personalized_split(mine)
        print("dry-run: cold(pseudo)", len(cold_tr), len(cold_ev), "pers(train users only)", len(p_tr), len(p_ev), p_users, flush=True)
    else:
        cold_tr, cold_ev = mine, ds[ds.uid.isin(te_users)].reset_index(drop=True)
        assert (len(cold_tr), cold_tr.uid.nunique()) == EXPECT["cold_train"]
        assert (len(cold_ev), cold_ev.uid.nunique()) == EXPECT["cold_test"], (len(cold_ev), cold_ev.uid.nunique())
        p_tr, p_ev, p_users = personalized_split(ds)
        assert (len(p_tr), len(p_ev), p_users) == (EXPECT["pers_train"], EXPECT["pers_eval"], EXPECT["pers_users"]), (len(p_tr), len(p_ev), p_users)
        print("개수 확인 통과", flush=True)

    res = {}
    for name, (tr, ev) in {"cold": (cold_tr, cold_ev), "pers": (p_tr, p_ev)}.items():
        print("학습·평가:", name, flush=True)
        res[name] = (ev, *run_condition(tr, ev, hp))
    if args.dry_run:
        for name, (ev, preds, coef, eps) in res.items():
            R, U = analyze(ev, preds)
            print(name, "방법 수", len(R), "B2 a,b", [round(x, 2) for x in coef], "B2 사용자 평균 MAE", round(R["B2"]["mae"][0][0], 3), "에폭", eps, flush=True)
            tables(name, ev, preds)
        print("dry-run 완료(문서 저장 없음, 표식 없음)")
        return

    FLAG.write_text("final evaluation executed once; do not rerun\n", encoding="utf-8")
    pd.to_pickle({n: {"uid": ev.uid.to_numpy(), "t": ev.t.to_numpy(), "y": ev.y.to_numpy(),
                      "preds": {str(k): v for k, v in preds.items() if not (isinstance(k, tuple) and k[0] == "_ep") and k != "_B2_coef"}}
                  for n, (ev, preds, coef, eps) in res.items()}, PRED_OUT)
    doc = ["# 5km 과제 최종 평가 결과 (cold-start test + personalized, 각 1회)", "",
           "생성: `42_final_eval_5km.py`. 방법·목록·결론의 위계는 `docs/evaluation_freeze_5km.md`(커밋 f1857fb, 태그 `freeze-5km-v1`)에 평가 전에 고정했고 그대로 한 번 실행했다. "
           "정의는 `docs/predeclaration_5km_final.md`(3fad402)·`docs/predeclaration_5km_addendum.md`(9fb0d8f), 하이퍼파라미터는 `docs/model_cv_results_5km.md`에 기록된 선택값(다시 고르지 않음). "
           "시각은 start_time 원값(기준 미확인)이며 상관관계까지만 주장한다. 사용자/세션 ID·행 값은 싣지 않았다.", "",
           "- **결론의 위계**: 주 결론 = 기본 입력 Ridge·GBM·MLP의 (모델 − B2) 3개 비교. 보조 = 추가 비교·기준치 간 비교·방향 일치율. 탐색적 = E-MLP 전부(주 결론에 쓰지 않음).",
           "- **다중 비교 주의**: 표에 비교가 많아 우연히 CI가 0을 벗어나거나 통과 기준을 만족하는 항목이 생길 수 있다. 통과 판정은 주 결론 3개에만 적용한다.",
           "- 판정: 통계적 통과 = 사용자 평균 MAE 차이(모델 − B2)의 95% CI 상한 < 0, 실용적 통과 = 점추정 개선 ≥ 10초. CI가 0을 포함한 차이는 동등성의 증명이 아니다.",
           f"- CI: 사용자 단위 cluster bootstrap({N_BOOT}회, seed {SEED}), 사용자를 복원추출하고 뽑힌 사용자의 모든 기준 시점을 유지, 모든 방법에 같은 추출(쌍대), 백분위 95% CI. 방향 밴드 ±{BAND:g}초.",
           f"- 환경: numpy {np.__version__}, pandas {pd.__version__}, torch {torch.__version__}. 실행 후 추가 분석은 모두 \"사후 분석\"으로 표시한다(이 문서에는 없음).",
           f"- 학습 자료: cold-start = train 평가 대상 {len(cold_tr):,}개·{cold_tr.uid.nunique()}명, 평가 = test {len(cold_ev):,}개·{cold_ev.uid.nunique()}명. "
           f"personalized = 학습 {len(p_tr):,}개, 평가 {len(p_ev):,}개·{p_users}명(test 사용자 포함). 개수는 `docs/dataset_counts_5km.md`와 일치(assert).", ""]
    for name, title in (("cold", "A. cold-start test (학습 = train 전체, 평가 = test 사용자)"), ("pers", "B. personalized (학습 = personalized 학습 기준 시점, 평가 = 뒤쪽 기준 시점)")):
        ev, preds, coef, eps = res[name]
        doc += tables(title, ev, preds)
        doc += [f"B2 계수(이 조건의 학습 자료에서 LAD): a = {coef[0]:.2f}초, b = {coef[1]:.3f}. 학습 자료 안 내부 검증으로 정한 에폭 수: " +
                ", ".join(f"{k} {v}" for k, v in eps.items()) + ".", ""]
    DOC_PATH.write_text("\n".join(doc), encoding="utf-8", newline="\n")
    print("saved", DOC_PATH, PRED_OUT)


if __name__ == "__main__":
    main()
