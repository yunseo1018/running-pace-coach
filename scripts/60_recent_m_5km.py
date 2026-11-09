# -*- coding: utf-8 -*-
"""
사후 탐색: 최근 기간 기준치 B2r. 사양: docs/posthoc_spec_recent_m.md (커밋 5f3be15, 결과 계산 전 고정).
- B2r = B2와 같은 식, m 대신 m_W(t 이전에 끝난 최근 W일 안 유효 5km의 T5 중앙값). 후보 W ∈ {90,180,365} × k ∈ {1,2,3}, n_W < k 규칙 (대체: 전체 기간 m / 보류: 예측 안 함).
- 선택은 train 사용자 단위 GroupKFold(5, seed 42) 폴드 밖으로만(폴드마다 a·b 재적합). test(352개·71명)는 선택에 쓰지 않으며 이미 열람한 자료의 탐색 결과다.
- 구현 세부(사양 2절의 "두 규칙의 예측은 같다"를 성립시키기 위해): 두 규칙 모두 a·b는 n_W ≥ k인 행에서 적합하고, 대체 규칙은 같은 a·b를 전체 기간 m의 x에 적용한다.
- 기존 B2(49c02f8)·시연 범위는 바꾸지 않는다. ID·행 값은 싣지 않는다(집계만). 사용법(저장소 루트): python scripts/60_recent_m_5km.py  (run_all에는 넣지 않음)
출력: docs/recent_m_5km.md
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import ridge_lib  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m39 = _load("39_baselines.py", "m39")
m43 = _load("43_within_person_5km.py", "m43")
m55 = _load("scripts/55_interval_calibration_5km.py", "m55")
m42 = m43.m42
md_table = m42.md_table
lad_fit = m39.lad_fit
DOC = Path("docs") / "recent_m_5km.md"
SPEC_COMMIT = "5f3be15"
DAY = 86400
WS, KS = (90, 180, 365), (1, 2, 3)
N_FOLDS, SEED, N_BOOT = 5, 42, 2000


def recent_features(rows, sess):
    """행(uid, t)마다 최근 W일 안 유효 5km 개수 n_W와 중앙값 m_W, 전체 기간 m_full. sess: {uid: (종료 시각 배열, T5 배열)}."""
    out = {W: (np.zeros(len(rows), int), np.full(len(rows), np.nan)) for W in WS}
    full = np.zeros(len(rows))
    for i, (u, t) in enumerate(zip(rows.uid.to_numpy(), rows.t.to_numpy())):
        e, v = sess[int(u)]
        past = e < t
        full[i] = np.median(v[past])
        for W in WS:
            sel = past & (e >= t - W * DAY)
            out[W][0][i] = int(sel.sum())
            out[W][1][i] = np.median(v[sel]) if sel.any() else np.nan
    return out, full


def umean(uid_inv, U, err, mask=None):
    """사용자 평균 MAE (mask 안 행만). 반환 (사용자별 평균 배열, 그 사용자의 행 수 배열)."""
    w = np.ones(len(err)) if mask is None else mask.astype(float)
    s = np.bincount(uid_inv, weights=np.where(w > 0, err, 0.0), minlength=U)
    n = np.bincount(uid_inv, weights=w, minlength=U)
    return s, n


def user_diff(uid_inv, U, e_a, e_b, mask):
    """mask 행에서 사용자별 (a 오차 평균 − b 오차 평균). 행이 없는 사용자는 제외. 반환 (차이 배열(사용자 U개, 없으면 NaN), mae_a, mae_b 사용자 평균)."""
    sa, n = umean(uid_inv, U, e_a, mask)
    sb, _ = umean(uid_inv, U, e_b, mask)
    ok = n > 0
    da = np.full(U, np.nan)
    db = np.full(U, np.nan)
    da[ok], db[ok] = sa[ok] / n[ok], sb[ok] / n[ok]
    return da - db, da, db


def boot_ci(d, idx):
    """사용자별 차이 d(NaN 포함) -> (평균, 평균 CI, 중앙값, 중앙값 CI, 사용자 수). 사용자 복원추출 idx(2,000 x U)."""
    ok = ~np.isnan(d)
    mean, med = float(np.nanmean(d)), float(np.nanmedian(d))
    # 행이 없는 사용자가 뽑히면 제외하고 평균(층별 CI와 같은 방식: 뽑힌 사용자 중 값이 있는 사용자)
    dd = d[idx]
    bm = np.nanmean(dd, axis=1)
    bmed = np.nanmedian(dd, axis=1)
    return mean, tuple(np.percentile(bm, [2.5, 97.5])), med, tuple(np.percentile(bmed, [2.5, 97.5])), int(ok.sum())


def cov_stats(inside, n_by_user_inv, U, idx):
    """풀링 포함률과 사용자 단위 bootstrap CI."""
    inv = n_by_user_inv
    n = np.bincount(inv, minlength=U).astype(float)
    k = np.bincount(inv, weights=inside.astype(float), minlength=U)
    boot = k[idx].sum(1) / n[idx].sum(1)
    return float(inside.mean()), tuple(np.percentile(boot, [2.5, 97.5]))


def verdict(lo, hi):
    if lo > 0:
        return "B2r이 B2보다 나쁨(CI 하한 > 0)"
    if hi < 0:
        return "B2r이 B2보다 좋음(CI 상한 < 0)"
    return "우열 확정 불가(동등성 입증 아님)"


def fmt_ci(lo, hi, nd=3):
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def main():
    d, trs, te, grp, cache = m43.load_base()
    ds = m43.main_dataset(d, grp, cache)
    m43.verify_against_final(ds, trs, te)
    # 사용자별 유효 5km (종료 시각, T5)
    valid = d[d.cand & np.array([not cache["ext"].get(int(s), False) for s in d.session_id])]
    sess = {}
    for u, g in valid.groupby("user_id"):
        e = (g.start_time.to_numpy(np.int64) + g.duration_sec.to_numpy(np.int64))
        v = np.array([cache["t5"][int(s)] for s in g.session_id])
        sess[int(u)] = (e, v)
    tr = ds[ds.uid.isin(trs)].sort_values(["uid", "t"]).reset_index(drop=True)
    tr_file = pd.read_pickle(Path("data") / "5km_dataset_train.pkl").sort_values(["uid", "t"]).reset_index(drop=True)
    assert len(tr) == 1720 and np.array_equal(tr.uid.to_numpy(), tr_file.uid.to_numpy()) and np.array_equal(tr.t.to_numpy(), tr_file.t.to_numpy()) and np.array_equal(tr.y.to_numpy(), tr_file.y.to_numpy())
    ts_ = ds[ds.uid.isin(te)].sort_values(["uid", "t"]).reset_index(drop=True)
    assert len(ts_) == 352 and ts_.uid.nunique() == 71
    ftr, full_tr = recent_features(tr, sess)
    fte, full_te = recent_features(ts_, sess)
    assert np.allclose(full_tr, tr.m.to_numpy()) and np.allclose(full_te, ts_.m.to_numpy()), "전체 기간 m 재계산이 데이터셋과 다름"
    p = json.loads((Path("demo") / "b2_params.json").read_text(encoding="utf-8"))

    # ---------------- 개발 자료: 폴드 밖 예측
    uid, x, y, ref = tr.uid.to_numpy(), tr.t5_ref_minus_m.to_numpy(float), tr.y.to_numpy(float), tr.t5_ref.to_numpy(float)
    users, inv = np.unique(uid, return_inverse=True)
    U = len(users)
    folds = list(ridge_lib.group_kfold_indices(uid, N_FOLDS, SEED))
    yhat_b2 = np.full(len(tr), np.nan)
    for trm, vam in folds:
        assert set(uid[trm]).isdisjoint(set(uid[vam]))
        a, b = lad_fit(x[trm], y[trm])
        yhat_b2[vam] = a + b * x[vam]
    res_b2 = y - yhat_b2
    q10, q90, _c, res55 = m55.oof_quantiles(tr_file)
    assert np.allclose(res_b2, res55) and abs(q10 - p["resid_q10"]) < 1e-9 and abs(q90 - p["resid_q90"]) < 1e-9, "B2 폴드 밖 잔차가 55번/시연 범위와 다름"
    err_b2 = np.abs(res_b2)
    idx_tr = np.random.default_rng(SEED).integers(0, U, size=(N_BOOT, U))

    cand = {}                 # (W, k) -> dict
    for W in WS:
        nW, mW = ftr[W]
        for k in KS:
            ok = nW >= k
            xr = np.where(ok, ref - mW, x)                     # 대체: n_W < k이면 전체 기간 m의 x
            yh = np.full(len(tr), np.nan)
            for trm, vam in folds:
                fit = trm & ok
                a, b = lad_fit(xr[fit], y[fit])
                yh[vam] = a + b * xr[vam]
            err = np.abs(y - yh)
            diff_c, mae_r_c, mae_b_c = user_diff(inv, U, err, err_b2, ok)           # 두 규칙이 모두 예측하는 행(n_W ≥ k)
            diff_all, mae_r_a, mae_b_a = user_diff(inv, U, err, err_b2, np.ones(len(tr), bool))
            cand[(W, k)] = dict(ok=ok, yh=yh, n_ok=int(ok.sum()), score=float(np.nanmean(diff_c)), mae_r_c=float(np.nanmean(mae_r_c)), mae_b_c=float(np.nanmean(mae_b_c)),
                                score_all=float(np.nanmean(diff_all)), mae_r_all=float(np.nanmean(mae_r_a)), users_c=int((~np.isnan(diff_c)).sum()))
    # 선택: 두 규칙이 같은 행에서 같은 예측이므로(구현 세부) 점수가 같다 -> 동률이면 대체
    best = min(cand, key=lambda wk: (cand[wk]["score"], wk))
    W_sel, k_sel = best
    rule_sel = "대체"
    c = cand[best]

    # ---------------- 선택된 설정: B2r 범위(자신의 폴드 밖 잔차)
    res_r = y - c["yh"]                                          # 대체: 모든 행 예측
    q10r, q90r = (float(v) for v in np.percentile(res_r, [10, 90]))
    in_b2 = (res_b2 >= q10) & (res_b2 <= q90)
    in_r = (res_r >= q10r) & (res_r <= q90r)
    err_r = np.abs(res_r)

    # ---------------- test (탐색 결과)
    nWt, mWt = fte[W_sel]
    ok_tr, ok_te = c["ok"], nWt >= k_sel
    nW_tr, mW_tr = ftr[W_sel]
    xr_tr = np.where(nW_tr >= k_sel, ref - mW_tr, x)
    a_r, b_r = lad_fit(xr_tr[nW_tr >= k_sel], y[nW_tr >= k_sel])
    a_full, b_full = lad_fit(x, y)
    assert abs(a_full - p["a"]) < 1e-9 and abs(b_full - p["b"]) < 1e-9, "B2 train 전체 적합이 49c02f8/시연과 다름"
    uid_t, y_t, x_t, ref_t = ts_.uid.to_numpy(), ts_.y.to_numpy(float), ts_.t5_ref_minus_m.to_numpy(float), ts_.t5_ref.to_numpy(float)
    users_t, inv_t = np.unique(uid_t, return_inverse=True)
    Ut = len(users_t)
    yh_b2_t = p["a"] + p["b"] * x_t
    saved = pd.read_pickle(m42.PRED_OUT)["cold"]
    s_ = pd.DataFrame({"uid": saved["uid"], "t": saved["t"], "p": saved["preds"]["B2"]}).sort_values(["uid", "t"])
    e_ = pd.DataFrame({"uid": ts_.uid, "t": ts_.t, "p": yh_b2_t}).sort_values(["uid", "t"])
    assert np.allclose(s_.p.to_numpy(), e_.p.to_numpy()), "B2 test 예측이 저장된 49c02f8 예측과 다름"
    xr_t = np.where(ok_te, ref_t - mWt, x_t)
    yh_r_t = a_r + b_r * xr_t
    err_b2_t, err_r_t = np.abs(y_t - yh_b2_t), np.abs(y_t - yh_r_t)
    idx_te = np.random.default_rng(SEED).integers(0, Ut, size=(N_BOOT, Ut))
    final_doc = Path("docs/final_results_5km.md").read_text(encoding="utf-8")
    sec = final_doc[final_doc.index("## A."): final_doc.index("## B.")]
    mae_doc = float(re.search(r"\| B2 \(LAD\) \| ([\d.]+) \[", sec).group(1))
    _dt, _da, mae_b2_t = user_diff(inv_t, Ut, err_b2_t, err_b2_t, np.ones(len(ts_), bool))
    assert abs(float(np.nanmean(mae_b2_t)) - mae_doc) < 5e-4, "B2 test 사용자 평균 MAE가 49c02f8과 다름"
    b2_dev_mae = float(np.nanmean(user_diff(inv, U, err_b2, err_b2, np.ones(len(tr), bool))[1]))
    assert round(b2_dev_mae, 1) == 209.3, b2_dev_mae                       # docs/b2_failure_5km.md 0절: train 폴드 밖 B2 사용자 평균 MAE 209.3

    def compare(err_rr, err_bb, inv_, U_, idx_, mask):
        dd, mr, mb = user_diff(inv_, U_, err_rr, err_bb, mask)
        mean, mci, med, medci, nu = boot_ci(dd, idx_)
        return dict(mean=mean, ci=mci, med=med, medci=medci, users=nu, n=int(mask.sum()), mae_r=float(np.nanmean(mr)), mae_b=float(np.nanmean(mb)))

    all_tr, all_te = np.ones(len(tr), bool), np.ones(len(ts_), bool)
    # (a) 대체(선택) 규칙: 둘 다 예측 = 모든 행. 보류 규칙: n_W ≥ k인 행만.
    a_dev_sub = compare(err_r, err_b2, inv, U, idx_tr, all_tr)
    a_dev_hold = compare(err_r, err_b2, inv, U, idx_tr, ok_tr)
    a_te_sub = compare(err_r_t, err_b2_t, inv_t, Ut, idx_te, all_te)
    a_te_hold = compare(err_r_t, err_b2_t, inv_t, Ut, idx_te, ok_te)
    # (c) 포함률·폭
    cov_dev_b2 = cov_stats(in_b2, inv, U, idx_tr)
    cov_dev_r = cov_stats(in_r, inv, U, idx_tr)
    in_b2_t = (y_t >= yh_b2_t + q10) & (y_t <= yh_b2_t + q90)
    in_r_t = (y_t >= yh_r_t + q10r) & (y_t <= yh_r_t + q90r)
    cov_te_b2 = cov_stats(in_b2_t, inv_t, Ut, idx_te)
    cov_te_r = cov_stats(in_r_t, inv_t, Ut, idx_te)
    assert abs(cov_te_b2[0] - p["test_coverage_cold"]) < 1e-9, "B2 test 포함률이 시연 값과 다름"
    # 보류 규칙의 범위(참고): 보류는 n_W ≥ k 행만 예측하므로 그 행의 폴드 밖 잔차로 정한다
    q10h, q90h = (float(v) for v in np.percentile(res_r[ok_tr], [10, 90]))
    in_h = (res_r[ok_tr] >= q10h) & (res_r[ok_tr] <= q90h)
    in_h_t = (y_t[ok_te] >= yh_r_t[ok_te] + q10h) & (y_t[ok_te] <= yh_r_t[ok_te] + q90h)

    # ---------------- 문서
    def rowa(label, a, ncell):
        return {"자료": label, "비교한 시점 수": f"{a['n']:,}", "사용자": a["users"], "B2 사용자 평균 MAE": f"{a['mae_b']:.3f}", "B2r 사용자 평균 MAE": f"{a['mae_r']:.3f}",
                "[주] 사용자별 (B2r − B2) 평균 [95% CI]": f"{a['mean']:+.3f} {fmt_ci(*a['ci'])}", "주 판정": verdict(*a["ci"]),
                "[보조] 사용자별 오차 차이 중앙값 기준 [95% CI]": f"{a['med']:+.3f} {fmt_ci(*a['medci'])}"}

    out = ["# 사후 탐색 결과: 최근 기간 기준치 B2r (5km 과제)", "",
           f"생성: `scripts/60_recent_m_5km.py`. 사양: `docs/posthoc_spec_recent_m.md`(커밋 {SPEC_COMMIT}, 결과 계산 전 고정). **사후 탐색**이며 상관관계까지만 주장한다. 선택은 train 폴드 밖(개발)으로만 했고 test cold-start(352개·71명)는 **선택에 쓰지 않았으나 이미 열람한 자료의 탐색 결과**다. "
           "주 결론(49c02f8)·B2 시연(식·파라미터)·명목 80% 범위·조건별 안내 규칙은 **바꾸지 않았다**. 모든 평가는 \"다음 7일 안에 적격 5km 기록이 관측된 경우\"에 조건부다. ID·행 값은 싣지 않았다.", "",
           "- B2r: B2와 같은 식에서 m 대신 m_W(t 이전에 끝난 최근 W일 안 유효 5km의 T5 중앙값). 사용자 평균 MAE의 단위는 초이고 차이가 음수이면 B2r 쪽이 낮다.",
           "- 개발 자료 = train 1,720개·278명, 사용자 단위 GroupKFold(5, seed 42), 폴드마다 학습 부분에서 a·b를 다시 적합한 폴드 밖 예측. 구현 세부: 두 규칙(대체/보류) 모두 a·b는 n_W ≥ k인 행에서 적합하고, 대체는 같은 a·b를 전체 기간 m의 x에 적용한다(그래서 n_W ≥ k인 행에서 두 규칙의 예측이 같다).",
           f"- **검증(assert)**: B2의 폴드 밖 잔차가 55번 결과와 같고 q10·q90이 `demo/b2_params.json`과 같음, B2 train 전체 적합 a = {p['a']:.3f}, b = {p['b']:.4f}가 시연·49c02f8과 같음, B2 test 예측이 저장된 49c02f8 예측과 같고 사용자 평균 MAE {mae_doc:.3f}가 `docs/final_results_5km.md`와 같음, 개발 자료 B2 사용자 평균 MAE {b2_dev_mae:.1f}가 `docs/b2_failure_5km.md`와 같음, 전체 기간 m 재계산이 데이터셋의 m과 같음.",
           "- 사용자 단위 bootstrap: 사용자 복원추출 2,000회, seed 42, 백분위 95% CI(모든 비교에 같은 추출). 사용자별 차이는 그 사용자의 비교한 시점들의 평균 MAE 차이다.", "",
           "## 선택된 설정 (개발 자료만으로)", "",
           f"- **W = {W_sel}일, k = {k_sel}, 규칙 = {rule_sel}**. 선택 점수(두 규칙이 모두 예측하는 행에서 사용자별 (B2r MAE − B2 MAE)의 평균) = {c['score']:+.3f}초.",
           "- 두 규칙의 예측은 n_W ≥ k인 행에서 같아 점수가 같으므로 사양대로 동률 규칙(대체: 예측 가능 시점을 줄이지 않음)을 택했다. 보류 규칙은 비교용으로 함께 보고한다.",
           f"- 기록이 있는 한 기준 기록은 항상 직전 28일 안에 있어 W ≥ 28이면 n_W ≥ 1이다. 그래서 k = 1이면 두 규칙 모두 모든 시점을 예측한다(n_W < 1이 없음).",
           "- **후보를 9개(규칙 포함 18개 조합) 보고 하나를 골랐으므로 개발 자료의 개선은 선택 편향으로 우연일 수 있다.**", "",
           "## (d) 9개 후보 전체의 개발 자료 표 (선택 과정 공개)", "",
           md_table([{"W(일)": W, "k": k, "예측 행(n_W ≥ k)": f"{cand[(W, k)]['n_ok']:,} ({100 * cand[(W, k)]['n_ok'] / len(tr):.1f}%)", "사용자(그 행)": cand[(W, k)]["users_c"],
                      "B2r 사용자 평균 MAE (그 행)": f"{cand[(W, k)]['mae_r_c']:.3f}", "B2 사용자 평균 MAE (같은 행)": f"{cand[(W, k)]['mae_b_c']:.3f}", "선택 점수 (B2r − B2, 같은 행)": f"{cand[(W, k)]['score']:+.3f}",
                      "대체 규칙: 전체 행 B2r − B2": f"{cand[(W, k)]['score_all']:+.3f}", "선택": "◀ 선택" if (W, k) == best else ""} for W in WS for k in KS]), "",
           f"- B2 전체 행(1,720개) 사용자 평균 MAE = {b2_dev_mae:.3f}초. \"예측 행\"은 보류 규칙이 예측하는 행이며 대체 규칙은 모든 행을 예측한다. 후보마다 행 집합이 달라 원 MAE는 후보 사이에서 비교하지 않고 같은 행의 B2와의 차이(선택 점수)로 비교했다.", "",
           "## (a) 둘 다 예측한 같은 시점에서 B2r − B2", "",
           md_table([rowa("train 폴드 밖 (개발) — 대체(선택): 모든 시점", a_dev_sub, 0), rowa(f"train 폴드 밖 (개발) — 보류: n_W ≥ {k_sel}인 시점", a_dev_hold, 0),
                     rowa("test cold-start (탐색, 이미 열람) — 대체(선택): 모든 시점", a_te_sub, 0), rowa(f"test cold-start (탐색, 이미 열람) — 보류: n_W ≥ {k_sel}인 시점", a_te_hold, 0)]), "",
           "- 주 판정은 평균 차이의 CI로만 한다. CI가 0을 포함하면 \"우열 확정 불가(동등성 입증 아님)\"이다. 중앙값은 \"사용자별 오차 차이 중앙값 기준\"의 보조 값이다.",
           "- test의 a·b는 train 1,720개 전체에서 적합했다(B2r a = %.3f초, b = %.4f; B2는 시연의 a = %.3f초, b = %.4f)." % (a_r, b_r, p["a"], p["b"]), "",
           "## (b) 예측 가능 시점 비율 (B2 대비)", "",
           md_table([{"자료": "train 폴드 밖 (개발)", "B2 예측 시점": f"{len(tr):,}", "B2r 대체(선택)": f"{len(tr):,} (100.0%)", "B2r 보류": f"{int(ok_tr.sum()):,} ({100 * ok_tr.mean():.1f}%)"},
                     {"자료": "test cold-start (탐색, 이미 열람)", "B2 예측 시점": f"{len(ts_):,}", "B2r 대체(선택)": f"{len(ts_):,} (100.0%)", "B2r 보류": f"{int(ok_te.sum()):,} ({100 * ok_te.mean():.1f}%)"}]), "",
           "## (c) 명목 80% 범위 포함률과 평균 폭", "",
           f"- B2 범위 = ŷ + [{q10:.1f}, {q90:.1f}]초(시연 그대로). B2r 범위 = ŷ_r + [{q10r:.1f}, {q90r:.1f}]초(B2r 자신의 폴드 밖 잔차의 10%·90% 분위수, 55번과 같은 절차). 보류 규칙의 범위(참고, n_W ≥ {k_sel}인 행의 잔차) = [{q10h:.1f}, {q90h:.1f}]초.", "",
           md_table([{"자료": "train 폴드 밖 (개발)", "기준치": "B2", "관측 수": f"{len(tr):,}", "사용자": U, "포함률 [95% CI]": f"{100 * cov_dev_b2[0]:.1f}% [{100 * cov_dev_b2[1][0]:.1f}%, {100 * cov_dev_b2[1][1]:.1f}%]", "평균 폭(초)": f"{q90 - q10:.1f}"},
                     {"자료": "train 폴드 밖 (개발)", "기준치": "B2r (대체)", "관측 수": f"{len(tr):,}", "사용자": U, "포함률 [95% CI]": f"{100 * cov_dev_r[0]:.1f}% [{100 * cov_dev_r[1][0]:.1f}%, {100 * cov_dev_r[1][1]:.1f}%]", "평균 폭(초)": f"{q90r - q10r:.1f}"},
                     {"자료": "test cold-start (탐색, 이미 열람)", "기준치": "B2", "관측 수": f"{len(ts_):,}", "사용자": Ut, "포함률 [95% CI]": f"{100 * cov_te_b2[0]:.1f}% [{100 * cov_te_b2[1][0]:.1f}%, {100 * cov_te_b2[1][1]:.1f}%]", "평균 폭(초)": f"{q90 - q10:.1f}"},
                     {"자료": "test cold-start (탐색, 이미 열람)", "기준치": "B2r (대체)", "관측 수": f"{len(ts_):,}", "사용자": Ut, "포함률 [95% CI]": f"{100 * cov_te_r[0]:.1f}% [{100 * cov_te_r[1][0]:.1f}%, {100 * cov_te_r[1][1]:.1f}%]", "평균 폭(초)": f"{q90r - q10r:.1f}"}]), "",
           "- 개발 자료의 포함률은 범위를 같은 폴드 밖 잔차로 정했으므로 구성상 약 80%다(독립 평가가 아님). test 포함률의 CI가 80%를 포함하면 \"명목 80%에서 벗어났다고 확정할 수 없음\"이다.", "",
           "## 해석 (사양 4절의 규칙)", "",
           f"- 개발 자료 (a): 대체(선택) 평균 차이 {a_dev_sub['mean']:+.3f} {fmt_ci(*a_dev_sub['ci'])} → **{verdict(*a_dev_sub['ci'])}**. test(탐색): {a_te_sub['mean']:+.3f} {fmt_ci(*a_te_sub['ci'])} → **{verdict(*a_te_sub['ci'])}**.",
           (f"- 9개 후보 모두 개발 자료의 선택 점수가 양수였다(가장 낮은 점수 {c['score']:+.3f}초; 같은 행에서 B2r의 점추정이 B2보다 높음). 어느 후보도 B2보다 낮은 점추정을 보이지 않았다." if all(v["score"] > 0 for v in cand.values()) else "- 일부 후보의 선택 점수가 음수였다(표 참고)."),
           "- 후보를 9개(규칙 포함 18개 조합) 보고 하나를 골랐으므로 선택된 설정의 개발 자료 값은 낙관적이며 우연일 수 있다. test는 선택에 쓰지 않았지만 이미 열람한 자료의 탐색 결과다.",
           "- **이 결과로 시연(B2, 범위, 안내)과 기존 결론(49c02f8)을 바꾸지 않는다.** 개선돼 보이더라도 새 자료로 확인하는 향후 과제로만 둔다. B2r이 B2보다 낫다고 쓰지 않으며, 최근 기간을 쓰는 것이 원인이라고 해석하지 않는다.", ""]
    DOC.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC)
    print("selected", best, rule_sel, "score", round(c["score"], 3))
    print("dev (a)", round(a_dev_sub["mean"], 3), a_dev_sub["ci"], "| test (a)", round(a_te_sub["mean"], 3), a_te_sub["ci"])


if __name__ == "__main__":
    main()
