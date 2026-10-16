# -*- coding: utf-8 -*-
"""
10월 3주차 4번: 코칭 판정 정의 문서의 혼동행렬 + 판정 보류 실험.
**모든 목표는 합성 시나리오이며 실제 사용자 목표 기반 코칭 성능이 아니다. 불일치율은 고정 규칙에 대한 판정 일치도다.**
설계는 실행 전에 docs/week3_predeclaration.md 8절에 고정했다(결과를 보고 바꾸지 않음). 새 예측 모델은 없고,
sklearn 의존이 없다(numpy / ridge_lib / gbt_lib / torch). 29번의 %TEMP% 예측 캐시는 쓰지 않고 예측을 새로 계산한다.
중단 조건: 이 스크립트가 재현한 강제 판정 불일치율·CI가 docs/coaching_results.md 3-2절과 다르면 중단.

산출물: docs/coaching_abstain_results.md (평가 행별 예측값은 파일로 저장하지 않는다)
"""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
import coaching_lib as cl
import ridge_lib as rl


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m21 = _load("m21", "21_baseline_integrated_report.py")
m25 = _load("m25", "25_tree_models.py")
m27 = _load("m27", "27_mlp_torch.py")
m28 = _load("m28", "28_model_comparison.py")
m29 = _load("m29", "29_coaching.py")
m24 = m25.m24

OUTPUT_MD_PATH = Path("docs") / "coaching_abstain_results.md"
DELTAS, DEADLINES = m29.DELTAS, m29.DEADLINES
MAIN_DELTA, MAIN_D = 0.05, 8
LEVELS = [0.80, 0.90]
ABS_PREDS = ["median5", "blend", "gbm", "mlp"]       # 보류 실험 대상(사전 선언)
CM_PREDS = ["median5", "hist_median", "const", "gbm", "mlp"]  # 혼동행렬 대상(사전 선언)
NAMES = {"median5": "median-5", "hist_median": "전체 이력 중앙값", "blend": "(d) 0.5·m5+0.5·전체이력중앙값 [탐색적 baseline]",
         "const": "상수 규칙(학습/이력 구간 최빈)", "gbm": "GBM22", "mlp": "MLP22"}
SHORT = {"median5": "median-5", "blend": "(d)", "gbm": "GBM22", "mlp": "MLP22", "hist_median": "전체이력중앙값", "const": "상수 규칙"}
N_BOOT, BOOT_SEED = 2000, 42
N_CTRL = 1000
CATS = cl.CATEGORIES
SPLITS = ["cold-start", "personalized"]


def log(*a):
    print(*a, flush=True)


def pct(x):
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


# ===================== 보정 잔차 (out-of-sample) =====================
def calibration_residuals(split, tr_s, p_train_s):
    """반환: {pred: 절대 잔차 배열}, 보정 자료 설명, 보정 윈도우 수."""
    if split == "cold-start":
        X, y, uid = m25.samples_to_xy(tr_s)
        med, hist = m25.median5_from_X(X), X[:, 18]
        res = {"median5": y - med, "blend": y - (0.5 * med + 0.5 * hist)}
        oof_g = np.full(len(y), np.nan)
        oof_m = np.full(len(y), np.nan)
        mp, ep = m28.MLP_COLD_PARAMS, m28.MLP_COLD_EPOCHS
        resid_mlp = m28.MLP_COLD_FORM == "median5_residual"
        for k, (tr, va) in enumerate(rl.group_kfold_splits(uid, m25.N_FOLDS, m25.CV_SEED)):
            log(f"  [cold-start OOF] fold {k + 1}/{m25.N_FOLDS}: train {len(tr)} / held-out {len(va)}")
            g = m25.build_gbt(m28.GBM_COLD_PARAMS, "absolute_error").fit(X[tr], y[tr])  # cold-start GBM22 = absolute_error 직접
            oof_g[va] = g.predict(X[va])
            model, scaler = m27.train_mlp_fixed_epochs(X[tr], (y[tr] - med[tr]) if resid_mlp else y[tr], mp["n_hidden_layers"],
                                                       mp["width"], mp["weight_decay"], m28.MLP_SEED, ep)
            oof_m[va] = (med[va] if resid_mlp else 0.0) + m27.predict_mlp(model, scaler, X[va])
        assert np.isfinite(oof_g).all() and np.isfinite(oof_m).all()
        res["gbm"], res["mlp"] = y - oof_g, y - oof_m
        desc = "train 사용자 전체 윈도우의 GroupKFold(ridge_lib, 5-fold, 시드 42) OOF 잔차(모델은 fold마다 해당 fold 사용자를 제외하고 학습)"
        return {k: np.abs(v) for k, v in res.items()}, desc, len(y)
    inner_tr, inner_val = m25.personalized_inner_split(p_train_s)
    Xi, yi, _ = m25.samples_to_xy(inner_tr)
    Xv, yv, _ = m25.samples_to_xy(inner_val)
    medi, medv, histv = m25.median5_from_X(Xi), m25.median5_from_X(Xv), Xv[:, 18]
    res = {"median5": yv - medv, "blend": yv - (0.5 * medv + 0.5 * histv)}
    log(f"  [personalized 보정] inner-train {len(inner_tr)} / inner-val {len(inner_val)}")
    g = m25.build_gbt(m28.GBM_PERS_PARAMS, "absolute_error").fit(Xi, yi - medi)  # personalized GBM22 = median5+잔차
    res["gbm"] = yv - (medv + g.predict(Xv))
    mp, ep = m28.MLP_PERS_PARAMS, m28.MLP_PERS_EPOCHS
    resid_mlp = m28.MLP_PERS_FORM == "median5_residual"
    model, scaler = m27.train_mlp_fixed_epochs(Xi, (yi - medi) if resid_mlp else yi, mp["n_hidden_layers"], mp["width"],
                                               mp["weight_decay"], m28.MLP_SEED, ep)
    res["mlp"] = yv - ((medv if resid_mlp else 0.0) + m27.predict_mlp(model, scaler, Xv))
    desc = ("history 윈도우를 사용자별 시간순 앞 80%(inner-train)/뒤 20%(inner-val)로 나눠, inner-train으로 학습한 모델의 inner-val 잔차 "
            "(inner-val은 학습에 쓰이지 않고 hold-out보다 이전 시점)")
    return {k: np.abs(v) for k, v in res.items()}, desc, len(yv)


# ===================== 통계 =====================
def make_W(nu):
    rng = np.random.default_rng(BOOT_SEED)
    return np.stack([np.bincount(rng.integers(0, nu, nu), minlength=nu) for _ in range(N_BOOT)]).astype(float)


def ratio_draws(W, num, den):
    n, d = W @ num, W @ den
    out = np.full(len(n), np.nan)
    ok = d > 0
    out[ok] = n[ok] / d[ok]
    return out


def ci(a):
    a = a[np.isfinite(a)]
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def fmt_ci(point, draws, as_pct=True, signed=False):
    lo, hi = ci(draws)
    if as_pct:
        f = (lambda x: f"{100 * x:+.1f}%p") if signed else (lambda x: f"{100 * x:.1f}%")
    else:
        f = lambda x: f"{x:.1f}"
    return f"{f(point)} [{f(lo)}, {f(hi)}]"


def parse_prior_rows():
    doc = Path("docs/coaching_results.md").read_text(encoding="utf-8")
    sec = doc[doc.index("### 3-2."):doc.index("### 3-3.")]
    rows = {}
    for line in sec.split("\n"):
        if not line.startswith("|") or line.startswith("|---") or "평가조건" in line:
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        rows[(c[0], int(c[1]), c[2], c[3])] = (c[4], c[5], int(c[6]), int(c[7]))
    return rows


def main():
    md = ["# 코칭 혼동행렬 + 판정 보류 실험 (10월 3주차 4번)\n\n"]
    md.append("> **모든 목표는 합성 시나리오다. 이 데이터에는 사용자가 입력한 실제 목표가 없으며, 아래 결과는 실제 사용자 목표 기반 코칭 성능이 아니다.** "
              "불일치율은 **고정 규칙에 대한 판정 일치도**이며 코칭 효과나 목표 달성의 검증이 아니다. 경과 시간 기준 페이스가 느린 세션을 체력 저하나 목표 실패로 해석하지 않는다. "
              "판정 정의는 [`coaching_definition.md`](coaching_definition.md), 사전 선언은 `week3_predeclaration.md` 8절, 기존 강제 판정 결과는 [`coaching_results.md`](coaching_results.md).\n\n")
    main_df = config.load_main_sessions()
    md.append(f"입력 메인 데이터: `{config.MAIN_DATA_PATH}` (세션 {len(main_df)}건, 사용자 {main_df['user_id'].nunique()}명, assert 통과). "
              "예측은 29번의 `%TEMP%` 캐시를 쓰지 않고 이번 실행에서 새로 계산했다(GBM22/MLP22는 기존 하이퍼파라미터·에폭, 재탐색 없음).\n\n")

    train = pd.read_pickle(m24.EXT_TRAIN_PATH)
    test = pd.read_pickle(m24.EXT_TEST_PATH)
    p_train = pd.read_pickle(m24.EXT_P_TRAIN_PATH)
    p_test = pd.read_pickle(m24.EXT_P_TEST_PATH)
    data_splits = {"cold-start": (train, test), "personalized": (p_train, p_test)}

    # ---- 상수 규칙용 학습/이력 구간 정보 (29번과 동일) ----
    train_ref = {}
    for split, smp in [("cold-start", train), ("personalized", p_train)]:
        X_s, y_s, _ = m25.samples_to_xy(smp)
        le = np.array([s["last_input_session_time"] for s in smp], dtype=float)
        ts = np.array([s["target_start_time"] for s in smp], dtype=float)
        assert (le < ts).all()
        train_ref[split] = {"cur": m25.median5_from_X(X_s), "y": y_s,
                            "weeks": np.array([cl.weeks_until_target(a, b) for a, b in zip(le, ts)])}

    D = {}      # 테스트 데이터/예측
    CAL = {}    # 보정 잔차
    for split in SPLITS:
        tr_s, te_s = data_splits[split]
        log(f"\n===== {split}: 테스트 예측 새로 계산 =====")
        D[split] = m29.build_split_data(split, tr_s, te_s)
        log(f"===== {split}: 보정 잔차 계산 =====")
        CAL[split] = calibration_residuals(split, tr_s, p_train if split == "personalized" else None)

    prior = parse_prior_rows()

    # ---- 보정 요약 ----
    idx_after_header = len(md)  # 0절(기존 결과 재현 검증)은 계산 후 여기에 삽입
    md.append("## 1. 구간 보정 (절대 잔차의 분위수 q, 구간 = 예측 ± q)\n")
    rows = []
    for split in SPLITS:
        res, desc, ncal = CAL[split]
        for p in ABS_PREDS:
            q80, q90 = np.quantile(res[p], 0.8), np.quantile(res[p], 0.9)
            rows.append({"평가조건": split, "예측기": SHORT[p], "보정 윈도우 수": ncal, "q80 (초/km)": round(float(q80), 3), "q90 (초/km)": round(float(q90), 3),
                         "보정 잔차 MAE (참고)": round(float(res[p].mean()), 2)})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n")
    for split in SPLITS:
        md.append(f"- **{split} 보정 자료**: {CAL[split][1]}.\n")
    md.append("- 구간은 (예측기, 평가 조건, 수준)마다 하나의 전역 상수 q를 쓰므로 구간 폭(2q)은 윈도우와 무관하게 일정하다. 최종 평가 모델은 보정용 모델과 학습 자료 크기가 달라(cold-start: OOF 학습은 train 사용자의 80%, personalized: 보정용은 history의 80%) 잔차 분포가 약간 다를 수 있고, personalized의 보정 구간은 hold-out보다 이른 시점이라 시간에 따른 분포 변화를 반영하지 못한다(사전 선언한 한계).\n\n")
    idx_q_note = len(md)  # q/폭 검증 노트는 계산 후 여기에 삽입

    # ---- 조건 루프 ----
    cm_main = {}            # (split, pred) -> matrix at main cond
    count_rows_other = []   # 건수 표
    abs_rows = []           # 보류 실험 전 조건
    width_errs = []
    repro_checked = 0
    for split in SPLITS:
        d = D[split]
        res_cal, _, _ = CAL[split]
        q = {(p, lv): float(np.quantile(res_cal[p], lv)) for p in ABS_PREDS for lv in LEVELS}
        for Dl in DEADLINES:
            keep = d["weeks"] <= Dl
            codes, nu = m29.user_codes(d["uid"][keep])
            W = make_W(nu)
            n_all = np.bincount(codes).astype(float)
            ti = train_ref[split]
            tkeep = ti["weeks"] <= Dl
            for delta in DELTAS:
                ref = m29.category_idx(d["y"], d["cur"], delta, d["weeks"], Dl)[keep]
                tref = m29.category_idx(ti["y"], ti["cur"], delta, ti["weeks"], Dl)[tkeep]
                const_cat = int(np.argmax(np.bincount(tref, minlength=4)))
                cat_pt = {p: m29.category_idx(d["preds"][p], d["cur"], delta, d["weeks"], Dl)[keep] for p in ["median5", "hist_median", "blend", "gbm", "mlp"]}
                cat_pt["const"] = np.full(keep.sum(), const_cat)
                is_main = abs(delta - MAIN_DELTA) < 1e-9 and Dl == MAIN_D

                # ---- 기존 결과 재현 검증(중단 조건) ----
                for p in ["median5", "hist_median", "blend", "gbm", "mlp", "const"]:
                    ind = (cat_pt[p] != ref).astype(float)
                    bs = m29.boot_rate_ci(ind, codes, nu)
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    name = m29.PRED_NAMES[p] if p != "const" else "상수 규칙(학습/이력 구간 최빈)"
                    key = (split, Dl, f"{delta:.0%}", name)
                    got = (pct(ind.mean()), f"[{pct(lo)},{pct(hi)}]", nu, int(keep.sum()))
                    want = prior[key]
                    if p == "const":
                        want = (want[0], want[1], want[2], want[3])
                    if got != want:
                        raise SystemExit(f"[중단] 기존 coaching_results.md와 불일치: {key} 재계산 {got} vs 문서 {want}")
                    repro_checked += 1

                # ---- 혼동행렬/건수 ----
                for p in CM_PREDS:
                    M = np.bincount(ref * 4 + cat_pt[p], minlength=16).reshape(4, 4)
                    if is_main:
                        cm_main[(split, p)] = M
                    else:
                        pass
                actual_counts = np.bincount(ref, minlength=4)
                row_a = {"평가조건": split, "데드라인": Dl, "δ": f"{delta:.0%}", "대상": "실제 판정(참조)"}
                row_a.update({c: int(actual_counts[i]) for i, c in enumerate(CATS)})
                row_a["n"] = int(keep.sum())
                if not is_main:
                    count_rows_other.append(row_a)
                    for p in CM_PREDS:
                        r = {"평가조건": split, "데드라인": Dl, "δ": f"{delta:.0%}", "대상": f"예측 판정: {SHORT[p]}"}
                        pc = np.bincount(cat_pt[p], minlength=4)
                        r.update({c: int(pc[i]) for i, c in enumerate(CATS)})
                        r["n"] = int(keep.sum())
                        count_rows_other.append(r)
                else:
                    cm_main[(split, "_actual")] = actual_counts

                # ---- 판정 보류 실험 ----
                for p in ABS_PREDS:
                    pred = d["preds"][p]
                    force_ind = (cat_pt[p] != ref).astype(float)
                    s_force = np.bincount(codes, weights=force_ind)
                    forced_draws = ratio_draws(W, s_force, n_all)
                    forced_pt = force_ind.mean()
                    for lv in LEVELS:
                        qq = q[(p, lv)]
                        lo_c = m29.category_idx(pred - qq, d["cur"], delta, d["weeks"], Dl)[keep]
                        hi_c = m29.category_idx(pred + qq, d["cur"], delta, d["weeks"], Dl)[keep]
                        assert (lo_c <= hi_c).all()
                        decided = lo_c == hi_c
                        assert (cat_pt[p][decided] == lo_c[decided]).all()  # 점예측 범주는 구간 안에 있으므로 판정 범주와 같다
                        dec_mis = ((lo_c != ref) & decided).astype(float)
                        n_dec = np.bincount(codes, weights=decided.astype(float))
                        s_dec_mis = np.bincount(codes, weights=dec_mis)
                        dec_draws = ratio_draws(W, s_dec_mis, n_dec)
                        abst_draws = 1.0 - ratio_draws(W, n_dec, n_all)
                        n_dec_tot = int(decided.sum())
                        abst_pt = 1.0 - decided.mean()
                        dec_pt = dec_mis.sum() / n_dec_tot if n_dec_tot > 0 else float("nan")
                        diff_draws = dec_draws - forced_draws
                        # 사후 통제(8-4): 판정된 윈도우에서 같은 윈도우의 상수 규칙 / median-5 판정 불일치율
                        c_ind = ((cat_pt["const"] != ref) & decided).astype(float)
                        m_ind = ((cat_pt["median5"] != ref) & decided).astype(float)
                        c_draws = ratio_draws(W, np.bincount(codes, weights=c_ind), n_dec)
                        m_draws = ratio_draws(W, np.bincount(codes, weights=m_ind), n_dec)
                        c_pt = c_ind.sum() / n_dec_tot if n_dec_tot > 0 else float("nan")
                        m_pt = m_ind.sum() / n_dec_tot if n_dec_tot > 0 else float("nan")
                        # 구간 폭 검증: 실제 상·하한의 폭 == 2q
                        w_emp = float(np.mean((pred[keep] + qq) - (pred[keep] - qq)))
                        width_errs.append(abs(w_emp - 2 * qq))
                        # 포함률 / 폭 (keep 윈도우 기준)
                        y_k = d["y"][keep]
                        cover = float(np.mean(np.abs(y_k - pred[keep]) <= qq))
                        # 통제: 같은 개수 무작위 보류
                        rng = np.random.default_rng(BOOT_SEED)
                        n = int(keep.sum())
                        n_abs = n - n_dec_tot
                        ctrl = np.empty(N_CTRL)
                        for r in range(N_CTRL):
                            idx = rng.permutation(n)
                            rest = idx[n_abs:]
                            ctrl[r] = force_ind[rest].mean() if len(rest) else np.nan
                        ctrl_mean = float(np.nanmean(ctrl))
                        c_lo, c_hi = np.nanpercentile(ctrl, [2.5, 97.5]) if len(ctrl) and np.isfinite(ctrl).any() else (np.nan, np.nan)
                        abs_rows.append({
                            "평가조건": split, "데드라인": Dl, "δ": f"{delta:.0%}", "예측기": SHORT[p], "수준": f"{int(lv * 100)}%",
                            "q": qq, "n": n, "판정 n": n_dec_tot, "#users": nu,
                            "보류율": abst_pt, "보류율_ci": fmt_ci(abst_pt, abst_draws),
                            "판정 불일치": dec_pt, "판정 불일치_ci": fmt_ci(dec_pt, dec_draws) if np.isfinite(dec_pt) else "-",
                            "강제 불일치": forced_pt, "강제 불일치_ci": fmt_ci(forced_pt, forced_draws),
                            "차이": dec_pt - forced_pt if np.isfinite(dec_pt) else float("nan"),
                            "차이_ci": fmt_ci(dec_pt - forced_pt, diff_draws, signed=True) if np.isfinite(dec_pt) else "-",
                            "포함률": cover, "평균 구간 폭": w_emp,
                            "상수_dec": c_pt, "상수_dec_ci": fmt_ci(c_pt, c_draws) if n_dec_tot > 0 else "-",
                            "m5_dec": m_pt, "m5_dec_ci": fmt_ci(m_pt, m_draws) if n_dec_tot > 0 else "-",
                            "예측-상수_ci": fmt_ci(dec_pt - c_pt, dec_draws - c_draws, signed=True) if n_dec_tot > 0 else "-",
                            "판정 범주 분포": " / ".join(f"{CATS[i]} {int(c)}" for i, c in enumerate(np.bincount(lo_c[decided], minlength=4))),
                            "통제 평균": ctrl_mean, "통제 lo": float(c_lo), "통제 hi": float(c_hi),
                            "main": is_main,
                        })
                log(f"  {split} D={Dl} δ={delta:.0%} 완료")
    log(f"[중단 조건] 기존 coaching_results.md 3-2절 {repro_checked}개 행(불일치율·CI·사용자 수·윈도우 수) 재현 일치")

    md[idx_after_header:idx_after_header] = [
        "## 0. 기존 결과 재현 검증 (중단 조건)\n",
        f"이 스크립트가 새로 계산한 강제 판정 불일치율·95% CI·사용자 수·윈도우 수가 `docs/coaching_results.md` 3-2절의 {repro_checked}개 행(6개 예측기/규칙 × 12개 조건)과 표시 자릿수에서 모두 일치했다 "
        "(`coaching_results.md`의 수치는 불변; 하나라도 다르면 이 스크립트는 중단되도록 되어 있다).\n\n"]
    idx_q_note += 2  # 위에서 2개 항목을 앞쪽에 삽입했으므로 보정

    # ---- 혼동행렬 (주 조건) ----
    md.append("## 2. 범주별 건수와 혼동행렬\n")
    md.append("행 = 실제 6번째 세션 페이스로 판정한 범주, 열 = 예측 페이스로 판정한 범주. 각 셀은 `건수 (행 정규화 %)`. 아래는 **주 조건 δ=5%·데드라인 8주**.\n\n")
    for split in SPLITS:
        act = cm_main[(split, "_actual")]
        n = int(act.sum())
        md.append(f"### 2-1 ({split}). δ5%·8주 (분모 {n}윈도우)\n")
        md.append("**실제 판정 범주의 분포**: " + ", ".join(f"{CATS[i]} {int(act[i])}건({100 * act[i] / n:.1f}%)" for i in range(4)) +
                  f". **다수 범주** = {CATS[int(np.argmax(act))]} ({100 * act.max() / n:.1f}%) — 이 범주만 항상 말하면 불일치율은 {100 * (1 - act.max() / n):.1f}%(평가 구간에서 사후에 고른 최빈값 기준 참고값; 상수 규칙은 학습/이력 구간의 최빈 범주를 쓴다).\n\n")
        for p in CM_PREDS:
            M = cm_main[(split, p)]
            rows = []
            for i in range(4):
                rs = M[i].sum()
                r = {"실제 \\ 예측": f"{CATS[i]} (n={int(rs)})"}
                for j in range(4):
                    r[CATS[j]] = f"{int(M[i, j])} ({100 * M[i, j] / rs:.1f}%)" if rs else f"{int(M[i, j])} (-)"
                rows.append(r)
            mis = 1 - np.trace(M) / M.sum()
            md.append(f"**{NAMES[p]}** — 불일치율 {100 * mis:.1f}% (대각선 밖 비율), 예측 판정 범주별 건수 {[int(x) for x in M.sum(axis=0)]}\n\n")
            md.append(m21.df_to_markdown(pd.DataFrame(rows)))
            md.append("\n\n")
    md.append("### 2-2. 나머지 조건: 범주별 건수 표 (혼동행렬 생략)\n")
    md.append(m21.df_to_markdown(pd.DataFrame(count_rows_other)))
    md.append("\n\n")

    # ---- 판정 보류 (주 조건) ----
    A = pd.DataFrame(abs_rows)
    md.append("## 3. 판정 보류 실험\n")
    md.append("판정 규칙: 구간 [예측−q, 예측+q]의 하한·상한을 각각 같은 방식(현재 능력=median-5, 같은 체크포인트)으로 범주로 변환해 **두 범주가 같으면 그 범주로 판정, 다르면 '판정 보류'**. "
              "구간 수준 80%/90%만(사전 고정). 판정 사례 불일치율의 CI는 사용자 단위 cluster bootstrap(2000회, seed 42, 판정 윈도우 수 합 분모의 비율), "
              "'차이'는 (판정 사례 불일치율 − 강제 판정 불일치율)로 같은 재표집에서 계산한 쌍체 CI. 포함률 = 테스트에서 |실제−예측| ≤ q인 윈도우 비율, 평균 구간 폭 = 2q(상수). "
              "통제 = 같은 개수의 윈도우를 무작위로 보류했을 때 나머지의 강제 판정 불일치율(1000회, seed 42, 윈도우 단위 무작위; 평균과 2.5/97.5 범위).\n\n")
    for split in SPLITS:
        sub = A[(A["평가조건"] == split) & A["main"]]
        md.append(f"### 3-1 ({split}). 주 조건 δ5%·8주 (분모 {int(sub['n'].iloc[0])}윈도우, {int(sub['#users'].iloc[0])}명)\n")
        rows = []
        for _, r in sub.iterrows():
            rows.append({"예측기": r["예측기"], "수준": r["수준"], "보류율 [CI]": r["보류율_ci"], "판정 n": r["판정 n"],
                         "판정 사례 불일치율 [CI]": r["판정 불일치_ci"], "강제 판정 불일치율 [CI]": r["강제 불일치_ci"],
                         "차이(판정−강제) [CI]": r["차이_ci"], "판정 범주 분포(건)": r["판정 범주 분포"], "구간 포함률": pct(r["포함률"]), "평균 구간 폭(초/km)": round(r["평균 구간 폭"], 3),
                         "통제: 무작위 보류 시 불일치율 평균 [95% 범위]": f"{pct(r['통제 평균'])} [{pct(r['통제 lo'])}, {pct(r['통제 hi'])}]"})
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n\n")
        md.append(f"#### 3-1b ({split}). [사후 추가 통제 — 결과를 본 뒤 추가, week3_predeclaration.md 8-4] 판정된 윈도우에서 같은 윈도우의 상수 규칙·median-5 불일치율" + "\n")
        md.append("판정된(보류되지 않은) 윈도우 부분집합에서, 예측기의 판정 사례 불일치율 옆에 같은 윈도우에 대한 상수 규칙(항상 학습/이력 구간 최빈 범주)과 median-5 판정의 불일치율을 둔다. "
                  "사용자 단위 cluster bootstrap(2000회, seed 42, 판정 윈도우 수 합 분모). 관찰용이며 인과·효과 해석은 하지 않는다.\n\n")
        rows = []
        for _, r in sub.iterrows():
            rows.append({"예측기": r["예측기"], "수준": r["수준"], "판정 n": r["판정 n"], "판정 사례 불일치율 [CI]": r["판정 불일치_ci"],
                         "같은 윈도우 상수 규칙 불일치율 [CI]": r["상수_dec_ci"], "같은 윈도우 median-5 불일치율 [CI]": r["m5_dec_ci"],
                         "예측기 − 상수 규칙 [CI]": r["예측-상수_ci"], "판정 범주 분포(건)": r["판정 범주 분포"]})
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n\n")

    md.append("### 3-2. 나머지 조건 요약 (보류율 / 판정 사례 불일치율 / 강제 불일치율 / 통제 평균)\n")
    rows = []
    for _, r in A[~A["main"]].iterrows():
        rows.append({"평가조건": r["평가조건"], "데드라인": r["데드라인"], "δ": r["δ"], "예측기": r["예측기"], "수준": r["수준"],
                     "보류율": pct(r["보류율"]), "판정 사례 불일치율": pct(r["판정 불일치"]), "강제 불일치율": pct(r["강제 불일치"]),
                     "통제 평균": pct(r["통제 평균"]), "같은 윈도우 상수 규칙 불일치율(사후)": pct(r["상수_dec"]), "같은 윈도우 median-5 불일치율(사후)": pct(r["m5_dec"]), "판정 n": r["판정 n"]})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n")

    # ---- 관찰 (사실만) ----
    md.append("## 4. 관찰 (사실만; 인과·효과 해석 없음)\n")
    for split in SPLITS:
        sub = A[(A["평가조건"] == split) & A["main"]]
        md.append(f"- **{split}, δ5%·8주**: 보류율 범위 {pct(sub['보류율'].min())}~{pct(sub['보류율'].max())}; "
                  f"판정 사례 불일치율 범위 {pct(sub['판정 불일치'].min())}~{pct(sub['판정 불일치'].max())} (강제 판정 {pct(sub['강제 불일치'].min())}~{pct(sub['강제 불일치'].max())}).\n")
        for _, r in sub.iterrows():
            below = r["판정 불일치"] < r["통제 lo"] if np.isfinite(r["판정 불일치"]) else False
            md.append(f"  - {r['예측기']} {r['수준']}: 보류율 {pct(r['보류율'])}, 판정 사례 불일치율 {pct(r['판정 불일치'])}, 통제 95% 범위 [{pct(r['통제 lo'])}, {pct(r['통제 hi'])}] → "
                      f"{'통제 범위보다 낮음' if below else '통제 범위 안 또는 위'}.\n")
    small = A[A["main"] & (A["판정 n"] < 100)]
    if len(small):
        md.append("- 판정한 윈도우 수가 100건 미만인 행(" + ", ".join(f"{r['평가조건']} {r['예측기']} {r['수준']}: {int(r['판정 n'])}건" for _, r in small.iterrows()) +
                  ")은 판정 사례 불일치율과 CI가 매우 불안정하다(100%·[100%, 100%] 같은 퇴화한 CI 포함).\n")
    md.append("- 판정된 윈도우는 구간 전체가 한 범주 안에 들어가는 윈도우이므로 판정 범주 분포(3-1절 표)가 양 끝 범주(앞서감/크게 뒤처짐)에 몰려 있는지는 표에서 직접 볼 수 있다. 판정 윈도우는 무작위 표본이 아니라 예측값에 따라 선택된 부분집합이다.\n")
    md.append("- 위 비교는 같은 보류율에서 무작위 보류와의 차이를 보는 관찰이며, 판정 보류가 코칭 품질을 높인다는 뜻이 아니다. 불일치율은 고정 규칙에 대한 판정 일치도다.\n\n")
    md.append("## 5. 한계\n")
    md.append("- 모든 목표·범주 경계·δ·데드라인이 합성 설정이다. 실제 사용자 목표·반응은 관찰하지 않았다(미확인).\n"
              "- 구간은 전역 상수 q(대칭, 윈도우 무관)라 사용자·페이스 수준에 따른 오차 크기 차이를 반영하지 않는다.\n"
              "- 보정 모델과 최종 평가 모델의 학습 자료 크기가 다르고(OOF 80% / history 80%), personalized 보정 구간은 hold-out보다 이른 시점이다.\n"
              "- 통제 비교는 윈도우 단위 무작위 보류이며 사용자 구조를 무시한다. 통제 범위는 불확실성 추정이 아니라 비교 기준이다.\n"
              "- 현재 능력=median-5 정의 때문에 median-5 예측기의 판정은 구조적으로 한 범주에 몰린다(`coaching_results.md` 3-5절).\n")
    # ---- q·구간 폭 검증 노트 (섹션 1 뒤) ----
    q_lines = ["- **q 값(초/km, 소수 셋째 자리)과 구간 폭**: 구간 = 예측 ± q이므로 폭 = 2q. 위 표의 q 값에서 80% 수준 q는 "
               + ", ".join(f"{sp} {min(float(np.quantile(CAL[sp][0][p], 0.8)) for p in ABS_PREDS):.3f}~{max(float(np.quantile(CAL[sp][0][p], 0.8)) for p in ABS_PREDS):.3f}" for sp in SPLITS)
               + "초/km → 폭 "
               + ", ".join(f"{sp} {2 * min(float(np.quantile(CAL[sp][0][p], 0.8)) for p in ABS_PREDS):.3f}~{2 * max(float(np.quantile(CAL[sp][0][p], 0.8)) for p in ABS_PREDS):.3f}" for sp in SPLITS)
               + "초/km; 90% 수준 q는 "
               + ", ".join(f"{sp} {min(float(np.quantile(CAL[sp][0][p], 0.9)) for p in ABS_PREDS):.3f}~{max(float(np.quantile(CAL[sp][0][p], 0.9)) for p in ABS_PREDS):.3f}" for sp in SPLITS)
               + "초/km → 폭 "
               + ", ".join(f"{sp} {2 * min(float(np.quantile(CAL[sp][0][p], 0.9)) for p in ABS_PREDS):.3f}~{2 * max(float(np.quantile(CAL[sp][0][p], 0.9)) for p in ABS_PREDS):.3f}" for sp in SPLITS)
               + f"초/km. **검증**: 모든 (예측기, 평가 조건, 수준, δ, 데드라인) 조합에서 실제 계산한 상한−하한의 평균과 2q의 최대 절대 오차 = {max(width_errs):.2e} (코드의 assert 통과).\n\n"]
    assert max(width_errs) < 1e-6
    md[idx_q_note:idx_q_note] = q_lines
    # ---- 문서 요약 (맨 앞) ----
    main_A = A[A["main"]]
    mdl = main_A[main_A["예측기"] != "median-5"]
    m5 = main_A[main_A["예측기"] == "median-5"]
    cur_med = float(np.median(np.concatenate([D[sp]["cur"] for sp in SPLITS])))
    widths = main_A["평균 구간 폭"]
    summary = [
        "## 요약\n",
        f"- **보류율 {pct(mdl['보류율'].min())}~{pct(mdl['보류율'].max())}** ((d)·GBM22·MLP22, 두 평가 조건, 80%/90% 수준, 주 조건 δ5%·8주; median-5는 {pct(m5['보류율'].min())}~{pct(m5['보류율'].max())}). "
        "**이 설계의 구간 폭은 범주 폭보다 훨씬 넓어 판정 보류가 거의 전부이며, 보류가 코칭 품질 향상을 뜻하지 않는다.** "
        f"구간 폭은 {widths.min():.3f}~{widths.max():.3f}초/km(= 2q)인 반면, 닫힌 범주 하나의 폭은 현재 페이스 중앙값({cur_med:.1f}초/km) 기준 '계획대로' "
        f"{(cl.ON_TRACK_MAX - cl.AHEAD_MAX) * cur_med:.1f}초/km(6%), '조금 뒤처짐' {(cl.SLIGHT_MAX - cl.ON_TRACK_MAX) * cur_med:.1f}초/km(7%)이다('앞서감'과 '크게 뒤처짐'은 한쪽이 열려 있음).\n",
        f"- 판정된 윈도우(보류되지 않은 {pct(1 - mdl['보류율'].max())}~{pct(1 - mdl['보류율'].min())})는 구간 전체가 한 범주에 들어가는 윈도우로, 예측값에 따라 선택된 부분집합이며(판정 범주 분포는 3-1절 표) 무작위 표본이 아니다. "
        "판정 사례 불일치율이 강제 판정보다 낮은 것은 이 선택된 부분집합에서의 관찰이다. 사후 통제(3-1b절, 결과를 본 뒤 추가)는 같은 윈도우의 상수 규칙·median-5 불일치율을 옆에 둔다.\n",
        f"- **사후 통제 관찰(3-1b절)**: (d)·GBM22·MLP22의 판정된 윈도우에서 같은 윈도우의 상수 규칙 불일치율은 {pct(mdl['상수_dec'].min())}~{pct(mdl['상수_dec'].max())}, "
        f"median-5 판정 불일치율은 {pct(mdl['m5_dec'].min())}~{pct(mdl['m5_dec'].max())}이고, (예측기 판정 사례 불일치율 − 상수 규칙) 점추정은 "
        f"{100 * (mdl['판정 불일치'] - mdl['상수_dec']).min():+.1f}~{100 * (mdl['판정 불일치'] - mdl['상수_dec']).max():+.1f}%p이다(CI는 표 참고). 판정된 윈도우의 대부분이 '앞서감'이므로 항상 '앞서감'을 말하는 상수 규칙과의 비교가 해당 부분집합의 기준이 된다는 점만 관찰로 기록한다.\n",
        "- 모든 목표는 합성 시나리오이며 불일치율은 고정 규칙에 대한 판정 일치도다(코칭 효과·목표 달성의 검증 아님). 기존 `coaching_results.md` 수치는 불변(0절).\n\n"]
    md[idx_after_header:idx_after_header] = summary
    OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
    log("\n저장:", OUTPUT_MD_PATH.resolve())


if __name__ == "__main__":
    main()
