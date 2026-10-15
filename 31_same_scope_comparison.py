# -*- coding: utf-8 -*-
"""
10월 3주차 2·3번: 같은 입력 범위 비교 + 신뢰구간 정비.
**실행 전에 docs/week3_predeclaration.md에 후보·지표·비교쌍·선택 규칙을 먼저 고정했다. 이 스크립트는 그것을 그대로 구현한다.**

그룹 (a) 최근 5회만: median-5, Ridge15, GBM15, MLP15 (GBM15/MLP15/Ridge15는 15피처로 CV 재선택)
그룹 (b) 전체 이력: 전체 이력 중앙값, (d), Ridge22, GBM22, MLP22 (기존 선택 그대로, 재탐색 없음)
모델 선택은 ridge_lib GroupKFold(cold-start) / 시간순 inner-split(personalized) 검증 MAE로만 하며 테스트 점수는 선택에 쓰지 않는다.
신뢰구간: 사용자 단위 cluster bootstrap(2000회, seed 42, 모든 방법에 같은 추출), 풀링 MAE / macro MAE(사용자별 MAE 평균) / 풀링 RMSE.
중단 조건: 그룹 (b)·레거시 6a의 수치가 docs/model_comparison.md와 소수 셋째 자리에서 다르면 즉시 중단.

산출물: docs/same_scope_comparison.md (평가 행별 예측값은 파일로 저장하지 않는다)
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
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

OUTPUT_MD_PATH = Path("docs") / "same_scope_comparison.md"
N_BOOT = 2000
BOOT_SEED = 42
MLP_SEEDS = [1, 2, 3, 42, 999]
REP_SEED = 42
CUR_SPLITS = ["cold-start", "personalized"]

GROUP_A = ["median5", "ridge15", "gbm15", "mlp15"]
GROUP_B = ["hist_median", "blend", "ridge22", "gbm22", "mlp22"]
NAMES = {
    "median5": "median-5", "ridge15": "Ridge15(15피처, 자체 CV alpha)", "gbm15": "GBM15", "mlp15": "MLP15",
    "hist_median": "전체 이력 중앙값", "blend": "(d) 0.5·m5+0.5·전체이력중앙값 [탐색적 baseline]",
    "ridge22": "Ridge22", "gbm22": "GBM22", "mlp22": "MLP22",
    "ridge15_legacy": "레거시 6a Ridge15(alpha=1e4, 22피처용 값)",
}
SHORT = {"median5": "median-5", "ridge15": "Ridge15", "gbm15": "GBM15", "mlp15": "MLP15", "hist_median": "전체이력중앙값",
         "blend": "(d)", "ridge22": "Ridge22", "gbm22": "GBM22", "mlp22": "MLP22"}
DOC_ROW = {  # docs/model_comparison.md 종합표의 행 이름 (불변 검증용)
    "median5": "3. median-5", "hist_median": "P1. 사용자 전체 이력 중앙값(사후추가)",
    "blend": "P4. 0.5*median5+0.5*전체이력중앙값(사후추가)", "ridge22": "6b. Ridge(22피처,확장)",
    "gbm22": "7. 직접구현 GBM(주결과)", "mlp22": "8. MLP(주결과)", "ridge15_legacy": "6a. Ridge(15피처,원복원)",
}


def log(*a):
    print(*a, flush=True)


# ===================== 선택 (15피처) =====================
def ridge_select(X, y, uid, split, p_train_samples=None):
    scores = {}
    if split == "cold-start":
        folds = list(rl.group_kfold_splits(uid, m25.N_FOLDS, m25.CV_SEED))
        for a in m25.RIDGE_ALPHA_GRID:
            maes = []
            for tr, va in folds:
                p = rl.RidgePipeline(a).fit(X[tr], y[tr]).predict(X[va])
                maes.append(np.mean(np.abs(y[va] - p)))
            scores[a] = float(np.mean(maes))
    else:
        inner_tr, inner_val = m25.personalized_inner_split(p_train_samples)
        Xi, yi, _ = m25.samples_to_xy(inner_tr)
        Xv, yv, _ = m25.samples_to_xy(inner_val)
        for a in m25.RIDGE_ALPHA_GRID:
            p = rl.RidgePipeline(a).fit(Xi[:, :15], yi).predict(Xv[:, :15])
            scores[a] = float(np.mean(np.abs(yv - p)))
    best = min(scores, key=scores.get)
    return best, scores


def _gbm_pred(params, form, X_a, y_a, X_b):
    med_a, med_b = m25.median5_from_X(X_a), m25.median5_from_X(X_b)
    model = m25.build_gbt(params, "absolute_error")
    if form == "abs":
        return model.fit(X_a, y_a).predict(X_b)
    return med_b + model.fit(X_a, y_a - med_a).predict(X_b)


def gbm_select(X, y, uid, split, p_train_samples=None):
    """형태(abs/resid)별로 격자를 CV/inner-val MAE로 선택, 두 형태 중 더 낮은 쪽을 주결과로."""
    res = {}
    if split == "cold-start":
        folds = list(rl.group_kfold_splits(uid, m25.N_FOLDS, m25.CV_SEED))
    else:
        inner_tr, inner_val = m25.personalized_inner_split(p_train_samples)
        Xi, yi, _ = m25.samples_to_xy(inner_tr)
        Xv, yv, _ = m25.samples_to_xy(inner_val)
        Xi, Xv = Xi[:, :15], Xv[:, :15]
    for form in ["abs", "resid"]:
        scores = {}
        for params in m25.HGB_PARAM_GRID:
            if split == "cold-start":
                maes = [np.mean(np.abs(y[va] - _gbm_pred(params, form, X[tr], y[tr], X[va]))) for tr, va in folds]
                mae = float(np.mean(maes))
            else:
                mae = float(np.mean(np.abs(yv - _gbm_pred(params, form, Xi, yi, Xv))))
            scores[m25._param_key(params)] = (params, mae)
            log(f"    GBM15 {split} {form} {params}: {mae:.3f}")
        bk = min(scores, key=lambda k: scores[k][1])
        res[form] = (scores[bk][0], scores[bk][1])
    best_form = min(res, key=lambda f: res[f][1])
    return best_form, res


def mlp_select(X, y, uid, split, p_train_samples=None):
    if split == "cold-start":
        tr, va = next(iter(rl.group_kfold_splits(uid, m25.N_FOLDS, m25.CV_SEED)))
        Xf, yf, Xv, yv = X[tr], y[tr], X[va], y[va]
    else:
        inner_tr, inner_val = m25.personalized_inner_split(p_train_samples)
        Xf, yf, _ = m25.samples_to_xy(inner_tr)
        Xv, yv, _ = m25.samples_to_xy(inner_val)
        Xf, Xv = Xf[:, :15], Xv[:, :15]
    medf, medv = m25.median5_from_X(Xf), m25.median5_from_X(Xv)
    res = {}
    for form, resid in [("direct", False), ("resid", True)]:
        log(f"    MLP15 {split} {form} 격자 탐색...")
        params, val_mae, epochs, _ = m27.select_mlp_hparams(Xf, yf, Xv, yv, resid, medf, medv)
        res[form] = (params, val_mae, epochs)
    best_form = min(res, key=lambda f: res[f][1])
    return best_form, res


def mlp_final(form, params, epochs, X_a, y_a, X_b, seeds):
    resid = form == "resid"
    med_a, med_b = m25.median5_from_X(X_a), m25.median5_from_X(X_b)
    preds = {}
    for sd in seeds:
        model, scaler = m27.train_mlp_fixed_epochs(X_a, (y_a - med_a) if resid else y_a, params["n_hidden_layers"],
                                                   params["width"], params["weight_decay"], sd, epochs)
        preds[sd] = (med_b if resid else 0.0) + m27.predict_mlp(model, scaler, X_b)
    return preds


# ===================== 평가 =====================
def user_stats(pred, y, codes, n_users):
    err = y - pred
    s_abs = np.bincount(codes, weights=np.abs(err), minlength=n_users)
    s_sq = np.bincount(codes, weights=err ** 2, minlength=n_users)
    n = np.bincount(codes, minlength=n_users).astype(float)
    return s_abs, s_sq, n


def stat_draws(st, W):
    s_abs, s_sq, n = st
    pooled = (W @ s_abs) / (W @ n)
    rmse = np.sqrt((W @ s_sq) / (W @ n))
    macro = (W @ (s_abs / n)) / W.sum(axis=1)
    return {"mae": pooled, "macro": macro, "rmse": rmse}


def point_stats(st):
    s_abs, s_sq, n = st
    return {"mae": s_abs.sum() / n.sum(), "macro": float(np.mean(s_abs / n)), "rmse": float(np.sqrt(s_sq.sum() / n.sum()))}


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def parse_doc_rows(split):
    doc = (Path("docs") / "model_comparison.md").read_text(encoding="utf-8")
    out = {}
    for key, row in DOC_ROW.items():
        m = re.search(r"\| " + re.escape(row) + r" \| " + re.escape(split) +
                      r" \| ([0-9.]+) \| \[([0-9.]+),([0-9.]+)\] \| ([0-9.]+) \| \[([0-9.]+),([0-9.]+)\] \|", doc)
        assert m, (key, split)
        out[key] = tuple(float(x) for x in m.groups())
    return out




def posthoc_section(results, md):
    """사후 기술통계(탐색): 결과를 본 뒤 추가한 분석(docs/week3_predeclaration.md 7절). 기술통계만, 인과 해석 없음."""
    md.append("## 7. 사후 기술통계(탐색) — 결과를 본 뒤 추가한 분석\n")
    md.append("`docs/week3_predeclaration.md` 7절에 '결과를 본 뒤 추가'로 기록했다. 기술통계이며 인과 해석을 하지 않는다. 사전 선언된 비교쌍이 아니다.\n\n")
    for split in CUR_SPLITS:
        r = results[split]
        st = r["stats"]
        n = st["median5"][2]
        md.append(f"### 7-1. 평가 사용자별 윈도우 수 분포 — {split} ({r['n_users']}명)\n")
        qs = [0, .1, .25, .5, .75, .9, 1]
        row = {f"p{int(q * 100)}" if 0 < q < 1 else ("최소" if q == 0 else "최대"): float(np.quantile(n, q)) for q in qs}
        row["평균"] = float(n.mean())
        md.append(m21.df_to_markdown(pd.DataFrame([row])))
        md.append("\n\n")
        md.append(f"### 7-2. 윈도우 수 하/중/상 3분위 사용자 그룹별 median-5 대비 MAE 차이 — {split}\n")
        md.append("사용자를 윈도우 수 오름차순(동률은 안정 정렬 순서)으로 정렬해 사용자 수가 같게 3등분. 차이 = median-5 − 모델(양수 = 모델이 더 낮은 오차). CI는 그룹 안에서 사용자를 복원추출(2000회, seed 42)한 percentile.\n\n")
        order = np.argsort(n, kind="stable")
        rows = []
        for gname, idx in zip(["하", "중", "상"], np.array_split(order, 3)):
            ns = len(idx)
            rng = np.random.default_rng(BOOT_SEED)
            Wg = np.stack([np.bincount(rng.integers(0, ns, ns), minlength=ns) for _ in range(N_BOOT)]).astype(float)
            sub = {k: tuple(a[idx] for a in st[k]) for k in ["median5", "gbm15", "mlp15"]}
            dr = {k: stat_draws(v, Wg) for k, v in sub.items()}
            pt = {k: point_stats(v) for k, v in sub.items()}
            row = {"그룹": gname, "#users": ns, "#windows": int(n[idx].sum()), "윈도우 수 범위": f"{int(n[idx].min())}~{int(n[idx].max())}",
                   "median-5 풀링 MAE": round(pt["median5"]["mae"], 3), "GBM15 풀링 MAE": round(pt["gbm15"]["mae"], 3), "MLP15 풀링 MAE": round(pt["mlp15"]["mae"], 3)}
            for k, nm in [("gbm15", "GBM15"), ("mlp15", "MLP15")]:
                for metric, mn in [("mae", "풀링"), ("macro", "macro")]:
                    d = dr["median5"][metric] - dr[k][metric]
                    lo, hi = ci(d)
                    row[f"median-5−{nm} {mn}"] = f"{pt['median5'][metric] - pt[k][metric]:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" (0 포함)" if lo <= 0 <= hi else "")
            rows.append(row)
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n\n")
        md.append(f"### 7-3. 문구 근거 확인: median-5 대비 22피처 모델의 풀링/macro MAE 차이 — {split} (전체 평가 사용자, 그룹 간 쌍이라 순위 아님)\n")
        rows = []
        for k in ["gbm22", "mlp22"]:
            row = {"모델": NAMES[k]}
            for metric, mn in [("mae", "풀링 MAE"), ("macro", "macro MAE")]:
                d = r["draws"]["median5"][metric] - r["draws"][k][metric]
                lo, hi = ci(d)
                row[f"median-5 − 모델 ({mn})"] = f"{r['pts']['median5'][metric] - r['pts'][k][metric]:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" (0 포함)" if lo <= 0 <= hi else "")
            rows.append(row)
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n\n")


def main():
    md = ["# 같은 입력 범위 비교 + 신뢰구간 정비 (10월 3주차 2·3번)\n\n"]
    md.append("**사전 선언**: `docs/week3_predeclaration.md`(실행 전 작성·고정). 이 문서의 후보·지표·비교쌍·선택 규칙은 결과를 본 뒤 바꾸지 않았다. "
              "그룹을 가로지르는 순위는 매기지 않는다. 새 모델 종류는 없다(기존 Ridge/GBM/MLP의 입력 피처 변형만).\n\n")
    main_df = config.load_main_sessions()
    md.append(f"입력 메인 데이터: `{config.MAIN_DATA_PATH}` (세션 {len(main_df)}건, 사용자 {main_df['user_id'].nunique()}명, assert 통과)\n\n")
    md.append("**cold-start 정의**: 학습에 없던 사용자이지만 타깃 이전 기록이 5회 이상 있는 조건(전체 이력 중앙값 피처는 5회보다 더 먼 과거까지 사용). "
              "기록이 전혀 없는 신규 사용자는 평가에 없다.\n\n")

    train = pd.read_pickle(m24.EXT_TRAIN_PATH)
    test = pd.read_pickle(m24.EXT_TEST_PATH)
    p_train = pd.read_pickle(m24.EXT_P_TRAIN_PATH)
    p_test = pd.read_pickle(m24.EXT_P_TEST_PATH)
    data = {"cold-start": (train, test), "personalized": (p_train, p_test)}

    # median-10/20 인덱스 전제 assert는 build_pace_by_user 안에 있다(여기서 한 번 호출해 확인)
    m28.build_pace_by_user(pd.read_pickle(config.USER_SEQUENCES_PATH))
    log("[assert] median-10/20 session_order 조밀 인덱스 전제 통과")

    rng_w = {}
    results = {}
    sel_rows = []
    for split in CUR_SPLITS:
        tr_s, te_s = data[split]
        X_tr, y_tr, uid_tr = m25.samples_to_xy(tr_s)
        X_te, y_te, uid_te = m25.samples_to_xy(te_s)
        n_te = len(te_s)
        log(f"\n===== {split}: train {len(tr_s)} / test {n_te} =====")
        X_tr15, X_te15 = X_tr[:, :15], X_te[:, :15]

        preds = {}
        # ---- 그룹 (b)의 기존 방법 + median-5 : 기존 선택 재사용 (재탐색 없음) ----
        base = m29.predict_all_models(tr_s, te_s, split)  # 문서 값과 assert 대조 포함
        preds["median5"], preds["hist_median"], preds["blend"] = base["median5"], base["hist_median"], base["blend"]
        preds["gbm22"], preds["mlp22"] = base["gbm"], base["mlp"]
        preds["ridge22"] = rl.RidgePipeline(m28.RIDGE_ALPHA_COLD if split == "cold-start" else m28.RIDGE_ALPHA_PERS).fit(X_tr, y_tr).predict(X_te)
        preds["ridge15_legacy"] = rl.RidgePipeline(m28.RIDGE_ALPHA_COLD if split == "cold-start" else m28.RIDGE_ALPHA_PERS).fit(X_tr15, y_tr).predict(X_te15)

        # ---- Ridge15 (자체 CV alpha) ----
        log("  Ridge15 alpha 선택...")
        a15, a15_scores = ridge_select(X_tr15, y_tr, uid_tr, split, tr_s)
        preds["ridge15"] = rl.RidgePipeline(a15).fit(X_tr15, y_tr).predict(X_te15)
        sel_rows.append({"평가조건": split, "방법": "Ridge15", "선택": f"alpha={a15}", "선택 기준 점수(CV/검증 MAE)": f"{a15_scores[a15]:.3f}"})

        # ---- GBM15 ----
        log("  GBM15 선택(형태 2 × 격자 8 × CV)...")
        g_form, g_res = gbm_select(X_tr15, y_tr, uid_tr, split, tr_s)
        g_params = g_res[g_form][0]
        preds["gbm15"] = _gbm_pred(g_params, g_form, X_tr15, y_tr, X_te15)
        sel_rows.append({"평가조건": split, "방법": "GBM15", "선택": f"{'absolute_error 직접' if g_form == 'abs' else 'median5+잔차'}, {g_params}",
                         "선택 기준 점수(CV/검증 MAE)": f"abs {g_res['abs'][1]:.3f} / resid {g_res['resid'][1]:.3f}"})

        # ---- MLP15 ----
        log("  MLP15 선택...")
        m_form, m_res = mlp_select(X_tr15, y_tr, uid_tr, split, tr_s)
        m_params, m_val, m_ep = m_res[m_form]
        mp = mlp_final(m_form, m_params, m_ep, X_tr15, y_tr, X_te15, MLP_SEEDS)
        preds["mlp15"] = mp[REP_SEED]
        seed_maes = [float(np.mean(np.abs(y_te - mp[s]))) for s in MLP_SEEDS]
        sel_rows.append({"평가조건": split, "방법": "MLP15", "선택": f"{'median5+잔차' if m_form == 'resid' else '직접'}, {m_params}, 에폭 {m_ep}",
                         "선택 기준 점수(CV/검증 MAE)": f"direct {m_res['direct'][1]:.3f} / resid {m_res['resid'][1]:.3f}; 5시드 test MAE 평균 {np.mean(seed_maes):.3f}±{np.std(seed_maes):.3f}"})

        # ---- 무결성 (i): 길이·순서 ----
        meta = pd.DataFrame({"user_id": uid_te, "sid": [s["target_session_id"] for s in te_s]})
        for k, p in preds.items():
            assert len(p) == n_te, f"예측 길이 불일치: {k}"
        codes, uniq = pd.factorize(uid_te)
        n_users = len(uniq)
        assert (pd.Series(uid_te).drop_duplicates().to_numpy() == uniq).all()
        stats = {k: user_stats(np.asarray(p, dtype=float), y_te, codes, n_users) for k, p in preds.items()}
        ref_n = stats["median5"][2]
        for k in stats:
            assert np.array_equal(stats[k][2], ref_n), f"방법 간 사용자별 윈도우 수 불일치: {k}"
        log(f"  [assert] 모든 방법 예측 {n_te}개, 사용자별 윈도우 수 동일, 윈도우 순서 동일(같은 test 샘플 리스트)")

        # ---- bootstrap (m21.cluster_bootstrap과 같은 난수열: default_rng(42), 반복마다 integers(0, n, n)) ----
        rng = np.random.default_rng(BOOT_SEED)
        W = np.stack([np.bincount(rng.integers(0, n_users, n_users), minlength=n_users) for _ in range(N_BOOT)]).astype(float)
        draws = {k: stat_draws(st, W) for k, st in stats.items()}
        pts = {k: point_stats(st) for k, st in stats.items()}

        # ---- 불변 검증 (중단 조건) ----
        doc_vals = parse_doc_rows(split)
        for key, (dmae, dmlo, dmhi, drmse, drlo, drhi) in doc_vals.items():
            p = pts[key]
            d = draws[key]
            lo, hi = ci(d["mae"])
            rlo, rhi = ci(d["rmse"])
            for a, b, what in [(p["mae"], dmae, "MAE"), (p["rmse"], drmse, "RMSE"), (lo, dmlo, "MAE CI lo"), (hi, dmhi, "MAE CI hi"),
                               (rlo, drlo, "RMSE CI lo"), (rhi, drhi, "RMSE CI hi")]:
                if abs(a - b) > 0.0006:
                    raise SystemExit(f"[중단] 불변 검증 실패: {split} {key} {what}: 재계산 {a:.4f} vs 문서 {b:.4f}")
        log("  [불변 검증] 그룹 (b)·median-5·레거시 6a의 MAE/RMSE/CI가 docs/model_comparison.md와 일치")

        results[split] = {"preds": preds, "stats": stats, "draws": draws, "pts": pts, "y": y_te, "n_users": n_users,
                          "n_win": n_te, "pace_median": float(np.median(y_te)), "mlp_seed_maes": seed_maes}

    # ===================== 문서 =====================
    md.append("## 0. 15피처 모델 선택 결과 (CV/검증 MAE만 사용, 테스트 미사용)\n")
    md.append(m21.df_to_markdown(pd.DataFrame(sel_rows)))
    md.append("\n\n")
    md.append("## 1. 통계 방법 (모든 표 공통)\n")
    for split in CUR_SPLITS:
        r = results[split]
        md.append(f"- {split}: 평가 사용자 {r['n_users']}명, 평가 윈도우 {r['n_win']}건, 타깃 페이스 중앙값 {r['pace_median']:.1f}초/km\n")
    md.append(f"- **재표집**: 평가 사용자를 복원추출(사용자 수만큼), {N_BOOT}회, seed {BOOT_SEED}. 뽑힌 사용자의 **모든 윈도우**를 함께 쓰고 모든 방법에 같은 추출을 적용(쌍체). 95% CI는 percentile.\n")
    md.append("- **풀링 MAE**: 윈도우 단위 평균(사용자 윈도우 수에 비례해 가중). **macro MAE**: 사용자별 MAE의 평균(사용자 1명 = 1표, 뽑힌 사용자는 중복 횟수만큼 가중). 둘은 다른 양이며 둘 다 보고한다. RMSE는 풀링만.\n")
    md.append("- 쌍 차이 = (앞 방법) − (뒤 방법). **양수면 뒤 방법이 더 낮은(좋은) 값.** CI가 0을 포함하면 'CI가 0을 포함 — 동등성(차이가 없음)의 입증이 아님'이다. 다중 비교 보정은 하지 않았다(쌍이 다수이므로 개별 CI를 확정 근거로 쓰지 않음).\n")
    md.append("- 개선폭은 분모를 구분한다: (1) 절대 개선(초/km), (2) 그룹 기준 방법의 MAE 대비 %, (3) 평가 윈도우 타깃 페이스 중앙값 대비 %.\n\n")

    def group_section(split, title, members, ref, note=""):
        r = results[split]
        md.append(f"### {title} — {split}\n")
        if note:
            md.append(note + "\n\n")
        rows = []
        for k in members:
            p, d = r["pts"][k], r["draws"][k]
            lo, hi = ci(d["mae"]); mlo, mhi = ci(d["macro"]); rlo, rhi = ci(d["rmse"])
            rows.append({"방법": NAMES[k], "풀링 MAE [95% CI]": f"{p['mae']:.3f} [{lo:.3f}, {hi:.3f}]",
                         "macro MAE [95% CI]": f"{p['macro']:.3f} [{mlo:.3f}, {mhi:.3f}]",
                         "풀링 RMSE [95% CI]": f"{p['rmse']:.3f} [{rlo:.3f}, {rhi:.3f}]",
                         "#users": r["n_users"], "#windows": r["n_win"]})
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n\n")
        # 순위 (그룹 내, MAE와 RMSE 따로)
        for key, label in [("mae", "풀링 MAE"), ("macro", "macro MAE"), ("rmse", "풀링 RMSE")]:
            order = sorted(members, key=lambda k: r["pts"][k][key])
            md.append(f"- {label} 순위(그룹 내 점추정): " + " < ".join(f"{NAMES[k]}({r['pts'][k][key]:.3f})" for k in order) + "\n")
        md.append("\n")
        # 개선폭
        rows = []
        for k in members:
            if k == ref:
                continue
            for metric, mname in [("mae", "풀링 MAE"), ("macro", "macro MAE")]:
                diff = r["draws"][ref][metric] - r["draws"][k][metric]
                lo, hi = ci(diff)
                gain = r["pts"][ref][metric] - r["pts"][k][metric]
                rows.append({"방법": NAMES[k], "지표": mname, f"절대 개선(vs {SHORT[ref]}) [95% CI]": f"{gain:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                             "ref MAE 대비 %": f"{100 * gain / r['pts'][ref][metric]:+.2f}%",
                             "페이스 중앙값 대비 %": f"{100 * gain / r['pace_median']:+.2f}%",
                             "해석": ("CI가 0을 포함 — 동등성의 입증이 아님" if lo <= 0 <= hi else ("ref보다 유의하게 낮음" if lo > 0 else "ref보다 유의하게 높음"))})
        md.append(f"**개선폭 (기준 방법 = {NAMES[ref]}, 양수 = 기준보다 낮은 오차)**\n\n")
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n\n")
        # 쌍 차이
        rows = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                row = {"쌍 (앞−뒤)": f"{SHORT[a]} − {SHORT[b]}"}
                for metric, mname in [("mae", "풀링 MAE"), ("macro", "macro MAE"), ("rmse", "풀링 RMSE")]:
                    diff = r["draws"][a][metric] - r["draws"][b][metric]
                    lo, hi = ci(diff)
                    pt = r["pts"][a][metric] - r["pts"][b][metric]
                    row[mname] = f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" (0 포함)" if lo <= 0 <= hi else "")
                rows.append(row)
        md.append("**그룹 내 모든 쌍의 차이 (사전 선언된 비교쌍 전부)**\n\n")
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n\n'(0 포함)' 표시는 CI가 0을 포함한다는 뜻이며 두 방법이 같다는 입증이 아니다.\n\n")

    md.append("## 2. 그룹 (a): 최근 5회만 쓰는 방법\n")
    for split in CUR_SPLITS:
        group_section(split, "그룹 (a)", GROUP_A, "median5",
                      note="입력: 과거 5세션 × (거리·페이스·심박) 15피처(median-5는 페이스만). Ridge15/GBM15/MLP15는 15피처로 CV 재선택.")
    md.append("## 3. 그룹 (b): 타깃 이전 전체 이력을 쓰는 방법\n")
    for split in CUR_SPLITS:
        group_section(split, "그룹 (b)", GROUP_B, "hist_median",
                      note="입력: 15피처 + 이력 요약 7개(전체 이력 중앙값·누적 세션 수 포함)를 쓰는 22피처 모델과, 전체 이력 중앙값을 쓰는 단순 규칙. (d)는 탐색적 baseline(9단계 결과를 본 뒤 추가).")

    md.append("## 4. 참고: 레거시 6a (순위 비교에 넣지 않음)\n")
    md.append("종합표 6a는 22피처용으로 선택된 alpha=1e4를 15피처에 그대로 쓴 값이다. 자체 CV alpha를 쓴 Ridge15와 다르며, 값은 `docs/model_comparison.md`와 동일하다(불변 검증 통과).\n\n")
    rows = []
    for split in CUR_SPLITS:
        r = results[split]
        for k in ["ridge15_legacy", "ridge15"]:
            p = r["pts"][k]
            rows.append({"평가조건": split, "방법": NAMES[k], "풀링 MAE": round(p["mae"], 3), "macro MAE": round(p["macro"], 3), "풀링 RMSE": round(p["rmse"], 3)})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n## 5. 불변 확인 (전/후)\n")
    md.append("그룹 (b)의 전체 이력 중앙값·(d)·Ridge22·GBM22·MLP22, median-5, 레거시 6a의 풀링 MAE/RMSE와 각 95% CI가 `docs/model_comparison.md` 종합표와 소수 셋째 자리(±0.0006)에서 모두 일치했다 — 두 평가 조건 모두. 바뀌었다면 이 스크립트는 중단되도록 되어 있다.\n")
    md.append("\n## 6. 구현 변경 (수치 불변)\n- `21_baseline_integrated_report.py` `cluster_bootstrap`: 방법 간 사용자별 윈도우 수 일치 assert 추가.\n"
              "- `28_model_comparison.py` `build_pace_by_user`: session_order 조밀 인덱스(0..n-1)와 시간 순서 assert 추가(median-10/20 전제).\n"
              "- 이 스크립트: 모든 방법의 예측 길이·사용자별 윈도우 수 assert. 평가 행별 예측값은 파일로 저장하지 않는다.\n")
    posthoc_section(results, md)
    OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
    log("\n저장:", OUTPUT_MD_PATH.resolve())


if __name__ == "__main__":
    main()
