# -*- coding: utf-8 -*-
"""
10월 2주차 9단계: 전체 종합 + LSTM 시도 여부 판단.

모든 방법(baseline 5종, Ridge, 직접 구현 GBM, MLP)을 **동일 평가 윈도우**
(7단계 확장 피처 윈도우, 24단계 출력 기준)에서 재평가해 하나의 종합표로
만든다. 모든 하이퍼파라미터는 6~8단계에서 이미 선택된 값을 그대로 쓰고
재탐색하지 않는다 — 이 스크립트는 "같은 조건에서 나란히 비교"가 목적이다.

LSTM 시도 여부는 결과를 보기 전에 고정한 규칙으로만 판단한다(4번 섹션):
history가 가장 긴 구간의 MAE 개선폭이 가장 짧은 구간보다 크고, 그 차이의
사용자 단위 bootstrap CI가 0을 포함하지 않을 때만 "시도 대상"으로 본다.
시도 대상이 되어도 이 스크립트는 구현하지 않고 계획만 적는다.

산출물: docs/model_comparison.md
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import gbt_lib as gl

_spec_m21 = importlib.util.spec_from_file_location("m21", "21_baseline_integrated_report.py")
m21 = importlib.util.module_from_spec(_spec_m21)
_spec_m21.loader.exec_module(m21)

_spec_m25 = importlib.util.spec_from_file_location("m25", "25_tree_models.py")
m25 = importlib.util.module_from_spec(_spec_m25)
_spec_m25.loader.exec_module(m25)

_spec_m27 = importlib.util.spec_from_file_location("m27", "27_mlp_torch.py")
m27 = importlib.util.module_from_spec(_spec_m27)
_spec_m27.loader.exec_module(m27)

m24 = m25.m24  # 25_tree_models.py가 이미 24단계 모듈을 로드해 재노출

OUTPUT_MD_PATH = Path("docs") / "model_comparison.md"

# ===== 이미 6~8단계에서 선택된(재탐색 없음) 하이퍼파라미터 =====
RIDGE_ALPHA_COLD = 10000.0
RIDGE_ALPHA_PERS = 10000.0
GBM_COLD_PARAMS = {"learning_rate": 0.1, "max_iter": 300, "max_depth": 3}   # cold-start 주결과: absolute_error 직접예측
GBM_PERS_PARAMS = {"learning_rate": 0.1, "max_iter": 300, "max_depth": 3}  # personalized 주결과: median5+잔차
MLP_COLD_PARAMS = {"n_hidden_layers": 2, "width": 64, "weight_decay": 0.0}
MLP_COLD_EPOCHS = 97
MLP_PERS_PARAMS = {"n_hidden_layers": 1, "width": 64, "weight_decay": 1e-4}
MLP_PERS_EPOCHS = 142
MLP_SEED = 42  # 8단계 보고의 "대표 시드"와 동일

RIDGE15_LABEL = "ridge_15feat"
RIDGE_LABEL = "ridge_ext"
GBM_LABEL = "gbm"
MLP_LABEL = "mlp"
# 9단계 보완(Part 1-2): ablation을 본 뒤 추가한 사후 baseline 4종 (고정, 더 추가 안 함)
HIST_MEDIAN_LABEL = "user_hist_median"
MEDIAN10_LABEL = "median10"
MEDIAN20_LABEL = "median20"
BLEND_LABEL = "blend_m5_hist"
POST_HOC_NAMES = [HIST_MEDIAN_LABEL, MEDIAN10_LABEL, MEDIAN20_LABEL, BLEND_LABEL]

ALL_LABELS = dict(m21.BASELINE_LABELS)
ALL_LABELS[RIDGE15_LABEL] = "6a. Ridge(15피처,원복원)"
ALL_LABELS[RIDGE_LABEL] = "6b. Ridge(22피처,확장)"
ALL_LABELS[GBM_LABEL] = "7. 직접구현 GBM(주결과)"
ALL_LABELS[MLP_LABEL] = "8. MLP(주결과)"
ALL_LABELS[HIST_MEDIAN_LABEL] = "P1. 사용자 전체 이력 중앙값(사후추가)"
ALL_LABELS[MEDIAN10_LABEL] = "P2. median-10(사후추가)"
ALL_LABELS[MEDIAN20_LABEL] = "P3. median-20(사후추가)"
ALL_LABELS[BLEND_LABEL] = "P4. 0.5*median5+0.5*전체이력중앙값(사후추가)"
ALL_NAMES = m21.BASELINE_NAMES + [RIDGE15_LABEL, RIDGE_LABEL, GBM_LABEL, MLP_LABEL] + POST_HOC_NAMES

HIST_BINS = [0, 10, 30, 60, 120, 1e9]
HIST_LABELS = ["5-9", "10-29", "30-59", "60-119", "120+"]


def with_original_15_X(samples):
    return [{**s, "X": s["X"][:15]} for s in samples]


def build_pace_by_user(user_sequences):
    """사용자별 session_order 순 avg_pace_sec_per_km 배열. session_order는 0부터 시작하는
    조밀한 시간순 인덱스라(02/17단계에서 이미 확인됨) 위치==session_order로 바로 슬라이싱 가능."""
    d = user_sequences.sort_values(["user_id", "session_order"])
    return {uid: grp["avg_pace_sec_per_km"].to_numpy() for uid, grp in d.groupby("user_id")}


def last_n_median_pred(samples, pace_by_user, n):
    """타깃 세션 바로 이전까지의 세션 중 최근 n개(모자라면 가능한 만큼만)의
    avg_pace_sec_per_km 중앙값. session_order < target_session_order만 쓰므로
    last_input_session_time 감사와 같은 시간 경계를 공유한다(새 리크 없음)."""
    preds, used_counts = [], []
    for s in samples:
        arr = pace_by_user[s["user_id"]]
        t = s["target_session_order"]
        window = arr[max(0, t - n):t]
        preds.append(float(np.median(window)))
        used_counts.append(len(window))
    return np.array(preds), np.array(used_counts)


def cluster_bootstrap_bucket_diff(df, model_name, bucket_col, bucket_a, bucket_b, n_boot=2000, seed=42):
    """사용자 단위 bootstrap으로 (median5-model 개선폭 in bucket_a) - (같은 값 in bucket_b)의
    분포를 구한다. 같은 사용자가 여러 버킷에 걸칠 수 있으므로, 사용자 리샘플은 한 번만 하고
    그 리샘플 결과를 양쪽 버킷에 동시에 적용한다(사용자 수준 상관을 보존)."""
    rng = np.random.default_rng(seed)
    user_ids = df["user_id"].unique()
    n_users = len(user_ids)

    def per_user_stats(sub_df):
        g = sub_df.groupby("user_id").agg(sum_abs=("abs_error", "sum"), n=("abs_error", "size"))
        g = g.reindex(user_ids).fillna(0)
        return g["sum_abs"].to_numpy(), g["n"].to_numpy()

    stats = {}
    for bucket in [bucket_a, bucket_b]:
        for name in ["median5", model_name]:
            sub = df[(df[bucket_col] == bucket) & (df["baseline"] == name)]
            stats[(bucket, name)] = per_user_stats(sub)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_users, size=n_users)
        improvements = {}
        for bucket in [bucket_a, bucket_b]:
            sum_abs_m5, n_m5 = stats[(bucket, "median5")]
            sum_abs_md, n_md = stats[(bucket, model_name)]
            total_m5 = sum_abs_m5[idx].sum()
            total_md = sum_abs_md[idx].sum()
            n_m5_tot = n_m5[idx].sum()
            n_md_tot = n_md[idx].sum()
            mae_m5 = total_m5 / n_m5_tot if n_m5_tot > 0 else np.nan
            mae_md = total_md / n_md_tot if n_md_tot > 0 else np.nan
            improvements[bucket] = mae_m5 - mae_md
        diffs[i] = improvements[bucket_a] - improvements[bucket_b]
    return diffs


def main():
    md = []
    md.append("# 전체 종합 + LSTM 시도 여부 판단 (10월 2주차 9단계)\n\n")

    main_df = config.load_main_sessions()
    md.append(f"입력 메인 데이터: `{config.MAIN_DATA_PATH}` (세션 {len(main_df)}건, "
              f"사용자 {main_df['user_id'].nunique()}명, assert 통과)\n\n")
    md.append("모든 하이퍼파라미터는 6~8단계에서 이미 선택된 값을 그대로 재사용했다(재탐색 없음). "
              "이 문서의 목적은 '같은 평가 윈도우에서 나란히 비교'다.\n\n")

    # ===== 입력 =====
    train_samples = pd.read_pickle(m24.EXT_TRAIN_PATH)
    test_samples = pd.read_pickle(m24.EXT_TEST_PATH)
    p_train_samples = pd.read_pickle(m24.EXT_P_TRAIN_PATH)
    p_test_samples = pd.read_pickle(m24.EXT_P_TEST_PATH)
    user_sequences = pd.read_pickle(config.USER_SEQUENCES_PATH)
    overall_avg_lookup = m21.build_user_overall_avg_lookup(user_sequences)

    X_tr, y_tr, uid_tr = m25.samples_to_xy(train_samples)
    X_te, y_te, _ = m25.samples_to_xy(test_samples)
    X_ptr, y_ptr, _ = m25.samples_to_xy(p_train_samples)
    X_pte, y_pte, _ = m25.samples_to_xy(p_test_samples)

    md.append("## 0. 입력 및 윈도우 수(6단계 대비 차이는 7단계 `docs/tree_results.md` 1-1절 참고)\n")
    md.append(f"- cold-start: train {len(train_samples)}건 / test {len(test_samples)}건\n")
    md.append(f"- personalized: history {len(p_train_samples)}건 / hold-out {len(p_test_samples)}건\n")
    md.append("- 5~6단계 원표는 04/06단계 윈도우(과거 정보 감사만 적용)를 썼고, 이 표는 7~8단계와 "
              "같은 24단계 확장 피처 윈도우(이력 감사 추가 적용)를 쓴다 — cold-start는 train "
              "2건/test 1건, personalized는 history 2건/holdout 1건(경계 이동 포함 3건) 적다. "
              "**baseline/Ridge 숫자도 이 윈도우에서 다시 계산했으므로 5~6단계 표의 숫자와 약간 "
              "다르다(소수점 셋째 자리 수준) — 모델 차이가 아니라 평가 대상 윈도우 차이다.**\n\n")

    # ===== 1. 모델 refit (이미 선택된 하이퍼파라미터, 재탐색 없음) =====
    print("Ridge(확장) refit 중...")
    from sklearn.linear_model import Ridge as SkRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    def ridge_pipeline(alpha):
        return Pipeline([("scaler", StandardScaler()), ("ridge", SkRidge(alpha=alpha))])

    ridge_cold_pred = ridge_pipeline(RIDGE_ALPHA_COLD).fit(X_tr, y_tr).predict(X_te)
    ridge_pers_pred = ridge_pipeline(RIDGE_ALPHA_PERS).fit(X_ptr, y_ptr).predict(X_pte)

    print("Ridge(15피처, 24단계 윈도우) refit 중 (9단계 보완: 피처구성 vs 윈도우구성 효과 분리용)...")
    # 6단계 Ridge는 04/06단계 윈도우(15피처)에서 학습했다. 여기서는 "같은 24단계 윈도우에서
    # 피처 수만 15로 줄인" 버전을 만들어, 6단계 숫자와의 차이 중 윈도우 구성 때문인 부분과
    # 피처 구성 때문인 부분을 분리해서 볼 수 있게 한다. alpha는 재탐색하지 않고 기존 값을
    # 그대로 쓴다(6단계에서 이미 이 alpha 범위는 결과에 영향이 거의 없음을 확인함).
    ridge_cold_15_pred = ridge_pipeline(RIDGE_ALPHA_COLD).fit(X_tr[:, :15], y_tr).predict(X_te[:, :15])
    ridge_pers_15_pred = ridge_pipeline(RIDGE_ALPHA_PERS).fit(X_ptr[:, :15], y_ptr).predict(X_pte[:, :15])

    print("직접구현 GBM refit 중 (cold=absolute_error, personalized=median5+잔차)...")
    gbm_cold_model = m25.build_gbt(GBM_COLD_PARAMS, "absolute_error").fit(X_tr, y_tr)
    gbm_cold_pred = gbm_cold_model.predict(X_te)
    med_ptr = m25.median5_from_X(X_ptr)
    med_pte = m25.median5_from_X(X_pte)
    gbm_pers_model = m25.build_gbt(GBM_PERS_PARAMS, "absolute_error")
    gbm_pers_model.fit(X_ptr, y_ptr - med_ptr)
    gbm_pers_pred = med_pte + gbm_pers_model.predict(X_pte)

    print("MLP refit 중 (시드=42, 이미 선택된 하이퍼파라미터/에폭)...")
    # 8단계 보고 기준: MLP는 cold-start/personalized 둘 다 median5+잔차 형태가
    # 검증(validation) MAE 기준 주결과로 선택됐다 -> 둘 다 잔차형으로 학습.
    med_tr_full = m25.median5_from_X(X_tr)
    med_te_full = m25.median5_from_X(X_te)
    mlp_cold_model, mlp_cold_scaler = m27.train_mlp_fixed_epochs(
        X_tr, y_tr - med_tr_full, MLP_COLD_PARAMS["n_hidden_layers"], MLP_COLD_PARAMS["width"],
        MLP_COLD_PARAMS["weight_decay"], MLP_SEED, MLP_COLD_EPOCHS,
    )
    mlp_cold_pred = med_te_full + m27.predict_mlp(mlp_cold_model, mlp_cold_scaler, X_te)

    mlp_pers_model, mlp_pers_scaler = m27.train_mlp_fixed_epochs(
        X_ptr, y_ptr - med_ptr, MLP_PERS_PARAMS["n_hidden_layers"], MLP_PERS_PARAMS["width"],
        MLP_PERS_PARAMS["weight_decay"], MLP_SEED, MLP_PERS_EPOCHS,
    )
    mlp_pers_pred = med_pte + m27.predict_mlp(mlp_pers_model, mlp_pers_scaler, X_pte)

    print("사후 추가 baseline 4종(ablation을 본 뒤 추가) 계산 중...")
    # (a) 사용자 전체 이력 중앙값: 24단계가 이미 이력 감사를 통과시켜 만든
    # user_history_median_pace 피처(X 22개 중 인덱스 18)를 그대로 재사용한다 — 새로 계산하면
    # 같은 값이 나오므로 중복 계산하지 않는다.
    USER_HIST_MEDIAN_COL = 18
    pace_by_user = build_pace_by_user(user_sequences)

    hist_median_cold_pred = X_te[:, USER_HIST_MEDIAN_COL]
    hist_median_pers_pred = X_pte[:, USER_HIST_MEDIAN_COL]

    med10_cold_pred, used10_cold = last_n_median_pred(test_samples, pace_by_user, 10)
    med10_pers_pred, used10_pers = last_n_median_pred(p_test_samples, pace_by_user, 10)
    med20_cold_pred, used20_cold = last_n_median_pred(test_samples, pace_by_user, 20)
    med20_pers_pred, used20_pers = last_n_median_pred(p_test_samples, pace_by_user, 20)

    med5_cold_for_blend = m25.median5_from_X(X_te)
    med5_pers_for_blend = m25.median5_from_X(X_pte)
    blend_cold_pred = 0.5 * med5_cold_for_blend + 0.5 * hist_median_cold_pred
    blend_pers_pred = 0.5 * med5_pers_for_blend + 0.5 * hist_median_pers_pred

    md.append("## 0-2. 사후 추가 baseline 4종 (ablation 결과를 본 뒤 추가 — 아래 4개로 고정, 더 추가하지 않음)\n")
    md.append("(a) 사용자 전체 이력 중앙값 (b) median-10 (c) median-20 (d) 0.5×median5+0.5×전체이력중앙값. "
              "모두 타깃 시작 시각 이전 세션만 사용한다 — (a)/(d)의 전체이력중앙값 부분은 24단계가 "
              "이미 '이력 감사'(가장 늦게 참조한 세션 종료 시각 < 타깃 시작 시각)를 통과시킨 "
              "`user_history_median_pace` 피처를 그대로 재사용했고, (b)/(c)는 session_order가 "
              "0부터 시작하는 조밀한 시간순 인덱스라는 점(02/17단계에서 이미 확인)을 이용해 "
              "`session_order < target_session_order`인 세션만 썼다(새로운 리크 경로 없음).\n")
    md.append(f"- **median-10 이력 부족(10개 미만) 윈도우 비율**: cold-start "
              f"{np.mean(used10_cold < 10):.1%} ({int(np.sum(used10_cold<10))}/{len(used10_cold)}), "
              f"personalized {np.mean(used10_pers < 10):.1%} ({int(np.sum(used10_pers<10))}/{len(used10_pers)}) "
              f"— 모자라면 있는 만큼만 중앙값을 계산했다(규칙).\n")
    md.append(f"- **median-20 이력 부족(20개 미만) 윈도우 비율**: cold-start "
              f"{np.mean(used20_cold < 20):.1%} ({int(np.sum(used20_cold<20))}/{len(used20_cold)}), "
              f"personalized {np.mean(used20_pers < 20):.1%} ({int(np.sum(used20_pers<20))}/{len(used20_pers)})\n\n")

    # ===== 2. 통합 =====
    print("통합표/부트스트랩 계산 중...")
    base_cold_df = m21.build_rows(with_original_15_X(test_samples), overall_avg_lookup)
    base_pers_df = m21.build_rows(with_original_15_X(p_test_samples), overall_avg_lookup)

    ridge_cold_df = m25.build_model_rows(test_samples, ridge_cold_pred, RIDGE_LABEL)
    ridge15_cold_df = m25.build_model_rows(test_samples, ridge_cold_15_pred, RIDGE15_LABEL)
    gbm_cold_df = m25.build_model_rows(test_samples, gbm_cold_pred, GBM_LABEL)
    mlp_cold_df = m25.build_model_rows(test_samples, mlp_cold_pred, MLP_LABEL)
    hist_median_cold_df = m25.build_model_rows(test_samples, hist_median_cold_pred, HIST_MEDIAN_LABEL)
    med10_cold_df = m25.build_model_rows(test_samples, med10_cold_pred, MEDIAN10_LABEL)
    med20_cold_df = m25.build_model_rows(test_samples, med20_cold_pred, MEDIAN20_LABEL)
    blend_cold_df = m25.build_model_rows(test_samples, blend_cold_pred, BLEND_LABEL)

    ridge_pers_df = m25.build_model_rows(p_test_samples, ridge_pers_pred, RIDGE_LABEL)
    ridge15_pers_df = m25.build_model_rows(p_test_samples, ridge_pers_15_pred, RIDGE15_LABEL)
    gbm_pers_df = m25.build_model_rows(p_test_samples, gbm_pers_pred, GBM_LABEL)
    mlp_pers_df = m25.build_model_rows(p_test_samples, mlp_pers_pred, MLP_LABEL)
    hist_median_pers_df = m25.build_model_rows(p_test_samples, hist_median_pers_pred, HIST_MEDIAN_LABEL)
    med10_pers_df = m25.build_model_rows(p_test_samples, med10_pers_pred, MEDIAN10_LABEL)
    med20_pers_df = m25.build_model_rows(p_test_samples, med20_pers_pred, MEDIAN20_LABEL)
    blend_pers_df = m25.build_model_rows(p_test_samples, blend_pers_pred, BLEND_LABEL)

    combined_cold_df = pd.concat([
        base_cold_df, ridge15_cold_df, ridge_cold_df, gbm_cold_df, mlp_cold_df,
        hist_median_cold_df, med10_cold_df, med20_cold_df, blend_cold_df,
    ], ignore_index=True)
    combined_pers_df = pd.concat([
        base_pers_df, ridge15_pers_df, ridge_pers_df, gbm_pers_df, mlp_pers_df,
        hist_median_pers_df, med10_pers_df, med20_pers_df, blend_pers_df,
    ], ignore_index=True)
    combined_cold_df["split"] = "cold-start"
    combined_pers_df["split"] = "personalized"

    md.append("## 1. 종합표 (동일 윈도우, 재탐색 없음)\n")
    all_rows = []
    boot_results = {}
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        boot_mae, boot_rmse = m21.cluster_bootstrap(df, ALL_NAMES)
        boot_results[split_name] = (boot_mae, boot_rmse)
        s2 = df.groupby("baseline").agg(mae=("abs_error", "mean"), rmse=("sq_error", lambda c: np.sqrt(c.mean())),
                                          n_users=("user_id", "nunique"), n_samples=("abs_error", "size"))
        n_merges2 = df[df["merge_affected"]].groupby("baseline").size().rename("n_merges")
        s2 = s2.join(n_merges2, how="left")
        s2["n_merges"] = s2["n_merges"].fillna(0).astype(int)
        for name in ALL_NAMES:
            r = s2.loc[name]
            mae_lo, mae_hi = np.percentile(boot_mae[name], [2.5, 97.5])
            rmse_lo, rmse_hi = np.percentile(boot_rmse[name], [2.5, 97.5])
            all_rows.append({
                "방법": ALL_LABELS[name], "평가조건": split_name, "MAE": r["mae"],
                "MAE_95%CI": f"[{mae_lo:.3f},{mae_hi:.3f}]", "RMSE": r["rmse"],
                "RMSE_95%CI": f"[{rmse_lo:.3f},{rmse_hi:.3f}]",
                "#users": int(r["n_users"]), "#eval sessions": int(r["n_samples"]), "#merges": int(r["n_merges"]),
            })
    md.append(m21.df_to_markdown(pd.DataFrame(all_rows)))
    md.append("\n")

    # ===== 3. 쌍체 차이 CI =====
    md.append("\n## 2. 쌍체 차이 95% CI\n")
    for split_name in ["cold-start", "personalized"]:
        boot_mae, boot_rmse = boot_results[split_name]
        md.append(f"\n### {split_name}\n")
        md.append("**MAE: median-5 대비 (전 방법)**\n")
        rows = []
        for name in ALL_NAMES:
            if name == "median5":
                continue
            diff = boot_mae["median5"] - boot_mae[name]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append({"방법": ALL_LABELS[name], "median5-이방법": diff.mean(), "하한": lo, "상한": hi,
                         "0포함(유의하지않음)": bool(lo <= 0 <= hi)})
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))

        md.append("\n**RMSE: median-5·사용자전체평균·최근5회 대비 (Ridge/GBM/MLP만)**\n")
        rows = []
        for name in [RIDGE_LABEL, GBM_LABEL, MLP_LABEL]:
            for other in ["median5", "user_overall_avg", "recent_avg"]:
                diff = boot_rmse[other] - boot_rmse[name]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                rows.append({"방법": ALL_LABELS[name], "비교": m21.BASELINE_LABELS[other],
                             "차이": diff.mean(), "하한": lo, "상한": hi, "0포함(유의하지않음)": bool(lo <= 0 <= hi)})
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n")

    # ----- 9단계 보완: 사후 추가 baseline vs GBM/MLP, 개선폭 비율, Ridge 변형 비교 -----
    md.append("\n## 2-2. 사후 추가 baseline(ablation을 본 뒤 추가) 쌍체 비교\n")
    md.append("차이 = (비교대상 지표) - (사후 baseline 지표). 양수면 사후 baseline이 더 낮은(좋은) 값.\n")
    share_rows = []
    for split_name in ["cold-start", "personalized"]:
        boot_mae, boot_rmse = boot_results[split_name]
        md.append(f"\n### {split_name}\n")
        rows = []
        for name in POST_HOC_NAMES:
            for ref in ["median5", GBM_LABEL, MLP_LABEL]:
                for metric, bt in [("MAE", boot_mae), ("RMSE", boot_rmse)]:
                    diff = bt[ref] - bt[name]
                    lo, hi = np.percentile(diff, [2.5, 97.5])
                    rows.append({"사후 baseline": ALL_LABELS[name], "비교대상": ALL_LABELS[ref],
                                 "지표": metric, "차이(비교대상-사후)": diff.mean(), "하한": lo, "상한": hi,
                                 "0포함": bool(lo <= 0 <= hi)})
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n")
        for model in [GBM_LABEL, MLP_LABEL]:
            gain_model = boot_mae["median5"] - boot_mae[model]
            for name in POST_HOC_NAMES:
                gain_rule = boot_mae["median5"] - boot_mae[name]
                ratio = gain_rule / gain_model
                lo, hi = np.percentile(ratio, [2.5, 97.5])
                share_rows.append({"평가조건": split_name, "모델": ALL_LABELS[model], "단순규칙": ALL_LABELS[name],
                                   "규칙 개선폭(vs median5)": gain_rule.mean(), "모델 개선폭(vs median5)": gain_model.mean(),
                                   "규칙/모델 비율(중앙값)": float(np.median(ratio)), "비율 95%CI": f"[{lo:.2f},{hi:.2f}]"})
    md.append("\n## 2-3. median-5 대비 GBM·MLP 개선폭 중 각 단순 규칙이 차지하는 비율\n")
    md.append("비율 = (median5 MAE - 규칙 MAE) / (median5 MAE - 모델 MAE). 같은 bootstrap 표본 안에서 계산. "
              "분모(모델 개선폭)가 작아 CI가 넓을 수 있다. 1에 가까울수록 모델 개선폭을 그 단순 규칙이 설명한다는 뜻(상관적 해석, 인과 아님).\n")
    md.append(m21.df_to_markdown(pd.DataFrame(share_rows)))
    md.append("\n")

    md.append("\n## 2-4. Ridge 표기: 15피처 vs 22피처, 6단계 숫자와의 차이\n")
    parts = []
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        a = df[df["baseline"] == RIDGE15_LABEL]
        b = df[df["baseline"] == RIDGE_LABEL]
        parts.append((split_name, a["abs_error"].mean(), np.sqrt(a["sq_error"].mean()),
                      b["abs_error"].mean(), np.sqrt(b["sq_error"].mean())))
    md.append("- 종합표의 **6a는 15피처(과거 5세션x3, 24단계 윈도우)**, **6b는 22피처(+이력요약 7개, 7단계 확장)** 이다. "
              "둘 다 표준화+Ridge(alpha=1e4) Pipeline, 학습 구간 fit.\n")
    for sp, m15, r15, m22, r22 in parts:
        md.append(f"- {sp}: 15피처 MAE={m15:.3f}/RMSE={r15:.3f}, 22피처 MAE={m22:.3f}/RMSE={r22:.3f}\n")
    md.append("- 6단계 문서(`docs/ridge_results.md`)의 Ridge 메인 숫자: cold-start MAE 69.928(RMSE 262.069), "
              "personalized MAE 62.054(RMSE 239.762). 같은 24단계 윈도우에서 15피처만 쓴 6a를 6단계 숫자와 비교하면:\n")
    md.append(f"- cold-start: 6a(15피처) MAE {parts[0][1]:.3f} vs 6단계 69.928 (차이 {parts[0][1]-69.928:+.3f}); "
              f"6b(22피처) {parts[0][3]:.3f} (6단계 대비 {parts[0][3]-69.928:+.3f})\n")
    md.append(f"- personalized: 6a(15피처) MAE {parts[1][1]:.3f} vs 6단계 62.054 (차이 {parts[1][1]-62.054:+.3f}); "
              f"6b(22피처) {parts[1][3]:.3f} (6단계 대비 {parts[1][3]-62.054:+.3f})\n")
    md.append("- 판독: 6a가 6단계 숫자와 비슷하면 6단계→현재 차이는 대부분 피처 구성(15→22) 때문, 6a도 6단계와 "
              "크게 다르면 윈도우 구성이나 학습 설정(예: alpha 선택 방식)의 영향이 섞여 있다. "
              "후자인 경우 그 원인은 **추정/미확인**이다(6단계 학습 설정을 여기서 재현해 대조하지 않음).\n\n")

    # ===== 4. MAE/RMSE 순위 (각각 따로) =====
    md.append("\n## 3. MAE/RMSE 순위 (따로 서술)\n")
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        s2 = df.groupby("baseline").agg(mae=("abs_error", "mean"), rmse=("sq_error", lambda c: np.sqrt(c.mean())))
        mae_rank = s2["mae"].sort_values()
        rmse_rank = s2["rmse"].sort_values()
        md.append(f"\n### {split_name}\n")
        mae_str = " < ".join(f"{ALL_LABELS[n]}({mae_rank[n]:.3f})" for n in mae_rank.index)
        rmse_str = " < ".join(f"{ALL_LABELS[n]}({rmse_rank[n]:.3f})" for n in rmse_rank.index)
        md.append(f"- **MAE 순위**: {mae_str}\n")
        md.append(f"- **RMSE 순위**: {rmse_str}\n")
        if list(mae_rank.index[:1]) != list(rmse_rank.index[:1]):
            md.append(f"  - **MAE 1위와 RMSE 1위가 다르다** — 위 bootstrap CI에서 상당수가 "
                      f"서로 통계적으로 구분되지 않으므로, '1위'라는 표현 자체를 신중하게 읽어야 한다.\n")

    # ===== 4. history 길이별 개선 (median-5 대비 GBM/MLP, CI 포함) =====
    print("history 길이별 개선 계산 중...")
    md.append("\n## 4. history 길이별 개선 (5단계와 같은 구간, median-5 대비 GBM/MLP)\n")
    bucket_mae_lookup = {}  # (split, bucket, method) -> (mae, n)
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        df["history_bucket"] = pd.cut(df["target_session_order"], bins=HIST_BINS, labels=HIST_LABELS, right=False)
        md.append(f"\n### {split_name}\n")
        rows = []
        for bucket in HIST_LABELS:
            bucket_df = df[df["history_bucket"] == bucket]
            if len(bucket_df) == 0:
                continue
            row = {"history_bucket": bucket, "n": int((bucket_df["baseline"] == "median5").sum())}
            for name in ["median5", GBM_LABEL, MLP_LABEL]:
                sub = bucket_df[bucket_df["baseline"] == name]
                mae = sub["abs_error"].mean()
                bucket_mae_lookup[(split_name, bucket, name)] = mae
                row[f"{ALL_LABELS[name]} MAE"] = mae
            for name in [GBM_LABEL, MLP_LABEL]:
                boot_b_mae, _ = m21.cluster_bootstrap(bucket_df, ["median5", name])
                diff = boot_b_mae["median5"] - boot_b_mae[name]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                row[f"median5-{ALL_LABELS[name]} 개선폭"] = diff.mean()
                row[f"{ALL_LABELS[name]} CI"] = f"[{lo:.3f},{hi:.3f}]"
            rows.append(row)
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n")

    # ===== 5. LSTM 시도 여부 (결과를 보기 전 고정한 규칙) =====
    print("LSTM 시도 여부 규칙 적용 중...")
    md.append("\n## 5. LSTM 시도 여부 판단\n")
    md.append("**규칙(결과를 보기 전에 고정)**: history가 가장 긴 구간(120+)의 median-5 대비 "
              "MAE 개선폭이 가장 짧은 구간(5-9)보다 크고, 그 차이의 사용자 단위 bootstrap CI가 "
              "0을 포함하지 않을 때만 '시도 대상'으로 판단한다. 아니면 '미시도'로 결정한다.\n")
    lstm_try = False
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        md.append(f"\n### {split_name}\n")
        for name in [GBM_LABEL, MLP_LABEL]:
            improve_long = bucket_mae_lookup[(split_name, "120+", "median5")] - bucket_mae_lookup[(split_name, "120+", name)]
            improve_short = bucket_mae_lookup[(split_name, "5-9", "median5")] - bucket_mae_lookup[(split_name, "5-9", name)]
            diffs = cluster_bootstrap_bucket_diff(df, name, "history_bucket", "120+", "5-9")
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            meets_rule = (improve_long > improve_short) and (lo > 0)
            lstm_try = lstm_try or meets_rule
            md.append(f"- **{ALL_LABELS[name]}**: 120+ 개선폭={improve_long:.3f}, 5-9 개선폭={improve_short:.3f}, "
                      f"차이={diffs.mean():.3f} (95% CI [{lo:.3f},{hi:.3f}]) "
                      f"-> 규칙 충족={meets_rule}\n")

    md.append("\n**사후 추가 baseline의 영향(9단계 보완)**: LSTM 규칙은 median-5 대비 GBM/MLP의 history 구간별 "
              "개선폭만 입력으로 쓰므로 사후 추가 baseline 4종은 결정에 영향을 주지 않는다(규칙·결정 불변).\n")
    if lstm_try:
        md.append("\n**결론: 시도 대상.** 규칙을 충족하는 조합이 있었다. **구현은 하지 않고 계획만 "
                  "적는다**:\n")
        md.append("- 입력 형태: 과거 5개보다 긴 가변 길이 시퀀스(예: 최근 20개 세션)를 그대로 "
                  "RNN/LSTM에 넣고, 7단계에서 쓴 요약 피처는 보조 입력으로만 결합(또는 생략 후 "
                  "시퀀스 자체로 대체 가능 여부 비교).\n")
        md.append("- 시퀀스 길이: history가 짧은 사용자가 많으므로(5단계 분포 참고) 가변 길이를 "
                  "패딩+마스킹으로 처리하거나, 최소 길이를 고정(예: 10)해 그 미만 사용자는 제외.\n")
        md.append("- 비교 기준: median-5와 7~8단계 주결과(GBM/MLP) 대비 MAE/RMSE 쌍체 차이 CI, "
                  "특히 **history 길이가 긴 구간에서만** LSTM이 추가 이득을 주는지가 핵심 질문.\n")
    else:
        md.append("\n**결론: 미시도.** 규칙을 충족하는 조합이 없었다 — 즉 history가 길어질수록 "
                  "개선폭이 통계적으로 유의하게 커진다는 근거가 (이 데이터·이 모델들에서는) "
                  "확인되지 않았다. 사용자당 세션 수가 적은 사용자가 많다는 점(CLAUDE.md에 "
                  "이미 명시된 우려)과 일치하는 결과로 보이며, 복잡한 시퀀스 모델을 추가할 "
                  "근거가 부족하다고 판단해 구현하지 않는다.\n")

    # ===== 6. 한계 =====
    md.append("\n## 6. 한계\n")
    md.append("① **개선폭 크기**: median-5 대비 절대 개선폭은 2.5~2.9초/km(상대 4~5%) 수준으로, "
              "통계적으로는 유의하지만 이 크기가 실제 코칭 문구를 바꿀 만큼 큰지는 별개의 "
              "판단이다(이미 6~8단계 보고에서 반복 명시).\n")
    md.append("② **극단 페이스/상위 1% 윈도우 집중**: RMSE 기준 비교의 상당 부분이 상위 1% 오차 "
              "윈도우(주로 극단 페이스)에 좌우된다(7단계 세분화 참고, SSE의 90% 이상). RMSE로 "
              "순위를 말할 때는 항상 이 전제를 같이 읽어야 한다.\n")
    md.append("③ **cold-start 초보자(6세션 미만) 제외**: 이 프로젝트의 N=5 설계상 세션 6개 미만인 "
              "사용자는 애초에 평가에 들어오지 않는다 — 가장 신규인 사용자에 대한 성능은 이 "
              "보고서들이 전혀 말해주지 않는다(미확인, 별도 설계 필요).\n")
    md.append("④ **날씨/계획 거리 없음**: `docs/feature_availability.md`에서 이미 확인했듯 날씨와 "
              "사용자가 입력한 계획 거리는 원본 데이터에 없어 모든 모델이 이 정보 없이 학습됐다 "
              "— 이게 성능 상한을 얼마나 낮추는지는 **미확인**이다.\n")
    md.append("⑤ **직접 구현 GBM과 sklearn 환경 이슈**: sklearn.tree의 import 성공 여부가 실행마다 "
              "달라지는 간헐적 현상을 7단계 보완에서 발견했고(원인 미확인), 재현성을 위해 "
              "import 성공 여부와 무관하게 `gbt_lib.py`(numpy 직접구현)를 고정 사용하도록 "
              "고쳤다. 이 환경의 근본 원인(Application Control Policy의 캐시/타이밍 문제로 "
              "추정)은 검증하지 않았다 — 다른 환경/다른 시점에 재현했을 때 실제 "
              "sklearn.HistGradientBoostingRegressor 결과와 수치가 얼마나 다를지는 **미확인**이다"
              "(0-1/0-2절 비교 참고, 직접 대조는 못 했음).\n")
    md.append("⑥ **입력 정보량이 병목일 수 있다는 것은 가설**: Ridge/GBM/MLP가 서로 성능이 "
              "비슷하고(통계적으로 구분 안 되는 경우가 많음) median-5 대비 개선폭도 비슷한 "
              "크기라는 점은 '모델 구조보다 입력 피처(과거 5세션 요약)가 성능을 제한하는 "
              "병목일 수 있다'는 가설과 일치하지만, 이를 직접 검증(예: 피처를 늘렸을 때 "
              "성능이 계속 오르는지 체계적으로 확인)하지는 않았다 — **추정/미확인**이다.\n")
    md.append("\n**확인하지 못한 원인은 모두 추정/미확인으로 남긴다.**\n")

    OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
    print(f"\n최종 저장 완료: {OUTPUT_MD_PATH.resolve()}")
    return {
        "combined_cold_df": combined_cold_df, "combined_pers_df": combined_pers_df,
        "boot_results": boot_results, "md": md, "lstm_try": lstm_try,
    }


if __name__ == "__main__":
    main()
