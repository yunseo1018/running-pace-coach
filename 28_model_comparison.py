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
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import gbt_lib as gl
from report_notes import MACRO_NOTE
import ridge_lib as rl

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
# MLP 하이퍼파라미터는 하드코딩하지 않고 27번이 저장한 선택 결과(검증 MAE로만 선택, 테스트 미사용)를 읽는다
# -> 27번 재실행 시 28/29번과 숫자가 어긋나는 일을 막는다. 27번을 먼저 실행해야 한다.
_mlp_sel = json.loads((Path("docs") / "mlp_selected_hyperparams.json").read_text(encoding="utf-8"))["splits"]
MLP_COLD_PARAMS = {k: _mlp_sel["cold-start"][k] for k in ("n_hidden_layers", "width", "weight_decay")}
MLP_COLD_EPOCHS = _mlp_sel["cold-start"]["n_epochs"]
MLP_COLD_FORM = _mlp_sel["cold-start"]["form"]
MLP_PERS_PARAMS = {k: _mlp_sel["personalized"][k] for k in ("n_hidden_layers", "width", "weight_decay")}
MLP_PERS_EPOCHS = _mlp_sel["personalized"]["n_epochs"]
MLP_PERS_FORM = _mlp_sel["personalized"]["form"]
assert MLP_COLD_FORM in ("median5_residual", "direct") and MLP_PERS_FORM in ("median5_residual", "direct")
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
    out = {}
    for uid, grp in d.groupby("user_id"):
        # 3주차 4번: 위치==session_order 전제를 assert로 검증(사용자별 0..n-1 조밀 인덱스 + 시작 시각 비감소)
        assert (grp["session_order"].to_numpy() == np.arange(len(grp))).all(), f"session_order가 조밀하지 않음: user {uid}"
        assert (np.diff(grp["start_time"].to_numpy()) >= 0).all(), f"session_order와 start_time 순서가 어긋남: user {uid}"
        out[uid] = grp["avg_pace_sec_per_km"].to_numpy()
    return out


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
    def ridge_pipeline(alpha):
        # ridge_lib 고정(sklearn import 성공 여부와 무관): 표준화+Ridge, 학습 구간에서만 fit
        return rl.RidgePipeline(alpha)

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
    def _mlp_pred(form, params, epochs, X_a, y_a, med_a, X_b, med_b):
        resid = form == "median5_residual"
        model, scaler = m27.train_mlp_fixed_epochs(
            X_a, (y_a - med_a) if resid else y_a, params["n_hidden_layers"], params["width"],
            params["weight_decay"], MLP_SEED, epochs,
        )
        return (med_b if resid else 0.0) + m27.predict_mlp(model, scaler, X_b)

    mlp_cold_pred = _mlp_pred(MLP_COLD_FORM, MLP_COLD_PARAMS, MLP_COLD_EPOCHS, X_tr, y_tr, med_tr_full, X_te, med_te_full)
    mlp_pers_pred = _mlp_pred(MLP_PERS_FORM, MLP_PERS_PARAMS, MLP_PERS_EPOCHS, X_ptr, y_ptr, med_ptr, X_pte, med_pte)

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

    md.append("## 문서 생성 메모\n")
    md.append("- 이 문서의 Ridge(6a/6b)는 `ridge_lib.RidgePipeline`(numpy, ManualStandardScaler + 닫힌형 Ridge)으로 계산한다 — "
              "sklearn import 성공 여부와 무관하게 고정(GBM을 gbt_lib로 고정한 것과 같은 방식).\n")
    md.append("- **생성 이력**: (1) 초기 버전(커밋 91c2e86, 7eeef37)은 sklearn Pipeline(StandardScaler, Ridge)으로 생성. "
              "(2) 커밋 330f5f7 버전은 그 시점에 sklearn import가 Windows 애플리케이션 제어 정책으로 차단돼 있어 "
              "**임시 하네스**(sklearn.Ridge/StandardScaler/Pipeline을 ridge_lib로 대체해 이 스크립트를 실행, 출력만 임시 경로로 저장 후 문서에 반영)로 생성했다. "
              "(3) 이 버전은 스크립트 자체에서 ridge_lib를 직접 쓰도록 고정한 뒤 재생성했다(하네스 불필요). "
              "(1)→(3) 사이 숫자 비교 결과는 해당 커밋 메시지/작업 보고 참고.\n\n")
    headline_idx = len(md)  # 헤드라인은 bootstrap 결과 계산 후 여기에 삽입(수치 동적)
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
        cur_df = combined_cold_df if split_name == "cold-start" else combined_pers_df
        full_mae = cur_df.groupby("baseline")["abs_error"].mean()  # 전체 표본 평균 MAE (점추정)
        for model in [GBM_LABEL, MLP_LABEL]:
            gain_model = boot_mae["median5"] - boot_mae[model]
            for name in POST_HOC_NAMES:
                gain_rule = boot_mae["median5"] - boot_mae[name]
                ratio = gain_rule / gain_model
                lo, hi = np.percentile(ratio, [2.5, 97.5])
                point_gm = full_mae["median5"] - full_mae[model]
                point_gr = full_mae["median5"] - full_mae[name]
                share_rows.append({"평가조건": split_name, "모델": ALL_LABELS[model], "단순규칙": ALL_LABELS[name],
                                   "규칙 개선폭(점추정)": point_gr, "모델 개선폭(점추정)": point_gm,
                                   "비율(점추정)": point_gr / point_gm,
                                   "참고: bootstrap 비율 중앙값": float(np.median(ratio)), "비율 95%CI(bootstrap)": f"[{lo:.2f},{hi:.2f}]"})
    md.append("\n## 2-3. median-5 대비 GBM·MLP 개선폭 중 각 단순 규칙이 차지하는 비율\n")
    md.append("**계산 방식(정정됨)**: 주 수치 '비율(점추정)' = 전체 평가 표본의 평균 MAE로 계산한 (median5 MAE − 규칙 MAE) / (median5 MAE − 모델 MAE). "
              "이전 버전은 bootstrap 반복마다 계산한 비율의 중앙값을 보고했는데(예: cold-start (d) vs GBM 1.058), 비율의 중앙값은 평균 MAE 개선폭의 비율(1.074)과 일반적으로 같지 않으므로 "
              "점추정으로 바꿨다. bootstrap 열은 참고용(중앙값)과 95% CI(비율 분포 percentile)다. "
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
              "둘 다 표준화+Ridge(alpha=1e4), `ridge_lib.RidgePipeline`(numpy), 학습 구간 fit.\n")
    md.append("- **레거시 6a(Ridge15, alpha=1e4 재사용)와 `docs/same_scope_comparison.md`의 Ridge15(자체 CV alpha=1e-4, 6단계 원본과 일치)는 서로 다른 값이다.**\n")
    for sp, m15, r15, m22, r22 in parts:
        md.append(f"- {sp}: 15피처 MAE={m15:.3f}/RMSE={r15:.3f}, 22피처 MAE={m22:.3f}/RMSE={r22:.3f}\n")
    md.append("- 6단계 문서(`docs/ridge_results.md`)의 Ridge 메인 숫자: cold-start MAE 69.928(RMSE 262.069), "
              "personalized MAE 62.054(RMSE 239.762). 같은 24단계 윈도우에서 15피처만 쓴 6a를 6단계 숫자와 비교하면:\n")
    md.append(f"- cold-start: 6a(15피처) MAE {parts[0][1]:.3f} vs 6단계 69.928 (차이 {parts[0][1]-69.928:+.3f}); "
              f"6b(22피처) {parts[0][3]:.3f} (6단계 대비 {parts[0][3]-69.928:+.3f})\n")
    md.append(f"- personalized: 6a(15피처) MAE {parts[1][1]:.3f} vs 6단계 62.054 (차이 {parts[1][1]-62.054:+.3f}); "
              f"6b(22피처) {parts[1][3]:.3f} (6단계 대비 {parts[1][3]-62.054:+.3f})\n")
    md.append("- **정정 및 원인 분해** (`docs/ridge_discrepancy.md`, 30_ridge_discrepancy.py): 이전 버전은 이 차이를 '피처 구성 때문'이라고 적었으나 틀렸다. "
              "6a는 6단계와 같은 15피처다. 6단계 설정(ridge_lib, 04/06 윈도우, alpha=0.0001)은 MAE 69.928493 / 62.053868을 그대로 재현했고(S0), "
              "윈도우를 24단계로 바꿔도 +0.001~0.002에 그쳤으며(S1~S3), **차이(+0.34 / +0.21)는 거의 전부 alpha 때문이다**: 종합표 6a는 7단계에서 22피처로 선택된 alpha=10000을 "
              "15피처에도 그대로 썼다(S4: ridge_lib로 alpha만 1e4로 바꾸면 70.2737 / 62.2698). 구현 전환(ridge_lib→sklearn)은 소수 3자리까지 같은 값(S5)이어서 영향이 확인되지 않았다 "
              "(S6 '구현만 전환' 단계는 sklearn import가 정책으로 차단된 실행에서는 미실행). "
              "따라서 6a는 15피처 자신의 CV로 고른 alpha가 아니라 22피처용 alpha를 쓴 값이며, 6단계 설정 값(69.929 / 62.056)이 더 적절한 '15피처 기준'이다. "
              "15→22피처 효과는 6단계 설정 기준 MAE −4.25(cold-start) / −3.74(personalized), 종합표 6a 기준 −4.59 / −3.96이다(후자는 alpha 차이가 섞인 값).\n\n")

    # ----- 헤드라인(수치는 이번 실행의 bootstrap 결과에서 동적으로 채움: 27번 재실행 시 어긋남 방지) -----
    def _ci(arr):
        lo_, hi_ = np.percentile(arr, [2.5, 97.5])
        return float(arr.mean()), float(lo_), float(hi_)

    hl = {}
    for sp in ["cold-start", "personalized"]:
        bm, _ = boot_results[sp]
        hl[sp] = {
            "gain_gbm": _ci(bm["median5"] - bm[GBM_LABEL]), "gain_mlp": _ci(bm["median5"] - bm[MLP_LABEL]),
            "d_minus_gbm": _ci(bm[GBM_LABEL] - bm[BLEND_LABEL]),  # GBM − (d): 음수면 GBM이 더 낮음
            "d_minus_mlp": _ci(bm[MLP_LABEL] - bm[BLEND_LABEL]),
        }
    gains = [hl[sp][k][0] for sp in hl for k in ("gain_gbm", "gain_mlp")]

    def _sig(c):  # CI가 0을 포함하는가
        return c[1] <= 0 <= c[2]

    hl_md = ["## 헤드라인 (확인된 사실 / 추정 구분)\n", "**확인된 사실 (이 문서의 표와 CI에서 직접 읽힌 것)**\n"]
    hl_md.append(f"- 학습 모델(GBM, MLP)은 두 평가 조건 모두에서 median-5보다 MAE가 낮고 쌍체 차이 CI가 0을 포함하지 않는다"
                 f"(개선폭 {min(gains):.1f}~{max(gains):.1f}초/km, 4~5%) " + MACRO_NOTE + ". 그러나 이것만으로 끝나는 결론이 아니다:\n")
    hl_md.append("- **사후 추가 baseline (median-10, median-20, 전체 이력 중앙값, 결합 규칙 (d))은 ablation 결과를 본 뒤 추가한 것이고, "
                 "후보 4개 중 가장 좋은 것(d)을 함께 보고하는 것이라 약간의 낙관(선택 편향) 가능성이 있다.** 이 4개는 고정했고 더 추가하지 않았다.\n")
    c = hl["cold-start"]
    hl_md.append(f"- **cold-start**: (d) 0.5×median-5+0.5×전체 이력 중앙값과 GBM/MLP의 MAE 차이: GBM−(d) {c['d_minus_gbm'][0]:+.3f} "
                 f"[{c['d_minus_gbm'][1]:+.3f}, {c['d_minus_gbm'][2]:+.3f}] ({'CI가 0을 포함해 구분되지 않음' if _sig(c['d_minus_gbm']) else 'CI가 0을 포함하지 않음'}), "
                 f"MLP−(d) {c['d_minus_mlp'][0]:+.3f} [{c['d_minus_mlp'][1]:+.3f}, {c['d_minus_mlp'][2]:+.3f}] "
                 f"({'CI가 0을 포함해 구분되지 않음' if _sig(c['d_minus_mlp']) else 'CI가 0을 포함하지 않음'}) (초/km, 양수면 (d)가 더 낮음).\n")
    p = hl["personalized"]
    hl_md.append(f"- **personalized**: GBM−(d) {p['d_minus_gbm'][0]:+.3f} [{p['d_minus_gbm'][1]:+.3f}, {p['d_minus_gbm'][2]:+.3f}] "
                 f"({'CI가 0을 포함' if _sig(p['d_minus_gbm']) else 'CI가 0을 포함하지 않음'}), "
                 f"MLP−(d) {p['d_minus_mlp'][0]:+.3f} [{p['d_minus_mlp'][1]:+.3f}, {p['d_minus_mlp'][2]:+.3f}] "
                 f"({'CI가 0을 포함' if _sig(p['d_minus_mlp']) else 'CI가 0을 포함하지 않음'}) — 음수는 모델이 (d)보다 낮다는 뜻이며 "
                 f"점추정으로는 모델이 약 {-p['d_minus_gbm'][0]:.1f}~{-p['d_minus_mlp'][0]:.1f}초/km 낮고, 유의성은 CI 기준으로만 말할 수 있다.\n")
    hl_md.append("- **코칭 판정(합성 시나리오, `docs/coaching_results.md`)**: 결과는 해당 문서의 요약 절 참고(예측기별 불일치율 차이와 CI는 거기서 계산).\n")
    hl_md.append("- MAE와 RMSE의 순위는 다르다(3절). 개선폭 비율은 점추정 기준이며 2-3절에 계산 방식을 명시했다.\n")
    hl_md.append("\n**추정/미확인**: (d)가 GBM/MLP만큼 좋은 이유(장기 페이스 수준 정보가 개선의 상당 부분을 설명한다는 가설)는 직접 검증하지 않았다. "
                 "personalized에서 GBM/MLP가 (d)보다 낮은 이유(시간 순서 정보 활용 등)도 미확인이다.\n\n")
    cold_mlp_mae = float(combined_cold_df[combined_cold_df["baseline"] == MLP_LABEL]["abs_error"].mean())
    hl_md.append("## 분할기 변경에 따른 민감도 (MLP cold-start)\n")
    hl_md.append("MLP의 cold-start 하이퍼파라미터 선택은 train 사용자 GroupKFold의 첫 fold를 검증으로 쓴다. 처음에는 sklearn GroupKFold(커밋 00c27d1 시점), "
                 "재현성 고정 후에는 ridge_lib GroupKFold로 바꿨고, **둘 다 유효한 CV 선택이다(어느 쪽이 더 낫다고 판단하지 않는다).** "
                 "fold 구성이 바뀌자 선택된 구조가 달라져(2×64, wd=0, 97에폭 → 2×32, wd=1e-4, 112에폭) cold-start MLP 결과가 약 0.2초/km 움직였다:\n\n")
    hl_md.append("| 분할기 | cold-start MLP test MAE | median-5 대비 개선폭(bootstrap 평균) |\n|---|---|---|\n"
                 "| sklearn GroupKFold (이전 문서 기록값) | 59.578 | 2.661 |\n"
                 f"| ridge_lib GroupKFold (현재) | {cold_mlp_mae:.3f} | {hl['cold-start']['gain_mlp'][0]:.3f} |\n\n")
    hl_md.append("personalized(시간순 분할)와 GBM·Ridge·baseline은 분할기와 무관하게 불변. 자세한 비교는 `docs/mlp_results.md` 6절. "
                 "변동 원인(검증 fold 데이터 차이 vs 조기종료·에폭 변화)은 분해하지 않았다 — 추정/미확인.\n\n")
    md[headline_idx:headline_idx] = hl_md

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
        top1 = mae_rank.index[0]
        if top1 in (GBM_LABEL, MLP_LABEL):
            dd = hl[split_name]["d_minus_gbm" if top1 == GBM_LABEL else "d_minus_mlp"]
            same = dd[1] <= 0 <= dd[2]
            md.append(f"  - MAE 1위는 {ALL_LABELS[top1]}이며 사후 추가 규칙 (d)와 "
                      f"{'**구분되지 않는다**' if same else '구분된다(1위가 (d)보다 유의하게 낮음)'}"
                      f"({'MLP' if top1 == MLP_LABEL else 'GBM'}−(d) {dd[0]:+.3f}, CI [{dd[1]:.3f}, {dd[2]:.3f}]"
                      f"{', CI가 0을 포함' if same else ', CI가 0을 포함하지 않음'}).\n")
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
    md.append("① **개선폭 크기**: median-5 대비 절대 개선폭은 2.6~3.0초/km(상대 4~5%) 수준으로, "
              "통계적으로는 유의하지만(" + MACRO_NOTE + ") 이 크기가 실제 코칭 문구를 바꿀 만큼 큰지는 별개의 "
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
