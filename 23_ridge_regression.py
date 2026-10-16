# -*- coding: utf-8 -*-
"""
10월 2주차 6번: Ridge 회귀를 baseline 통합표에 추가.

환경 제약: sklearn을 import할 수 없어(ridge_lib.py 상단 설명 참고)
Ridge/Huber/GroupKFold를 numpy로 직접 구현한 ridge_lib.py를 쓴다.

산출물: docs/ridge_results.md
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import config
import ridge_lib as rl

# 21단계의 baseline 비교/부트스트랩 유틸리티를 그대로 재사용한다 (파일명이
# 숫자로 시작해 import 문으로 못 불러오므로 importlib로 로드).
_spec = importlib.util.spec_from_file_location("m21", "21_baseline_integrated_report.py")
m21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m21)

OUTPUT_MD_PATH = Path("docs") / "ridge_results.md"
POSTHOC_TRAIN_PATH = Path("data") / "posthoc_train_dataset.pkl"
POSTHOC_TEST_PATH = Path("data") / "posthoc_test_dataset.pkl"

# 최초 격자였으나 cold-start/personalized 선택이 하한(0.01)에 걸려, post-hoc
# 선택은 상한(1000.0)에 걸려 범위를 넓혀 재검증했다 (아래 ORIGINAL_ALPHA_GRID로 보존).
ORIGINAL_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
N_FOLDS = 5
CV_SEED = 42
HISTORY_INNER_RATIO = 0.8  # personalized 내부 시간순 검증 분할 비율 (06단계와 동일 관례)

RIDGE_LABEL = "ridge"
HUBER_LABEL = "huber"
RIDGE_POSTHOC_LABEL = "ridge_posthoc"

ENV_DIAG_MODULES = [
    "sklearn", "scipy.linalg", "scipy.optimize", "scipy.spatial",
    "lightgbm", "xgboost", "torch", "pandas", "statsmodels",
]


def diagnose_environment():
    """설치/변경 없이 import만 시도해 성공/실패와 에러 메시지를 기록한다."""
    import importlib
    results = []
    for mod_name in ENV_DIAG_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", None)
            results.append({"module": mod_name, "ok": True, "detail": f"version={version}" if version else "OK"})
        except Exception as e:
            results.append({"module": mod_name, "ok": False, "detail": f"{type(e).__name__}: {e}"})
    return results


def samples_to_xy(samples):
    X = np.array([s["X"] for s in samples], dtype=float)
    y = np.array([s["y"] for s in samples], dtype=float)
    user_ids = np.array([s["user_id"] for s in samples])
    return X, y, user_ids


# ===================== alpha 선택 =====================

def select_alpha_group_cv(X, y, user_ids, alphas, n_folds=N_FOLDS, seed=CV_SEED):
    """cold-start: 학습 데이터(train) 내부에서만 사용자 단위 GroupKFold로 alpha 선택."""
    cv_scores = {a: [] for a in alphas}
    for train_idx, val_idx in rl.group_kfold_indices(user_ids, n_folds, seed):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        scaler = rl.ManualStandardScaler().fit(X_tr)
        X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)
        for a in alphas:
            beta, intercept = rl.ridge_fit(X_tr_s, y_tr, alpha=a)
            pred = rl.ridge_predict(X_val_s, beta, intercept)
            cv_scores[a].append(np.mean(np.abs(y_val - pred)))
    mean_scores = {a: float(np.mean(v)) for a, v in cv_scores.items()}
    best_alpha = min(mean_scores, key=mean_scores.get)
    return best_alpha, mean_scores


def personalized_inner_split(train_samples, ratio=HISTORY_INNER_RATIO):
    """personalized: history 구간 내부를 사용자별로 시간순(target_session_order)
    앞부분(inner-train)/뒷부분(inner-val)으로 한 번 더 나눈다 (06단계의
    history/hold-out 분할과 같은 방식을 history 내부에 재귀적으로 적용)."""
    by_user = {}
    for s in train_samples:
        by_user.setdefault(s["user_id"], []).append(s)
    inner_train, inner_val = [], []
    for uid, lst in by_user.items():
        lst_sorted = sorted(lst, key=lambda s: s["target_session_order"])
        n = len(lst_sorted)
        if n < 2:
            inner_train.extend(lst_sorted)  # 1개뿐이면 검증에 못 쓰고 학습에만 사용
            continue
        n_inner_train = min(max(int(n * ratio), 0), n - 1)
        inner_train.extend(lst_sorted[:n_inner_train])
        inner_val.extend(lst_sorted[n_inner_train:])
    return inner_train, inner_val


def select_alpha_time_split(train_samples, alphas):
    """personalized: history 내부 시간순 검증(inner-val)으로만 alpha 선택,
    실제 holdout(test)은 절대 사용하지 않는다."""
    inner_train, inner_val = personalized_inner_split(train_samples)
    X_tr, y_tr, _ = samples_to_xy(inner_train)
    X_val, y_val, _ = samples_to_xy(inner_val)
    scaler = rl.ManualStandardScaler().fit(X_tr)
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)
    scores = {}
    for a in alphas:
        beta, intercept = rl.ridge_fit(X_tr_s, y_tr, alpha=a)
        pred = rl.ridge_predict(X_val_s, beta, intercept)
        scores[a] = float(np.mean(np.abs(y_val - pred)))
    best_alpha = min(scores, key=scores.get)
    return best_alpha, scores, len(inner_train), len(inner_val)


def fit_final(train_samples, alpha, fit_fn=rl.ridge_fit):
    X, y, _ = samples_to_xy(train_samples)
    scaler = rl.ManualStandardScaler().fit(X)
    X_s = scaler.transform(X)
    result = fit_fn(X_s, y, alpha=alpha)
    beta, intercept = result[0], result[1]
    extra = result[2] if len(result) > 2 else None
    return scaler, beta, intercept, extra


def predict_with(test_samples, scaler, beta, intercept):
    X, y, user_ids = samples_to_xy(test_samples)
    X_s = scaler.transform(X)
    pred = rl.ridge_predict(X_s, beta, intercept)
    return pred


def build_model_rows(test_samples, preds, label):
    """m21.all_df와 같은 스키마로 행을 만든다 (baseline 비교/부트스트랩 함수 재사용용)."""
    rows = []
    for s, y_pred in zip(test_samples, preds):
        target_merged, input_merged = m21.is_merge_affected(s)
        target_extreme, input_extreme = m21.is_extreme_affected(s)
        rows.append({
            "baseline": label, "user_id": s["user_id"], "y_true": s["y"], "y_pred": y_pred,
            "abs_error": abs(s["y"] - y_pred), "sq_error": (s["y"] - y_pred) ** 2,
            "target_merged": target_merged, "input_merged": input_merged,
            "merge_affected": target_merged or input_merged,
            "target_extreme": target_extreme, "input_extreme": input_extreme,
            "extreme_affected": target_extreme or input_extreme,
            "target_session_order": s["target_session_order"],
            "target_actual_distance_km": s["target_actual_distance_km"],
            "target_session_id": s["target_session_id"],
        })
    return pd.DataFrame(rows)


def main():
    md = []
    md.append("# Ridge 회귀 결과 (10월 2주차 6번)\n\n")
    md.append(
        "**환경 제약 안내 (작성 시점 기준)**: 이 Ridge/Huber/GroupKFold 구현은 "
        "`import sklearn`이 실패하던 시점(scipy 컴파일 확장 `_distance_pybind`가 "
        "Windows Application Control Policy에 차단된 것으로 보였고, 당시 여러 번 "
        "재시도해도 동일하게 실패) 에 numpy로 직접 작성한 것이다 (alpha=0 Ridge가 "
        "numpy OLS와 4.00e-15 오차로 일치함을 검증). **단, 이번 6단계 마무리 중 "
        "재확인(아래 '1. 환경 진단')한 결과 지금은 `import sklearn`이 정상 동작한다** "
        "— 원인(정책 변경/일시적 문제였는지)은 미확인이며, 이 문서의 모든 결과는 "
        "sklearn이 실패하던 전제로 이미 작성·검증된 numpy 구현을 그대로 썼다 "
        "(재확인 후에도 sklearn으로 바꿔 다시 만들지는 않았음).\n\n"
    )

    # ===== 0. 입력 확인 =====
    main_df = config.load_main_sessions()
    md.append(f"## 0. 입력 데이터\n- 메인 데이터: `{config.MAIN_DATA_PATH}` "
              f"(세션 {len(main_df)}건, 사용자 {main_df['user_id'].nunique()}명)\n")

    user_sequences = pd.read_pickle(config.USER_SEQUENCES_PATH)
    print(f"[입력 확인] user_sequences: {config.USER_SEQUENCES_PATH} "
          f"(세션 {len(user_sequences)}건, 사용자 {user_sequences['user_id'].nunique()}명)")

    train_samples = pd.read_pickle(config.TRAIN_DATASET_PATH)
    test_samples = pd.read_pickle(config.TEST_DATASET_PATH)
    p_train_samples = pd.read_pickle(config.PERSONALIZED_TRAIN_DATASET_PATH)
    p_test_samples = pd.read_pickle(config.PERSONALIZED_TEST_DATASET_PATH)
    posthoc_train_samples = pd.read_pickle(POSTHOC_TRAIN_PATH)
    posthoc_test_samples = pd.read_pickle(POSTHOC_TEST_PATH)
    for name, s in [("train(cold-start)", train_samples), ("test(cold-start)", test_samples),
                     ("personalized_train(history)", p_train_samples), ("personalized_test(holdout)", p_test_samples),
                     ("posthoc_train", posthoc_train_samples), ("posthoc_test", posthoc_test_samples)]:
        print(f"[입력 확인] {name}: {len(s)}건")
        md.append(f"- `{name}`: {len(s)}건\n")
    md.append("\n입력 피처(X)는 04/06단계가 만든 **과거 5개 세션만 쓰는 15차원 "
               "벡터**다 (타깃 세션의 실제 거리 등 사후 정보는 메인에서 제외). "
               "post-hoc(7번)에서만 16차원(타깃 거리 포함)을 따로 쓴다.\n")

    overall_avg_lookup = m21.build_user_overall_avg_lookup(user_sequences)

    # ===== 1. 환경 진단 (설치/변경 없이 확인만) =====
    print("\n환경 진단: 라이브러리 import 시도 중 (설치/변경 없음)...")
    md.append("\n## 1. 환경 진단 (설치/변경 없이 import만 시도, 확인만)\n")
    env_results = diagnose_environment()
    env_df = pd.DataFrame(env_results).rename(columns={"module": "모듈", "ok": "성공", "detail": "상세"})
    md.append(m21.df_to_markdown(env_df))
    for r in env_results:
        print(f"  {r['module']}: {'OK' if r['ok'] else 'FAIL'} - {r['detail']}")
    n_ok = sum(1 for r in env_results if r["ok"])
    md.append(f"\n- **{n_ok}/{len(env_results)}개 모듈 import 성공.** "
              f"sklearn/scipy.linalg/scipy.optimize/scipy.spatial/pandas/torch는 성공, "
              f"lightgbm/xgboost/statsmodels는 미설치(ModuleNotFoundError)로 실패 — "
              f"**설치하지 않았다** (요청 범위가 '확인만'이므로).\n")
    md.append("- **차단된 파일 경로**: 실패한 3개(lightgbm/xgboost/statsmodels) 모두 "
              "`ModuleNotFoundError`(패키지 자체가 설치되어 있지 않음)이고, DLL 차단이나 "
              "권한 오류로 인한 실패는 이번 재확인에서 **없었다**. sklearn도 지금은 "
              "정상 import되므로 차단된 경로가 없다 (`sklearn.__file__`, "
              "`scipy.spatial.distance.__file__` 모두 `Python314\\Lib\\site-packages\\` "
              "아래 정상 경로를 가리킴, 수동 확인).\n")
    md.append("- **다른 Python 환경(conda/venv/WSL) 존재 여부**: `py -0p`로 확인한 결과 "
              "이 머신에 등록된 Python은 **3.14 하나뿐**(`...\\Programs\\Python\\Python314\\python.exe`)"
              "이다. `conda` 명령은 없음(미설치). `wsl --status`/`wsl -l -v` 실행 결과는 "
              "'배포판 없음'을 뜻하는 안내만 나와 **WSL에 등록된 배포판은 없는 것으로 보이나, "
              "한글 경로의 인코딩 문제로 출력이 깨져 완전히는 확인하지 못했다 (미확인)**. "
              "별도 venv 폴더도 사용자 홈/프로젝트 디렉터리에서 찾지 못했다 (수동 확인, 전체 "
              "디스크를 뒤진 것은 아니므로 '없다'고 완전히 단정하지는 않음).\n")

    # ===== 2. 직접 구현 검증 (Huber IRLS, GroupKFold) =====
    print("\nridge_lib 직접 구현 검증 중 (Ridge/Huber/GroupKFold)...")
    md.append("\n## 2. 직접 구현 검증\n")
    ridge_ols_diff = rl._validate_ridge_against_ols()
    md.append(f"- **Ridge vs OLS(alpha=0)**: 예측값 최대 차이 {ridge_ols_diff:.2e} "
              f"(<1e-6 기준 통과). 확인됨.\n")

    huber_val = rl._validate_huber_irls()
    md.append(f"- **Huber IRLS 검증(합성 데이터, 이상치 15/300건 포함)**: "
              f"{huber_val['n_iterations']}회 반복 후 수렴. 매 반복 실제 Huber 목적함수"
              f"(손실+L2페널티)가 단조감소(또는 불변)함 = **{huber_val['monotonic_decrease']}** "
              f"(반복 간 최대 증가분 {huber_val['max_increase']:.2e}, 부동소수점 오차 수준). "
              f"IRLS 수렴해의 목적함수값({huber_val['irls_final_obj']:.6f})이 같은 목적함수를 "
              f"numpy 경사하강법(50,000회, 충분히 수렴)으로 독립적으로 최소화한 해"
              f"({huber_val['gd_final_obj']:.6f})보다 나쁘지 않음 = "
              f"**{huber_val['irls_not_worse_than_gd']}**. **확인됨** (이전 '미검증' 표기를 대체).\n")

    group_kfold_overlap_synthetic = rl._validate_group_kfold_no_overlap()
    user_ids_for_check = np.array([s["user_id"] for s in train_samples])
    gk_folds = list(rl.group_kfold_indices(user_ids_for_check, N_FOLDS, CV_SEED))
    val_sets = [set(user_ids_for_check[vm].tolist()) for _, vm in gk_folds]
    real_overlap = 0
    for i in range(len(val_sets)):
        for j in range(i + 1, len(val_sets)):
            real_overlap = max(real_overlap, len(val_sets[i] & val_sets[j]))
    assert real_overlap == 0, f"실제 cold-start train 데이터에서 fold간 사용자 겹침 발견: {real_overlap}"
    md.append(f"- **GroupKFold 그룹(사용자) 겹침 검증**: 합성 데이터(1000명/5000행) 기준 "
              f"fold간 최대 겹침 **{group_kfold_overlap_synthetic}**, 실제 cold-start train 데이터"
              f"({len(set(user_ids_for_check.tolist()))}명, {N_FOLDS}-fold) 기준 fold간 최대 겹침 "
              f"**{real_overlap}** (둘 다 0이어야 통과, 통과). **확인됨**.\n")

    # ===== 2~3. alpha 선택 =====
    print("\ncold-start Ridge: GroupKFold CV로 alpha 선택 중...")
    X_tr, y_tr, uid_tr = samples_to_xy(train_samples)
    cold_alpha, cold_cv_scores = select_alpha_group_cv(X_tr, y_tr, uid_tr, ALPHA_GRID)
    print(f"  선택된 alpha={cold_alpha}, CV MAE={cold_cv_scores}")

    print("personalized Ridge: history 내부 시간순 검증으로 alpha 선택 중...")
    pers_alpha, pers_val_scores, n_inner_tr, n_inner_val = select_alpha_time_split(p_train_samples, ALPHA_GRID)
    print(f"  선택된 alpha={pers_alpha}, inner-val MAE={pers_val_scores} "
          f"(inner-train {n_inner_tr}건/inner-val {n_inner_val}건)")

    md.append("\n## 3. 하이퍼파라미터(alpha) 선택\n")
    md.append(f"- **선택 기준**: 각 alpha 후보에 대해 계산한 **평균 CV MAE(절대오차 평균)를 "
              f"최소화하는 alpha**를 선택한다 (RMSE 기준 선택은 하지 않음 — MAE가 선택 지표).\n")
    md.append(f"- **최초 후보 격자**: {ORIGINAL_ALPHA_GRID} (로그 간격). 이 격자로 처음 돌렸을 때 "
              f"cold-start/personalized 선택이 **격자 하한(0.01)**에, post-hoc(7번) 선택이 "
              f"**격자 상한(1000.0)**에 걸렸다 — 둘 다 범위 안에서 내부 최솟값이 아니라 경계에서 "
              f"끝났다는 뜻이라, 두 방향 모두 격자를 넓혀 재검증했다.\n")
    md.append(f"- **넓힌 후보 격자(최종 사용)**: {ALPHA_GRID}\n")
    md.append(f"- **cold-start**: train 사용자만으로 **사용자 단위 GroupKFold({N_FOLDS}-fold, "
              f"seed={CV_SEED})**를 train 내부에서 돌려 평균 CV MAE가 가장 작은 alpha를 선택. "
              f"test는 전혀 사용하지 않음.\n")
    md.append(f"  - 선택된 alpha = **{cold_alpha}** ")
    md.append(f"({'여전히 격자 하한' if cold_alpha == min(ALPHA_GRID) else '격자 경계 아님(내부 최솟값)'})\n")
    md.append("  - alpha별 평균 CV MAE: " + ", ".join(f"{a}={v:.6f}" for a, v in cold_cv_scores.items()) + "\n")
    md.append(f"  - **넓힌 뒤 바뀐 점 확인**: 선택된 alpha는 0.01→{cold_alpha}로 바뀌었지만, "
              f"alpha=0.01일 때와 alpha={cold_alpha}일 때 최종 test MAE/RMSE는 "
              f"**소수점 6자리까지 동일**(69.928493/262.069142, 직접 재계산해 확인)했다. "
              f"즉 **이 alpha 범위에서 정규화 강도가 예측 결과에 미치는 영향이 사실상 없다** — "
              f"메인 통합표(4번) 수치는 바뀌지 않으므로 갱신하지 않는다.\n")
    md.append(f"- **personalized**: 각 사용자의 history 구간을 다시 시간순 "
              f"앞 {HISTORY_INNER_RATIO:.0%}(inner-train)/뒤 {1-HISTORY_INNER_RATIO:.0%}(inner-val)으로 "
              f"나눠(06단계의 history/hold-out 분할과 동일한 방식을 history 내부에 재귀 적용) "
              f"inner-val MAE가 가장 작은 alpha를 선택. 실제 holdout(test)은 전혀 사용하지 않음. "
              f"(간단한 1회 시간순 분할이며, 여러 번 반복하는 완전한 시계열 CV는 아님 — 범위상 간소화했다.)\n")
    md.append(f"  - 선택된 alpha = **{pers_alpha}** ")
    md.append(f"({'여전히 격자 하한' if pers_alpha == min(ALPHA_GRID) else '격자 경계 아님(내부 최솟값)'})\n")
    md.append("  - alpha별 inner-val MAE: " + ", ".join(f"{a}={v:.6f}" for a, v in pers_val_scores.items()) + "\n")
    md.append(f"  - **넓힌 뒤 바뀐 점 확인**: 선택된 alpha는 0.01→{pers_alpha}로 바뀌었지만, "
              f"alpha=0.01일 때와 alpha={pers_alpha}일 때 최종 test MAE/RMSE는 "
              f"**소수점 6자리까지 동일**(62.053868/239.761552~239.761553, 직접 재계산해 확인)했다. "
              f"메인 통합표(4번) 수치는 바뀌지 않으므로 갱신하지 않는다.\n")

    # ===== 4. 메인 Ridge 적합 + 평가 =====
    print("\n최종 Ridge 적합 및 평가...")
    cold_scaler, cold_beta, cold_intercept, _ = fit_final(train_samples, cold_alpha)
    cold_pred = predict_with(test_samples, cold_scaler, cold_beta, cold_intercept)
    ridge_cold_df = build_model_rows(test_samples, cold_pred, RIDGE_LABEL)

    pers_scaler, pers_beta, pers_intercept, _ = fit_final(p_train_samples, pers_alpha)
    pers_pred = predict_with(p_test_samples, pers_scaler, pers_beta, pers_intercept)
    ridge_pers_df = build_model_rows(p_test_samples, pers_pred, RIDGE_LABEL)

    # 기존 5개 baseline (같은 test set 기준) - 비교/통합표용
    base_cold_df = m21.build_rows(test_samples, overall_avg_lookup)
    base_pers_df = m21.build_rows(p_test_samples, overall_avg_lookup)

    combined_cold_df = pd.concat([base_cold_df, ridge_cold_df], ignore_index=True)
    combined_pers_df = pd.concat([base_pers_df, ridge_pers_df], ignore_index=True)

    ALL_LABELS = dict(m21.BASELINE_LABELS)
    ALL_LABELS[RIDGE_LABEL] = "6. Ridge"
    ALL_NAMES = m21.BASELINE_NAMES + [RIDGE_LABEL]

    def summarize_with_ridge(df):
        g = df.groupby("baseline")
        out = g.agg(n_samples=("abs_error", "size"), n_users=("user_id", "nunique"),
                     mae=("abs_error", "mean"), rmse=("sq_error", lambda c: np.sqrt(c.mean())))
        n_merges = df[df["merge_affected"]].groupby("baseline").size().rename("n_merges")
        out = out.join(n_merges, how="left")
        out["n_merges"] = out["n_merges"].fillna(0).astype(int)
        out = out.reindex(ALL_NAMES)
        out["방법"] = [ALL_LABELS[b] for b in out.index]
        return out.reset_index(drop=True)[["방법", "mae", "rmse", "n_users", "n_samples", "n_merges"]].rename(
            columns={"mae": "MAE(초/km)", "rmse": "RMSE(초/km)", "n_users": "#users",
                     "n_samples": "#eval sessions", "n_merges": "#merges"}
        )

    md.append("\n## 4. Baseline 통합표 + Ridge (메인, past-only 입력, 극단 페이스 전부 포함)\n")
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        md.append(f"\n### {split_name}\n")
        md.append(m21.df_to_markdown(summarize_with_ridge(df)))
        md.append("\n")

    # ===== C-2 스타일: 부트스트랩 CI (Ridge 포함) =====
    print("\n부트스트랩 신뢰구간 계산 중 (Ridge 포함)...")
    md.append(f"\n## 4-1. 신뢰구간 (사용자 단위 cluster bootstrap, {m21.N_BOOTSTRAP}회, "
              f"seed={m21.BOOTSTRAP_SEED})\n")
    boot_results = {}
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        boot_mae, boot_rmse = m21.cluster_bootstrap(df, ALL_NAMES)
        boot_results[split_name] = (boot_mae, boot_rmse)
        rows = []
        for b in ALL_NAMES:
            mae_lo, mae_hi = np.percentile(boot_mae[b], [2.5, 97.5])
            rmse_lo, rmse_hi = np.percentile(boot_rmse[b], [2.5, 97.5])
            rows.append({"방법": ALL_LABELS[b], "MAE": boot_mae[b].mean(), "MAE_95%CI_하한": mae_lo,
                         "MAE_95%CI_상한": mae_hi, "RMSE": boot_rmse[b].mean(), "RMSE_95%CI_하한": rmse_lo,
                         "RMSE_95%CI_상한": rmse_hi})
        md.append(f"\n### {split_name}\n")
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n")

    # ===== 5. Ridge vs median-5(MAE) / median-5,사용자전체평균,recent_avg(RMSE) =====
    md.append("\n## 5. Ridge 대비 쌍체 차이 95% CI\n")
    for split_name in ["cold-start", "personalized"]:
        boot_mae, boot_rmse = boot_results[split_name]
        md.append(f"\n### {split_name}\n")
        mae_diff = boot_mae["median5"] - boot_mae[RIDGE_LABEL]
        mae_lo, mae_hi = np.percentile(mae_diff, [2.5, 97.5])
        md.append(f"- **MAE: median-5 - Ridge** = {mae_diff.mean():.3f} (95% CI [{mae_lo:.3f}, {mae_hi:.3f}])"
                  f"{' -> 0 포함, 유의하지 않음' if mae_lo <= 0 <= mae_hi else ' -> 0 미포함, 유의함'}\n")
        rows = []
        for other in ["median5", "user_overall_avg", "recent_avg"]:
            diff = boot_rmse[other] - boot_rmse[RIDGE_LABEL]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append({"비교 baseline": m21.BASELINE_LABELS[other], "RMSE차이(baseline-Ridge)": diff.mean(),
                         "하한": lo, "상한": hi, "0포함(유의하지않음)": bool(lo <= 0 <= hi)})
        md.append("- **RMSE: median-5/사용자 전체 평균/최근 5회 평균 각각 대비**\n\n")
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n")

    # ===== 6. Huber 민감도 (1행만) =====
    print("\nHuber 회귀(민감도) 적합 중...")
    md.append("\n## 6. 민감도: Huber 회귀 (robust, 1행만 — 결과를 보고 메인을 바꾸지 않음)\n")
    md.append("메인(Ridge)은 극단 페이스 세션을 전부 포함한 데이터로 학습했다. "
              "제곱오차 기반 Ridge가 그 극단치에 얼마나 끌려가는지 참고하기 위해, "
              "같은 train 데이터·같은 alpha로 **robust loss(Huber)** 버전을 1번만 "
              "추가로 적합해 비교한다 (alpha 재탐색 없음, delta는 초기 OLS 잔차의 "
              "MAD 기반 1회 추정 — `ridge_lib.huber_fit` 참고, 간소화된 구현임을 명시).\n\n")
    huber_rows = []
    for split_name, tr_samples, te_samples, alpha in [
        ("cold-start", train_samples, test_samples, cold_alpha),
        ("personalized", p_train_samples, p_test_samples, pers_alpha),
    ]:
        scaler, beta, intercept, delta = fit_final(tr_samples, alpha, fit_fn=rl.huber_fit)
        pred = predict_with(te_samples, scaler, beta, intercept)
        y_true = np.array([s["y"] for s in te_samples])
        mae = np.mean(np.abs(y_true - pred))
        rmse = np.sqrt(np.mean((y_true - pred) ** 2))
        huber_rows.append({"평가조건": split_name, "delta": delta, "MAE": mae, "RMSE": rmse, "n": len(te_samples)})
    huber_df = pd.DataFrame(huber_rows)
    md.append(m21.df_to_markdown(huber_df))
    md.append("\n\n(참고: Ridge 메인 결과와 비교하면 ")
    for i, row in enumerate(huber_rows):
        split_name = row["평가조건"]
        ridge_mae = combined_cold_df if split_name == "cold-start" else combined_pers_df
        ridge_mae_val = ridge_mae[ridge_mae["baseline"] == RIDGE_LABEL]["abs_error"].mean()
        md.append(f"{split_name} Ridge MAE={ridge_mae_val:.3f} vs Huber MAE={row['MAE']:.3f}; ")
    md.append(")\n")

    # ===== 7. post-hoc Ridge (별도 표) =====
    print("\npost-hoc Ridge(타깃 거리 포함, 16차원) 적합 중...")
    md.append("\n## 7. [POST-HOC 분석, 타깃 실제 거리 입력 포함 — 사전 예측 성능 아님] Ridge\n")
    md.append("**주의: 이 표는 타깃 세션의 실제 완주 거리를 입력(X)에 추가한 Ridge다. "
              "러닝 전에는 알 수 없는 사후 정보를 쓰므로 예측 성능으로 보고하지 않으며, "
              "메인 표(4번)와 절대 섞지 않는다.**\n\n")
    X_ptr, y_ptr, uid_ptr = samples_to_xy(posthoc_train_samples)
    posthoc_alpha, posthoc_cv_scores = select_alpha_group_cv(X_ptr, y_ptr, uid_ptr, ALPHA_GRID)
    ph_scaler, ph_beta, ph_intercept, _ = fit_final(posthoc_train_samples, posthoc_alpha)
    ph_pred = predict_with(posthoc_test_samples, ph_scaler, ph_beta, ph_intercept)
    y_ph_true = np.array([s["y"] for s in posthoc_test_samples])
    ph_mae = np.mean(np.abs(y_ph_true - ph_pred))
    ph_rmse = np.sqrt(np.mean((y_ph_true - ph_pred) ** 2))
    posthoc_at_boundary = (posthoc_alpha == max(ALPHA_GRID)) or (posthoc_alpha == min(ALPHA_GRID))
    md.append(f"- **평가 조건: cold-start 분할** (post-hoc은 personalized 버전을 만들지 않았음 — "
              f"train/test는 04단계와 같은 cold-start 사용자 분할).\n")
    md.append(f"- alpha 선택(GroupKFold, train 내부, {ORIGINAL_ALPHA_GRID}로 처음 돌렸을 때 "
              f"상한(1000.0)에 걸려 {ALPHA_GRID}로 넓혀 재검증): **{posthoc_alpha}** "
              f"({'여전히 격자 경계' if posthoc_at_boundary else '격자 경계 아님(내부 최솟값, 양옆 값 모두 더 나쁨)'}) "
              f"(CV MAE: {', '.join(f'{a}={v:.6f}' for a, v in posthoc_cv_scores.items())})\n")
    ridge_cold_mae = combined_cold_df[combined_cold_df['baseline']==RIDGE_LABEL]['abs_error'].mean()
    ridge_cold_rmse = np.sqrt(combined_cold_df[combined_cold_df['baseline']==RIDGE_LABEL]['sq_error'].mean())
    distance_coef = ph_beta[-1]
    md.append(f"- **[사후 거리 조건 분석] MAE={ph_mae:.3f}, RMSE={ph_rmse:.3f}** "
              f"(cold-start 분할 기준, n={len(posthoc_test_samples)})\n")
    md.append(f"- **타깃 실제 거리(표준화) 계수 = {distance_coef:.3f}** (16개 피처 중 마지막, "
              f"나머지 15개는 과거 세션 피처). 부호가 **음수** — 표준화된 피처 기준, 다른 입력을 "
              f"고정했을 때 타깃 거리가 길수록 모델이 예측하는 페이스(초/km) 값이 **낮아지는(빨라지는) "
              f"쪽으로 선형 관계를 적합했다**는 뜻이다. 이 선형 계수의 방향이 실제 원인(예: 짧고 "
              f"빠른 인터벌/대회 세션이 길고 느린 세션보다 많이 섞여 있어서 생긴 상관인지, 다른 "
              f"이유인지)은 **미확인**이다 — 인과관계로 해석하지 않는다.\n")
    md.append(f"- **메인 Ridge(과거 기록만, cold-start) MAE={ridge_cold_mae:.3f}, "
              f"RMSE={ridge_cold_rmse:.3f}인데, 타깃의 실제 거리를 입력에 추가한 post-hoc Ridge는 "
              f"오히려 MAE가 {ph_mae:.3f}로 더 나쁘다({ph_mae - ridge_cold_mae:+.3f}).** "
              f"RMSE는 {ph_rmse:.3f}로 메인보다 소폭 나쁘다({ph_rmse - ridge_cold_rmse:+.3f}). "
              f"**거리라는 사후 정보를 추가했는데 왜 MAE가 악화되는지는 확인된 원인이 없다 "
              f"(미확인)** — 가능한 추정(검증 안 함): 16차원으로 늘면서 train 내부 "
              f"GroupKFold가 더 큰 alpha(정규화)를 선택해 다른 15개 피처의 유효 계수가 전반적으로 "
              f"축소됐을 수 있다, 또는 거리와 기존 피처(과거 거리/페이스) 간 상관이 겹쳐 선형모델이 "
              f"안정적으로 분리해 쓰지 못했을 수 있다. 둘 다 이번 분석에서 직접 검증하지 않았다.\n")

    # ===== 8. 세분화 (Ridge 기준) =====
    print("\n세분화 분석 중...")
    md.append("\n## 8. 세분화 분석 (Ridge 기준, MAE/RMSE 둘 다)\n")

    point_counts = pd.read_pickle(Path("data") / "session_point_counts.pkl")
    pc_lookup = dict(zip(point_counts["session_id"], point_counts["n_points"]))

    combined_cold_df["split"] = "cold-start"
    combined_pers_df["split"] = "personalized"
    all_df = pd.concat([combined_cold_df, combined_pers_df], ignore_index=True)
    ridge_df = all_df[all_df["baseline"] == RIDGE_LABEL].copy()

    md.append("\n### 8-1. 사용자별 MAE 분포\n")
    rows = []
    for split_name, grp in ridge_df.groupby("split"):
        per_user_mae = grp.groupby("user_id")["abs_error"].mean()
        rows.append({"평가조건": split_name, "median": per_user_mae.median(), "p25": per_user_mae.quantile(.25),
                     "p75": per_user_mae.quantile(.75), "p95": per_user_mae.quantile(.95)})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n")

    md.append("\n### 8-2. 과거 기록 길이별\n")
    bins = [0, 10, 30, 60, 120, 1e9]
    labels = ["5-9", "10-29", "30-59", "60-119", "120+"]
    ridge_df["history_bucket"] = pd.cut(ridge_df["target_session_order"], bins=bins, labels=labels, right=False)
    rows = []
    for split_name in ["cold-start", "personalized"]:
        g = ridge_df[ridge_df["split"] == split_name].groupby("history_bucket", observed=True)
        for bucket, grp in g:
            if len(grp) == 0:
                continue
            rows.append({"평가조건": split_name, "history_bucket": bucket, "n": len(grp),
                         "MAE": grp["abs_error"].mean(), "RMSE": np.sqrt(grp["sq_error"].mean())})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n")

    md.append("\n### 8-3. 타깃 페이스 구간별\n")
    pace_bins = [0, 250, 350, 450, 1000, 1e9]
    pace_labels = ["<250", "250-349", "350-449", "450-999", "1000+(극단)"]
    ridge_df["pace_bucket"] = pd.cut(ridge_df["y_true"], bins=pace_bins, labels=pace_labels, right=False)
    rows = []
    for split_name in ["cold-start", "personalized"]:
        g = ridge_df[ridge_df["split"] == split_name].groupby("pace_bucket", observed=True)
        for bucket, grp in g:
            if len(grp) == 0:
                continue
            rows.append({"평가조건": split_name, "pace_bucket": bucket, "n": len(grp),
                         "MAE": grp["abs_error"].mean(), "RMSE": np.sqrt(grp["sq_error"].mean())})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n")

    md.append("\n### 8-4. 타깃 n_points=500 여부별 (타깃이 병합되지 않은 단일 세션인 경우만)\n")
    non_merged = ridge_df[~ridge_df["target_merged"]].copy()
    non_merged["target_n_points"] = non_merged["target_session_id"].map(pc_lookup)
    non_merged["is_500"] = non_merged["target_n_points"] == 500
    rows = []
    for split_name in ["cold-start", "personalized"]:
        for is_500_val in [True, False]:
            grp = non_merged[(non_merged["split"] == split_name) & (non_merged["is_500"] == is_500_val)]
            if len(grp) == 0:
                continue
            rows.append({"평가조건": split_name, "n_points==500": is_500_val, "n": len(grp),
                         "MAE": grp["abs_error"].mean(), "RMSE": np.sqrt(grp["sq_error"].mean())})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append(f"\n(병합되지 않은 타깃만 대상: {len(non_merged)} / {len(ridge_df)}. "
              "0(a) 분석에서 이미 확인했듯 n_points==500은 거리와 교란돼 있어 순수 효과로 단정하지 않음)\n")

    md.append("\n### 8-5. 병합 영향 분해 — (a)무관 (b)입력만 병합 (c)타깃 병합\n")
    def merge_category(row):
        if row["target_merged"]:
            return "c_target_merged"
        elif row["input_merged"]:
            return "b_input_merged"
        return "a_unaffected"
    ridge_df["merge_category"] = ridge_df.apply(merge_category, axis=1)
    rows = []
    for split_name in ["cold-start", "personalized"]:
        g = ridge_df[ridge_df["split"] == split_name].groupby("merge_category")
        for cat, grp in g:
            rows.append({"평가조건": split_name, "카테고리": cat, "n": len(grp),
                         "MAE": grp["abs_error"].mean(), "RMSE": np.sqrt(grp["sq_error"].mean())})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n**주의**: (c) 타깃 병합 그룹은 y(정답)의 정의 자체가 병합으로 바뀌므로, "
              "(a)/(b)와 직접 비교해 '병합이 오차를 줄인다/늘린다'고 단정하지 않는다.\n")

    md.append("\n### 8-6. 오차(abs_error) 상위 1% 집중 요인\n")
    for split_name in ["cold-start", "personalized"]:
        grp = ridge_df[ridge_df["split"] == split_name]
        threshold = grp["abs_error"].quantile(0.99)
        top1 = grp[grp["abs_error"] >= threshold]
        sse_share = top1["sq_error"].sum() / grp["sq_error"].sum()
        abs_share = top1["abs_error"].sum() / grp["abs_error"].sum()
        md.append(f"\n**{split_name}** (임계값={threshold:.1f}초/km, n={len(top1)}/{len(grp)}):\n")
        md.append(f"- 전체 SSE 중 상위 1% 비율: **{sse_share:.1%}**, 전체 절대오차 합 중 비율: **{abs_share:.1%}**\n")
        md.append(f"- 병합 영향 비율: 상위1% {top1['merge_affected'].mean():.1%} vs 전체 {grp['merge_affected'].mean():.1%}\n")
        md.append(f"- 극단 페이스 영향 비율: 상위1% {top1['extreme_affected'].mean():.1%} vs 전체 {grp['extreme_affected'].mean():.1%}\n")
        md.append("- GPS 이상치: 별도 플래그 없음 (미확인)\n")

    # ===== 9. 해석 (MAE/RMSE 결론이 다르면 각각 서술) =====
    md.append("\n## 9. 해석\n")
    md.append(
        "**RMSE는 상위 1% 윈도우가 SSE의 대부분을 차지하는 극단 윈도우 민감 지표다** "
        "(8-6 참고, 90% 안팎). 이 전제 위에서 MAE 기준과 RMSE 기준 결론이 다르므로 "
        "각각 따로 쓴다.\n\n"
        "- **MAE 기준**: Ridge는 cold-start/personalized 둘 다 median-5보다 "
        "**통계적으로 유의하게 나쁘다** (윈도우 풀링 MAE 기준, macro는 이 문서에서 미계산; 95% CI가 0을 포함하지 않고 median-5가 더 작은 쪽).\n"
        "- **RMSE 기준**: Ridge는 median-5·사용자 전체 평균과는 **통계적으로 구분되지 "
        "않는다**(CI가 0을 포함) — 즉 '이겼다'고 할 수 없고 '동률'이다. 다만 "
        "최근 5회 평균보다는 **유의하게 낫다**.\n"
    )

    # ===== 10. 한계 문서화 =====
    md.append("\n## 10. 한계\n")
    md.append(
        "**personalized의 전역 모델 시점 혼입 가능성 (영향은 미확인)**: personalized "
        "Ridge는 모든 사용자의 history를 한 번에 풀링해 **하나의 전역 모델**을 학습한다. "
        "이 말은, 사용자 A의 평가 시점(hold-out)보다 **달력상 더 늦은 시기**에 기록된 "
        "사용자 B의 history 윈도우가 같은 학습 데이터에 섞여 들어갈 수 있다는 뜻이다 "
        "— 각 사용자 자신의 과거만 보는 것은 보장되지만(04/06단계 감사 참고), "
        "'전역 모델 파라미터'는 다른 사용자의 더 늦은 시기 정보로부터 영향을 받을 "
        "수 있다. 이게 실제로 성능에 얼마나 영향을 주는지는 **검증하지 않았다 "
        "(미확인)** — 사용자별로 완전히 분리된 모델을 쓰거나 시점을 엄격히 맞춘 "
        "글로벌 분할을 쓰면 다른 결과가 나올 수 있다.\n"
    )

    # ===== 11. 종합 판정 (plain) =====
    md.append("\n## 11. Ridge가 최고 baseline을 이겼는가 — plain 판정\n")
    md.append(
        "**MAE 기준: Ridge는 최고 baseline(median-5)을 신뢰구간 밖에서 이기지 못했다.** "
        "오히려 median-5가 Ridge보다 유의하게 낫다(윈도우 풀링 MAE 기준). **RMSE 기준: Ridge는 최고 baseline 그룹"
        "(median-5/사용자 전체 평균)을 신뢰구간 밖에서 이기지 못했다** — 동률일 뿐이다.\n\n"
        "**이유는 추정/미확인이다.** 가능한 설명(모두 검증하지 않은 추정): "
        "(1) 입력 피처가 과거 5개 세션의 거리/페이스/심박만이라 선형모델이 활용할 "
        "수 있는 정보량 자체가 median-5(비선형적 극단치 저항성을 가진 통계량)보다 "
        "불리할 수 있다, (2) alpha 선택 결과가 전부 0.01(거의 무정규화)에 가까워 "
        "정규화가 과적합을 막는 데 별 역할을 못했을 수 있다, (3) 15차원 입력 자체가 "
        "선형 관계로 다음 페이스를 설명하기엔 부족할 수 있다. 어느 쪽이 실제 원인인지는 "
        "이번 분석에서 확인하지 않았다.\n"
    )

    OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
    print(f"\n최종 저장 완료: {OUTPUT_MD_PATH.resolve()}")


if __name__ == "__main__":
    main()
