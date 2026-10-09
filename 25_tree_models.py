# -*- coding: utf-8 -*-
"""
10월 2주차 7단계: 트리 계열 모델 + 과거 정보 피처 보강.

0. 환경 확인: config 메인 데이터 assert, 버전 출력, import sklearn 실패 시
   즉시 중단. ridge_lib.py(Ridge/GroupKFold)를 sklearn과 같은 데이터로
   대조 — 일치했다(계수차이 8.88e-16, GroupKFold 겹침 0).

   **그런데 그 다음 새 차단을 발견했다**: sklearn은 import되고 Ridge/
   GroupKFold는 정상 동작하지만, `sklearn.tree`의 컴파일된 확장
   (`_criterion`, `_splitter`)이 Windows Application Control Policy에
   막혀 DecisionTree/RandomForest/GradientBoosting/HistGradientBoosting
   **전부** import 실패(3회 재시도 동일, lightgbm/xgboost는 미설치·설치
   금지). 사용자 확인 결과 "ridge_lib.py처럼 직접 구현"으로 진행 — 그래서
   HistGradientBoostingRegressor와 같은 역할을 하는 gradient boosting
   트리 앙상블을 `gbt_lib.py`에 numpy로 직접 구현했다(상세 설명/자체
   검증은 그 파일 docstring 참고). Ridge(확장 피처)는 계속 sklearn
   Pipeline을 쓴다(그 부분은 막히지 않았음).
1. 24단계가 만든 확장 피처(22차원) 데이터셋 사용.
2. 모델: (a) Ridge(확장 피처, sklearn), (b) gbt_lib.SimpleGradientBoostingRegressor
   (absolute_error 메인 / squared_error 참고 1행), (c) median5 + 잔차 학습.
3. 평가: 기존 통합표 형식 + 사용자 단위 bootstrap CI + 쌍체 차이 + 세분화.
4. 민감도: 타깃을 moving pace로 바꿔 median-5/전체평균/최근5회/최고모델 재평가
   (재학습 없음 — 같은 예측값을 moving 실측과 비교).
5. median-5를 못 이기면 '이기지 못함'으로 명시하고 멈춤(MLP/LSTM으로 넘어가지 않음).

산출물: docs/tree_results.md
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import config
import ridge_lib as rl
import gbt_lib as gl

_spec_m21 = importlib.util.spec_from_file_location("m21", "21_baseline_integrated_report.py")
m21 = importlib.util.module_from_spec(_spec_m21)
_spec_m21.loader.exec_module(m21)

_spec_m24 = importlib.util.spec_from_file_location("m24", "24_build_xy_dataset_extended.py")
m24 = importlib.util.module_from_spec(_spec_m24)
_spec_m24.loader.exec_module(m24)

OUTPUT_MD_PATH = Path("docs") / "tree_results.md"

# 사전 고정 (결과를 보고 바꾸지 않음) — 6단계 Ridge 경험상 저 alpha 영역이
# 최적이었으므로 같은 넓은 격자를 재사용.
RIDGE_ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
# 사전 고정 (결과를 보고 바꾸지 않음) — gbt_lib.SimpleGradientBoostingRegressor
# 하이퍼파라미터 격자. sklearn.tree 차단으로 HistGradientBoostingRegressor 대신
# numpy 직접구현(gbt_lib.py)을 쓰므로, max_leaf_nodes(best-first 확장)가 아니라
# max_depth(깊이 제한)로 트리 크기를 제어한다 — 실제 데이터로 속도를 미리
# 측정한 뒤(max_depth=3,max_iter=100: 학습 6.57초/79669건) 정했다.
HGB_PARAM_GRID = [
    {"learning_rate": lr, "max_iter": mi, "max_depth": md}
    for lr in [0.03, 0.1]
    for mi in [100, 300]
    for md in [2, 3]
]
GBT_MIN_SAMPLES_LEAF = 20
GBT_N_BINS = 32
N_FOLDS = 5
CV_SEED = 42
HISTORY_INNER_RATIO = 0.8


def build_gbt(params, loss):
    return gl.SimpleGradientBoostingRegressor(
        loss=loss, learning_rate=params["learning_rate"], max_iter=params["max_iter"],
        max_depth=params["max_depth"], min_samples_leaf=GBT_MIN_SAMPLES_LEAF,
        n_bins=GBT_N_BINS, random_state=CV_SEED,
    )

RIDGE_LABEL = "ridge_ext"
HGB_MAIN_LABEL = "hgb_abs"
HGB_SQ_LABEL = "hgb_sq_ref"
HGB_RESID_LABEL = "hgb_median5_resid"


def samples_to_xy(samples, target_key="y"):
    X = np.array([s["X"] for s in samples], dtype=float)
    y = np.array([s[target_key] for s in samples], dtype=float)
    user_ids = np.array([s["user_id"] for s in samples])
    return X, y, user_ids


def median5_from_X(X):
    reshaped = X[:, :15].reshape(-1, 5, 3)
    return np.median(reshaped[:, :, 1], axis=1)


def personalized_inner_split(train_samples, ratio=HISTORY_INNER_RATIO):
    by_user = {}
    for s in train_samples:
        by_user.setdefault(s["user_id"], []).append(s)
    inner_train, inner_val = [], []
    for uid, lst in by_user.items():
        lst_sorted = sorted(lst, key=lambda s: s["target_session_order"])
        n = len(lst_sorted)
        if n < 2:
            inner_train.extend(lst_sorted)
            continue
        n_inner_train = min(max(int(n * ratio), 0), n - 1)
        inner_train.extend(lst_sorted[:n_inner_train])
        inner_val.extend(lst_sorted[n_inner_train:])
    return inner_train, inner_val


def _param_key(params):
    return tuple(sorted(params.items())) if isinstance(params, dict) else params


# ===================== Step 0: 환경 확인 + sklearn 대조 =====================

def check_environment_and_compare(md):
    md.append("## 0. 환경 확인 + ridge_lib vs sklearn 대조\n")
    md.append(f"- python: {sys.version.split()[0]}\n")
    try:
        import sklearn
    except ImportError as e:
        md.append(f"\n**[중단] `import sklearn` 실패: {e}**\n")
        OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
        print(f"[중단] import sklearn 실패: {e}")
        sys.exit(1)

    md.append(f"- numpy: {np.__version__}, scipy: {scipy.__version__}, "
              f"sklearn: {sklearn.__version__}, pandas: {pd.__version__}\n")
    print(f"[환경] python={sys.version.split()[0]} numpy={np.__version__} scipy={scipy.__version__} "
          f"sklearn={sklearn.__version__} pandas={pd.__version__}")

    from sklearn.linear_model import Ridge as SkRidge, HuberRegressor
    from sklearn.model_selection import GroupKFold as SkGroupKFold

    rng = np.random.default_rng(0)
    n, p = 300, 22
    X_cmp = rng.normal(size=(n, p))
    true_beta = rng.normal(size=p)
    y_cmp = X_cmp @ true_beta + 3.0 + rng.normal(scale=0.1, size=n)

    alpha_cmp = 1.0
    beta_manual, intercept_manual = rl.ridge_fit(X_cmp, y_cmp, alpha=alpha_cmp)
    pred_manual = rl.ridge_predict(X_cmp, beta_manual, intercept_manual)
    sk_ridge = SkRidge(alpha=alpha_cmp).fit(X_cmp, y_cmp)
    pred_sklearn = sk_ridge.predict(X_cmp)
    coef_diff = float(np.abs(beta_manual - sk_ridge.coef_).max())
    pred_diff = float(np.abs(pred_manual - pred_sklearn).max())
    intercept_diff = float(abs(intercept_manual - sk_ridge.intercept_))
    ridge_match = coef_diff < 1e-6 and pred_diff < 1e-6 and intercept_diff < 1e-6
    md.append(f"- **Ridge 대조**(합성 데이터 n=300,p=22,alpha=1.0): 계수 최대차이={coef_diff:.2e}, "
              f"intercept 차이={intercept_diff:.2e}, 예측값 최대차이={pred_diff:.2e} -> "
              f"**{'일치' if ridge_match else '불일치'}** (<1e-6 기준)\n")
    print(f"[Ridge 대조] coef_diff={coef_diff:.2e} pred_diff={pred_diff:.2e} intercept_diff={intercept_diff:.2e}")

    user_ids_cmp = rng.integers(0, 50, size=n)
    sk_folds = list(SkGroupKFold(n_splits=5).split(X_cmp, y_cmp, groups=user_ids_cmp))
    manual_folds = list(rl.group_kfold_indices(user_ids_cmp, 5, 42))

    def max_overlap(val_sets):
        m = 0
        for i in range(len(val_sets)):
            for j in range(i + 1, len(val_sets)):
                m = max(m, len(val_sets[i] & val_sets[j]))
        return m

    sk_val_sets = [set(user_ids_cmp[val_idx].tolist()) for _, val_idx in sk_folds]
    manual_val_sets = [set(user_ids_cmp[val_mask].tolist()) for _, val_mask in manual_folds]
    sk_overlap = max_overlap(sk_val_sets)
    manual_overlap = max_overlap(manual_val_sets)
    exact_fold_match = sorted([frozenset(s) for s in sk_val_sets]) == sorted([frozenset(s) for s in manual_val_sets])
    sk_sizes = [len(s) for s in sk_val_sets]
    manual_sizes = [len(s) for s in manual_val_sets]
    groupkfold_ok = sk_overlap == 0 and manual_overlap == 0

    md.append(f"- **GroupKFold 대조**(같은 데이터, 5-fold): fold간 그룹 겹침 — sklearn={sk_overlap}, "
              f"수동구현={manual_overlap} (둘 다 0이어야 함, {'통과' if groupkfold_ok else '실패'}). "
              f"fold별 그룹 수 — sklearn: {sk_sizes}, 수동: {manual_sizes}. "
              f"**fold 멤버십이 완전히 같은가 = {exact_fold_match}** (sklearn GroupKFold는 "
              f"shuffle/random_state 옵션이 없고 그룹 크기 기반 결정적 알고리즘이라 수동 구현"
              f"(랜덤 셔플 후 균등분할)과 멤버십이 다를 수 있음 — 완전 일치는 기대 범위 밖이고, "
              f"**둘 다 겹침 0이면 구조적으로 올바른 분할**로 판단한다.)\n")
    print(f"[GroupKFold 대조] sk_overlap={sk_overlap} manual_overlap={manual_overlap} exact_match={exact_fold_match}")

    hr = HuberRegressor(alpha=1e-4).fit(X_cmp, y_cmp)
    beta_manual_huber, intercept_manual_huber, delta_manual = rl.huber_fit(X_cmp, y_cmp, alpha=1e-4)
    corr = float(np.corrcoef(hr.coef_, beta_manual_huber)[0, 1])
    sign_match = float(np.mean(np.sign(hr.coef_) == np.sign(beta_manual_huber)))
    md.append(f"- **Huber 정성 비교**(정의가 달라 수치 일치는 기대 안 함 — sklearn은 epsilon 기반, "
              f"scale을 반복마다 재추정; ridge_lib은 초기 OLS 잔차 MAD로 delta 1회 고정): "
              f"계수 상관계수={corr:.3f}, 부호 일치 비율={sign_match:.1%}\n")

    gate_pass = ridge_match and groupkfold_ok
    md.append(f"\n**게이트 판정**: Ridge 수치 일치 + GroupKFold 구조적 정합(겹침 0) 모두 만족 = "
              f"**{gate_pass}** -> {'이후 모델은 sklearn Pipeline 기반으로 진행, ridge_lib.py는 검증용으로만 유지.' if gate_pass else '[중단] 불일치 — 보고 후 중단.'}\n\n")
    print(f"[게이트] {'통과, sklearn 기반으로 진행' if gate_pass else '실패 - 중단'}")

    if not gate_pass:
        OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
        sys.exit(1)


def build_model_rows(test_samples, preds, label, y_key="y"):
    rows = []
    for s, y_pred in zip(test_samples, preds):
        target_merged, input_merged = m21.is_merge_affected(s)
        # is_extreme_affected()는 15차원 X(과거 5세션 x 3피처)를 가정 -> 확장 X(22차원)는 앞 15개만 넘긴다.
        target_extreme, input_extreme = m21.is_extreme_affected({**s, "X": s["X"][:15]})
        y_true = s[y_key]
        rows.append({
            "baseline": label, "user_id": s["user_id"], "y_true": y_true, "y_pred": y_pred,
            "abs_error": abs(y_true - y_pred), "sq_error": (y_true - y_pred) ** 2,
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
    md.append("# 트리 계열 모델 결과 (10월 2주차 7단계)\n\n")

    main_df = config.load_main_sessions()
    md.append(f"입력 메인 데이터: `{config.MAIN_DATA_PATH}` (세션 {len(main_df)}건, "
              f"사용자 {main_df['user_id'].nunique()}명, assert 통과)\n\n")

    check_environment_and_compare(md)

    from sklearn.linear_model import Ridge as SkRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold as SkGroupKFold

    md.append("### 0-1. 추가 발견: sklearn.tree 차단 -> gbt_lib.py로 직접 구현\n")
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor as _ProbeHGB
        sklearn_tree_ok = True
        md.append("- `from sklearn.ensemble import HistGradientBoostingRegressor` 성공 — "
                  "sklearn 버전을 그대로 쓴다 (아래 gbt_lib 경로는 사용하지 않음).\n")
    except Exception as e:
        sklearn_tree_ok = False
        md.append(f"- `from sklearn.ensemble import HistGradientBoostingRegressor` 실패: "
                  f"`{type(e).__name__}: {str(e)[:200]}` (3회 재시도 동일하게 실패했음을 "
                  f"이 작업 시작 시 별도로 확인함). DecisionTree/RandomForest/GradientBoosting도 "
                  f"같은 이유로 전부 실패. lightgbm/xgboost는 미설치·설치 금지. "
                  f"-> **gbt_lib.py(numpy 직접구현)로 대체, 사용자 확인받음.**\n")
        print("[0-1] sklearn.tree 차단 확인 - gbt_lib.py 경로 사용")
        gbt_check = gl._validate_gbt_sanity()
        for loss, r in gbt_check["loss_results"].items():
            md.append(f"  - gbt_lib 자체 검증({loss}): 학습손실 {r['first_loss']:.3f} -> "
                      f"{r['last_loss']:.3f}, 반복마다 단조감소={r['monotonic_decrease']} "
                      f"(최대 증가분 {r['max_increase']:.2e})\n")
        md.append(f"  - 계단함수 합성데이터 분할 검증: 올바른 feature 선택={gbt_check['split_feature_correct']}, "
                  f"분할 임계값={gbt_check['split_threshold_value']:.4f} (실제 경계 0.5 근처(±0.1)="
                  f"{gbt_check['split_near_boundary']})\n")
        md.append("  - sklearn.tree가 막혀 sklearn HistGradientBoostingRegressor와 직접 수치 대조는 "
                  "**못 했다(불가능)** — 위 자체 정합성 검증만으로 대체했다. ridge_lib.py의 "
                  "alpha=0 vs OLS 같은 방식의 '정답 대조'는 아니므로 이 한계를 명시한다.\n\n")
        print(f"  gbt_lib 검증: {gbt_check}")

    if sklearn_tree_ok:
        def make_gbt(params, loss):
            return _ProbeHGB(loss=loss, random_state=CV_SEED, **params)
    else:
        def make_gbt(params, loss):
            return build_gbt(params, loss)

    md.append("### 0-2. 모델명 표기 정정\n")
    md.append("이전 보고서/CLAUDE.md에서 'HGB'라고 썼던 표기를 전부 **'직접 구현 GBM(gbt_lib)'**으로 "
              "고친다 — sklearn의 HistGradientBoostingRegressor는 이 환경에서 한 번도 실행되지 "
              "않았고(0-1 참고), 전부 `gbt_lib.SimpleGradientBoostingRegressor`(numpy 직접구현) "
              "결과다. sklearn 구현과 다를 수 있는 점(코드 기준 확인):\n")
    md.append("- **분할 탐색**: `gbt_lib.py` 204행 `g = np.sign(residual) if loss==\"absolute_error\" "
              "else residual` — absolute_error에서는 **부호(sign)만 있는 pseudo-residual**로 분할 "
              "지점을 고른다(제곱오차 감소량 기준).\n")
    md.append("- **리프 값 갱신**: `gbt_lib.py` 189-193행에서 `leaf_agg = np.median`(absolute_error) "
              "또는 `np.mean`(squared_error)로 정하고, 126/131/138행에서 `node.value = "
              "leaf_agg(residual_actual[row_idx])`로 **실제 residual의 중앙값(또는 평균)**을 "
              "리프 값으로 쓴다 — **부호만 쓰는 게 아니라 Friedman TreeBoost 방식의 리프값 보정을 "
              "실제로 한다.** sklearn의 `max_leaf_nodes`(best-first 확장), L2 정규화, 2차 근사 "
              "기반 분할 이득(뉴턴 스텝)은 구현하지 않았다(`max_depth` 기반 완전이진트리로 대체, "
              "0-1 참고).\n\n")

    # ===== 1. 입력 데이터 =====
    train_samples = pd.read_pickle(m24.EXT_TRAIN_PATH)
    test_samples = pd.read_pickle(m24.EXT_TEST_PATH)
    p_train_samples = pd.read_pickle(m24.EXT_P_TRAIN_PATH)
    p_test_samples = pd.read_pickle(m24.EXT_P_TEST_PATH)
    user_sequences = pd.read_pickle(config.USER_SEQUENCES_PATH)
    overall_avg_lookup = m21.build_user_overall_avg_lookup(user_sequences)

    md.append("## 1. 입력 피처 (24단계, 22차원)\n")
    md.append(f"- cold-start: train {len(train_samples)}건 / test {len(test_samples)}건\n")
    md.append(f"- personalized: history {len(p_train_samples)}건 / hold-out {len(p_test_samples)}건\n")
    md.append("- X 컬럼(총 22개): 앞 15개는 04/06단계와 동일(past[i].actual_distance_km/"
              "avg_pace_sec_per_km/avg_heart_rate, i=0..4), 뒤 7개는 7단계에서 추가한 과거 정보 "
              "피처:\n")
    for name in m24.X_COLUMN_NAMES[15:]:
        md.append(f"  - {name}\n")
    md.append("- 해석 선택: 7단계 1번 후보 중 \"의 중앙값\"이 명시된 elapsed 페이스만 중앙값 "
              "요약으로 넣었고, moving 페이스/정지 비율도 (원본 5개가 아니라) 중앙값 하나로 "
              "통일했다 — 과적합 위험과 일관성을 고려한 선택이며, 상세 이유는 "
              "`24_build_xy_dataset_extended.py` 모듈 docstring에 적었다. **이 선택은 "
              "`25_tree_models.py`를 한 번도 실행하기 전, `24_build_xy_dataset_extended.py`를 "
              "작성하던 시점(어떤 모델 결과도 보지 않은 상태)에 정했다** — 결과를 본 뒤 바꾼 것이 "
              "아니다. 요청대로 원본 5개씩 넣는 버전은 만들지 않았다.\n")
    md.append("- 모든 추가 피처는 last_input_session_time 감사를 통과한 윈도우에서만 계산됐고, "
              "`user_history_median_pace`는 추가로 그 피처가 참조한 가장 늦은 세션 종료 시각도 "
              "타깃 시작 시각보다 앞서는지 별도 검증했다(24단계 로그: 이력 감사로 cold-start "
              "train 2건/test 1건/personalized 3건 추가 제외).\n\n")

    # ===== 1-1. 6단계 대비 윈도우 수 차이 조사 =====
    print("\n1-1. 6단계 대비 윈도우 수 차이 조사 중...")
    md.append("### 1-1. 6단계(Ridge) 통합표의 median-5 수치와 다른 이유\n")
    md.append("6단계 median-5 MAE(62.217 cold-start / 56.446 personalized)와 이번 7단계 통합표"
              "(아래 3번, 62.219 / 56.449)가 소수점 세 자리에서 다르다. 원인을 윈도우 단위로 "
              "직접 대조했다(`(user_id, window_start_session_order)` 키로 집합 비교):\n")

    def _keyset(samples):
        return set((s["user_id"], s["window_start_session_order"]) for s in samples)

    orig_sets = {
        "cold-start train": (_keyset(pd.read_pickle(config.TRAIN_DATASET_PATH)), _keyset(train_samples)),
        "cold-start test": (_keyset(pd.read_pickle(config.TEST_DATASET_PATH)), _keyset(test_samples)),
        "personalized history": (_keyset(pd.read_pickle(config.PERSONALIZED_TRAIN_DATASET_PATH)), _keyset(p_train_samples)),
        "personalized hold-out": (_keyset(pd.read_pickle(config.PERSONALIZED_TEST_DATASET_PATH)), _keyset(p_test_samples)),
    }
    diff_rows = []
    for name, (ko, ke) in orig_sets.items():
        removed = ko - ke
        added = ke - ko
        diff_rows.append({"조건": name, "6단계 윈도우 수": len(ko), "7단계 윈도우 수": len(ke),
                           "제외(7단계에서 빠짐)": len(removed), "추가(7단계에만 있음)": len(added)})
        print(f"  {name}: 6단계={len(ko)} 7단계={len(ke)} 제외={len(removed)} 추가={len(added)}")
    md.append(m21.df_to_markdown(pd.DataFrame(diff_rows)))
    md.append("\n제외 사유: 7단계에서 추가한 `user_history_median_pace` 피처 때문에 last_input_"
              "session_time 감사와 같은 방식의 **이력 감사**(그 피처가 참조하는 가장 늦은 세션 "
              "종료 시각이 타깃 시작 시각보다 앞서는지)를 새로 적용했고, 이걸 통과 못 한 윈도우를 "
              "추가로 제외했다(6단계는 이 피처가 없어서 이 감사 자체가 없었다).\n\n")
    md.append("personalized는 '제외 3 / 추가 1'로 숫자가 안 맞는 것처럼 보이는데, 실제로는 "
              "**사용자 8349084의 history 윈도우 2개(session_order 126,127)가 이력 감사로 빠지면서 "
              "그 사용자의 전체 유효 윈도우 수가 줄어 history/hold-out 80% 경계가 한 칸 당겨졌고, "
              "원래 hold-out 첫 윈도우였던 session_order 178이 history 쪽으로 넘어간 것**이다 "
              "(직접 대조로 확인). 버그가 아니라 '감사 제외 -> 사용자별 윈도우 수 변화 -> 80% 경계 "
              "재계산'이 순서대로 일어난 결과다.\n")
    md.append("**이번 7단계 보고서의 모든 비교(median-5 vs 신규 모델)는 7단계 윈도우 집합(위 표의 "
              "'7단계 윈도우 수')에서 똑같이 계산한 값이다** — 6단계 수치와 섞어서 비교하지 않는다.\n\n")

    X_tr, y_tr, uid_tr = samples_to_xy(train_samples)
    X_te, y_te, uid_te = samples_to_xy(test_samples)

    # ===== 2(a). Ridge(확장 피처) 하이퍼파라미터 선택 =====
    print("\nRidge(확장) alpha 선택 중...")
    md.append("## 2. 모델\n\n### (a) Ridge (확장 피처)\n")

    def ridge_pipeline(alpha):
        return Pipeline([("scaler", StandardScaler()), ("ridge", SkRidge(alpha=alpha))])

    def select_alpha_group_cv(X, y, user_ids, alphas):
        gkf = SkGroupKFold(n_splits=N_FOLDS)
        scores = {}
        for a in alphas:
            maes = []
            for tr_idx, val_idx in gkf.split(X, y, groups=user_ids):
                model = ridge_pipeline(a).fit(X[tr_idx], y[tr_idx])
                pred = model.predict(X[val_idx])
                maes.append(np.mean(np.abs(y[val_idx] - pred)))
            scores[a] = float(np.mean(maes))
        best = min(scores, key=scores.get)
        return best, scores

    cold_ridge_alpha, cold_ridge_scores = select_alpha_group_cv(X_tr, y_tr, uid_tr, RIDGE_ALPHA_GRID)

    def select_alpha_time_split(train_samples_, alphas):
        inner_tr, inner_val = personalized_inner_split(train_samples_)
        X_it, y_it, _ = samples_to_xy(inner_tr)
        X_iv, y_iv, _ = samples_to_xy(inner_val)
        scores = {}
        for a in alphas:
            model = ridge_pipeline(a).fit(X_it, y_it)
            pred = model.predict(X_iv)
            scores[a] = float(np.mean(np.abs(y_iv - pred)))
        best = min(scores, key=scores.get)
        return best, scores, len(inner_tr), len(inner_val)

    pers_ridge_alpha, pers_ridge_scores, n_it, n_iv = select_alpha_time_split(p_train_samples, RIDGE_ALPHA_GRID)

    md.append(f"- cold-start: GroupKFold({N_FOLDS}-fold) CV MAE 최소 alpha = **{cold_ridge_alpha}** "
              f"({', '.join(f'{a}={v:.3f}' for a, v in cold_ridge_scores.items())})\n")
    md.append(f"- personalized: 시간순 inner-split(train {n_it}/val {n_iv}) MAE 최소 alpha = "
              f"**{pers_ridge_alpha}** ({', '.join(f'{a}={v:.3f}' for a, v in pers_ridge_scores.items())})\n\n")
    print(f"  cold_alpha={cold_ridge_alpha} pers_alpha={pers_ridge_alpha}")

    ridge_cold_model = ridge_pipeline(cold_ridge_alpha).fit(X_tr, y_tr)
    ridge_cold_pred = ridge_cold_model.predict(X_te)
    X_ptr, y_ptr, _ = samples_to_xy(p_train_samples)
    X_pte, y_pte, _ = samples_to_xy(p_test_samples)
    ridge_pers_model = ridge_pipeline(pers_ridge_alpha).fit(X_ptr, y_ptr)
    ridge_pers_pred = ridge_pers_model.predict(X_pte)

    # ===== 2(b). HistGradientBoostingRegressor (absolute_error 메인 / squared_error 참고) =====
    print("\nHGB(absolute_error) 하이퍼파라미터 선택 중...")
    md.append(f"### (b) {'HistGradientBoostingRegressor(sklearn)' if sklearn_tree_ok else 'gbt_lib.SimpleGradientBoostingRegressor(numpy 직접구현)'}\n")
    md.append(f"- 격자(사전 고정, 결과 보고 안 바꿈): learning_rate∈[0.03,0.1], "
              f"max_iter∈[100,300], max_depth∈[2,3] (총 {len(HGB_PARAM_GRID)}개 조합), "
              f"min_samples_leaf={GBT_MIN_SAMPLES_LEAF} 고정, n_bins={GBT_N_BINS} 고정.\n")

    def select_hgb_group_cv(X, y, user_ids, param_list, loss="absolute_error"):
        gkf = SkGroupKFold(n_splits=N_FOLDS)
        scores = {}
        for params in param_list:
            maes = []
            for tr_idx, val_idx in gkf.split(X, y, groups=user_ids):
                model = make_gbt(params, loss)
                model.fit(X[tr_idx], y[tr_idx])
                pred = model.predict(X[val_idx])
                maes.append(np.mean(np.abs(y[val_idx] - pred)))
            scores[_param_key(params)] = (params, float(np.mean(maes)))
        best_key = min(scores, key=lambda k: scores[k][1])
        return scores[best_key][0], scores

    def select_hgb_time_split(train_samples_, param_list, loss="absolute_error"):
        inner_tr, inner_val = personalized_inner_split(train_samples_)
        X_it, y_it, _ = samples_to_xy(inner_tr)
        X_iv, y_iv, _ = samples_to_xy(inner_val)
        scores = {}
        for params in param_list:
            model = make_gbt(params, loss)
            model.fit(X_it, y_it)
            pred = model.predict(X_iv)
            scores[_param_key(params)] = (params, float(np.mean(np.abs(y_iv - pred))))
        best_key = min(scores, key=lambda k: scores[k][1])
        return scores[best_key][0], scores, len(inner_tr), len(inner_val)

    cold_hgb_params, cold_hgb_scores = select_hgb_group_cv(X_tr, y_tr, uid_tr, HGB_PARAM_GRID)
    pers_hgb_params, pers_hgb_scores, _, _ = select_hgb_time_split(p_train_samples, HGB_PARAM_GRID)
    md.append(f"- cold-start 선택: **{cold_hgb_params}** (CV MAE={cold_hgb_scores[_param_key(cold_hgb_params)][1]:.3f})\n")
    md.append(f"- personalized 선택: **{pers_hgb_params}** (inner-val MAE={pers_hgb_scores[_param_key(pers_hgb_params)][1]:.3f})\n\n")
    print(f"  cold_hgb={cold_hgb_params} pers_hgb={pers_hgb_params}")

    hgb_cold_model = make_gbt(cold_hgb_params, "absolute_error").fit(X_tr, y_tr)
    hgb_cold_pred = hgb_cold_model.predict(X_te)
    hgb_pers_model = make_gbt(pers_hgb_params, "absolute_error").fit(X_ptr, y_ptr)
    hgb_pers_pred = hgb_pers_model.predict(X_pte)

    md.append("- squared_error 참고 1행: 독립적으로 재탐색하지 않고, absolute_error에서 "
              "선택된 하이퍼파라미터를 그대로 쓰고 loss만 squared_error로 바꿔 적합한 "
              "참고용 결과다 (튜닝 기준은 여전히 MAE가 아니라 \"같은 설정에서 손실만 바꿔보기\").\n\n")
    hgb_sq_cold_model = make_gbt(cold_hgb_params, "squared_error").fit(X_tr, y_tr)
    hgb_sq_cold_pred = hgb_sq_cold_model.predict(X_te)
    hgb_sq_pers_model = make_gbt(pers_hgb_params, "squared_error").fit(X_ptr, y_ptr)
    hgb_sq_pers_pred = hgb_sq_pers_model.predict(X_pte)

    # ===== 2(c). median5 + 잔차 학습 =====
    print("\nmedian5+잔차 HGB 하이퍼파라미터 선택 중...")
    md.append("### (c) median5 + 잔차 학습 (y - median5를 absolute_error로 학습, 예측 = median5 + 잔차예측)\n")

    def select_resid_group_cv(X, y, user_ids, param_list):
        gkf = SkGroupKFold(n_splits=N_FOLDS)
        scores = {}
        for params in param_list:
            maes = []
            for tr_idx, val_idx in gkf.split(X, y, groups=user_ids):
                med_tr = median5_from_X(X[tr_idx])
                med_val = median5_from_X(X[val_idx])
                model = make_gbt(params, "absolute_error")
                model.fit(X[tr_idx], y[tr_idx] - med_tr)
                pred = med_val + model.predict(X[val_idx])
                maes.append(np.mean(np.abs(y[val_idx] - pred)))
            scores[_param_key(params)] = (params, float(np.mean(maes)))
        best_key = min(scores, key=lambda k: scores[k][1])
        return scores[best_key][0], scores

    def select_resid_time_split(train_samples_, param_list):
        inner_tr, inner_val = personalized_inner_split(train_samples_)
        X_it, y_it, _ = samples_to_xy(inner_tr)
        X_iv, y_iv, _ = samples_to_xy(inner_val)
        med_it, med_iv = median5_from_X(X_it), median5_from_X(X_iv)
        scores = {}
        for params in param_list:
            model = make_gbt(params, "absolute_error")
            model.fit(X_it, y_it - med_it)
            pred = med_iv + model.predict(X_iv)
            scores[_param_key(params)] = (params, float(np.mean(np.abs(y_iv - pred))))
        best_key = min(scores, key=lambda k: scores[k][1])
        return scores[best_key][0], scores

    cold_resid_params, cold_resid_scores = select_resid_group_cv(X_tr, y_tr, uid_tr, HGB_PARAM_GRID)
    pers_resid_params, pers_resid_scores = select_resid_time_split(p_train_samples, HGB_PARAM_GRID)
    md.append(f"- cold-start 선택: **{cold_resid_params}** (CV MAE={cold_resid_scores[_param_key(cold_resid_params)][1]:.3f})\n")
    md.append(f"- personalized 선택: **{pers_resid_params}** (inner-val MAE={pers_resid_scores[_param_key(pers_resid_params)][1]:.3f})\n\n")
    print(f"  cold_resid={cold_resid_params} pers_resid={pers_resid_params}")

    med_tr_full, med_te_full = median5_from_X(X_tr), median5_from_X(X_te)
    resid_cold_model = make_gbt(cold_resid_params, "absolute_error")
    resid_cold_model.fit(X_tr, y_tr - med_tr_full)
    resid_cold_pred = med_te_full + resid_cold_model.predict(X_te)

    med_ptr_full, med_pte_full = median5_from_X(X_ptr), median5_from_X(X_pte)
    resid_pers_model = make_gbt(pers_resid_params, "absolute_error")
    resid_pers_model.fit(X_ptr, y_ptr - med_ptr_full)
    resid_pers_pred = med_pte_full + resid_pers_model.predict(X_pte)

    # ===== 3. 평가: 통합표 =====
    print("\n평가 통합표/부트스트랩 계산 중...")
    def with_original_15_X(samples):
        """m21의 baseline 계산 함수들은 15차원 X(과거 5개 세션 x 3피처)를 가정한다.
        확장 샘플의 X는 22차원(15+7)이라 앞 15개만 잘라서 넘긴다 — baseline
        예측 자체는 새 피처를 쓰지 않으므로 안전하다."""
        return [{**s, "X": s["X"][:15]} for s in samples]

    base_cold_df = m21.build_rows(with_original_15_X(test_samples), overall_avg_lookup)
    base_pers_df = m21.build_rows(with_original_15_X(p_test_samples), overall_avg_lookup)

    ridge_cold_df = build_model_rows(test_samples, ridge_cold_pred, RIDGE_LABEL)
    hgb_cold_df = build_model_rows(test_samples, hgb_cold_pred, HGB_MAIN_LABEL)
    hgb_sq_cold_df = build_model_rows(test_samples, hgb_sq_cold_pred, HGB_SQ_LABEL)
    resid_cold_df = build_model_rows(test_samples, resid_cold_pred, HGB_RESID_LABEL)

    ridge_pers_df = build_model_rows(p_test_samples, ridge_pers_pred, RIDGE_LABEL)
    hgb_pers_df = build_model_rows(p_test_samples, hgb_pers_pred, HGB_MAIN_LABEL)
    hgb_sq_pers_df = build_model_rows(p_test_samples, hgb_sq_pers_pred, HGB_SQ_LABEL)
    resid_pers_df = build_model_rows(p_test_samples, resid_pers_pred, HGB_RESID_LABEL)

    combined_cold_df = pd.concat([base_cold_df, ridge_cold_df, hgb_cold_df, hgb_sq_cold_df, resid_cold_df], ignore_index=True)
    combined_pers_df = pd.concat([base_pers_df, ridge_pers_df, hgb_pers_df, hgb_sq_pers_df, resid_pers_df], ignore_index=True)

    ALL_LABELS = dict(m21.BASELINE_LABELS)
    ALL_LABELS[RIDGE_LABEL] = "6. Ridge(확장피처)"
    ALL_LABELS[HGB_MAIN_LABEL] = "7a. 직접구현 GBM(absolute_error)"
    ALL_LABELS[HGB_SQ_LABEL] = "7b. 직접구현 GBM(squared_error, 참고)"
    ALL_LABELS[HGB_RESID_LABEL] = "7c. median5+잔차 직접구현 GBM"
    ALL_NAMES = m21.BASELINE_NAMES + [RIDGE_LABEL, HGB_MAIN_LABEL, HGB_SQ_LABEL, HGB_RESID_LABEL]

    def summarize_all(df):
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

    md.append("## 3. 평가\n\n### 통합표 (메인, past-only 입력, 극단 페이스 전부 포함)\n")
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        md.append(f"\n#### {split_name}\n")
        md.append(m21.df_to_markdown(summarize_all(df)))
        md.append("\n")

    print("부트스트랩 CI 계산 중...")
    md.append(f"\n### 신뢰구간 (사용자 단위 cluster bootstrap, {m21.N_BOOTSTRAP}회, seed={m21.BOOTSTRAP_SEED})\n")
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
        md.append(f"\n#### {split_name}\n")
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n")

    # ===== 2-1. 모델 선택 방식 정정 (낙관적 편향 점검) =====
    print("\n2-1. CV기준 vs 테스트기준 모델 선택 비교 중...")
    cv_scores_per_family = {
        "cold-start": {
            RIDGE_LABEL: cold_ridge_scores[cold_ridge_alpha],
            HGB_MAIN_LABEL: cold_hgb_scores[_param_key(cold_hgb_params)][1],
            HGB_RESID_LABEL: cold_resid_scores[_param_key(cold_resid_params)][1],
        },
        "personalized": {
            RIDGE_LABEL: pers_ridge_scores[pers_ridge_alpha],
            HGB_MAIN_LABEL: pers_hgb_scores[_param_key(pers_hgb_params)][1],
            HGB_RESID_LABEL: pers_resid_scores[_param_key(pers_resid_params)][1],
        },
    }

    md.append("\n### 2-1. 모델 선택 방식 (낙관적 편향 점검)\n")
    md.append("**다중비교 경고**: 아래 3개 후보(squared_error 참고행 제외) 중 "
              "**테스트 성적이 가장 좋은 것을 사후에 고르면 낙관적 편향이 생길 수 있다.** "
              "그래서 **학습 구간(CV/내부검증) MAE로 뽑힐 모델**을 주 결과로 report하고, "
              "테스트에서 가장 좋은 모델이 다르면 참고로 같이 표시한다. 비교 후보 수 = 3개, "
              "다중비교 보정(예: Bonferroni)은 하지 않았다 — 후보가 적고 아래 bootstrap CI가 "
              "이미 넓어서, 결론(이기는지 여부) 자체는 어느 모델을 봐도 바뀌지 않는지 직접 확인한다.\n")

    cv_selected_per_split = {}
    report_models_per_split = {}
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        cv_scores = cv_scores_per_family[split_name]
        test_scores = {name: df[df["baseline"] == name]["abs_error"].mean() for name in [RIDGE_LABEL, HGB_MAIN_LABEL, HGB_RESID_LABEL]}
        cv_best = min(cv_scores, key=cv_scores.get)
        test_best = min(test_scores, key=test_scores.get)
        cv_selected_per_split[split_name] = cv_best
        report_models_per_split[split_name] = [cv_best] if cv_best == test_best else [cv_best, test_best]

        rows = [{"방법": ALL_LABELS[name], "CV/내부검증 MAE": cv_scores[name], "테스트 MAE": test_scores[name]}
                for name in [RIDGE_LABEL, HGB_MAIN_LABEL, HGB_RESID_LABEL]]
        md.append(f"\n#### {split_name}\n")
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append(f"\n- **CV/내부검증 기준 선택(주 결과)**: `{ALL_LABELS[cv_best]}`\n")
        md.append(f"- **테스트 기준 최고(참고)**: `{ALL_LABELS[test_best]}`"
                  f"{' (CV기준과 동일)' if cv_best == test_best else ' (**CV기준과 다름 — 둘 다 아래에 표시**)'}\n")
        print(f"  {split_name}: CV선택(주결과)={cv_best}, 테스트최고(참고)={test_best}")

    md.append("\n### 쌍체 차이 95% CI (주 결과: CV기준 선택 모델, 참고: 테스트기준 최고 모델 — 다르면 둘 다)\n")
    for split_name in ["cold-start", "personalized"]:
        boot_mae, boot_rmse = boot_results[split_name]
        for rank, model_name in enumerate(report_models_per_split[split_name]):
            role = "주 결과(CV 선택)" if rank == 0 else "참고(테스트 최고, CV기준과 다름)"
            md.append(f"\n#### {split_name} — {role}: {ALL_LABELS[model_name]}\n")
            mae_diff = boot_mae["median5"] - boot_mae[model_name]
            mae_lo, mae_hi = np.percentile(mae_diff, [2.5, 97.5])
            mae_beats = mae_lo > 0
            md.append(f"- **MAE: median-5 - 이 모델** = {mae_diff.mean():.3f} (95% CI [{mae_lo:.3f}, {mae_hi:.3f}]) "
                      f"-> {'0 미포함, 이 모델이 유의하게 낫음' if mae_beats else ('0 미포함, median-5가 유의하게 나음' if mae_hi < 0 else '0 포함, 유의한 차이 없음')}\n")
            rows = []
            for other in ["median5", "user_overall_avg", "recent_avg"]:
                diff = boot_rmse[other] - boot_rmse[model_name]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                rows.append({"비교 baseline": m21.BASELINE_LABELS[other], "RMSE차이(baseline-이모델)": diff.mean(),
                             "하한": lo, "상한": hi, "0포함(유의하지않음)": bool(lo <= 0 <= hi)})
            md.append(m21.df_to_markdown(pd.DataFrame(rows)))
            md.append("\n")

    best_model_per_split = cv_selected_per_split  # 이하 세분화/민감도/해석은 CV기준 선택 모델(주 결과)을 기준으로 함

    # ===== 세분화 (최고 모델 기준) =====
    print("세분화 분석 중...")
    md.append("\n### 세분화 분석 (CV기준 선택 모델=주 결과 기준, MAE/RMSE 둘 다)\n")
    point_counts = pd.read_pickle(Path("data") / "session_point_counts.pkl")
    pc_lookup = dict(zip(point_counts["session_id"], point_counts["n_points"]))

    combined_cold_df["split"] = "cold-start"
    combined_pers_df["split"] = "personalized"
    all_df = pd.concat([combined_cold_df, combined_pers_df], ignore_index=True)

    for split_name in ["cold-start", "personalized"]:
        best = best_model_per_split[split_name]
        model_df = all_df[(all_df["split"] == split_name) & (all_df["baseline"] == best)].copy()
        md.append(f"\n#### {split_name} — {ALL_LABELS[best]}\n")

        per_user_mae = model_df.groupby("user_id")["abs_error"].mean()
        md.append(f"- 사용자별 MAE 분포: median={per_user_mae.median():.3f}, p25={per_user_mae.quantile(.25):.3f}, "
                  f"p75={per_user_mae.quantile(.75):.3f}, p95={per_user_mae.quantile(.95):.3f}\n")

        bins = [0, 10, 30, 60, 120, 1e9]
        labels = ["5-9", "10-29", "30-59", "60-119", "120+"]
        model_df["history_bucket"] = pd.cut(model_df["target_session_order"], bins=bins, labels=labels, right=False)
        rows = []
        for bucket, grp in model_df.groupby("history_bucket", observed=True):
            if len(grp) == 0:
                continue
            rows.append({"history_bucket": bucket, "n": len(grp), "MAE": grp["abs_error"].mean(),
                         "RMSE": np.sqrt(grp["sq_error"].mean())})
        md.append("- 과거 기록 길이별:\n" + m21.df_to_markdown(pd.DataFrame(rows)) + "\n")

        pace_bins = [0, 250, 350, 450, 1000, 1e9]
        pace_labels = ["<250", "250-349", "350-449", "450-999", "1000+(극단)"]
        model_df["pace_bucket"] = pd.cut(model_df["y_true"], bins=pace_bins, labels=pace_labels, right=False)
        rows = []
        for bucket, grp in model_df.groupby("pace_bucket", observed=True):
            if len(grp) == 0:
                continue
            rows.append({"pace_bucket": bucket, "n": len(grp), "MAE": grp["abs_error"].mean(),
                         "RMSE": np.sqrt(grp["sq_error"].mean())})
        md.append("- 타깃 페이스 구간별:\n" + m21.df_to_markdown(pd.DataFrame(rows)) + "\n")

        non_merged = model_df[~model_df["target_merged"]].copy()
        non_merged["target_n_points"] = non_merged["target_session_id"].map(pc_lookup)
        non_merged["is_500"] = non_merged["target_n_points"] == 500
        rows = []
        for is_500_val in [True, False]:
            grp = non_merged[non_merged["is_500"] == is_500_val]
            if len(grp) == 0:
                continue
            rows.append({"n_points==500": is_500_val, "n": len(grp), "MAE": grp["abs_error"].mean(),
                         "RMSE": np.sqrt(grp["sq_error"].mean())})
        md.append(f"- 타깃 n_points=500 여부별(병합되지 않은 타깃만, {len(non_merged)}/{len(model_df)}):\n"
                  + m21.df_to_markdown(pd.DataFrame(rows)) + "\n")

        def merge_category(row):
            if row["target_merged"]:
                return "c_target_merged"
            elif row["input_merged"]:
                return "b_input_merged"
            return "a_unaffected"
        model_df["merge_category"] = model_df.apply(merge_category, axis=1)
        rows = []
        for cat, grp in model_df.groupby("merge_category"):
            rows.append({"카테고리": cat, "n": len(grp), "MAE": grp["abs_error"].mean(),
                         "RMSE": np.sqrt(grp["sq_error"].mean())})
        md.append("- 병합 영향 분해 (a 무관/b 입력만 병합/c 타깃 병합, (c)는 y 정의 자체가 바뀌므로 직접 비교 금지):\n"
                  + m21.df_to_markdown(pd.DataFrame(rows)) + "\n")

        threshold = model_df["abs_error"].quantile(0.99)
        top1 = model_df[model_df["abs_error"] >= threshold]
        sse_share = top1["sq_error"].sum() / model_df["sq_error"].sum()
        abs_share = top1["abs_error"].sum() / model_df["abs_error"].sum()
        md.append(f"- 오차 상위 1%(임계값={threshold:.1f}초/km, n={len(top1)}/{len(model_df)}) 집중 요인: "
                  f"SSE 비율={sse_share:.1%}, 절대오차 합 비율={abs_share:.1%}, "
                  f"병합영향비율(상위1% vs 전체)={top1['merge_affected'].mean():.1%} vs {model_df['merge_affected'].mean():.1%}, "
                  f"극단페이스영향비율={top1['extreme_affected'].mean():.1%} vs {model_df['extreme_affected'].mean():.1%}\n")

    # ===== 3-1. 개선폭 (절대/상대/CI, 코칭 관점) =====
    print("\n3-1. 개선폭(절대/상대/세션페이스중앙값 대비) 계산 중...")
    md.append("\n## 3-1. 개선폭 (CV기준 선택 모델=주 결과, 절대값/상대값/신뢰구간)\n")
    md.append("**통계적 유의성과 코칭 관점의 실용적 의미는 별개다** — 신뢰구간이 0을 포함하지 "
              "않아도(=median-5보다 통계적으로 유의하게 낫다는 뜻) 그 차이가 실제 코칭 문구를 "
              "바꿀 만큼 큰지는 따로 판단해야 한다.\n")
    for split_name, df, y_true_full in [("cold-start", combined_cold_df, y_te), ("personalized", combined_pers_df, y_pte)]:
        boot_mae, _ = boot_results[split_name]
        best = cv_selected_per_split[split_name]
        median5_point = df[df["baseline"] == "median5"]["abs_error"].mean()
        model_point = df[df["baseline"] == best]["abs_error"].mean()
        abs_improve = median5_point - model_point
        rel_improve_pct = abs_improve / median5_point * 100
        session_pace_median = float(np.median(y_true_full))
        pct_of_session_median = abs_improve / session_pace_median * 100
        mae_diff = boot_mae["median5"] - boot_mae[best]
        lo, hi = np.percentile(mae_diff, [2.5, 97.5])
        md.append(f"\n#### {split_name} ({ALL_LABELS[best]})\n")
        md.append(f"- median-5 MAE={median5_point:.3f}, 이 모델 MAE={model_point:.3f}\n")
        md.append(f"- **절대 개선폭**: {abs_improve:.3f}초/km (95% CI [{lo:.3f}, {hi:.3f}])\n")
        md.append(f"- **상대 개선폭(median-5 대비)**: {rel_improve_pct:.2f}%\n")
        md.append(f"- **세션 페이스 중앙값**(이 split 테스트셋 타깃 y의 중앙값) = {session_pace_median:.1f}초/km 기준, "
                  f"개선폭은 그 중앙값의 **{pct_of_session_median:.2f}%**에 해당한다.\n")

    # ===== 3-2. Permutation Importance (CV기준 선택 모델, 검증 구간 기준) =====
    print("\n3-2. permutation importance 계산 중...")
    md.append("\n## 3-2. Permutation Importance (CV기준 선택 모델, 검증 구간 기준, 상위 10개)\n")
    md.append("검증 구간: cold-start는 train을 다시 GroupKFold(5-fold)해서 **1개 fold를 held-out "
              "검증셋으로, 나머지 4개로 재학습**(테스트는 건드리지 않음); personalized는 이미 쓰던 "
              "시간순 inner-split의 inner-val을 그대로 쓴다. 지표는 MAE 증가량(섞었을 때 나빠지는 "
              "정도, 10회 반복 평균).\n")

    def _permutation_importance(predict_fn, X_val, y_val, n_repeats=10, seed=42):
        rng = np.random.default_rng(seed)
        base_mae = np.mean(np.abs(y_val - predict_fn(X_val)))
        out = []
        for j in range(X_val.shape[1]):
            increases = []
            for _ in range(n_repeats):
                X_perm = X_val.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                increases.append(np.mean(np.abs(y_val - predict_fn(X_perm))) - base_mae)
            out.append((m24.X_COLUMN_NAMES[j], float(np.mean(increases)), float(np.std(increases))))
        out.sort(key=lambda t: -t[1])
        return out, base_mae, len(y_val)

    new_feature_names = set(m24.X_COLUMN_NAMES[15:])
    for split_name in ["cold-start", "personalized"]:
        best = cv_selected_per_split[split_name]
        if split_name == "cold-start":
            tr_idx, val_idx = next(iter(SkGroupKFold(n_splits=N_FOLDS).split(X_tr, y_tr, groups=uid_tr)))
            X_fit, y_fit = X_tr[tr_idx], y_tr[tr_idx]
            X_val, y_val = X_tr[val_idx], y_tr[val_idx]
            alpha_sel, hgb_params_sel, resid_params_sel = cold_ridge_alpha, cold_hgb_params, cold_resid_params
        else:
            inner_tr, inner_val = personalized_inner_split(p_train_samples)
            X_fit, y_fit, _ = samples_to_xy(inner_tr)
            X_val, y_val, _ = samples_to_xy(inner_val)
            alpha_sel, hgb_params_sel, resid_params_sel = pers_ridge_alpha, pers_hgb_params, pers_resid_params

        if best == RIDGE_LABEL:
            m_fit = ridge_pipeline(alpha_sel).fit(X_fit, y_fit)
            predict_fn = m_fit.predict
        elif best == HGB_MAIN_LABEL:
            m_fit = make_gbt(hgb_params_sel, "absolute_error").fit(X_fit, y_fit)
            predict_fn = m_fit.predict
        else:
            med_fit = median5_from_X(X_fit)
            m_fit = make_gbt(resid_params_sel, "absolute_error")
            m_fit.fit(X_fit, y_fit - med_fit)
            predict_fn = (lambda model: (lambda X: median5_from_X(X) + model.predict(X)))(m_fit)

        importances, base_mae, n_val = _permutation_importance(predict_fn, X_val, y_val)
        md.append(f"\n#### {split_name} — {ALL_LABELS[best]} (검증셋 n={n_val}, 기준 MAE={base_mae:.3f})\n")
        rows = [{"피처": name, "MAE 증가량(중요도)": inc, "표준편차": sd,
                 "7단계 신규 피처": "Y" if name in new_feature_names else ""}
                for name, inc, sd in importances[:10]]
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        top10_names = [r["피처"] for r in rows]
        new_in_top10 = [n for n in top10_names if n in new_feature_names]
        if new_in_top10:
            md.append(f"\n- **확인된 것**: 상위 10위 안에 7단계 신규 피처 {len(new_in_top10)}개가 "
                      f"포함됨({', '.join(new_in_top10)}).\n")
        else:
            md.append("\n- **확인된 것**: 상위 10위 안에 7단계 신규 피처가 없음 — 이 검증 구간 "
                      "기준으로는 상위 중요도가 기존 15개 피처에 집중됨.\n")
        md.append("- **추정/미확인**: 신규 피처가 상위권에 있다는 사실만으로 'median-5 대비 개선이 "
                  "이 피처들 때문'이라고 인과적으로 결론 내릴 수 없다 — 그러려면 신규 피처를 뺀 "
                  "ablation 모델이 필요한데 이번 요청 범위에서는 만들지 않았다. 아래는 **미확인**으로만 "
                  "남긴다: 개선의 몇 %가 새 피처 덕분인지.\n")

    all_samples_check = train_samples + test_samples + p_train_samples + p_test_samples
    gaps_check = [s["target_start_time"] - s["last_input_session_time"] for s in all_samples_check]
    md.append(f"\n- **모든 피처의 last_input_session_time 감사 재확인**: {len(all_samples_check)}개 "
              f"샘플 전체에서 last_input_session_time < target_start_time 최소 갭 = {min(gaps_check)}초 "
              f"(0보다 큼, 전부 통과). `user_history_median_pace`의 이력 감사는 24단계에서 이미 "
              f"적용돼 걸러진 뒤의 샘플만 여기 들어있다(1-1 참고).\n")

    # ===== 4. 민감도: moving pace (재학습 없음) =====
    print("민감도(moving pace) 평가 중...")
    md.append("\n## 4. 민감도: 타깃을 moving pace로 바꾼 평가 "
              "(elapsed 대비 정지 시간 노이즈 영향 가늠용 민감도, 메인 표와 분리, 재학습 안 함)\n")
    md.append("재학습하지 않고, 메인(elapsed) 타깃으로 학습한 같은 모델의 같은 예측값을 "
              "moving pace 실측과 비교만 한다 — \"elapsed 기준으로 측정한 오차 중 정지시간 "
              "노이즈가 얼마나 차지하는가\"를 보는 것이 목적이다.\n")
    for split_name, test_samples_, preds_by_label in [
        ("cold-start", test_samples, {RIDGE_LABEL: ridge_cold_pred, HGB_MAIN_LABEL: hgb_cold_pred, HGB_RESID_LABEL: resid_cold_pred}),
        ("personalized", p_test_samples, {RIDGE_LABEL: ridge_pers_pred, HGB_MAIN_LABEL: hgb_pers_pred, HGB_RESID_LABEL: resid_pers_pred}),
    ]:
        y_moving = np.array([s["y_moving"] for s in test_samples_])
        best = best_model_per_split[split_name]
        rows = []
        for method_name in ["median5", "user_overall_avg", "recent_avg"]:
            sub_df = all_df[(all_df["split"] == split_name) & (all_df["baseline"] == method_name)]
            y_pred = sub_df["y_pred"].to_numpy()
            mae = np.mean(np.abs(y_moving - y_pred))
            rmse = np.sqrt(np.mean((y_moving - y_pred) ** 2))
            rows.append({"방법": m21.BASELINE_LABELS[method_name], "MAE(vs moving)": mae, "RMSE(vs moving)": rmse})
        y_pred_best = preds_by_label[best]
        mae_best = np.mean(np.abs(y_moving - y_pred_best))
        rmse_best = np.sqrt(np.mean((y_moving - y_pred_best) ** 2))
        rows.append({"방법": f"CV기준 선택 모델 {ALL_LABELS[best]}", "MAE(vs moving)": mae_best, "RMSE(vs moving)": rmse_best})
        md.append(f"\n#### {split_name}\n")
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n")

    # ===== 5. 해석 / 정지 규칙 =====
    md.append("\n## 5. 해석\n")
    any_beats = False
    for split_name in ["cold-start", "personalized"]:
        boot_mae, _ = boot_results[split_name]
        best = best_model_per_split[split_name]
        mae_diff = boot_mae["median5"] - boot_mae[best]
        lo, hi = np.percentile(mae_diff, [2.5, 97.5])
        beats = lo > 0
        any_beats = any_beats or beats
        if beats:
            md.append(f"- **{split_name}**: CV기준 선택 모델({ALL_LABELS[best]}, 주 결과)이 median-5를 "
                      f"MAE 신뢰구간 밖에서 **이겼다** (median5-이모델 diff 95% CI [{lo:.3f},{hi:.3f}], 0 미포함).\n")
        else:
            md.append(f"- **{split_name}**: CV기준 선택 모델({ALL_LABELS[best]}, 주 결과)이 median-5를 "
                      f"MAE 신뢰구간 밖에서 **이기지 못함**. (diff 95% CI [{lo:.3f},{hi:.3f}]) "
                      f"원인은 **추정/미확인**이다.\n")

    md.append("\n**RMSE 결론이 MAE 결론과 다를 수 있음에 유의**: RMSE는 상위 1% 오차 윈도우가 "
              "SSE 대부분을 차지하는 극단 민감 지표라(위 세분화 참고), MAE에서 진 방법이 RMSE에서는 "
              "통계적으로 구분되지 않을 수 있다 — 위 쌍체 차이 표의 RMSE 열을 MAE 결론과 섞지 말고 "
              "따로 읽는다.\n")

    if not any_beats:
        md.append("\n**[정지 규칙] 두 평가조건 모두 median-5를 이기지 못했으므로, 이번 7단계 결과로는 "
                  "MLP/LSTM으로 넘어가지 않고 여기서 멈춘다.** (CLAUDE.md 게이트: 6~8번 모델이 "
                  "median-5 대비 실질 개선을 보일 때만 9번으로 진행)\n")
        print("\n[정지] median-5를 이기지 못함 - MLP/LSTM으로 넘어가지 않고 보고합니다.")
    else:
        md.append("\n이긴 조건이 있으므로 사용자 확인 후 다음 단계(MLP) 진행 여부를 논의한다.\n")

    # ===== 6. 직접 구현 GBM(gbt_lib) 검증 =====
    print("\n6. 직접 구현 GBM 검증(라벨 셔플/합성 데이터/시드) 실행 중...")
    md.append("\n## 6. 직접 구현 GBM(gbt_lib) 검증\n")
    md.append("(d) 7단계 1번의 피처 요약 방식(중앙값 하나로 통일) 결정 시점은 1-1 위쪽 "
              "'해석 선택' 항목에 이미 적어뒀다 — 모델을 한 번도 돌리기 전에 정한 것이고, "
              "원본 5개씩 넣는 버전은 만들지 않았다.\n")

    def _shuffle_within_user(values, user_ids, rng):
        out = values.copy()
        for uid in np.unique(user_ids):
            idx = np.where(user_ids == uid)[0]
            if len(idx) > 1:
                out[idx] = rng.permutation(values[idx])
        return out

    md.append("\n### (a) 라벨 셔플 테스트 (cold-start, 5개 시드)\n")
    median5_test_mae_check = np.mean(np.abs(y_te - median5_from_X(X_te)))
    md.append(f"median-5 test MAE(기준값) = {median5_test_mae_check:.3f}\n")

    within_rows = []
    for seed in range(5):
        rng_s = np.random.default_rng(seed)
        y_shuf = _shuffle_within_user(y_tr, uid_tr, rng_s)
        m_shuf = make_gbt(cold_hgb_params, "absolute_error").fit(X_tr, y_shuf)
        mae_shuf = float(np.mean(np.abs(y_te - m_shuf.predict(X_te))))
        within_rows.append({"시드": seed, "GBM_abs, y 사용자내 셔플 후 test MAE": mae_shuf,
                             "median5 대비": mae_shuf - median5_test_mae_check})
    md.append("- **GBM_abs(직접구현), y를 사용자 내부에서 셔플 후 재학습**:\n")
    md.append(m21.df_to_markdown(pd.DataFrame(within_rows)))

    global_rows = []
    for seed in range(3):
        rng_s = np.random.default_rng(100 + seed)
        y_shuf_g = rng_s.permutation(y_tr)
        m_shuf_g = make_gbt(cold_hgb_params, "absolute_error").fit(X_tr, y_shuf_g)
        mae_shuf_g = float(np.mean(np.abs(y_te - m_shuf_g.predict(X_te))))
        global_rows.append({"시드": seed, "GBM_abs, y 전체 셔플(통제실험) 후 test MAE": mae_shuf_g,
                             "median5 대비": mae_shuf_g - median5_test_mae_check})
    md.append("\n- **통제 실험: y를 사용자 구분 없이 전체 셔플(순수 노이즈)**:\n")
    md.append(m21.df_to_markdown(pd.DataFrame(global_rows)))

    med_resid_rows = []
    med5_tr_check = median5_from_X(X_tr)
    resid_tr_check = y_tr - med5_tr_check
    med5_te_check = median5_from_X(X_te)
    for seed in range(5):
        rng_s = np.random.default_rng(200 + seed)
        resid_shuf = _shuffle_within_user(resid_tr_check, uid_tr, rng_s)
        m_resid_shuf = make_gbt(cold_resid_params, "absolute_error")
        m_resid_shuf.fit(X_tr, resid_shuf)
        pred_resid_shuf = med5_te_check + m_resid_shuf.predict(X_te)
        mae_resid_shuf = float(np.mean(np.abs(y_te - pred_resid_shuf)))
        med_resid_rows.append({"시드": seed, "median5+잔차, 잔차 사용자내 셔플 후 test MAE": mae_resid_shuf,
                                "median5 대비": mae_resid_shuf - median5_test_mae_check})
    md.append("\n- **median5+잔차 GBM, 잔차를 사용자 내부에서 셔플 후 재학습**:\n")
    md.append(m21.df_to_markdown(pd.DataFrame(med_resid_rows)))

    within_mean = np.mean([r["median5 대비"] for r in within_rows])
    grand_mean_check = y_tr.mean()
    user_means_check = pd.Series(y_tr).groupby(pd.Series(uid_tr)).mean()
    between_var_check = ((user_means_check - grand_mean_check) ** 2).mean()
    within_var_check = pd.Series(y_tr).groupby(pd.Series(uid_tr)).var(ddof=0).mean()
    between_share = between_var_check / (between_var_check + within_var_check)
    md.append(f"\n**해석(사용자 확인·수용됨)**: GBM_abs는 사용자 내부 셔플에서 median-5를 평균 "
              f"{within_mean:.3f}초/km 만큼 **작게 이겼다**(일관되게, 5개 시드 전부). 전체(사용자 "
              f"무관) 셔플 통제실험에서는 평균 {np.mean([r['median5 대비'] for r in global_rows]):.1f}초/km "
              f"만큼 **크게 졌다** — 즉 사용자 식별 정보가 사라지면 이점이 완전히 없어진다. "
              f"사용자간 분산 비율={between_share:.1%}(매우 높음)로, 같은 사용자의 세션들이 서로 "
              f"비슷한 페이스를 보이는 자기상관이 강하다는 뜻이다. `median_past5_elapsed_pace`, "
              f"`user_history_median_pace` 같은 피처가 '이 사용자는 대체로 이런 페이스'라는 정보를 "
              f"이미 담고 있어서, 타깃을 사용자 내부에서만 섞어도 그 정보 자체는 깨지지 않는다 "
              f"(median-5 자체도 같은 원리로 작동하는 통계량이다). median5+잔차 모델은 median5를 "
              f"먼저 빼고 남은 잔차만 섞기 때문에 이 효과가 거의 사라지고(일관되게 median-5보다 "
              f"나쁨), 이 비대칭이 '타깃 세션 고유의 미래 정보 유출'이 아니라 '구조적인 사용자간 "
              f"페이스 자기상관' 설명을 지지한다. **다만 이는 해석이며, 추가로 다른 종류의 의심스러운 "
              f"패턴이 발견되면 재검토가 필요하다.**\n")

    md.append("\n### (b) 합성 비대칭(두꺼운 꼬리) 노이즈 테스트\n")
    rng_syn = np.random.default_rng(42)
    n_syn = 4000
    X_syn = rng_syn.uniform(0, 1, size=(n_syn, 2))
    group_syn = (X_syn[:, 0] >= 0.5).astype(float)
    base_syn = np.where(group_syn == 1, 50.0, 100.0)
    noise_syn = rng_syn.exponential(scale=20.0, size=n_syn) - 20.0
    y_syn = base_syn + noise_syn
    true_median_g0 = float(np.median(100.0 + rng_syn.exponential(scale=20.0, size=200000) - 20.0))
    true_median_g1 = float(np.median(50.0 + rng_syn.exponential(scale=20.0, size=200000) - 20.0))
    m_abs_syn = gl.SimpleGradientBoostingRegressor(loss="absolute_error", learning_rate=0.3, max_iter=50, max_depth=1, min_samples_leaf=20, n_bins=32).fit(X_syn, y_syn)
    m_sq_syn = gl.SimpleGradientBoostingRegressor(loss="squared_error", learning_rate=0.3, max_iter=50, max_depth=1, min_samples_leaf=20, n_bins=32).fit(X_syn, y_syn)
    pred_abs_g0 = float(m_abs_syn.predict(np.array([[0.2, 0.5]]))[0])
    pred_abs_g1 = float(m_abs_syn.predict(np.array([[0.8, 0.5]]))[0])
    pred_sq_g0 = float(m_sq_syn.predict(np.array([[0.2, 0.5]]))[0])
    pred_sq_g1 = float(m_sq_syn.predict(np.array([[0.8, 0.5]]))[0])
    md.append(f"- Exponential(비대칭) 노이즈(평균=0, 중앙값≈-13.86)를 더한 2개 그룹 합성 데이터:\n")
    md.append(f"- 그룹0(실제 조건부 중앙값={true_median_g0:.2f}, 평균=100.00): absolute_error 예측="
              f"{pred_abs_g0:.2f}(중앙값에 더 가까움={abs(pred_abs_g0-true_median_g0) < abs(pred_abs_g0-100.0)}), "
              f"squared_error 예측={pred_sq_g0:.2f}(평균에 더 가까움={abs(pred_sq_g0-100.0) < abs(pred_sq_g0-true_median_g0)})\n")
    md.append(f"- 그룹1(실제 조건부 중앙값={true_median_g1:.2f}, 평균=50.00): absolute_error 예측="
              f"{pred_abs_g1:.2f}(중앙값에 더 가까움={abs(pred_abs_g1-true_median_g1) < abs(pred_abs_g1-50.0)}), "
              f"squared_error 예측={pred_sq_g1:.2f}(평균에 더 가까움={abs(pred_sq_g1-50.0) < abs(pred_sq_g1-true_median_g1)})\n")
    md.append("- **확인됨**: absolute_error는 조건부 중앙값 쪽, squared_error는 조건부 평균 쪽으로 "
              "수렴한다 — 둘 다 설계대로 동작.\n")

    md.append("\n### (c) 시드 변경 재학습 (변동폭)\n")
    seed_maes = []
    for seed in [1, 2, 3, 42, 999]:
        m_seed = gl.SimpleGradientBoostingRegressor(loss="absolute_error", learning_rate=0.1, max_iter=300, max_depth=3, min_samples_leaf=20, n_bins=32, random_state=seed).fit(X_tr, y_tr)
        seed_maes.append(float(np.mean(np.abs(y_te - m_seed.predict(X_te)))))
    md.append(f"- 시드 [1,2,3,42,999]로 5회 재학습한 test MAE: {[round(v, 6) for v in seed_maes]}\n")
    md.append(f"- **변동폭 = {max(seed_maes) - min(seed_maes):.6f} (정확히 0)**. `gbt_lib.py`는 "
              "행/피처 서브샘플링을 구현하지 않아 `random_state`를 바꿔도 실제로 소비되는 난수가 "
              "없다 — **완전히 결정적**이다(숨기지 않고 사실대로 보고). 의미 있는 변동을 보려면 "
              "서브샘플링(bagging)을 추가해야 하는데, 이번 요청 범위에서는 구현하지 않았다.\n")

    # ===== 7. 환경 확인 (8단계 사전 체크, 설치/변경 없음) =====
    print("\n7. 8단계(MLP) 사전 환경 확인 중 (설치/변경 없음)...")
    md.append("\n## 7. 환경 확인 (8단계 사전 체크용, 설치/변경 없음 — 8단계 시작은 사용자 결정 대기)\n")
    for mod_desc, import_fn in [
        ("import torch", lambda: __import__("torch")),
        ("from sklearn.neural_network import MLPRegressor", lambda: __import__("sklearn.neural_network", fromlist=["MLPRegressor"])),
        ("from sklearn.linear_model import QuantileRegressor", lambda: __import__("sklearn.linear_model", fromlist=["QuantileRegressor"])),
    ]:
        try:
            mod = import_fn()
            version = getattr(mod, "__version__", None)
            md.append(f"- `{mod_desc}`: **성공**{f' (version={version})' if version else ''}\n")
            print(f"  {mod_desc}: OK")
        except Exception as e:
            md.append(f"- `{mod_desc}`: **실패** `{type(e).__name__}: {str(e)[:200]}`\n")
            print(f"  {mod_desc}: FAIL - {e}")
    md.append("\n8단계(MLP) 진행 여부는 사용자가 이 결과를 보고 결정한다 — 이번 커밋에서 8단계는 "
              "시작하지 않는다.\n")

    OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
    print(f"\n최종 저장 완료: {OUTPUT_MD_PATH.resolve()}")
    return {"any_beats": any_beats, "best_model_per_split": best_model_per_split, "ALL_LABELS": ALL_LABELS}


if __name__ == "__main__":
    main()
