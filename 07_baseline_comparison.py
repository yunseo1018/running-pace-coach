# -*- coding: utf-8 -*-
"""
7단계 (v2, 5주차): Baseline 5종을 cold-start/personalized 두 평가 방식으로 비교.

CLAUDE.md "비교 모델 및 Baseline (v2)" 순서대로 5개 baseline을 구현한다.
전부 "과거 5개 세션의 페이스만 보고 단순하게 예측하면 얼마나 틀리는지"를 보는
목적이라, X의 16번째 값(target.distance_km, 거리 조건값)은 쓰지 않고 앞 15개
(과거 5세션 x [distance_km, avg_pace_sec_per_km, avg_heart_rate])만 사용한다.

1. recent_avg   : 과거 5개 세션 페이스의 평균
2. last_session : 과거 5개 중 가장 최근(5번째) 세션의 페이스 그대로
3. median5      : 과거 5개 세션 페이스의 중앙값 (극단치에 덜 흔들림)
4. user_overall_avg : 그 사용자의 "지금까지의 모든 세션"(과거 5개보다 더 이전 것까지
   전부) 페이스 평균 — 과거 5개만 보는 1번과 달리 전체 이력을 사용하므로
   data/user_sequences.pkl에서 사용자별 전체 이력을 따로 불러와야 한다.
5. linear_trend : 과거 5개 세션의 (순서, 페이스)에 단순 선형회귀(최소제곱)를
   적용해 바로 다음 시점의 페이스를 추정 (빨라지는/느려지는 추세 반영).
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

USER_SEQUENCES_PATH = Path("data") / "user_sequences.pkl"
COLD_START_TEST_PATH = Path("data") / "test_dataset.pkl"
PERSONALIZED_TEST_PATH = Path("data") / "personalized_test_dataset.pkl"

N_PAST_SESSIONS = 5
N_FEATURES_PER_PAST_SESSION = 3
PACE_FEATURE_INDEX = 1  # [distance, pace, heart_rate] 중 pace의 위치


def extract_past_paces(x_flat):
    """flatten된 X 벡터(길이 16)에서 과거 5개 세션의 avg_pace_sec_per_km만 뽑는다.
    target.distance_km(16번째 값)은 이 baseline들에서는 쓰지 않는다."""
    past_part = np.array(x_flat[: N_PAST_SESSIONS * N_FEATURES_PER_PAST_SESSION])
    return past_part.reshape(N_PAST_SESSIONS, N_FEATURES_PER_PAST_SESSION)[:, PACE_FEATURE_INDEX]


def build_user_overall_avg_lookup(user_sequences_df):
    """
    (user_id, target_session_order) -> "그 세션 이전까지의 모든 세션" 페이스 평균
    을 조회할 수 있는 dict를 만든다.

    expanding().mean()은 현재 행까지 포함한 누적평균이므로, shift(1)로 한 칸
    밀어서 "현재 행 제외, 그 이전까지의 누적평균"으로 만든다 (타겟 세션 자신의
    값이 평균에 섞이면 leakage이므로 반드시 shift 필요).
    """
    df = user_sequences_df.sort_values(["user_id", "session_order"]).copy()
    df["cum_mean_before"] = (
        df.groupby("user_id")["avg_pace_sec_per_km"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    lookup = {
        (row.user_id, row.session_order): row.cum_mean_before
        for row in df.itertuples()
    }
    return lookup


def predict_all_baselines(sample, overall_avg_lookup):
    past_paces = extract_past_paces(sample["X"])

    x_idx = np.arange(N_PAST_SESSIONS)
    slope, intercept = np.polyfit(x_idx, past_paces, 1)
    linear_trend_pred = slope * N_PAST_SESSIONS + intercept

    overall_avg_pred = overall_avg_lookup.get(
        (sample["user_id"], sample["target_session_order"])
    )

    return {
        "recent_avg": past_paces.mean(),
        "last_session": past_paces[-1],
        "median5": np.median(past_paces),
        "user_overall_avg": overall_avg_pred,
        "linear_trend": linear_trend_pred,
    }


BASELINE_NAMES = ["recent_avg", "last_session", "median5", "user_overall_avg", "linear_trend"]
BASELINE_LABELS = {
    "recent_avg": "1. 최근 5회 평균",
    "last_session": "2. last-session (가장 최근 세션)",
    "median5": "3. median-5 (최근 5개 중앙값)",
    "user_overall_avg": "4. 사용자 전체 평균",
    "linear_trend": "5. 단순 선형 추세",
}


def evaluate_split(dataset_path, overall_avg_lookup, label):
    samples = pd.read_pickle(dataset_path)

    rows = []
    for sample in samples:
        preds = predict_all_baselines(sample, overall_avg_lookup)
        # target_session_order는 항상 5 이상(과거 5개 세션이 보장됨)이므로
        # user_overall_avg는 항상 최소 5개 이전 세션의 평균을 가진다 (None/NaN 불가능).
        assert pd.notna(preds["user_overall_avg"]), \
            f"user_overall_avg 조회 실패: user_id={sample['user_id']}, target_session_order={sample['target_session_order']}"
        for name in BASELINE_NAMES:
            y_pred = preds[name]
            y_true = sample["y"]
            rows.append(
                {
                    "baseline": name,
                    "user_id": sample["user_id"],
                    "abs_error": abs(y_true - y_pred),
                    "sq_error": (y_true - y_pred) ** 2,
                }
            )

    result_df = pd.DataFrame(rows)

    summary_rows = []
    for name in BASELINE_NAMES:
        sub = result_df[result_df["baseline"] == name]
        per_user_mae = sub.groupby("user_id")["abs_error"].mean()
        summary_rows.append(
            {
                "baseline": BASELINE_LABELS[name],
                "overall_mae": sub["abs_error"].mean(),
                "overall_rmse": np.sqrt(sub["sq_error"].mean()),
                "per_user_mae_mean": per_user_mae.mean(),
                "per_user_mae_median": per_user_mae.median(),
                "per_user_mae_std": per_user_mae.std(),
            }
        )
    summary_df = pd.DataFrame(summary_rows).set_index("baseline")

    print("=" * 70)
    print(f"[{label}] baseline 5종 비교 (샘플 {len(samples)}건, "
          f"사용자 {result_df['user_id'].nunique()}명)")
    print(summary_df.to_string())
    print()

    return summary_df


def main():
    user_sequences_df = pd.read_pickle(USER_SEQUENCES_PATH)
    overall_avg_lookup = build_user_overall_avg_lookup(user_sequences_df)

    cold_start_summary = evaluate_split(COLD_START_TEST_PATH, overall_avg_lookup,
                                         "cold-start (사용자 단위 80/20)")
    personalized_summary = evaluate_split(PERSONALIZED_TEST_PATH, overall_avg_lookup,
                                           "personalized (사용자 내 시간순 temporal hold-out)")

    print("=" * 70)
    print("[최종 비교] 두 평가 방식에서 각 baseline의 전체 MAE/RMSE:")
    combined = pd.DataFrame(
        {
            "cold_start_MAE": cold_start_summary["overall_mae"],
            "cold_start_RMSE": cold_start_summary["overall_rmse"],
            "personalized_MAE": personalized_summary["overall_mae"],
            "personalized_RMSE": personalized_summary["overall_rmse"],
        }
    )
    print(combined.to_string())


if __name__ == "__main__":
    main()
