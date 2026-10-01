# -*- coding: utf-8 -*-
"""
5단계: Baseline 1 - 최근 N회 평균.

예측값 = 과거 5개 세션의 avg_pace_sec_per_km 평균 (그대로 다음 세션 예측값으로 사용).
학습이 필요 없는 규칙 기반 baseline이라 train 데이터는 쓰지 않고, test(평가용)
데이터에 대해서만 예측/평가한다.

5주차(v2) 업데이트: X가 [과거 5세션 x 3피처](15) + [target.distance_km](1) = 16개로
늘어났지만, 이 baseline은 여전히 "과거 5개 세션의 페이스 평균"만 쓰므로 X의 앞 15개
값 중 pace 위치만 읽는다 (target.distance_km은 이 baseline에서는 사용하지 않음).

CLAUDE.md 데이터 분할 규칙(v2): cold-start(사용자 단위 80/20)와 personalized
(사용자 내 시간순 history/temporal hold-out) 두 가지 평가를 각각 따로 계산해서
따로 보고한다 (하나로 합치지 않는다).

평가 지표:
- 전체 MAE, RMSE
- 사용자별 오차(MAE) 분산도 (전체 평균만 보고 끝내지 않음)
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

COLD_START_TEST_PATH = Path("data") / "test_dataset.pkl"
PERSONALIZED_TEST_PATH = Path("data") / "personalized_test_dataset.pkl"

# X는 [distance_km, avg_pace_sec_per_km, avg_heart_rate] x 5세션(15) +
# target.distance_km(1) 순서로 flatten되어 있다 (04/06단계 참고).
# 이 baseline은 과거 세션들의 avg_pace_sec_per_km(각 3개 블록의 1번 인덱스)만 쓴다.
N_PAST_SESSIONS = 5
N_FEATURES_PER_PAST_SESSION = 3
PACE_FEATURE_INDEX = 1  # [distance, pace, heart_rate] 중 pace의 위치


def extract_past_paces(x_flat):
    """flatten된 X 벡터(길이 16)에서 과거 5개 세션의 avg_pace_sec_per_km만 뽑는다."""
    past_part = np.array(x_flat[: N_PAST_SESSIONS * N_FEATURES_PER_PAST_SESSION])
    x_reshaped = past_part.reshape(N_PAST_SESSIONS, N_FEATURES_PER_PAST_SESSION)
    return x_reshaped[:, PACE_FEATURE_INDEX]


def evaluate(dataset_path, label):
    samples = pd.read_pickle(dataset_path)

    records = []
    for sample in samples:
        past_paces = extract_past_paces(sample["X"])
        y_pred = past_paces.mean()  # baseline: 최근 5개 평균
        y_true = sample["y"]
        records.append(
            {
                "user_id": sample["user_id"],
                "y_true": y_true,
                "y_pred": y_pred,
                "abs_error": abs(y_true - y_pred),
                "sq_error": (y_true - y_pred) ** 2,
            }
        )

    result_df = pd.DataFrame(records)
    overall_mae = result_df["abs_error"].mean()
    overall_rmse = np.sqrt(result_df["sq_error"].mean())
    per_user_mae = result_df.groupby("user_id")["abs_error"].mean()

    print("=" * 60)
    print(f"Baseline 1: 최근 5회 평균 (recency average) — {label}")
    print(f"샘플 수: {len(result_df)} (사용자 {result_df['user_id'].nunique()}명)")
    print("-" * 60)
    print(f"전체 MAE  (초/km): {overall_mae:.3f}")
    print(f"전체 RMSE (초/km): {overall_rmse:.3f}")
    print("-" * 60)
    print("사용자별 MAE 분포:")
    print(per_user_mae.describe())
    print(f"사용자별 MAE 표준편차: {per_user_mae.std():.3f}")
    print("-" * 60)

    return {
        "label": label,
        "n_samples": len(result_df),
        "n_users": result_df["user_id"].nunique(),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "per_user_mae_mean": per_user_mae.mean(),
        "per_user_mae_median": per_user_mae.median(),
        "per_user_mae_std": per_user_mae.std(),
    }


def main():
    cold_start_result = evaluate(COLD_START_TEST_PATH, "cold-start (사용자 단위 80/20)")
    personalized_result = evaluate(PERSONALIZED_TEST_PATH, "personalized (사용자 내 시간순 temporal hold-out)")

    print("=" * 60)
    print("[비교] 같은 baseline(최근 5회 평균)을 두 평가 방식으로 각각 평가한 결과:")
    comparison_df = pd.DataFrame([cold_start_result, personalized_result]).set_index("label")
    comparison_df = comparison_df[["n_samples", "n_users", "overall_mae", "overall_rmse",
                                    "per_user_mae_mean", "per_user_mae_median", "per_user_mae_std"]]
    print(comparison_df.to_string())


if __name__ == "__main__":
    main()
