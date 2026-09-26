# -*- coding: utf-8 -*-
"""
5단계: Baseline 1 - 최근 N회 평균.

예측값 = 과거 5개 세션의 avg_pace_sec_per_km 평균 (그대로 다음 세션 예측값으로 사용).
학습이 필요 없는 규칙 기반 baseline이라, train_dataset은 쓰지 않고
test_dataset에 대해서만 예측/평가한다 (train은 나중에 다른 모델(선형회귀 등)의
비교 기준을 맞추기 위해 그대로 둔다).

평가 지표 (CLAUDE.md 규칙):
- 전체 MAE, RMSE
- 사용자별 오차(MAE) 분산도 (전체 평균만 보고 끝내지 않음)
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TEST_DATASET_PATH = Path("data") / "test_dataset.pkl"

# 04단계에서 X는 [distance_km, avg_pace_sec_per_km, avg_heart_rate] x 5세션
# 순서로 flatten되어 있다. avg_pace_sec_per_km은 각 세션 블록에서 1번 인덱스.
N_PAST_SESSIONS = 5
N_FEATURES_PER_SESSION = 3
PACE_FEATURE_INDEX = 1  # [distance, pace, heart_rate] 중 pace의 위치


def extract_past_paces(x_flat):
    """flatten된 X 벡터(길이 15)에서 과거 5개 세션의 avg_pace_sec_per_km만 뽑는다."""
    x_reshaped = np.array(x_flat).reshape(N_PAST_SESSIONS, N_FEATURES_PER_SESSION)
    return x_reshaped[:, PACE_FEATURE_INDEX]


def main():
    test_samples = pd.read_pickle(TEST_DATASET_PATH)

    records = []
    for sample in test_samples:
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

    # 사용자별 MAE를 따로 구해서 분포 확인 (전체 평균만 보고 끝내지 않음).
    per_user_mae = result_df.groupby("user_id")["abs_error"].mean()

    print("=" * 60)
    print("Baseline 1: 최근 5회 평균 (recency average)")
    print(f"test 샘플 수: {len(result_df)} (사용자 {result_df['user_id'].nunique()}명)")
    print("-" * 60)
    print(f"전체 MAE  (초/km): {overall_mae:.3f}")
    print(f"전체 RMSE (초/km): {overall_rmse:.3f}")
    print("-" * 60)
    print("사용자별 MAE 분포 (사용자마다 MAE를 따로 구한 값들의 분포):")
    print(per_user_mae.describe())
    print("-" * 60)
    print(f"사용자별 MAE의 표준편차: {per_user_mae.std():.3f}")
    print("사용자별 MAE가 가장 큰 5명 (오차가 큰 사용자):")
    print(per_user_mae.sort_values(ascending=False).head(5))
    print("사용자별 MAE가 가장 작은 5명:")
    print(per_user_mae.sort_values(ascending=True).head(5))
    print("-" * 60)

    sample = result_df.iloc[0]
    print(f"[샘플 확인] user_id={sample['user_id']}: "
          f"실제값={sample['y_true']:.3f}, 예측값(최근5회 평균)={sample['y_pred']:.3f}, "
          f"절대오차={sample['abs_error']:.3f}")


if __name__ == "__main__":
    main()
