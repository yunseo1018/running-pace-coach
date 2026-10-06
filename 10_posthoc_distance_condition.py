# -*- coding: utf-8 -*-
"""
사후 거리 조건 분석 (post-hoc distance-conditioned analysis) — 러닝 전 예측 성능 아님.

배경: 10월 1주차(v2)에 "타깃 세션의 거리를 계획값이라 부르며 입력에 넣는"
방식을 썼다. 10월 2주차에 교수 피드백으로, 그 값(actual_distance_km)은
실제로 GPS로 계산한 완주 거리라 러닝이 끝나야 확정되는 결과값이고, 러닝 전
예측에는 쓸 수 없다는 정정을 받았다 (docs/feature_availability.md 참고).

이 스크립트는 그 v2 방식을 **삭제하지 않고**, "사후(post-hoc) 거리 조건
분석"이라는 별도 실험으로 보존한다. 목적은 하나뿐이다:
"타깃 세션이 실제로 몇 km였는지 안다고 가정하면, 그 조건에서 페이스 변동
(인터벌처럼 짧고 빠른지, 장거리처럼 길고 느린지)이 얼마나 설명되는가"를
보는 사후 분석이며, **러닝 전 예측 성능으로 보고하지 않는다.**

X(사후) = 과거 5개 세션의 (actual_distance_km, avg_pace_sec_per_km, avg_heart_rate) (15)
          + 타깃 세션의 actual_distance_km (1, 사후 정보) = 16개
y = 타깃 세션의 avg_pace_sec_per_km
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

INPUT_PATH = Path("data") / "user_sequences.pkl"
TRAIN_USERS_PATH = Path("data") / "train_user_ids.txt"
TEST_USERS_PATH = Path("data") / "test_user_ids.txt"
TRAIN_OUTPUT_PATH = Path("data") / "posthoc_train_dataset.pkl"
TEST_OUTPUT_PATH = Path("data") / "posthoc_test_dataset.pkl"

N_PAST_SESSIONS = 5
PAST_FEATURE_COLUMNS = ["actual_distance_km", "avg_pace_sec_per_km", "avg_heart_rate"]

# 타깃(6번째) 세션에서 "사후 정보"로서만 X에 포함하는 값 — actual_distance_km
# (예전 v2에서는 "계획값"이라 불렀으나, 실제로는 완주 후 확정되는 결과값이다).
TARGET_POSTHOC_COLUMN = "actual_distance_km"
TARGET_POSTHOC_DISPLAY_NAME = "actual_distance_km"

TARGET_FORBIDDEN_COLUMNS = ["avg_heart_rate", "duration_sec", "avg_pace_sec_per_km"]
TARGET_COLUMN = "avg_pace_sec_per_km"

X_COLUMN_NAMES = [
    f"past[{i}].{col}" for i in range(N_PAST_SESSIONS) for col in PAST_FEATURE_COLUMNS
] + [f"target.{TARGET_POSTHOC_DISPLAY_NAME} (사후 정보, 러닝 전 예측에는 사용 금지)"]


def load_user_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def build_windows_for_user(user_df):
    samples = []
    n_sessions = len(user_df)
    for start in range(0, n_sessions - N_PAST_SESSIONS):
        past_window = user_df.iloc[start : start + N_PAST_SESSIONS]
        target_row = user_df.iloc[start + N_PAST_SESSIONS]

        past_indices = set(past_window["session_order"].tolist())
        target_index = target_row["session_order"]
        assert target_index not in past_indices
        assert target_index == max(past_indices) + 1

        past_values = past_window[PAST_FEATURE_COLUMNS].to_numpy().flatten()
        target_posthoc_value = np.array([target_row[TARGET_POSTHOC_COLUMN]])
        x_values = np.concatenate([past_values, target_posthoc_value])

        samples.append(
            {
                "user_id": target_row["user_id"],
                "window_start_session_order": start,
                "target_session_order": target_index,
                "X": x_values,
                "y": target_row[TARGET_COLUMN],
            }
        )
    return samples


def build_dataset(df, user_ids):
    subset = df[df["user_id"].isin(user_ids)]
    all_samples = []
    for user_id, user_df in subset.groupby("user_id"):
        all_samples.extend(build_windows_for_user(user_df.sort_values("session_order")))
    return all_samples


def main():
    df = pd.read_pickle(INPUT_PATH)
    train_user_ids = load_user_ids(TRAIN_USERS_PATH)
    test_user_ids = load_user_ids(TEST_USERS_PATH)

    train_samples = build_dataset(df, train_user_ids)
    test_samples = build_dataset(df, test_user_ids)

    pd.to_pickle(train_samples, TRAIN_OUTPUT_PATH)
    pd.to_pickle(test_samples, TEST_OUTPUT_PATH)

    print("=" * 60)
    print("!!! 이 데이터셋은 '사후 거리 조건 분석' 전용이다. 러닝 전 예측 성능으로 쓰지 말 것 !!!")
    print("=" * 60)
    print("X 컬럼 목록 (총 16개, 마지막 1개가 타깃 세션의 사후 정보):")
    for name in X_COLUMN_NAMES:
        print(f"  - {name}")
    target_feature_names = [name for name in X_COLUMN_NAMES if name.startswith("target.")]
    assert len(target_feature_names) == 1
    for forbidden in TARGET_FORBIDDEN_COLUMNS:
        assert forbidden not in target_feature_names[0]
    print(f"  -> 타깃 관련 피처는 {TARGET_POSTHOC_DISPLAY_NAME} 하나뿐이고, "
          f"{TARGET_FORBIDDEN_COLUMNS}는 포함되지 않음 (assert 통과).")
    print("-" * 60)
    print(f"train 샘플 수: {len(train_samples)}, test 샘플 수: {len(test_samples)}")
    print(f"저장 위치: {TRAIN_OUTPUT_PATH.resolve()}, {TEST_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
