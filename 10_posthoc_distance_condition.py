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

10월 2주차 5번 재배선: config.MAIN_DATA_PATH 기준 user_sequences.pkl을
읽고, 04단계와 동일한 과거 정보 감사(last_input_session_time 검증,
위반 윈도우 제외) + 기존 train/test 배정과 새 eligible 집합의 교집합을
적용해 04/06과 일관되게 맞췄다. 06단계처럼 분석용 메타데이터
(target_merged_count 등)도 같이 담는다.
"""

import numpy as np
import pandas as pd

import config

INPUT_PATH = config.USER_SEQUENCES_PATH
TRAIN_OUTPUT_PATH = config.PERSONALIZED_TRAIN_DATASET_PATH.parent / "posthoc_train_dataset.pkl"
TEST_OUTPUT_PATH = config.PERSONALIZED_TRAIN_DATASET_PATH.parent / "posthoc_test_dataset.pkl"

N_PAST_SESSIONS = config.N_PAST_SESSIONS
PAST_FEATURE_COLUMNS = ["actual_distance_km", "avg_pace_sec_per_km", "avg_heart_rate"]

# 타깃(6번째) 세션에서 "사후 정보"로서만 X에 포함하는 값 — actual_distance_km
# (예전 v2에서는 "계획값"이라 불렀으나, 실제로는 완주 후 확정되는 결과값이다).
TARGET_POSTHOC_COLUMN = "actual_distance_km"

TARGET_FORBIDDEN_COLUMNS = ["avg_heart_rate", "duration_sec", "avg_pace_sec_per_km"]
TARGET_COLUMN = "avg_pace_sec_per_km"

X_COLUMN_NAMES = [
    f"past[{i}].{col}" for i in range(N_PAST_SESSIONS) for col in PAST_FEATURE_COLUMNS
] + ["target.actual_distance_km (사후 정보, 러닝 전 예측에는 사용 금지)"]


def build_windows_for_user(user_df):
    """04단계와 동일한 과거 정보 감사(last_input_session_time 검증, 위반 시 제외)를
    적용한 뒤, 타깃 세션의 actual_distance_km을 '사후 정보'로 X 끝에 추가한다."""
    samples = []
    n_audit_excluded = 0
    n_sessions = len(user_df)
    for start in range(0, n_sessions - N_PAST_SESSIONS):
        past_window = user_df.iloc[start : start + N_PAST_SESSIONS]
        target_row = user_df.iloc[start + N_PAST_SESSIONS]

        past_indices = set(past_window["session_order"].tolist())
        target_index = target_row["session_order"]
        assert target_index not in past_indices
        assert target_index == max(past_indices) + 1

        last_past_row = past_window.iloc[-1]
        last_input_session_time = last_past_row["start_time"] + last_past_row["duration_sec"]
        target_start_time = target_row["start_time"]
        if last_input_session_time >= target_start_time:
            n_audit_excluded += 1
            continue

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
                "last_input_session_time": last_input_session_time,
                "target_start_time": target_start_time,
                "target_merged_count": int(target_row["merged_count"]),
                "target_actual_distance_km": float(target_row["actual_distance_km"]),
                "past_merged_counts": past_window["merged_count"].astype(int).tolist(),
                "target_session_id": int(target_row["session_id"]),
            }
        )
    return samples, n_audit_excluded


def build_dataset(df, user_ids):
    subset = df[df["user_id"].isin(user_ids)]
    all_samples = []
    total_audit_excluded = 0
    for user_id, user_df in subset.groupby("user_id"):
        samples, n_excluded = build_windows_for_user(user_df.sort_values("session_order"))
        all_samples.extend(samples)
        total_audit_excluded += n_excluded
    return all_samples, total_audit_excluded


def main():
    df = pd.read_pickle(INPUT_PATH)
    print(f"[입력 확인] user_sequences: {INPUT_PATH} (세션 {len(df)}건, 사용자 {df['user_id'].nunique()}명)")

    eligible_ids = set(df["user_id"].unique())
    train_user_ids, test_user_ids, excluded = config.intersect_train_test_with_eligible(eligible_ids)
    print(f"[재배선] 기존 train/test 배정과 새 eligible({len(eligible_ids)}명) 교집합: "
          f"train {len(train_user_ids)}명, test {len(test_user_ids)}명, 제외 {len(excluded)}명 -> {sorted(excluded)}")

    train_samples, train_audit_excluded = build_dataset(df, train_user_ids)
    test_samples, test_audit_excluded = build_dataset(df, test_user_ids)

    pd.to_pickle(train_samples, TRAIN_OUTPUT_PATH)
    pd.to_pickle(test_samples, TEST_OUTPUT_PATH)

    print("=" * 60)
    print("!!! 이 데이터셋은 '사후 거리 조건 분석' 전용이다. 러닝 전 예측 성능으로 쓰지 말 것 !!!")
    print("=" * 60)
    print(f"[과거 정보 감사] train에서 제외: {train_audit_excluded}건, test에서 제외: {test_audit_excluded}건")
    print("X 컬럼 목록 (총 16개, 마지막 1개가 타깃 세션의 사후 정보):")
    for name in X_COLUMN_NAMES:
        print(f"  - {name}")
    target_feature_names = [name for name in X_COLUMN_NAMES if name.startswith("target.")]
    assert len(target_feature_names) == 1
    for forbidden in TARGET_FORBIDDEN_COLUMNS:
        assert forbidden not in target_feature_names[0]
    print(f"  -> 타깃 관련 피처는 actual_distance_km 하나뿐이고, {TARGET_FORBIDDEN_COLUMNS}는 포함되지 않음 (assert 통과).")
    print("-" * 60)
    print(f"train 샘플 수: {len(train_samples)}, test 샘플 수: {len(test_samples)}")
    print(f"저장 위치: {TRAIN_OUTPUT_PATH.resolve()}, {TEST_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
