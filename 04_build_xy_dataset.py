# -*- coding: utf-8 -*-
"""
4단계 (v4, 10월 2주차 5번 재배선): 메인 데이터(중복제거->병합 완료) 기준으로
(과거 5개 세션 기록만) -> 6번째 세션 평균 페이스 데이터셋을 다시 만든다.

재배선: INPUT_PATH를 config.USER_SEQUENCES_PATH로 쓴다 (02단계가 이제
메인 데이터를 기준으로 만든 user_sequences.pkl). train/test 사용자는
03단계가 748명 기준으로 저장해둔 train_user_ids.txt/test_user_ids.txt를
**재분할하지 않고 그대로 재사용**하되, 메인 데이터의 새 eligible
집합(742명)과 교집합해 탈락한 사용자만 제외한다 (제외된 사용자 id를
명시적으로 출력).

X = 과거 5개 세션 각각의 (actual_distance_km, 평균 페이스, 평균 심박) (15개)
y = 6번째 세션의 avg_pace_sec_per_km (결과값, 숫자 하나)
그 외 메타데이터(분석용, X에는 안 들어감): target_merged_count,
target_actual_distance_km(사후 분석 전용), past_merged_counts.
"""

import numpy as np
import pandas as pd

import config

INPUT_PATH = config.USER_SEQUENCES_PATH
TRAIN_OUTPUT_PATH = config.TRAIN_DATASET_PATH
TEST_OUTPUT_PATH = config.TEST_DATASET_PATH

N_PAST_SESSIONS = config.N_PAST_SESSIONS

PAST_FEATURE_COLUMNS = ["actual_distance_km", "avg_pace_sec_per_km", "avg_heart_rate"]
TARGET_COLUMN = "avg_pace_sec_per_km"

X_COLUMN_NAMES = [
    f"past[{i}].{col}" for i in range(N_PAST_SESSIONS) for col in PAST_FEATURE_COLUMNS
]


def build_windows_for_user(user_df):
    """과거 정보 감사 통과 윈도우만 생성. (X,y) 외에 분석용 메타데이터도 같이 담는다."""
    samples = []
    n_audit_excluded = 0
    n_sessions = len(user_df)
    for start in range(0, n_sessions - N_PAST_SESSIONS):
        past_window = user_df.iloc[start : start + N_PAST_SESSIONS]
        target_row = user_df.iloc[start + N_PAST_SESSIONS]

        past_indices = set(past_window["session_order"].tolist())
        target_index = target_row["session_order"]
        assert target_index not in past_indices, "타깃 세션이 과거 윈도우에 포함되어 있습니다!"
        assert target_index == max(past_indices) + 1, "타깃 세션이 과거 윈도우 바로 다음 세션이 아닙니다!"

        last_past_row = past_window.iloc[-1]
        last_input_session_time = last_past_row["start_time"] + last_past_row["duration_sec"]
        target_start_time = target_row["start_time"]
        if last_input_session_time >= target_start_time:
            n_audit_excluded += 1
            continue

        x_values = past_window[PAST_FEATURE_COLUMNS].to_numpy().flatten()  # shape (15,) — 타깃 세션 값 없음
        y_value = target_row[TARGET_COLUMN]

        samples.append(
            {
                "user_id": target_row["user_id"],
                "window_start_session_order": start,
                "target_session_order": target_index,
                "X": x_values,
                "y": y_value,
                "last_input_session_time": last_input_session_time,
                "target_start_time": target_start_time,
                # 분석용 메타데이터 (X에는 포함되지 않음, leakage 아님 — 평가/분해 전용)
                "target_merged_count": int(target_row["merged_count"]),
                "target_actual_distance_km": float(target_row["actual_distance_km"]),  # 사후 분석 전용
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
        user_df = user_df.sort_values("session_order")
        samples, n_audit_excluded = build_windows_for_user(user_df)
        all_samples.extend(samples)
        total_audit_excluded += n_audit_excluded
    return all_samples, total_audit_excluded


def main():
    df = pd.read_pickle(INPUT_PATH)
    print(f"[입력 확인] user_sequences: {INPUT_PATH} (세션 {len(df)}건, 사용자 {df['user_id'].nunique()}명)")

    eligible_ids = set(df["user_id"].unique())
    train_user_ids, test_user_ids, excluded = config.intersect_train_test_with_eligible(eligible_ids)
    print(f"[재배선] 기존 train/test 배정(748명 기준)과 새 eligible({len(eligible_ids)}명) 교집합:")
    print(f"  train: {len(train_user_ids)}명, test: {len(test_user_ids)}명, "
          f"제외된 사용자 {len(excluded)}명 -> {sorted(excluded)}")
    print("-" * 60)

    train_samples, train_audit_excluded = build_dataset(df, train_user_ids)
    test_samples, test_audit_excluded = build_dataset(df, test_user_ids)

    pd.to_pickle(train_samples, TRAIN_OUTPUT_PATH)
    pd.to_pickle(test_samples, TEST_OUTPUT_PATH)

    print("[과거 정보 감사] last_input_session_time >= target_start_time인 윈도우를 제외함:")
    print(f"  train에서 제외: {train_audit_excluded}건, test에서 제외: {test_audit_excluded}건")
    print("-" * 60)
    print("[leakage 체크 1] X에 실제로 들어간 컬럼 목록 (총 15개, 전부 '과거' 세션 값):")
    for name in X_COLUMN_NAMES:
        print(f"  - {name}")
    print(f"[leakage 체크 2] y로 사용한 컬럼: '{TARGET_COLUMN}' (타깃 세션 1개, 숫자 하나)")
    n_target_columns_in_x = sum(1 for name in X_COLUMN_NAMES if name.startswith("target."))
    assert n_target_columns_in_x == 0, "X에 타깃 세션 관련 컬럼이 포함되어 있습니다!"
    print(f"  X_COLUMN_NAMES 중 'target.'으로 시작하는 항목 수 = {n_target_columns_in_x} (0이어야 정상, 통과)")
    print("-" * 60)

    print(f"train 샘플 수: {len(train_samples)} (사용자 {len(train_user_ids)}명)")
    print(f"test 샘플 수: {len(test_samples)} (사용자 {len(test_user_ids)}명)")
    print("-" * 60)

    all_samples = train_samples + test_samples
    gaps = [s["target_start_time"] - s["last_input_session_time"] for s in all_samples]
    print(f"[과거 정보 감사] 전체 {len(all_samples)}개 샘플에서 "
          f"last_input_session_time < target_start_time 검증 (assert 전부 통과).")
    print(f"  갭 초 단위: min={min(gaps)}, max={max(gaps)}, mean={sum(gaps) / len(gaps):.1f}")
    print("-" * 60)

    sample = train_samples[0]
    x_reshaped_past = np.array(sample["X"]).reshape(N_PAST_SESSIONS, len(PAST_FEATURE_COLUMNS))
    print(f"[샘플 확인] user_id={sample['user_id']}, target_session_order={sample['target_session_order']}")
    for i, row in enumerate(x_reshaped_past):
        print(f"    세션[{i}] actual_distance_km={row[0]:.3f}, avg_pace_sec_per_km={row[1]:.3f}, avg_heart_rate={row[2]:.3f}")
    print(f"  y = {sample['y']:.3f}, target_merged_count={sample['target_merged_count']}, "
          f"past_merged_counts={sample['past_merged_counts']}")
    print("-" * 60)

    train_sample_user_ids = {s["user_id"] for s in train_samples}
    test_sample_user_ids = {s["user_id"] for s in test_samples}
    print(f"[분할 확인] train/test 샘플 user_id 교집합 크기 -> "
          f"{len(train_sample_user_ids & test_sample_user_ids)} (0이어야 정상)")
    print(f"저장 위치: {TRAIN_OUTPUT_PATH.resolve()}, {TEST_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
