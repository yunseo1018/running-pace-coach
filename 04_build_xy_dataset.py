# -*- coding: utf-8 -*-
"""
4단계: (과거 5개 세션 -> 다음(6번째) 세션의 평균 페이스) 형태의 (X, y) 데이터셋을 만든다.

CLAUDE.md 데이터 누수(leakage) 금지 규칙:
6번째(타겟) 세션 자신의 거리/심박/날씨는 절대 X에 들어가면 안 된다.
=> X는 오직 "과거 5개" 세션 구간(윈도우 시작 ~ 윈도우 시작+4)의 값만 사용하고,
   타겟 세션(윈도우 시작+5)의 값은 y를 만들 때만 딱 한 번 읽는다.
   아래 build_windows_for_user()에서 이 경계가 어떻게 지켜지는지 주석 참고.
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
TRAIN_OUTPUT_PATH = Path("data") / "train_dataset.pkl"
TEST_OUTPUT_PATH = Path("data") / "test_dataset.pkl"

N_PAST_SESSIONS = 5  # CLAUDE.md: N=5로 고정

# 입력(X) 피처로 쓰는, "과거 세션 자신의" 컬럼들.
# 주의: 이 목록에 6번째(타겟) 세션에서만 알 수 있는 값(거리/심박/날씨)이
# 들어가면 안 된다 -> 하지만 여기 있는 값들은 "과거 세션"의 값이라 문제 없음.
FEATURE_COLUMNS = ["distance_km", "avg_pace_sec_per_km", "avg_heart_rate"]
TARGET_COLUMN = "avg_pace_sec_per_km"


def load_user_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def build_windows_for_user(user_df):
    """
    한 사용자의 시간순 정렬된 세션들(user_df, session_order로 이미 정렬됨)에서
    슬라이딩 윈도우로 (X, y) 샘플들을 만든다.

    세션이 L개면 (L - N_PAST_SESSIONS)개의 샘플이 나온다.
    예: L=6 -> 1개, L=10 -> 5개.

    윈도우 i (i = 0, 1, ..., L-N_PAST_SESSIONS-1):
      - 과거 세션 구간: user_df[i : i+N_PAST_SESSIONS]      (인덱스 i ~ i+4)
      - 타겟 세션:      user_df[i+N_PAST_SESSIONS]           (인덱스 i+5, 딱 하나)
    => 과거 구간과 타겟 세션의 인덱스 범위가 절대 겹치지 않는다 (leakage 방지 핵심).
    """
    samples = []
    n_sessions = len(user_df)
    for start in range(0, n_sessions - N_PAST_SESSIONS):
        past_window = user_df.iloc[start : start + N_PAST_SESSIONS]
        target_row = user_df.iloc[start + N_PAST_SESSIONS]

        # past_window의 세션 인덱스 집합과 target_row의 세션 인덱스가
        # 겹치지 않는지 코드로도 확인 (leakage 방지 assert).
        past_indices = set(past_window["session_order"].tolist())
        target_index = target_row["session_order"]
        assert target_index not in past_indices, "타겟 세션이 과거 윈도우에 포함되어 있습니다!"
        assert target_index == max(past_indices) + 1, "타겟 세션이 과거 윈도우 바로 다음 세션이 아닙니다!"

        x_values = past_window[FEATURE_COLUMNS].to_numpy().flatten()  # shape (5*3,)
        y_value = target_row[TARGET_COLUMN]

        samples.append(
            {
                "user_id": target_row["user_id"],
                "window_start_session_order": start,
                "target_session_order": target_index,
                "X": x_values,
                "y": y_value,
            }
        )
    return samples


def build_dataset(df, user_ids):
    subset = df[df["user_id"].isin(user_ids)]
    all_samples = []
    for user_id, user_df in subset.groupby("user_id"):
        user_df = user_df.sort_values("session_order")
        all_samples.extend(build_windows_for_user(user_df))
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
    print("[leakage 체크 1] X에 실제로 들어간 컬럼 (과거 5개 세션 각각에 대해 반복):")
    print(f"  {FEATURE_COLUMNS}  x 과거 {N_PAST_SESSIONS}개 세션 = X 길이 {len(FEATURE_COLUMNS) * N_PAST_SESSIONS}")
    print(f"[leakage 체크 2] y로 사용한 컬럼: '{TARGET_COLUMN}' (타겟 세션 1개, 숫자 하나)")
    print("  -> X 피처 목록에 '타겟 세션'의 값이 포함될 방법이 없음:")
    print("     X는 과거 윈도우(session_order가 target보다 작은 5개)에서만 값을 읽고,")
    print("     타겟 세션의 행 자체는 y를 뽑을 때만 접근함 (build_windows_for_user 참고).")
    print("  -> 코드 내 assert로도 매 샘플마다 '타겟 인덱스가 과거 윈도우에 없음'을 검증함 (통과).")
    print("-" * 60)

    print(f"train 샘플 수: {len(train_samples)} (사용자 {len(train_user_ids)}명)")
    print(f"test 샘플 수: {len(test_samples)} (사용자 {len(test_user_ids)}명)")
    print("-" * 60)

    # 임의의 사용자 1명, 샘플 1개를 실제 숫자로 확인.
    sample = train_samples[0]
    print(f"[샘플 확인] user_id={sample['user_id']}, "
          f"window_start_session_order={sample['window_start_session_order']}, "
          f"target_session_order={sample['target_session_order']}")
    print(f"  X (과거 5개 세션 x [distance_km, avg_pace_sec_per_km, avg_heart_rate]):")
    x_reshaped = np.array(sample["X"]).reshape(N_PAST_SESSIONS, len(FEATURE_COLUMNS))
    for i, row in enumerate(x_reshaped):
        print(f"    세션[{i}] distance_km={row[0]:.3f}, avg_pace_sec_per_km={row[1]:.3f}, avg_heart_rate={row[2]:.3f}")
    print(f"  y (타겟 세션의 avg_pace_sec_per_km) = {sample['y']:.3f}")
    print("-" * 60)

    # train/test가 3단계에서 만든 user_id 기준으로 잘 나뉘었는지 확인.
    train_sample_user_ids = {s["user_id"] for s in train_samples}
    test_sample_user_ids = {s["user_id"] for s in test_samples}
    print("[분할 확인] 데이터셋에 실제로 등장한 user_id가 3단계 분할과 일치하는가?")
    print(f"  train 샘플의 user_id가 모두 train_user_ids 안에 있는가? -> "
          f"{train_sample_user_ids.issubset(train_user_ids)}")
    print(f"  test 샘플의 user_id가 모두 test_user_ids 안에 있는가? -> "
          f"{test_sample_user_ids.issubset(test_user_ids)}")
    print(f"  train/test 샘플 user_id 교집합 크기 -> {len(train_sample_user_ids & test_sample_user_ids)} (0이어야 정상)")
    print("-" * 60)
    print(f"저장 위치: {TRAIN_OUTPUT_PATH.resolve()}, {TEST_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
