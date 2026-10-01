# -*- coding: utf-8 -*-
"""
4단계 (v2, 5주차 재정의): (과거 5개 세션 + 6번째 세션의 "계획값") -> 6번째 세션 평균 페이스

5주차 수정 배경: 기존(1~4주차) 버전은 6번째 세션이 어떤 거리를 뛸 건지 모델이
전혀 모르는 상태로 페이스를 맞춰야 했다. 그런데 페이스는 거리에 크게 좌우되므로
(짧고 빠른 인터벌 vs 길고 느린 장거리), target의 변동성이 "체력 변화"가 아니라
"오늘 무슨 운동을 하기로 했는지"에서 나올 수 있다는 지적을 받았다.

그래서 "계획값(러닝 시작 전에 이미 정해져 있는 것)"과 "결과값(러닝이 끝나야
알 수 있는 것)"을 구분한다:
  - 계획값 -> 입력에 넣어도 leakage 아님: 6번째 세션의 거리(오늘 몇 km 뛸지는
    보통 뛰기 전에 정해놓는다)
  - 결과값 -> 입력에 절대 넣으면 안 됨: 6번째 세션의 심박수, duration(소요 시간),
    평균 페이스(이건 애초에 y이므로 당연히 제외)

그 결과 X = [과거 5개 세션 각각의 (거리, 평균 페이스, 평균 심박)] + [6번째 세션의 거리]
         y = 6번째 세션의 평균 페이스
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

# 과거 세션에서 쓰는 피처 (세션 자신의 거리/페이스/심박 - 전부 "그 세션의 결과값"이지만,
# 그 세션은 이미 끝난 과거 세션이므로 leakage가 아니다. 타겟 세션에 대해서만
# 결과값 사용이 금지된다.)
PAST_FEATURE_COLUMNS = ["distance_km", "avg_pace_sec_per_km", "avg_heart_rate"]

# 타겟(6번째) 세션에서 유일하게 입력에 포함하는 값 - "계획값"으로 취급하는 거리.
TARGET_PLAN_COLUMN = "distance_km"

# 타겟 세션에서 입력에 절대 포함하면 안 되는 "결과값" 컬럼들
# (여기 있는 컬럼들이 X 어디에도 안 들어갔는지 코드로 검증한다).
TARGET_FORBIDDEN_COLUMNS = ["avg_heart_rate", "duration_sec", "avg_pace_sec_per_km"]

TARGET_COLUMN = "avg_pace_sec_per_km"

# X 벡터(flatten된 numpy array)의 각 위치가 어떤 의미인지 설명하는 컬럼 이름 목록.
# 과거 세션 5개 x 3피처 = 15개 + 타겟 세션 거리 1개 = 총 16개.
X_COLUMN_NAMES = [
    f"past[{i}].{col}" for i in range(N_PAST_SESSIONS) for col in PAST_FEATURE_COLUMNS
] + [f"target.{TARGET_PLAN_COLUMN}"]


def load_user_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def build_windows_for_user(user_df):
    """
    한 사용자의 시간순 정렬된 세션들(user_df, session_order로 이미 정렬됨)에서
    슬라이딩 윈도우로 (X, y) 샘플들을 만든다.

    세션이 L개면 (L - N_PAST_SESSIONS)개의 샘플이 나온다.

    윈도우 i (i = 0, 1, ..., L-N_PAST_SESSIONS-1):
      - 과거 세션 구간: user_df[i : i+N_PAST_SESSIONS]  (인덱스 i ~ i+4) -> 거리/페이스/심박 전부 사용
      - 타겟 세션:      user_df[i+N_PAST_SESSIONS]       (인덱스 i+5, 딱 하나)
                        -> "거리"만 계획값으로 X에 포함, 나머지(심박/duration/페이스)는 절대 안 읽음
    """
    samples = []
    n_sessions = len(user_df)
    for start in range(0, n_sessions - N_PAST_SESSIONS):
        past_window = user_df.iloc[start : start + N_PAST_SESSIONS]
        target_row = user_df.iloc[start + N_PAST_SESSIONS]

        # leakage 방지 assert: 과거 윈도우와 타겟 세션의 인덱스가 겹치지 않고,
        # 타겟이 과거 윈도우 바로 다음 세션인지 확인.
        past_indices = set(past_window["session_order"].tolist())
        target_index = target_row["session_order"]
        assert target_index not in past_indices, "타겟 세션이 과거 윈도우에 포함되어 있습니다!"
        assert target_index == max(past_indices) + 1, "타겟 세션이 과거 윈도우 바로 다음 세션이 아닙니다!"

        past_values = past_window[PAST_FEATURE_COLUMNS].to_numpy().flatten()  # shape (5*3,)
        target_plan_value = np.array([target_row[TARGET_PLAN_COLUMN]])  # shape (1,)
        x_values = np.concatenate([past_values, target_plan_value])  # shape (16,)

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
    print("[leakage 체크 1] X에 실제로 들어간 컬럼 목록 (총 16개):")
    for name in X_COLUMN_NAMES:
        print(f"  - {name}")
    print(f"[leakage 체크 2] y로 사용한 컬럼: '{TARGET_COLUMN}' (타겟 세션 1개, 숫자 하나)")
    print("-" * 60)
    print("[leakage 체크 3] 타겟(6번째) 세션에서 X에 들어간 것 / 안 들어간 것:")
    print(f"  포함(계획값으로 취급): target.{TARGET_PLAN_COLUMN}")
    print(f"  제외(결과값, 입력에 넣으면 leakage): {TARGET_FORBIDDEN_COLUMNS}")
    # 코드로도 검증: X_COLUMN_NAMES 중 'target.'으로 시작하는 항목이
    # TARGET_PLAN_COLUMN 하나뿐이고, forbidden 컬럼 이름이 전혀 등장하지 않는지 확인.
    target_feature_names = [name for name in X_COLUMN_NAMES if name.startswith("target.")]
    assert target_feature_names == [f"target.{TARGET_PLAN_COLUMN}"], \
        "타겟 세션 피처가 계획값(거리) 하나가 아닙니다!"
    for forbidden in TARGET_FORBIDDEN_COLUMNS:
        assert f"target.{forbidden}" not in X_COLUMN_NAMES, \
            f"금지된 타겟 결과값 '{forbidden}'이 X 컬럼에 포함되어 있습니다!"
    print("  -> 코드 assert로 확인: X 컬럼 중 'target.'으로 시작하는 건 "
          f"target.{TARGET_PLAN_COLUMN} 하나뿐이고, 금지 목록은 전혀 없음 (통과).")
    print("-" * 60)

    print(f"train 샘플 수: {len(train_samples)} (사용자 {len(train_user_ids)}명)")
    print(f"test 샘플 수: {len(test_samples)} (사용자 {len(test_user_ids)}명)")
    print("-" * 60)

    # 임의의 사용자 1명, 샘플 1개를 실제 숫자로 확인.
    sample = train_samples[0]
    print(f"[샘플 확인] user_id={sample['user_id']}, "
          f"window_start_session_order={sample['window_start_session_order']}, "
          f"target_session_order={sample['target_session_order']}")
    x_reshaped_past = np.array(sample["X"][: N_PAST_SESSIONS * len(PAST_FEATURE_COLUMNS)]).reshape(
        N_PAST_SESSIONS, len(PAST_FEATURE_COLUMNS)
    )
    print("  X - 과거 5개 세션 (distance_km, avg_pace_sec_per_km, avg_heart_rate):")
    for i, row in enumerate(x_reshaped_past):
        print(f"    세션[{i}] distance_km={row[0]:.3f}, avg_pace_sec_per_km={row[1]:.3f}, avg_heart_rate={row[2]:.3f}")
    target_distance_in_x = sample["X"][-1]
    print(f"  X - 타겟(6번째) 세션의 거리(계획값으로 포함): target.distance_km = {target_distance_in_x:.3f}")
    print(f"  (참고) X에는 타겟 세션의 avg_heart_rate/duration_sec/avg_pace_sec_per_km가 "
          f"전혀 없음 — 위 '세션[0]~[4]'와 'target.distance_km' 외에 다른 값이 없는 것이 그 증거.")
    print(f"  y (타겟 세션의 avg_pace_sec_per_km, 결과값) = {sample['y']:.3f}")
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
