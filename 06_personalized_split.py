# -*- coding: utf-8 -*-
"""
6단계 (v2, 5주차): Personalized 평가용 분할 — 사용자 내 시간순 history/temporal hold-out.

CLAUDE.md 배경: 기존 cold-start(user-disjoint 80/20, 03단계) 분할은
"처음 보는 사용자에게 얼마나 잘 일반화되는지"만 측정한다. 이 프로젝트의 실제
목표("한 사용자의 기록이 쌓일수록 코칭이 정교해지는지")를 측정하려면, 같은
사용자 안에서 "과거 기록(history)"과 "그 다음에 온 세션(temporal hold-out)"을
나눠야 한다. 그래서 전체 748명 사용자를 전부 사용하되(= 03단계의 사용자
단위 train/test 분할과는 무관하게), 각 사용자의 (과거5 -> 타겟) 윈도우들을
타겟 세션이 "그 사용자 안에서 몇 번째로 나온 윈도우인지" 기준으로
앞부분 80%는 history(=personalized train), 뒷부분 20%는 temporal
hold-out(=personalized test)으로 나눈다.

중요: 이 분할은 03단계(cold-start user_id 분할)와 독립적이다. 같은 사용자가
cold-start에서는 train(또는 test) 쪽에 있어도, 여기서는 그 사용자의 세션이
history/hold-out으로 다시 나뉜다 — "다른 질문"에 대한 "다른 분할"이기 때문에
의도적으로 이렇게 설계했다.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

INPUT_PATH = Path("data") / "user_sequences.pkl"
TRAIN_OUTPUT_PATH = Path("data") / "personalized_train_dataset.pkl"
TEST_OUTPUT_PATH = Path("data") / "personalized_test_dataset.pkl"

N_PAST_SESSIONS = 5
PAST_FEATURE_COLUMNS = ["distance_km", "avg_pace_sec_per_km", "avg_heart_rate"]
TARGET_PLAN_COLUMN = "distance_km"
TARGET_COLUMN = "avg_pace_sec_per_km"

# history(앞부분) : temporal hold-out(뒷부분) 비율. 03단계 cold-start와 같은
# 80/20을 써서 두 평가 결과를 비교하기 쉽게 맞췄다 (CLAUDE.md에 구체적 비율이
# 명시되진 않았지만, 기존 관례를 따르는 게 비교에 유리하다고 판단).
HISTORY_RATIO = 0.8


def build_windows_for_user(user_df):
    """04단계와 동일한 (X, y) 윈도우 생성 로직 (v2: 타겟 세션 거리를 계획값으로 X에 포함)."""
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
        target_plan_value = np.array([target_row[TARGET_PLAN_COLUMN]])
        x_values = np.concatenate([past_values, target_plan_value])

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


def split_history_and_holdout(user_windows):
    """
    한 사용자의 윈도우 리스트(이미 target_session_order 순 = 시간순)를
    앞부분 history / 뒷부분 temporal hold-out으로 나눈다.
    사용자마다 hold-out이 최소 1개는 생기도록 보장한다 (평가에 쓸 샘플이 있어야 하므로).
    """
    n_windows = len(user_windows)
    n_history = int(n_windows * HISTORY_RATIO)
    # hold-out이 0개가 되지 않도록, 그리고 history가 전부 사라지지도 않도록 보정.
    n_history = min(max(n_history, 0), n_windows - 1) if n_windows >= 2 else 0
    history = user_windows[:n_history]
    holdout = user_windows[n_history:]
    return history, holdout


def main():
    df = pd.read_pickle(INPUT_PATH)

    personalized_train = []
    personalized_test = []

    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("session_order")
        windows = build_windows_for_user(user_df)
        # build_windows_for_user는 이미 start(=시간순) 오름차순으로 생성하므로
        # windows 자체가 시간순으로 정렬돼 있다.
        history, holdout = split_history_and_holdout(windows)
        personalized_train.extend(history)
        personalized_test.extend(holdout)

    pd.to_pickle(personalized_train, TRAIN_OUTPUT_PATH)
    pd.to_pickle(personalized_test, TEST_OUTPUT_PATH)

    print("=" * 60)
    print(f"전체 사용자 수: {df['user_id'].nunique()}")
    print(f"personalized train(history) 샘플 수: {len(personalized_train)}")
    print(f"personalized test(temporal hold-out) 샘플 수: {len(personalized_test)}")
    print("-" * 60)

    # 검증: 같은 사용자 안에서 history의 모든 target_session_order가
    # holdout의 모든 target_session_order보다 작은지 (시간 역전이 없는지) 확인.
    train_by_user = {}
    for s in personalized_train:
        train_by_user.setdefault(s["user_id"], []).append(s["target_session_order"])
    test_by_user = {}
    for s in personalized_test:
        test_by_user.setdefault(s["user_id"], []).append(s["target_session_order"])

    violations = 0
    for user_id, test_indices in test_by_user.items():
        train_indices = train_by_user.get(user_id, [])
        if train_indices and max(train_indices) >= min(test_indices):
            violations += 1
    print(f"[검증] history의 타겟 인덱스가 hold-out의 타겟 인덱스보다 항상 더 이른 사용자 수 위반: {violations} (0이어야 정상)")
    print(f"[검증] hold-out 샘플이 1개 이상 있는 사용자 수: {len(test_by_user)} / 전체 {df['user_id'].nunique()}")
    print("-" * 60)

    # 샘플 사용자 하나로 history/hold-out 경계 확인 (목록이 길 수 있어 개수/범위로 요약).
    sample_user_id = next(iter(test_by_user))
    sample_train_indices = sorted(train_by_user.get(sample_user_id, []))
    sample_test_indices = sorted(test_by_user.get(sample_user_id, []))
    print(f"[샘플 확인] user_id={sample_user_id}")
    if sample_train_indices:
        print(f"  history(train): {len(sample_train_indices)}개, "
              f"target_session_order {sample_train_indices[0]} ~ {sample_train_indices[-1]}")
    else:
        print("  history(train): 0개 (세션이 적어 history 없음)")
    print(f"  hold-out(test): {len(sample_test_indices)}개, "
          f"target_session_order {sample_test_indices[0]} ~ {sample_test_indices[-1]}")
    print("-" * 60)
    print(f"저장 위치: {TRAIN_OUTPUT_PATH.resolve()}, {TEST_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
