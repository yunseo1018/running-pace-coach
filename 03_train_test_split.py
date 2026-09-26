# -*- coding: utf-8 -*-
"""
3단계: 사용자 단위로 train/test를 80/20으로 분할한다.

CLAUDE.md 규칙: 같은 사용자의 세션이 train/test 양쪽에 걸치면 안 됨.
그래서 세션 단위가 아니라 user_id 단위로 셔플/분할한다.
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

TRAIN_RATIO = 0.8
# 재현성을 위한 고정 시드. CLAUDE.md에 명시되진 않았지만, 같은 결과를
# 재현할 수 있어야 보고서/실험 비교가 의미 있어서 고정값으로 둔다.
RANDOM_SEED = 42


def main():
    df = pd.read_pickle(INPUT_PATH)

    user_ids = df["user_id"].unique()
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(user_ids)

    n_train = int(len(shuffled) * TRAIN_RATIO)
    train_user_ids = shuffled[:n_train]
    test_user_ids = shuffled[n_train:]

    # 두 집합이 겹치지 않는지, 그리고 합쳤을 때 전체 사용자를 다 포함하는지 검증.
    train_set = set(train_user_ids)
    test_set = set(test_user_ids)
    assert train_set.isdisjoint(test_set), "train/test 사용자 집합이 겹칩니다!"
    assert train_set | test_set == set(user_ids), "일부 사용자가 분할에서 빠졌습니다!"

    with open(TRAIN_USERS_PATH, "w", encoding="utf-8") as f:
        f.writelines(f"{uid}\n" for uid in train_user_ids)
    with open(TEST_USERS_PATH, "w", encoding="utf-8") as f:
        f.writelines(f"{uid}\n" for uid in test_user_ids)

    train_session_count = df[df["user_id"].isin(train_set)].shape[0]
    test_session_count = df[df["user_id"].isin(test_set)].shape[0]

    print("=" * 60)
    print(f"전체 사용자 수: {len(user_ids)}")
    print(f"train 사용자 수: {len(train_user_ids)} ({len(train_user_ids) / len(user_ids):.1%})")
    print(f"test 사용자 수: {len(test_user_ids)} ({len(test_user_ids) / len(user_ids):.1%})")
    print("-" * 60)
    print(f"train 세션 수: {train_session_count}")
    print(f"test 세션 수: {test_session_count}")
    print("-" * 60)
    print(f"train/test 사용자 교집합 크기: {len(train_set & test_set)} (0이어야 정상)")
    print(f"저장 위치: {TRAIN_USERS_PATH.resolve()}, {TEST_USERS_PATH.resolve()}")


if __name__ == "__main__":
    main()
