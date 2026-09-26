# -*- coding: utf-8 -*-
"""
2단계: 1단계에서 만든 run 세션 표를 사용자별로 시간순 정렬하고,
과거 5개 + 예측 대상 1개 = 최소 6개 세션이 안 나오는 사용자는 제외한다.

CLAUDE.md 규칙: N=5로 고정, 세션이 6개 미만인 사용자는 데이터셋에서 제외.
"""

import io
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

INPUT_PATH = Path("data") / "run_sessions.pkl"
OUTPUT_PATH = Path("data") / "user_sequences.pkl"

MIN_SESSIONS_REQUIRED = 6  # 과거 5개 + 예측 대상 1개


def main():
    df = pd.read_pickle(INPUT_PATH)

    n_users_before = df["user_id"].nunique()

    # 사용자별로 start_time(세션 시작 시각) 기준 시간순 정렬.
    # 같은 사용자의 세션들이 실제로 시간 순서대로 이어지도록 하는 것이 목적이며,
    # 이후 3~4단계의 "과거 5개 → 다음 세션" 구성이 이 순서에 의존한다.
    df = df.sort_values(["user_id", "start_time"]).reset_index(drop=True)

    # 사용자 내에서 몇 번째 세션인지 순번을 매긴다 (0부터 시작).
    # 나중에 슬라이딩 윈도우를 만들 때 "연속된 세션인지"를 이 순번으로 확인한다.
    df["session_order"] = df.groupby("user_id").cumcount()

    # 세션 수 계산 후, 6개 미만인 사용자를 제외.
    session_counts = df.groupby("user_id").size()
    eligible_user_ids = session_counts[session_counts >= MIN_SESSIONS_REQUIRED].index
    df_eligible = df[df["user_id"].isin(eligible_user_ids)].reset_index(drop=True)

    n_users_after = df_eligible["user_id"].nunique()

    df_eligible.to_pickle(OUTPUT_PATH)

    print("=" * 60)
    print(f"필터 전 사용자 수: {n_users_before}")
    print(f"6세션 미만으로 제외된 사용자 수: {n_users_before - n_users_after}")
    print(f"필터 후 사용자 수 (run 세션 {MIN_SESSIONS_REQUIRED}개 이상): {n_users_after}")
    print(f"필터 후 세션 수: {len(df_eligible)}")
    print("-" * 60)

    per_user_counts = df_eligible.groupby("user_id").size()
    print("사용자당 세션 수 분포:")
    print(per_user_counts.describe())
    print("-" * 60)

    # 정렬이 실제로 시간순으로 잘 되었는지 임의의 사용자 하나로 확인.
    sample_user_id = df_eligible["user_id"].iloc[0]
    sample = df_eligible[df_eligible["user_id"] == sample_user_id][
        ["user_id", "session_order", "start_time", "avg_pace_sec_per_km"]
    ]
    print(f"샘플 사용자({sample_user_id})의 시간순 정렬 확인:")
    print(sample.head(8))
    is_sorted = sample["start_time"].is_monotonic_increasing
    print(f"start_time이 순번대로 증가하는가? -> {is_sorted}")
    print("-" * 60)
    print(f"저장 위치: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
