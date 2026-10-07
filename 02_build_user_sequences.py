# -*- coding: utf-8 -*-
"""
2단계 (v3, 10월 2주차 5번 재배선): 메인 데이터(중복제거->병합 완료)를
사용자별로 시간순 정렬하고, 과거 5개 + 예측 대상 1개 = 최소 6개 세션이
안 나오는 사용자는 제외한다.

CLAUDE.md 규칙: N=5로 고정, 세션이 6개 미만인 사용자는 데이터셋에서 제외.

재배선: INPUT_PATH를 하드코딩하지 않고 config.py의 MAIN_DATA_PATH를
그대로 쓴다 (01단계 run_sessions.pkl이 아니라 중복제거->병합이 끝난
run_sessions_deduped_merged.pkl). config.load_main_sessions()가 파일명/
세션수/사용자수를 출력하고 세션 수가 105,709가 아니면 즉시 중단한다.
"""

from pathlib import Path

import config

OUTPUT_PATH = config.USER_SEQUENCES_PATH


def main():
    df = config.load_main_sessions()

    n_users_before = df["user_id"].nunique()

    df_eligible, eligible_ids = config.build_eligible_sequences(df)

    n_users_after = len(eligible_ids)

    df_eligible.to_pickle(OUTPUT_PATH)

    print("=" * 60)
    print(f"필터 전 사용자 수: {n_users_before}")
    print(f"6세션 미만으로 제외된 사용자 수: {n_users_before - n_users_after}")
    print(f"필터 후 사용자 수 (run 세션 {config.MIN_SESSIONS_REQUIRED}개 이상): {n_users_after}")
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
