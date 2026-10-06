# -*- coding: utf-8 -*-
"""
4단계 (v3, 10월 2주차 복원): (과거 5개 세션 기록만) -> 6번째 세션 평균 페이스

10월 2주차 교수 피드백 정정: 1주차(v2)에서 "타깃 세션의 거리를 계획값이라
부르며 입력에 넣었다"는 오류가 있었다. 실제로 그 값은 GPS(haversine)로
계산한 **완주 거리**(actual_distance_km)이며, 이는 러닝이 끝나야 확정되는
결과값이다. 이름을 "계획값"으로 바꿔도 예측 시점에 존재하지 않는 정보라는
사실은 변하지 않는다 (자세한 근거는 docs/feature_availability.md 참고 —
원본 파일 전체를 스캔해도 러닝 전에 확보되는 계획/목표 거리 필드는 없음을
확정했다).

그래서 메인 예측 태스크를 다시 "과거 5개 세션 기록만 입력"으로 복원한다.
타깃(6번째) 세션의 거리/시간/심박은 어떤 형태로도 입력에 넣지 않는다.
(v2처럼 타깃 세션의 완주 거리를 입력에 넣는 실험은 삭제하지 않고
10_posthoc_distance_condition.py로 옮겨 "사후 거리 조건 분석"이라는
이름으로만 분리 보존한다 — 러닝 전 예측 성능으로는 보고하지 않음.)

X = 과거 5개 세션 각각의 (actual_distance_km, 평균 페이스, 평균 심박) (15개)
y = 6번째 세션의 avg_pace_sec_per_km (결과값, 숫자 하나)

참고: 내부 DataFrame 컬럼명은 (01단계 이후 쭉 써온) distance_km이지만,
의미상 이 값은 GPS로 계산한 "완주 후 확정되는 실제 거리"이므로
docs/feature_availability.md의 명명을 따라 X 컬럼 설명/출력에서는
actual_distance_km으로 표기한다.
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

# 과거 세션에서 쓰는 피처 (내부 DataFrame 컬럼명 -> 보고용 표시명).
# 과거(이미 끝난) 세션의 값이므로 leakage가 아니다 — 금지되는 것은
# "타깃 세션 자신의" 결과값뿐이다 (아래 main()의 leakage 체크 참고).
PAST_FEATURE_COLUMNS = ["distance_km", "avg_pace_sec_per_km", "avg_heart_rate"]
PAST_FEATURE_DISPLAY_NAMES = {
    "distance_km": "actual_distance_km",
    "avg_pace_sec_per_km": "avg_pace_sec_per_km",
    "avg_heart_rate": "avg_heart_rate",
}

TARGET_COLUMN = "avg_pace_sec_per_km"

# X 벡터(flatten된 numpy array)의 각 위치가 어떤 의미인지 설명하는 컬럼 이름 목록.
# 과거 세션 5개 x 3피처 = 15개. 타깃 세션 관련 값은 전혀 없다 (v2와의 핵심 차이).
X_COLUMN_NAMES = [
    f"past[{i}].{PAST_FEATURE_DISPLAY_NAMES[col]}" for i in range(N_PAST_SESSIONS) for col in PAST_FEATURE_COLUMNS
]


def load_user_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def build_windows_for_user(user_df):
    """
    한 사용자의 시간순 정렬된 세션들(user_df, session_order로 이미 정렬됨)에서
    슬라이딩 윈도우로 (X, y) 샘플들을 만든다.

    세션이 L개면 (L - N_PAST_SESSIONS)개의 샘플이 나온다.

    윈도우 i (i = 0, 1, ..., L-N_PAST_SESSIONS-1):
      - 과거 세션 구간: user_df[i : i+N_PAST_SESSIONS]  (인덱스 i ~ i+4) -> X 전부 여기서만 읽음
      - 타깃 세션:      user_df[i+N_PAST_SESSIONS]       (인덱스 i+5, 딱 하나) -> y만 여기서 읽음,
                        X 쪽에서는 이 행을 전혀 참조하지 않는다.
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

        x_values = past_window[PAST_FEATURE_COLUMNS].to_numpy().flatten()  # shape (15,) — 타깃 세션 값 없음
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
    print("[leakage 체크 1] X에 실제로 들어간 컬럼 목록 (총 15개, 전부 '과거' 세션 값):")
    for name in X_COLUMN_NAMES:
        print(f"  - {name}")
    print(f"[leakage 체크 2] y로 사용한 컬럼: '{TARGET_COLUMN}' (타깃 세션 1개, 숫자 하나)")
    print("-" * 60)
    print("[leakage 체크 3] 타깃(6번째) 세션 값이 X에 하나도 없는지 확인:")
    n_target_columns_in_x = sum(1 for name in X_COLUMN_NAMES if name.startswith("target."))
    assert n_target_columns_in_x == 0, "X에 타깃 세션 관련 컬럼이 포함되어 있습니다! (v2 방식과 섞임)"
    print(f"  X_COLUMN_NAMES 중 'target.'으로 시작하는 항목 수 = {n_target_columns_in_x} (0이어야 정상, 통과)")
    print("  -> X는 build_windows_for_user()에서 past_window(인덱스 i~i+4)만 읽고,")
    print("     target_row(인덱스 i+5)는 y를 뽑을 때만 접근함. 거리를 포함한 어떤 값도")
    print("     타깃 세션에서 X로 흘러들어갈 코드 경로가 없음.")
    print("-" * 60)

    print(f"train 샘플 수: {len(train_samples)} (사용자 {len(train_user_ids)}명)")
    print(f"test 샘플 수: {len(test_samples)} (사용자 {len(test_user_ids)}명)")
    print("-" * 60)

    # 임의의 사용자 1명, 샘플 1개를 실제 숫자로 확인.
    sample = train_samples[0]
    print(f"[샘플 확인] user_id={sample['user_id']}, "
          f"window_start_session_order={sample['window_start_session_order']}, "
          f"target_session_order={sample['target_session_order']}")
    x_reshaped_past = np.array(sample["X"]).reshape(N_PAST_SESSIONS, len(PAST_FEATURE_COLUMNS))
    print("  X - 과거 5개 세션 (actual_distance_km, avg_pace_sec_per_km, avg_heart_rate):")
    for i, row in enumerate(x_reshaped_past):
        print(f"    세션[{i}] actual_distance_km={row[0]:.3f}, avg_pace_sec_per_km={row[1]:.3f}, avg_heart_rate={row[2]:.3f}")
    print(f"  (X 길이 = {len(sample['X'])}, 타깃 세션 값은 전혀 포함되지 않음)")
    print(f"  y (타깃 세션의 avg_pace_sec_per_km, 결과값) = {sample['y']:.3f}")
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
