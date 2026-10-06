# -*- coding: utf-8 -*-
"""
6단계 (v3, 10월 2주차 X 정의 동기화): Personalized 평가용 분할 — 사용자 내 시간순 history/temporal hold-out.

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

10월 2주차 수정: X에 타깃 세션의 거리를 포함하던 v2 방식은 leakage였음이
확인되어(docs/feature_availability.md 참고) 04단계와 동일하게 "과거 5개
세션만" 쓰도록 되돌렸다. 타깃 세션 거리를 쓰는 실험은
10_posthoc_distance_condition.py로만 분리 보존한다.

10월 2주차 과거 정보 감사 (추가): 04단계와 동일하게 샘플마다
last_input_session_time/target_start_time을 저장하고
last_input_session_time < target_start_time을 검증한다. 실제로 검증해보니
일부 세션(duration_sec이 비정상적으로 긴 이상 세션)이 다음 세션 시작
이후까지 "안 끝난 것"처럼 겹치는 경우가 있어(전체 윈도우의 약 2.45%),
이런 윈도우는 사용자 확인 후 데이터셋에서 제외하기로 했다 (04단계와 동일한
처리 방식, build_windows_for_user() 참고).

허용 학습 자료 (personalized 조건): personalized_train_dataset.pkl(=각
사용자의 history 구간, 시간순 앞부분 80%)만 "학습 자료"로 허용되고,
personalized_test_dataset.pkl(=뒷부분 temporal hold-out)의 정보는 같은
사용자의 history 쪽으로도 흘러들어가면 안 된다. split_history_and_holdout()이
시간순으로 자른 경계를 그대로 쓰므로, history는 항상 hold-out보다 과거다
(main()의 "시간 역전 없음" 검증 참고).
"""

import io
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

INPUT_PATH = Path("data") / "user_sequences.pkl"
TRAIN_OUTPUT_PATH = Path("data") / "personalized_train_dataset.pkl"
TEST_OUTPUT_PATH = Path("data") / "personalized_test_dataset.pkl"

N_PAST_SESSIONS = 5
PAST_FEATURE_COLUMNS = ["actual_distance_km", "avg_pace_sec_per_km", "avg_heart_rate"]
TARGET_COLUMN = "avg_pace_sec_per_km"

# history(앞부분) : temporal hold-out(뒷부분) 비율. 03단계 cold-start와 같은
# 80/20을 써서 두 평가 결과를 비교하기 쉽게 맞췄다 (CLAUDE.md에 구체적 비율이
# 명시되진 않았지만, 기존 관례를 따르는 게 비교에 유리하다고 판단).
HISTORY_RATIO = 0.8


def build_windows_for_user(user_df):
    """04단계와 동일한 (X, y) 윈도우 생성 로직 (v3: 과거 5개 세션만, 타깃 세션 값은 X에 없음).
    last_input_session_time >= target_start_time인 윈도우는 과거 정보 감사 실패로
    제외한다 (04단계와 동일, 반환값에 제외 건수 포함)."""
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

        x_values = past_window[PAST_FEATURE_COLUMNS].to_numpy().flatten()  # shape (15,)

        samples.append(
            {
                "user_id": target_row["user_id"],
                "window_start_session_order": start,
                "target_session_order": target_index,
                "X": x_values,
                "y": target_row[TARGET_COLUMN],
                "last_input_session_time": last_input_session_time,
                "target_start_time": target_start_time,
            }
        )
    return samples, n_audit_excluded


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
    total_audit_excluded = 0

    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("session_order")
        windows, n_audit_excluded = build_windows_for_user(user_df)
        total_audit_excluded += n_audit_excluded
        # build_windows_for_user는 이미 start(=시간순) 오름차순으로 생성하므로
        # windows 자체가 시간순으로 정렬돼 있다 (일부 윈도우가 감사로 제외돼도
        # 순서 자체는 그대로 유지됨).
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
    print(f"[과거 정보 감사] last_input_session_time >= target_start_time로 제외된 윈도우: "
          f"{total_audit_excluded}건")

    all_samples = personalized_train + personalized_test
    gaps = [s["target_start_time"] - s["last_input_session_time"] for s in all_samples]
    print(f"[과거 정보 감사] 남은 {len(all_samples)}개 샘플 전부에서 "
          f"last_input_session_time < target_start_time 확인 (제외 후 전부 통과).")
    print(f"  갭(target_start_time - last_input_session_time) 초 단위: "
          f"min={min(gaps)}, max={max(gaps)}, mean={sum(gaps) / len(gaps):.1f}")
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
