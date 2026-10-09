# -*- coding: utf-8 -*-
"""
10월 2주차 7-1단계: 트리 모델용 확장 피처 데이터셋 생성.

04/06단계와 동일한 (과거 5개 세션 -> 6번째 세션) 윈도우 생성/과거 정보
감사(last_input_session_time)/분할 로직을 그대로 재사용하고, X의 앞 15개
컬럼도 04/06과 완전히 같은 순서(past[i].actual_distance_km/avg_pace_sec_per_km/
avg_heart_rate, i=0..4)로 유지한다 — 그래야 21단계의 baseline 계산 함수
(extract_past_paces, predict_all, is_merge_affected 등)를 그대로 재사용할
수 있다.

=== 추가 피처 7개 (CLAUDE.md 7단계 1번 후보 목록, 해석 선택 명시) ===
후보 목록 중 "과거 5회 elapsed 페이스의 중앙값"만 "~의 중앙값"이 명시돼
있고 나머지는 집계 방식이 적혀있지 않았다. 이미 X에 과거 5개의 elapsed
페이스 원본값이 전부 들어있으므로(중복 추가 금지 규칙), elapsed만 중앙값
요약으로 넣는 게 맞다고 해석했다. moving 페이스/정지 비율은 X에 원본이
전혀 없는 새 정보라 "5개 원본 그대로 추가"와 "중앙값 요약만 추가" 둘 다
말이 되는데, 이번 구현은 **전부 중앙값 하나로 요약**하는 쪽을 선택했다
(이유: 과거 윈도우가 5개뿐인 사용자가 많아 차원을 너무 늘리면 과적합
위험이 커지고, elapsed와 다루는 방식을 통일하는 게 일관적이라고 판단함 —
이 해석이 의도와 다르면 알려주면 원본 5개씩 추가하는 버전으로 바꿀 수 있음).

1. median_past5_elapsed_pace: 과거 5개 avg_pace_sec_per_km의 중앙값.
2. median_past5_moving_pace: 과거 5개 moving_pace(정지 구간 제외 이동 페이스)의 중앙값.
3. median_past5_stop_ratio: 과거 5개 gap_ratio(=1-moving/elapsed)의 중앙값.
4. user_history_median_pace: 타깃 시작 시각 **이전 모든 세션**(과거 5개보다
   넓은 전체 이력)의 avg_pace_sec_per_km 중앙값 (기존 baseline 4번 "사용자
   전체 평균"은 평균이었고 이건 중앙값이라 다른 통계치).
5. past5_pace_mad: 과거 5개 elapsed 페이스의 median absolute deviation
   (median(|x_i - median(x)|)) — 변동성 지표.
6. days_since_last_session: (타깃 시작 시각 - 마지막 과거 세션 종료 시각) / 86400.
   타깃 러닝이 "시작되기 전" 앱이 이미 아는 값(현재 시각 - 마지막 기록 시각)이라
   leakage 아님.
7. cumulative_session_count: 타깃 이전 누적 세션 수 (= target_session_order,
   즉 session_order가 0부터 시작하므로 그 값 자체가 "이전 세션 개수"와 같음).

=== 과거 정보 감사 (CLAUDE.md 요구사항, 새 이력 기반 피처에도 적용) ===
user_history_median_pace는 과거 5개보다 넓은 범위(전체 이력)를 쓰므로,
이 피처가 참조한 세션 중 가장 늦은 "종료 시각"(start_time+duration_sec)도
타깃 시작 시각보다 앞서는지 별도로 검증한다(기존 last_input_session_time
감사와 같은 방식, 단 과거 5개 윈도우가 아니라 전체 이력 기준). 위반하는
윈도우는 제외하고 건수를 출력한다.

=== 타깃 ===
y = avg_pace_sec_per_km (elapsed, 메인, 04/06과 동일)
y_moving = moving_pace (민감도 평가 전용, 메인 타깃 아님 — 7단계 4번 항목)
"""

import numpy as np
import pandas as pd

import config

MOVING_PATH = config.MAIN_DATA_PATH.parent / "run_sessions_deduped_merged_with_moving.pkl"

EXT_TRAIN_PATH = config.TRAIN_DATASET_PATH.parent / "extended_train_dataset.pkl"
EXT_TEST_PATH = config.TRAIN_DATASET_PATH.parent / "extended_test_dataset.pkl"
EXT_P_TRAIN_PATH = config.TRAIN_DATASET_PATH.parent / "extended_personalized_train_dataset.pkl"
EXT_P_TEST_PATH = config.TRAIN_DATASET_PATH.parent / "extended_personalized_test_dataset.pkl"

N_PAST = config.N_PAST_SESSIONS
PAST_FEATURE_COLUMNS = ["actual_distance_km", "avg_pace_sec_per_km", "avg_heart_rate"]
TARGET_COLUMN = "avg_pace_sec_per_km"
TARGET_MOVING_COLUMN = "moving_pace"
HISTORY_RATIO = 0.8  # 06단계와 동일한 personalized history/hold-out 비율

NEW_FEATURE_NAMES = [
    "median_past5_elapsed_pace",
    "median_past5_moving_pace",
    "median_past5_stop_ratio",
    "user_history_median_pace",
    "past5_pace_mad",
    "days_since_last_session",
    "cumulative_session_count",
]
X_COLUMN_NAMES = [
    f"past[{i}].{c}" for i in range(N_PAST) for c in PAST_FEATURE_COLUMNS
] + NEW_FEATURE_NAMES


def build_history_median_lookup(full_df):
    """사용자별 전체 이력 기준 (타깃 이전까지의) 중앙값 페이스와, 그 이력이
    참조한 세션 중 가장 늦은 종료 시각을 (user_id, session_order)별로 미리 계산."""
    d = full_df.sort_values(["user_id", "session_order"]).copy()
    d["end_time"] = d["start_time"] + d["duration_sec"]
    d["hist_median_before"] = d.groupby("user_id")["avg_pace_sec_per_km"].transform(
        lambda s: s.expanding().median().shift(1)
    )
    d["hist_max_end_before"] = d.groupby("user_id")["end_time"].transform(
        lambda s: s.expanding().max().shift(1)
    )
    return {
        (r.user_id, r.session_order): (r.hist_median_before, r.hist_max_end_before)
        for r in d.itertuples()
    }


def build_windows_for_user(user_df, history_lookup):
    samples = []
    n_audit_excluded = 0
    n_history_audit_excluded = 0
    n_sessions = len(user_df)
    for start in range(0, n_sessions - N_PAST):
        past_window = user_df.iloc[start : start + N_PAST]
        target_row = user_df.iloc[start + N_PAST]

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

        hist_median, hist_max_end = history_lookup[(target_row["user_id"], target_index)]
        if pd.isna(hist_median) or hist_max_end >= target_start_time:
            n_history_audit_excluded += 1
            continue

        orig_15 = past_window[PAST_FEATURE_COLUMNS].to_numpy().flatten()
        past_elapsed = past_window["avg_pace_sec_per_km"].to_numpy()
        past_moving = past_window["moving_pace"].to_numpy()
        past_stop_ratio = past_window["gap_ratio"].to_numpy()

        median_elapsed = float(np.median(past_elapsed))
        median_moving = float(np.median(past_moving))
        median_stop_ratio = float(np.median(past_stop_ratio))
        mad_elapsed = float(np.median(np.abs(past_elapsed - median_elapsed)))
        days_since_last = (target_start_time - last_input_session_time) / 86400.0
        cumulative_count = float(target_index)

        new_feats = np.array(
            [median_elapsed, median_moving, median_stop_ratio, float(hist_median),
             mad_elapsed, days_since_last, cumulative_count]
        )
        x_values = np.concatenate([orig_15, new_feats])
        assert x_values.shape[0] == len(X_COLUMN_NAMES)

        samples.append(
            {
                "user_id": target_row["user_id"],
                "window_start_session_order": start,
                "target_session_order": target_index,
                "X": x_values,
                "y": target_row[TARGET_COLUMN],
                "y_moving": target_row[TARGET_MOVING_COLUMN],
                "last_input_session_time": last_input_session_time,
                "target_start_time": target_start_time,
                "target_merged_count": int(target_row["merged_count"]),
                "target_actual_distance_km": float(target_row["actual_distance_km"]),
                "past_merged_counts": past_window["merged_count"].astype(int).tolist(),
                "target_session_id": int(target_row["session_id"]),
            }
        )
    return samples, n_audit_excluded, n_history_audit_excluded


def build_dataset(df, user_ids, history_lookup):
    subset = df[df["user_id"].isin(user_ids)]
    all_samples = []
    total_audit = 0
    total_hist_audit = 0
    for user_id, user_df in subset.groupby("user_id"):
        user_df = user_df.sort_values("session_order")
        samples, n1, n2 = build_windows_for_user(user_df, history_lookup)
        all_samples.extend(samples)
        total_audit += n1
        total_hist_audit += n2
    return all_samples, total_audit, total_hist_audit


def split_history_and_holdout(user_windows):
    """06단계와 동일한 personalized history/hold-out 분할 로직."""
    n_windows = len(user_windows)
    n_history = int(n_windows * HISTORY_RATIO)
    n_history = min(max(n_history, 0), n_windows - 1) if n_windows >= 2 else 0
    return user_windows[:n_history], user_windows[n_history:]


def main():
    df = config.load_main_sessions()
    user_sequences, eligible_ids = config.build_eligible_sequences(df)
    print(f"[입력 확인] eligible 사용자 시퀀스: 세션 {len(user_sequences)}건, 사용자 {len(eligible_ids)}명")

    moving_df = pd.read_pickle(MOVING_PATH)[["session_id", "moving_pace", "gap_ratio"]]
    user_sequences = user_sequences.merge(moving_df, on="session_id", how="left", validate="one_to_one")
    n_missing_moving = user_sequences["moving_pace"].isna().sum()
    assert n_missing_moving == 0, f"moving_pace 병합 실패: {n_missing_moving}건 누락"
    print(f"[입력 확인] moving_pace/gap_ratio 병합 완료 (누락 0건, 통과)")

    history_lookup = build_history_median_lookup(user_sequences)

    # ===== cold-start: 03단계 배정 재사용 + eligible 교집합 (04단계와 동일) =====
    train_user_ids, test_user_ids, excluded = config.intersect_train_test_with_eligible(eligible_ids)
    print(f"[재배선] cold-start train/test: train {len(train_user_ids)}명, "
          f"test {len(test_user_ids)}명, 제외 {len(excluded)}명 -> {sorted(excluded)}")

    train_samples, train_audit, train_hist_audit = build_dataset(user_sequences, train_user_ids, history_lookup)
    test_samples, test_audit, test_hist_audit = build_dataset(user_sequences, test_user_ids, history_lookup)
    pd.to_pickle(train_samples, EXT_TRAIN_PATH)
    pd.to_pickle(test_samples, EXT_TEST_PATH)
    print(f"[cold-start] train {len(train_samples)}건(과거정보 감사 제외 {train_audit}건, "
          f"이력 감사 제외 {train_hist_audit}건), test {len(test_samples)}건(제외 {test_audit}건, "
          f"이력 감사 제외 {test_hist_audit}건)")

    # ===== personalized: 06단계와 동일한 history/hold-out 분할 =====
    p_train_samples, p_test_samples = [], []
    total_p_audit, total_p_hist_audit = 0, 0
    for user_id, user_df in user_sequences.groupby("user_id"):
        user_df = user_df.sort_values("session_order")
        windows, n1, n2 = build_windows_for_user(user_df, history_lookup)
        total_p_audit += n1
        total_p_hist_audit += n2
        history, holdout = split_history_and_holdout(windows)
        p_train_samples.extend(history)
        p_test_samples.extend(holdout)
    pd.to_pickle(p_train_samples, EXT_P_TRAIN_PATH)
    pd.to_pickle(p_test_samples, EXT_P_TEST_PATH)
    print(f"[personalized] history {len(p_train_samples)}건, hold-out {len(p_test_samples)}건 "
          f"(과거정보 감사 제외 {total_p_audit}건, 이력 감사 제외 {total_p_hist_audit}건)")

    print("-" * 60)
    print(f"[leakage 체크] X 컬럼 목록 (총 {len(X_COLUMN_NAMES)}개):")
    for name in X_COLUMN_NAMES:
        print(f"  - {name}")
    n_target_cols = sum(1 for n in X_COLUMN_NAMES if n.startswith("target."))
    assert n_target_cols == 0
    print(f"  'target.'으로 시작하는 컬럼 수 = {n_target_cols} (0이어야 정상, 통과)")
    print(f"[leakage 체크] y = '{TARGET_COLUMN}' (메인), y_moving = '{TARGET_MOVING_COLUMN}' (민감도 전용)")

    all_samples = train_samples + test_samples + p_train_samples + p_test_samples
    gaps = [s["target_start_time"] - s["last_input_session_time"] for s in all_samples]
    assert min(gaps) > 0
    print(f"[감사] last_input_session_time < target_start_time 전부 통과 (min gap={min(gaps)}초)")

    sample = train_samples[0]
    print("-" * 60)
    print(f"[샘플 확인] user_id={sample['user_id']}, target_session_order={sample['target_session_order']}")
    for name, val in zip(X_COLUMN_NAMES, sample["X"]):
        print(f"    {name} = {val:.3f}")
    print(f"  y(elapsed) = {sample['y']:.3f}, y_moving = {sample['y_moving']:.3f}")
    print(f"저장 위치: {EXT_TRAIN_PATH.resolve()} 등 4개 파일")


if __name__ == "__main__":
    main()
