# -*- coding: utf-8 -*-
"""
세션 병합 ablation: 병합 전/후 baseline(최근 5회 평균) 성능을 cold-start/
personalized 두 방식으로 각각 비교한다.

공정한 비교를 위한 설계:
- 병합은 세션 수를 줄이기 때문에, 병합 후 "6세션 이상" 조건을 만족하는
  사용자 집합(E_after)이 병합 전 집합(E_before)과 달라질 수 있다 (일부
  사용자가 병합으로 세션이 줄어 6개 미만이 되어 탈락할 수 있음).
- "병합 자체의 효과"만 보려면 모집단이 바뀌면 안 되므로, 두 집합의
  교집합(E_common = E_before ∩ E_after)에 속한 사용자만으로 병합 전/후를
  둘 다 다시 평가한다. (E_before/E_after 각각의 전체 탈락 규모는 참고로 같이 보여줌)
- cold-start 분할은 03단계에서 이미 고정한 train/test user_id를 그대로
  재사용하되(새로 셔플하지 않음), E_common으로 교집합만 취한다.
- personalized 분할(history 80%/hold-out 20%)은 E_common 사용자 전체에
  대해 병합 전/후 버전을 각각 새로 만든다.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RUN_SESSIONS_PATH = Path("data") / "run_sessions.pkl"
MERGED_SESSIONS_PATH = Path("data") / "run_sessions_merged.pkl"
TRAIN_USERS_PATH = Path("data") / "train_user_ids.txt"
TEST_USERS_PATH = Path("data") / "test_user_ids.txt"

N_PAST_SESSIONS = 5
MIN_SESSIONS_REQUIRED = 6
PAST_FEATURE_COLUMNS = ["distance_km", "avg_pace_sec_per_km", "avg_heart_rate"]
TARGET_COLUMN = "avg_pace_sec_per_km"
HISTORY_RATIO = 0.8


def load_user_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def build_eligible_sequences(session_df):
    """02단계와 동일: user_id별 시간순 정렬 + session_order 부여 + 6세션 미만 제외."""
    df = session_df.sort_values(["user_id", "start_time"]).reset_index(drop=True)
    df["session_order"] = df.groupby("user_id").cumcount()
    counts = df.groupby("user_id").size()
    eligible_ids = set(counts[counts >= MIN_SESSIONS_REQUIRED].index)
    return df[df["user_id"].isin(eligible_ids)].reset_index(drop=True), eligible_ids


def build_windows_for_user(user_df):
    """04단계의 (과거5 -> 타겟) 윈도우 생성 (이 ablation에서는 baseline 1만 쓰므로
    target.distance_km 등 추가 피처 없이 pace만 쓰는 단순 형태로 구성)."""
    samples = []
    n = len(user_df)
    for start in range(0, n - N_PAST_SESSIONS):
        past_window = user_df.iloc[start : start + N_PAST_SESSIONS]
        target_row = user_df.iloc[start + N_PAST_SESSIONS]
        samples.append(
            {
                "user_id": target_row["user_id"],
                "target_session_order": target_row["session_order"],
                "past_paces": past_window["avg_pace_sec_per_km"].to_numpy(),
                "y": target_row[TARGET_COLUMN],
            }
        )
    return samples


def build_all_windows(df, user_ids):
    subset = df[df["user_id"].isin(user_ids)]
    samples = []
    for user_id, user_df in subset.groupby("user_id"):
        samples.extend(build_windows_for_user(user_df.sort_values("session_order")))
    return samples


def split_personalized(df, user_ids):
    subset = df[df["user_id"].isin(user_ids)]
    history, holdout = [], []
    for user_id, user_df in subset.groupby("user_id"):
        windows = build_windows_for_user(user_df.sort_values("session_order"))
        n = len(windows)
        n_history = min(max(int(n * HISTORY_RATIO), 0), n - 1) if n >= 2 else 0
        history.extend(windows[:n_history])
        holdout.extend(windows[n_history:])
    return history, holdout


def evaluate_recent_avg(samples):
    if not samples:
        return None
    rows = []
    for s in samples:
        y_pred = s["past_paces"].mean()
        y_true = s["y"]
        rows.append({"user_id": s["user_id"], "abs_error": abs(y_true - y_pred), "sq_error": (y_true - y_pred) ** 2})
    result_df = pd.DataFrame(rows)
    per_user_mae = result_df.groupby("user_id")["abs_error"].mean()
    return {
        "n_samples": len(result_df),
        "n_users": result_df["user_id"].nunique(),
        "mae": result_df["abs_error"].mean(),
        "rmse": np.sqrt(result_df["sq_error"].mean()),
        "per_user_mae_mean": per_user_mae.mean(),
        "per_user_mae_median": per_user_mae.median(),
    }


def main():
    raw_sessions = pd.read_pickle(RUN_SESSIONS_PATH)
    merged_sessions = pd.read_pickle(MERGED_SESSIONS_PATH)

    before_df, eligible_before = build_eligible_sequences(raw_sessions)
    after_df, eligible_after = build_eligible_sequences(merged_sessions)

    common_users = eligible_before & eligible_after

    print("=" * 70)
    print(f"병합 전 세션 수: {len(raw_sessions)} / 병합 후 세션 수: {len(merged_sessions)}")
    print(f"6세션 이상 자격 사용자 - 병합 전: {len(eligible_before)}, 병합 후: {len(eligible_after)}")
    print(f"병합으로 인해 6세션 미만이 되어 새로 탈락한 사용자: "
          f"{len(eligible_before - eligible_after)}")
    print(f"공정 비교를 위해 사용하는 공통 사용자(E_common): {len(common_users)}")
    print("-" * 70)

    train_user_ids = load_user_ids(TRAIN_USERS_PATH) & common_users
    test_user_ids = load_user_ids(TEST_USERS_PATH) & common_users
    print(f"cold-start 재평가 대상: train {len(train_user_ids)}명 / test {len(test_user_ids)}명 (공통 사용자로 교집합)")
    print("-" * 70)

    results = {}

    for version_name, seq_df in [("병합 전", before_df), ("병합 후", after_df)]:
        cold_start_samples = build_all_windows(seq_df, test_user_ids)
        results[(version_name, "cold-start")] = evaluate_recent_avg(cold_start_samples)

        _, personalized_holdout = split_personalized(seq_df, common_users)
        results[(version_name, "personalized")] = evaluate_recent_avg(personalized_holdout)

    print("[baseline 1: 최근 5회 평균] 병합 전/후 x cold-start/personalized 비교")
    rows = []
    for (version_name, split_name), r in results.items():
        rows.append(
            {
                "버전": version_name,
                "평가방식": split_name,
                "샘플수": r["n_samples"],
                "사용자수": r["n_users"],
                "MAE": r["mae"],
                "RMSE": r["rmse"],
                "사용자별MAE_mean": r["per_user_mae_mean"],
                "사용자별MAE_median": r["per_user_mae_median"],
            }
        )
    comparison_df = pd.DataFrame(rows).set_index(["평가방식", "버전"]).sort_index()
    print(comparison_df.to_string())


if __name__ == "__main__":
    main()
