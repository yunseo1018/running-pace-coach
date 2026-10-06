# -*- coding: utf-8 -*-
"""
세션 병합 전/후 비교를 평가 샘플 (a)(b)(c)로 나눠서 보기 (CLAUDE.md 10월 2주차 3번, 5번 항목).

CLAUDE.md 경고: "병합은 타깃의 거리·시간·표본 수도 바꾸므로 전후 평균
오차만으로 정제 효과를 확정하지 않는다." 그래서 전체 평균만 비교하지 않고,
평가 샘플(윈도우)을 다음 세 그룹으로 나눠 각각 MAE/RMSE/샘플 수를 본다:

(a) 병합과 무관한 샘플: 과거 5개 세션과 타깃 세션 모두 병합에 영향받지 않음
(b) 입력 세션이 병합의 영향을 받은 샘플: 과거 5개 중 적어도 하나가 병합 대상이었음
(c) 타깃 세션 자체가 병합된 샘플: 타깃 세션 자체가 병합(2개 이상 묶임)되었음

"병합 전"(09_merge_ablation.py와 동일한 원본 세션 시퀀스)과 "병합 후"
(08단계가 만든 run_sessions_merged.pkl 기반 시퀀스)에서 각각 윈도우를
만들고, 윈도우를 구성하는 세션들이 chain(merged_session_ids)에 속하는지로
분류한다. 09단계와 마찬가지로 09_merge_ablation.py의 02단계 로직
(6세션 이상 필터)을 그대로 재사용하고, 과거 정보 감사(last_input_session_time)도
04/06단계와 동일하게 적용해 "메인 파이프라인과 같은 기준"으로 평가한다.

주의: "병합 전"과 "병합 후"의 윈도우는 서로 1:1로 대응되지 않는다 (병합이
세션 수/경계를 바꾸므로). 그래서 이 표는 "짝지어진 비교"가 아니라, 각
버전 안에서 "병합의 영향을 받은 샘플"과 "안 받은 샘플"의 오차가 어떻게
다른지를 보여주는 것이 목적이다.
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
HISTORY_RATIO = 0.8


def load_user_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def build_eligible_sequences(session_df):
    df = session_df.sort_values(["user_id", "start_time"]).reset_index(drop=True)
    df["session_order"] = df.groupby("user_id").cumcount()
    counts = df.groupby("user_id").size()
    eligible_ids = set(counts[counts >= MIN_SESSIONS_REQUIRED].index)
    return df[df["user_id"].isin(eligible_ids)].reset_index(drop=True), eligible_ids


def build_windows_with_category(user_df, chain_member_lookup):
    """
    과거 정보 감사(last_input_session_time < target_start_time)를 통과한
    윈도우만 만들고, 각 윈도우를 (a)/(b)/(c)로 분류한다.
    chain_member_lookup: session_id -> True/False (병합 체인(merged_count>1)에
    속한 세션이면 True). "병합 후" 데이터에서는 merged_count 컬럼을 직접 쓰므로
    이 lookup 대신 row["merged_count"] > 1을 쓴다 (호출부에서 분기).
    """
    samples = []
    n = len(user_df)
    for start in range(0, n - N_PAST_SESSIONS):
        past_window = user_df.iloc[start : start + N_PAST_SESSIONS]
        target_row = user_df.iloc[start + N_PAST_SESSIONS]

        last_past_row = past_window.iloc[-1]
        last_input_session_time = last_past_row["start_time"] + last_past_row["duration_sec"]
        target_start_time = target_row["start_time"]
        if last_input_session_time >= target_start_time:
            continue  # 과거 정보 감사 실패 윈도우는 제외 (04/06단계와 동일 기준)

        past_paces = past_window["avg_pace_sec_per_km"].to_numpy()
        y_true = target_row["avg_pace_sec_per_km"]

        target_affected = chain_member_lookup(target_row)
        input_affected = any(chain_member_lookup(r) for _, r in past_window.iterrows())

        if target_affected:
            category = "c_target_merged"
        elif input_affected:
            category = "b_input_merged"
        else:
            category = "a_unaffected"

        samples.append({"user_id": target_row["user_id"], "past_paces": past_paces, "y": y_true, "category": category})
    return samples


def evaluate_by_category(samples):
    rows = []
    for s in samples:
        y_pred = s["past_paces"].mean()
        y_true = s["y"]
        rows.append(
            {
                "category": s["category"],
                "abs_error": abs(y_true - y_pred),
                "sq_error": (y_true - y_pred) ** 2,
            }
        )
    df = pd.DataFrame(rows)
    summary = df.groupby("category").agg(
        n_samples=("abs_error", "size"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda s: np.sqrt(s.mean())),
    )
    return summary.reindex(["a_unaffected", "b_input_merged", "c_target_merged"])


def main():
    run_sessions = pd.read_pickle(RUN_SESSIONS_PATH)
    merged_sessions = pd.read_pickle(MERGED_SESSIONS_PATH)
    train_user_ids = load_user_ids(TRAIN_USERS_PATH)
    test_user_ids = load_user_ids(TEST_USERS_PATH)

    # "병합 전" 쪽에서 쓸 chain 멤버십: merged_session_ids에 들어있던 원본 session_id -> True
    chain_member_ids = set()
    for ids in merged_sessions.loc[merged_sessions["merged_count"] > 1, "merged_session_ids"]:
        chain_member_ids.update(ids)

    before_df, eligible_before = build_eligible_sequences(run_sessions)
    after_df, eligible_after = build_eligible_sequences(merged_sessions)

    results = {}

    for version_name, seq_df, is_merged_version in [("병합 전", before_df, False), ("병합 후", after_df, True)]:
        if is_merged_version:
            lookup = lambda r: r["merged_count"] > 1
        else:
            lookup = lambda r: r["session_id"] in chain_member_ids

        for split_name, user_ids in [("cold-start", test_user_ids), ("personalized", None)]:
            if split_name == "cold-start":
                subset = seq_df[seq_df["user_id"].isin(user_ids)]
                all_samples = []
                for user_id, g in subset.groupby("user_id"):
                    all_samples.extend(build_windows_with_category(g.sort_values("session_order"), lookup))
            else:
                # personalized: 각 사용자 history 80% / hold-out 20%, hold-out만 평가.
                all_samples = []
                for user_id, g in seq_df.groupby("user_id"):
                    windows = build_windows_with_category(g.sort_values("session_order"), lookup)
                    n = len(windows)
                    n_history = min(max(int(n * HISTORY_RATIO), 0), n - 1) if n >= 2 else 0
                    all_samples.extend(windows[n_history:])

            summary = evaluate_by_category(all_samples)
            results[(version_name, split_name)] = summary

    print("=" * 70)
    print("[5] 병합 전/후 x (a)병합무관 (b)입력세션 병합영향 (c)타깃세션 병합됨 비교")
    print("    (주의: 병합 전/후는 1:1 대응이 아님 — 각 버전 내 카테고리 간 비교가 핵심)")
    for (version_name, split_name), summary in results.items():
        print("-" * 70)
        print(f"[{version_name} / {split_name}]")
        print(summary.to_string())


if __name__ == "__main__":
    main()
