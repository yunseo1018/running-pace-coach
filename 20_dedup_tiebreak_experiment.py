# -*- coding: utf-8 -*-
"""
실험: dedup tie-break을 "동률 시 n_points -> duration_sec(긴 쪽) -> session_id"로
바꿨을 때 세션/윈도우/사용자 수와 baseline MAE/RMSE가 정확히 얼마나 바뀌는지 비교.

**프로덕션 파일(data/run_sessions_deduped*.pkl)은 건드리지 않는다** — 이 실험은
별도 경로에 결과를 만들고, 기존(현재 적용 중인) 결과와 비교만 한다.
적용 여부는 사용자가 결정한다.
"""

import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RUN_SESSIONS_PATH = Path("data") / "run_sessions.pkl"
GPS_ENDPOINTS_PATH = Path("data") / "session_gps_endpoints.pkl"
CORR_TABLE_PATH = Path("docs") / "merge_audit" / "dedup_correspondence_table.csv"
TRAIN_USERS_PATH = Path("data") / "train_user_ids.txt"
TEST_USERS_PATH = Path("data") / "test_user_ids.txt"
CURRENT_DEDUPED_MERGED_PATH = Path("data") / "run_sessions_deduped_merged.pkl"

MAX_GAP_SEC = 180
MAX_GAP_DISTANCE_M = 200
MAX_IMPLIED_SPEED_MPS = 3.0
EARTH_RADIUS_KM = 6371.0088

N_PAST_SESSIONS = 5
MIN_SESSIONS_REQUIRED = 6
HISTORY_RATIO = 0.8
BASELINE_NAMES = ["recent_avg", "last_session", "median5", "user_overall_avg", "linear_trend"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def should_merge(prev_row, next_row):
    gap_sec = next_row["start_time"] - (prev_row["start_time"] + prev_row["duration_sec"])
    if gap_sec < 0 or gap_sec > MAX_GAP_SEC:
        return False
    gap_km = haversine_km(prev_row["end_lat"], prev_row["end_lon"], next_row["start_lat"], next_row["start_lon"])
    gap_m = gap_km * 1000
    if gap_m > MAX_GAP_DISTANCE_M:
        return False
    speed = float("inf") if gap_sec == 0 and gap_m > 0 else (0.0 if gap_sec == 0 else gap_m / gap_sec)
    if speed > MAX_IMPLIED_SPEED_MPS:
        return False
    return True


def merge_chain(chain_rows):
    first, last = chain_rows[0], chain_rows[-1]
    total_distance_km = sum(r["actual_distance_km"] for r in chain_rows)
    merged_duration_sec = (last["start_time"] + last["duration_sec"]) - first["start_time"]
    total_duration_for_hr = sum(r["duration_sec"] for r in chain_rows)
    weighted_hr = sum(r["avg_heart_rate"] * r["duration_sec"] for r in chain_rows) / total_duration_for_hr
    return {
        "user_id": first["user_id"], "session_id": first["session_id"], "start_time": first["start_time"],
        "actual_distance_km": total_distance_km, "duration_sec": merged_duration_sec,
        "avg_pace_sec_per_km": merged_duration_sec / total_distance_km, "avg_heart_rate": weighted_hr,
        "gender": first["gender"], "merged_count": len(chain_rows),
    }


def apply_merge(df_with_gps):
    all_rows = []
    for user_id, g in df_with_gps.groupby("user_id"):
        g = g.sort_values("start_time").reset_index(drop=True)
        rows = g.to_dict("records")
        current_chain = [rows[0]]
        for row in rows[1:]:
            if should_merge(current_chain[-1], row):
                current_chain.append(row)
            else:
                all_rows.append(merge_chain(current_chain))
                current_chain = [row]
        all_rows.append(merge_chain(current_chain))
    return pd.DataFrame(all_rows)


def load_user_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def build_eligible_sequences(session_df):
    df = session_df.sort_values(["user_id", "start_time"]).reset_index(drop=True)
    df["session_order"] = df.groupby("user_id").cumcount()
    counts = df.groupby("user_id").size()
    eligible_ids = set(counts[counts >= MIN_SESSIONS_REQUIRED].index)
    return df[df["user_id"].isin(eligible_ids)].reset_index(drop=True), eligible_ids


def build_windows_for_user(user_df):
    samples, n_excluded = [], 0
    n = len(user_df)
    for start in range(0, n - N_PAST_SESSIONS):
        past_window = user_df.iloc[start : start + N_PAST_SESSIONS]
        target_row = user_df.iloc[start + N_PAST_SESSIONS]
        last_past_row = past_window.iloc[-1]
        last_input_session_time = last_past_row["start_time"] + last_past_row["duration_sec"]
        target_start_time = target_row["start_time"]
        if last_input_session_time >= target_start_time:
            n_excluded += 1
            continue
        samples.append({
            "user_id": target_row["user_id"], "target_session_order": target_row["session_order"],
            "past_paces": past_window["avg_pace_sec_per_km"].to_numpy(), "y": target_row["avg_pace_sec_per_km"],
        })
    return samples, n_excluded


def build_cold_start_windows(df, user_ids):
    samples, total_excluded = [], 0
    for user_id, g in df[df["user_id"].isin(user_ids)].groupby("user_id"):
        s, n = build_windows_for_user(g.sort_values("session_order"))
        samples.extend(s)
        total_excluded += n
    return samples, total_excluded


def build_personalized_holdout_windows(df):
    samples, total_excluded = [], 0
    for user_id, g in df.groupby("user_id"):
        windows, n_excluded = build_windows_for_user(g.sort_values("session_order"))
        total_excluded += n_excluded
        n = len(windows)
        n_history = min(max(int(n * HISTORY_RATIO), 0), n - 1) if n >= 2 else 0
        samples.extend(windows[n_history:])
    return samples, total_excluded


def build_user_overall_avg_lookup(df):
    d = df.sort_values(["user_id", "session_order"]).copy()
    d["cum_mean_before"] = d.groupby("user_id")["avg_pace_sec_per_km"].transform(lambda s: s.expanding().mean().shift(1))
    return {(r.user_id, r.session_order): r.cum_mean_before for r in d.itertuples()}


def evaluate_baselines(samples, overall_avg_lookup):
    rows = []
    for s in samples:
        past_paces = s["past_paces"]
        x_idx = np.arange(N_PAST_SESSIONS)
        slope, intercept = np.polyfit(x_idx, past_paces, 1)
        preds = {
            "recent_avg": past_paces.mean(), "last_session": past_paces[-1], "median5": np.median(past_paces),
            "user_overall_avg": overall_avg_lookup.get((s["user_id"], s["target_session_order"])),
            "linear_trend": slope * N_PAST_SESSIONS + intercept,
        }
        for name in BASELINE_NAMES:
            y_pred = preds[name]
            rows.append({"baseline": name, "user_id": s["user_id"], "abs_error": abs(s["y"] - y_pred), "sq_error": (s["y"] - y_pred) ** 2})
    df = pd.DataFrame(rows)
    out = df.groupby("baseline").agg(n_samples=("abs_error", "size"), mae=("abs_error", "mean"), rmse=("sq_error", lambda c: np.sqrt(c.mean())))
    return out.reindex(BASELINE_NAMES)


def evaluate_pipeline(session_df, train_user_ids, test_user_ids, label):
    seq_df, eligible = build_eligible_sequences(session_df)
    overall_avg_lookup = build_user_overall_avg_lookup(seq_df)
    cold_start_users = {u for u in test_user_ids if u in eligible}
    cold_samples, cold_excluded = build_cold_start_windows(seq_df, cold_start_users)
    personalized_samples, personalized_excluded = build_personalized_holdout_windows(seq_df)
    print(f"[{label}] 세션 수: {len(session_df)}, 자격 사용자: {len(eligible)}명")
    print(f"  cold-start: 샘플 {len(cold_samples)}건, 제외 {cold_excluded}건")
    print(f"  personalized: 샘플 {len(personalized_samples)}건, 제외 {personalized_excluded}건")
    cold_summary = evaluate_baselines(cold_samples, overall_avg_lookup)
    personalized_summary = evaluate_baselines(personalized_samples, overall_avg_lookup)
    print(cold_summary.to_string())
    print(personalized_summary.to_string())
    return {
        "n_sessions": len(session_df), "n_users": len(eligible),
        "cold_n": len(cold_samples), "cold_excluded": cold_excluded, "cold_summary": cold_summary,
        "personalized_n": len(personalized_samples), "personalized_excluded": personalized_excluded,
        "personalized_summary": personalized_summary,
    }


def main():
    run_sessions = pd.read_pickle(RUN_SESSIONS_PATH)
    gps_endpoints = pd.read_pickle(GPS_ENDPOINTS_PATH)
    corr = pd.read_csv(CORR_TABLE_PATH)
    duration_lookup = dict(zip(run_sessions["session_id"], run_sessions["duration_sec"]))
    train_user_ids = load_user_ids(TRAIN_USERS_PATH)
    test_user_ids = load_user_ids(TEST_USERS_PATH)

    clusters = corr.groupby("kept_session_id")["removed_session_id"].apply(list).to_dict()

    new_removed_ids = set()
    for old_kept, removed_list in clusters.items():
        members = [old_kept] + removed_list
        members_sorted = sorted(members, key=lambda sid: (-duration_lookup[sid], sid))
        for r in members_sorted[1:]:
            new_removed_ids.add(r)

    old_removed_ids = set(corr["removed_session_id"])
    print(f"기존 제거 건수: {len(old_removed_ids)}, 새 규칙 제거 건수: {len(new_removed_ids)}")
    print(f"제거 대상 자체가 달라진 세션 수(대칭차): {len(old_removed_ids ^ new_removed_ids)}")

    new_deduped = run_sessions[~run_sessions["session_id"].isin(new_removed_ids)].reset_index(drop=True)
    new_deduped_with_gps = new_deduped.merge(gps_endpoints, on="session_id", how="inner")
    new_deduped_merged = apply_merge(new_deduped_with_gps)

    print("=" * 70)
    current = pd.read_pickle(CURRENT_DEDUPED_MERGED_PATH)
    current_result = evaluate_pipeline(current, train_user_ids, test_user_ids, "기존(동률시 start_time 빠른 쪽)")
    print("=" * 70)
    new_result = evaluate_pipeline(new_deduped_merged, train_user_ids, test_user_ids, "신규(동률시 duration 긴 쪽)")

    print("=" * 70)
    print("[최종 비교] 기존 vs 신규 tie-break")
    print(f"세션 수: {current_result['n_sessions']} -> {new_result['n_sessions']} "
          f"(변화 {new_result['n_sessions'] - current_result['n_sessions']})")
    print(f"자격 사용자 수: {current_result['n_users']} -> {new_result['n_users']} "
          f"(변화 {new_result['n_users'] - current_result['n_users']})")
    print(f"cold-start 샘플 수: {current_result['cold_n']} -> {new_result['cold_n']} "
          f"(변화 {new_result['cold_n'] - current_result['cold_n']})")
    print(f"cold-start 제외: {current_result['cold_excluded']} -> {new_result['cold_excluded']}")
    print(f"personalized 샘플 수: {current_result['personalized_n']} -> {new_result['personalized_n']} "
          f"(변화 {new_result['personalized_n'] - current_result['personalized_n']})")
    print(f"personalized 제외: {current_result['personalized_excluded']} -> {new_result['personalized_excluded']}")

    for split_name, cur_s, new_s in [("cold-start", current_result["cold_summary"], new_result["cold_summary"]),
                                       ("personalized", current_result["personalized_summary"], new_result["personalized_summary"])]:
        print(f"\n--- {split_name} MAE/RMSE 변화 ---")
        diff_df = pd.DataFrame({
            "mae_기존": cur_s["mae"], "mae_신규": new_s["mae"], "mae_변화": new_s["mae"] - cur_s["mae"],
            "rmse_기존": cur_s["rmse"], "rmse_신규": new_s["rmse"], "rmse_변화": new_s["rmse"] - cur_s["rmse"],
        })
        print(diff_df.to_string())


if __name__ == "__main__":
    main()
