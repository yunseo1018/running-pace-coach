# -*- coding: utf-8 -*-
"""
중복 제거 + 병합 파이프라인 재구성 및 평가 (CLAUDE.md 10월 2주차 3-2번, 4~7).

=== 4. 중복 제거는 병합보다 "먼저" 적용한다 ===
이유: 중복 기록은 "같은 운동이 두 번 찍힌 것"이고, 병합은 "한 운동이
여러 조각으로 나뉘어 찍힌 것"이다. 서로 다른 현상이라 순서가 섞이면
문제가 생긴다 — 예를 들어 중복 세션 하나가 실제 다음 조각과 시간상
가깝게 끼어 있으면, 병합 로직이 "진짜 조각"과 "중복 사본" 중 하나와
잘못 체인을 만들 수 있다. 따라서 (1) 먼저 중복을 제거해 "같은 운동은
한 번만" 남긴 뒤, (2) 그 정리된 시퀀스에 병합 규칙을 적용해 "쪼개진
조각"만 다시 합친다.

=== 5. 재구성 후 과거 정보 감사 ===
중복 제거 + 병합을 모두 적용한 뒤 last_input_session_time 감사를 다시
돌려, 윈도우 제외 없이 통과하는지 확인한다. 아직 남는 겹침이 있다면
건수와 (14단계처럼) 패턴을 보고한다.

=== 6. user_id 7996903 민감도 확인 ===
제외된 2,700개 중 450건(16.7%)이 몰려 있던 사용자 하나를 포함/제외했을 때
baseline 결과가 얼마나 달라지는지 확인한다 (기본 결과에서는 제외하지 않음 —
이 사용자도 정당한 사용자이고, 중복 제거로 이미 문제가 해결됐으므로
인위적으로 뺄 이유가 없다. 민감도 확인 목적일 뿐).

=== 7. 중복 제거 전/후 baseline 5종 비교 ===
"전" = 08단계가 만든 run_sessions_merged.pkl (병합만 적용, 중복 제거 없음).
"후" = 이 스크립트가 만드는 "중복 제거 -> 병합" 결과.
둘 다 동일한 과거 정보 감사를 적용한 뒤 baseline 5종을 cold-start/
personalized로 평가해 MAE/RMSE/사용자 수/윈도우 수를 비교한다.
"""

import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RUN_SESSIONS_PATH = Path("data") / "run_sessions.pkl"
MERGED_SESSIONS_PATH = Path("data") / "run_sessions_merged.pkl"  # "전": 병합만, 중복제거 없음
DEDUPED_SESSIONS_PATH = Path("data") / "run_sessions_deduped.pkl"  # 16단계 출력 (중복 제거만, 병합 전)
GPS_ENDPOINTS_PATH = Path("data") / "session_gps_endpoints.pkl"
TRAIN_USERS_PATH = Path("data") / "train_user_ids.txt"
TEST_USERS_PATH = Path("data") / "test_user_ids.txt"

DEDUPED_MERGED_OUTPUT = Path("data") / "run_sessions_deduped_merged.pkl"  # "후": 중복제거 -> 병합

N_PAST_SESSIONS = 5
MIN_SESSIONS_REQUIRED = 6
HISTORY_RATIO = 0.8
RANDOM_SEED = 42

# 08_session_merge.py와 동일한 병합 임계값 (사전 고정값, 그대로 재사용).
MAX_GAP_SEC = 180
MAX_GAP_DISTANCE_M = 200
MAX_IMPLIED_SPEED_MPS = 3.0
EARTH_RADIUS_KM = 6371.0088

SENSITIVITY_USER_ID = 7996903  # 14단계에서 제외 윈도우의 16.7%가 몰려있던 사용자


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def should_merge(prev_row, next_row):
    """08_session_merge.py의 should_merge()와 동일 (감사 일관성을 위해 재구현)."""
    gap_sec = next_row["start_time"] - (prev_row["start_time"] + prev_row["duration_sec"])
    if gap_sec < 0 or gap_sec > MAX_GAP_SEC:
        return False
    gap_distance_km = haversine_km(prev_row["end_lat"], prev_row["end_lon"], next_row["start_lat"], next_row["start_lon"])
    gap_distance_m = gap_distance_km * 1000
    if gap_distance_m > MAX_GAP_DISTANCE_M:
        return False
    implied_speed_mps = float("inf") if gap_sec == 0 and gap_distance_m > 0 else (0.0 if gap_sec == 0 else gap_distance_m / gap_sec)
    if implied_speed_mps > MAX_IMPLIED_SPEED_MPS:
        return False
    return True


def merge_chain(chain_rows):
    first, last = chain_rows[0], chain_rows[-1]
    total_distance_km = sum(r["actual_distance_km"] for r in chain_rows)
    merged_duration_sec = (last["start_time"] + last["duration_sec"]) - first["start_time"]
    total_duration_for_hr = sum(r["duration_sec"] for r in chain_rows)
    weighted_hr = sum(r["avg_heart_rate"] * r["duration_sec"] for r in chain_rows) / total_duration_for_hr
    return {
        "user_id": first["user_id"],
        "session_id": first["session_id"],
        "start_time": first["start_time"],
        "actual_distance_km": total_distance_km,
        "duration_sec": merged_duration_sec,
        "avg_pace_sec_per_km": merged_duration_sec / total_distance_km,
        "avg_heart_rate": weighted_hr,
        "gender": first["gender"],
        "merged_count": len(chain_rows),
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
    """과거 정보 감사 통과 윈도우만 생성, 실패 건수도 같이 반환."""
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
        samples.append(
            {
                "user_id": target_row["user_id"],
                "target_session_order": target_row["session_order"],
                "past_paces": past_window["avg_pace_sec_per_km"].to_numpy(),
                "y": target_row["avg_pace_sec_per_km"],
            }
        )
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


BASELINE_NAMES = ["recent_avg", "last_session", "median5", "user_overall_avg", "linear_trend"]


def evaluate_baselines(samples, overall_avg_lookup):
    rows = []
    for s in samples:
        past_paces = s["past_paces"]
        x_idx = np.arange(N_PAST_SESSIONS)
        slope, intercept = np.polyfit(x_idx, past_paces, 1)
        preds = {
            "recent_avg": past_paces.mean(),
            "last_session": past_paces[-1],
            "median5": np.median(past_paces),
            "user_overall_avg": overall_avg_lookup.get((s["user_id"], s["target_session_order"])),
            "linear_trend": slope * N_PAST_SESSIONS + intercept,
        }
        for name in BASELINE_NAMES:
            y_pred = preds[name]
            rows.append({"baseline": name, "user_id": s["user_id"], "abs_error": abs(s["y"] - y_pred), "sq_error": (s["y"] - y_pred) ** 2})
    df = pd.DataFrame(rows)
    out = df.groupby("baseline").agg(n_samples=("abs_error", "size"), mae=("abs_error", "mean"), rmse=("sq_error", lambda c: np.sqrt(c.mean())))
    return out.reindex(BASELINE_NAMES)


def evaluate_pipeline(session_df, train_user_ids, test_user_ids, label, user_filter=None):
    seq_df, eligible = build_eligible_sequences(session_df)
    if user_filter is not None:
        seq_df = seq_df[seq_df["user_id"] != user_filter]
        eligible = {u for u in eligible if u != user_filter}

    overall_avg_lookup = build_user_overall_avg_lookup(seq_df)

    cold_start_users = {u for u in test_user_ids if u in eligible}
    cold_start_samples, cold_excluded = build_cold_start_windows(seq_df, cold_start_users)
    personalized_samples, personalized_excluded = build_personalized_holdout_windows(seq_df)

    print("=" * 70)
    print(f"[{label}] 세션 수: {len(session_df)}, 자격 사용자: {len(eligible)}명")
    print(f"  과거 정보 감사 제외 - cold-start: {cold_excluded}건, personalized: {personalized_excluded}건")

    cold_summary = evaluate_baselines(cold_start_samples, overall_avg_lookup)
    personalized_summary = evaluate_baselines(personalized_samples, overall_avg_lookup)

    print(f"  [cold-start] 샘플 {len(cold_start_samples)}건, 사용자 {len(cold_start_users)}명")
    print(cold_summary.to_string())
    print(f"  [personalized] 샘플 {len(personalized_samples)}건, 사용자 {seq_df['user_id'].nunique()}명")
    print(personalized_summary.to_string())

    return {
        "cold_summary": cold_summary, "personalized_summary": personalized_summary,
        "n_users": len(eligible), "cold_n_samples": len(cold_start_samples),
        "personalized_n_samples": len(personalized_samples),
        "cold_excluded": cold_excluded, "personalized_excluded": personalized_excluded,
    }


def main():
    run_sessions = pd.read_pickle(RUN_SESSIONS_PATH)
    merged_only = pd.read_pickle(MERGED_SESSIONS_PATH)  # "전"
    deduped = pd.read_pickle(DEDUPED_SESSIONS_PATH)
    gps_endpoints = pd.read_pickle(GPS_ENDPOINTS_PATH)
    train_user_ids = load_user_ids(TRAIN_USERS_PATH)
    test_user_ids = load_user_ids(TEST_USERS_PATH)

    # "후": 중복 제거된 세션에 병합 규칙을 새로 적용.
    deduped_with_gps = deduped.merge(gps_endpoints, on="session_id", how="inner")
    deduped_merged = apply_merge(deduped_with_gps)
    deduped_merged.to_pickle(DEDUPED_MERGED_OUTPUT)
    print(f"중복 제거 -> 병합 결과: {len(deduped)}건(중복 제거 후) -> {len(deduped_merged)}건(병합 후) "
          f"-> 저장: {DEDUPED_MERGED_OUTPUT.resolve()}")

    raw_stage = evaluate_pipeline(run_sessions, train_user_ids, test_user_ids, "0. 원본 run (정제 없음)")
    before = evaluate_pipeline(merged_only, train_user_ids, test_user_ids, "중복 제거 전 (병합만)")
    after = evaluate_pipeline(deduped_merged, train_user_ids, test_user_ids, "중복 제거 후 (중복제거->병합)")

    # ===== 6. 단계별 건수표: 원본 run -> dedup 후 -> 병합 후 =====
    dedup_only_stage = evaluate_pipeline(deduped, train_user_ids, test_user_ids, "1. dedup 후 (병합 전)")
    n_chains_merged_only = int((merged_only["merged_count"] > 1).sum())
    n_chains_deduped_merged = int((deduped_merged["merged_count"] > 1).sum())

    print("=" * 70)
    print("[6] 단계별 건수표: 원본 run -> dedup 후 -> 병합 후")
    stage_table = pd.DataFrame(
        [
            {
                "단계": "0. 원본 run", "세션수": len(run_sessions), "병합체인수": 0,
                "사용자수": raw_stage["n_users"],
                "cold-start 윈도우": raw_stage["cold_n_samples"], "cold-start 제외": raw_stage["cold_excluded"],
                "personalized 윈도우": raw_stage["personalized_n_samples"], "personalized 제외": raw_stage["personalized_excluded"],
            },
            {
                "단계": "1. dedup 후(병합 전)", "세션수": len(deduped), "병합체인수": 0,
                "사용자수": dedup_only_stage["n_users"],
                "cold-start 윈도우": dedup_only_stage["cold_n_samples"], "cold-start 제외": dedup_only_stage["cold_excluded"],
                "personalized 윈도우": dedup_only_stage["personalized_n_samples"], "personalized 제외": dedup_only_stage["personalized_excluded"],
            },
            {
                "단계": "2a. 병합만(dedup 없음)", "세션수": len(merged_only), "병합체인수": n_chains_merged_only,
                "사용자수": before["n_users"],
                "cold-start 윈도우": before["cold_n_samples"], "cold-start 제외": before["cold_excluded"],
                "personalized 윈도우": before["personalized_n_samples"], "personalized 제외": before["personalized_excluded"],
            },
            {
                "단계": "2b. dedup->병합 (최종)", "세션수": len(deduped_merged), "병합체인수": n_chains_deduped_merged,
                "사용자수": after["n_users"],
                "cold-start 윈도우": after["cold_n_samples"], "cold-start 제외": after["cold_excluded"],
                "personalized 윈도우": after["personalized_n_samples"], "personalized 제외": after["personalized_excluded"],
            },
        ]
    ).set_index("단계")
    print(stage_table.to_string())

    print("=" * 70)
    print("[7] 최종 비교 (전=병합만 / 후=중복제거->병합):")
    rows = []
    for label, result in [("전", before), ("후", after)]:
        for split_name, summary in [("cold-start", result["cold_summary"]), ("personalized", result["personalized_summary"])]:
            for baseline_name in BASELINE_NAMES:
                r = summary.loc[baseline_name]
                rows.append({
                    "버전": label, "평가방식": split_name, "baseline": baseline_name,
                    "n_samples": int(r["n_samples"]), "mae": r["mae"], "rmse": r["rmse"],
                })
    comparison_df = pd.DataFrame(rows).set_index(["평가방식", "baseline", "버전"]).sort_index()
    print(comparison_df.to_string())

    # ===== 6. user_id 7996903 민감도 확인 (최종 "후" 파이프라인에서) =====
    print("=" * 70)
    print(f"[6] 민감도 확인: user_id={SENSITIVITY_USER_ID} 포함/제외 비교 ('후' 파이프라인, baseline 1만)")
    with_user = evaluate_pipeline(deduped_merged, train_user_ids, test_user_ids, "포함 (전체)")
    without_user = evaluate_pipeline(deduped_merged, train_user_ids, test_user_ids, "제외", user_filter=SENSITIVITY_USER_ID)
    print("-" * 70)
    print(f"recent_avg 기준 비교:")
    print(f"  cold-start: 포함 MAE={with_user['cold_summary'].loc['recent_avg', 'mae']:.3f} / "
          f"제외 MAE={without_user['cold_summary'].loc['recent_avg', 'mae']:.3f}")
    print(f"  personalized: 포함 MAE={with_user['personalized_summary'].loc['recent_avg', 'mae']:.3f} / "
          f"제외 MAE={without_user['personalized_summary'].loc['recent_avg', 'mae']:.3f}")


if __name__ == "__main__":
    main()
