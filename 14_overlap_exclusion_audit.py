# -*- coding: utf-8 -*-
"""
시간 겹침으로 제외된 2,700개 윈도우의 원인 분류 (CLAUDE.md 10월 2주차 3번 보완).

04/06단계에서 last_input_session_time >= target_start_time인 윈도우를
train/test/personalized 데이터셋에서 제외했다. 이 스크립트는 그 2,700개를
다시 찾아내 더 자세히 들여다본다:
- 겹침 길이 분포
- 겹치는 쪽(마지막 과거 세션)의 소요 시간/거리
- "같은 운동의 중복 기록"으로 보이는지 — 과거 세션의 시작 시각/좌표와
  타깃 세션의 시작 시각/좌표가 거의 같은지 확인
- 제외된 윈도우가 특정 사용자에 몰려 있는지

원인은 확인된 것만 "확인됨"으로 쓰고, 패턴에 기반한 추정은 "추정"으로,
근거가 불충분하면 "미확인"으로 명확히 구분해서 표기한다.
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

OUTPUT_DIR = Path("docs") / "merge_audit"

N_PAST_SESSIONS = 5
MIN_SESSIONS_REQUIRED = 6
RANDOM_SEED = 42
EARTH_RADIUS_KM = 6371.0088

# "거의 동시에 시작" / "거의 같은 장소"로 보는 임계값. 이 자체는 패턴 분류를
# 위한 보조 기준일 뿐이며, 이 값 이하라고 해서 "확인됨"이라 단정하지 않고
# "추정"으로만 표기한다 (원본 데이터에 실측 근거(예: 중복 업로드 로그)가
# 없어 완전히 확인할 방법이 없기 때문).
NEAR_SIMULTANEOUS_START_SEC = 60
NEAR_SAME_LOCATION_M = 50


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def classify_pattern(start_time_diff_sec, start_distance_m, overlap_sec, last_duration_sec):
    """패턴 기반 분류 (추정일 뿐, 확정 아님)."""
    if start_time_diff_sec <= NEAR_SIMULTANEOUS_START_SEC and start_distance_m <= NEAR_SAME_LOCATION_M:
        return "추정: 같은 운동의 중복 기록 (시작 시각·위치가 거의 동일)"
    if overlap_sec > 0.5 * last_duration_sec:
        return "추정: 과거 세션의 duration이 비정상적으로 길어 다음 세션 시작을 덮음"
    return "미확인"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_sessions = pd.read_pickle(RUN_SESSIONS_PATH)
    gps_endpoints = pd.read_pickle(GPS_ENDPOINTS_PATH)
    df = run_sessions.merge(gps_endpoints, on="session_id", how="inner")

    df = df.sort_values(["user_id", "start_time"]).reset_index(drop=True)
    df["session_order"] = df.groupby("user_id").cumcount()
    counts = df.groupby("user_id").size()
    eligible_ids = set(counts[counts >= MIN_SESSIONS_REQUIRED].index)
    df = df[df["user_id"].isin(eligible_ids)].reset_index(drop=True)

    excluded_rows = []
    for user_id, g in df.groupby("user_id"):
        g = g.sort_values("session_order").reset_index(drop=True)
        n = len(g)
        for start in range(0, n - N_PAST_SESSIONS):
            last_past = g.iloc[start + N_PAST_SESSIONS - 1]
            target = g.iloc[start + N_PAST_SESSIONS]
            last_input_session_time = last_past["start_time"] + last_past["duration_sec"]
            target_start_time = target["start_time"]
            if last_input_session_time >= target_start_time:
                start_time_diff_sec = target_start_time - last_past["start_time"]
                start_distance_m = (
                    haversine_km(
                        last_past["start_lat"], last_past["start_lon"],
                        target["start_lat"], target["start_lon"],
                    )
                    * 1000
                )
                overlap_sec = last_input_session_time - target_start_time
                excluded_rows.append(
                    {
                        "user_id": user_id,
                        "last_past_session_id": last_past["session_id"],
                        "target_session_id": target["session_id"],
                        "overlap_sec": overlap_sec,
                        "last_past_distance_km": last_past["actual_distance_km"],
                        "last_past_duration_sec": last_past["duration_sec"],
                        "target_distance_km": target["actual_distance_km"],
                        "target_duration_sec": target["duration_sec"],
                        "start_time_diff_sec": start_time_diff_sec,
                        "start_distance_m": start_distance_m,
                        "pattern": classify_pattern(
                            start_time_diff_sec, start_distance_m, overlap_sec, last_past["duration_sec"]
                        ),
                    }
                )

    excluded_df = pd.DataFrame(excluded_rows)
    excluded_path = OUTPUT_DIR / "excluded_overlap_windows.csv"
    excluded_df.to_csv(excluded_path, index=False)

    print("=" * 70)
    print(f"[6] 시간 겹침으로 제외된 윈도우: {len(excluded_df)}건 (04/06단계 제외 건수와 일치해야 함)")
    print(f"    전체 결과 -> {excluded_path}")
    print("-" * 70)
    print("겹침 길이(overlap_sec) 분포:")
    print(excluded_df["overlap_sec"].describe())
    print("-" * 70)
    print("겹치는 쪽(마지막 과거 세션)의 거리/소요시간 분포:")
    print(excluded_df[["last_past_distance_km", "last_past_duration_sec"]].describe())
    print("-" * 70)
    print("패턴 분류 결과 (추정일 뿐 확정 아님, '미확인'은 패턴에 들어맞지 않음을 의미):")
    print(excluded_df["pattern"].value_counts())
    print("-" * 70)

    per_user_counts = excluded_df.groupby("user_id").size().sort_values(ascending=False)
    print(f"제외된 윈도우가 발생한 사용자 수: {len(per_user_counts)} / 전체 {df['user_id'].nunique()}")
    print("제외 건수가 가장 많은 사용자 10명:")
    print(per_user_counts.head(10))
    per_user_path = OUTPUT_DIR / "excluded_windows_per_user.csv"
    per_user_counts.rename("excluded_count").to_csv(per_user_path)
    print(f"사용자별 제외 건수 전체 -> {per_user_path}")
    print("-" * 70)

    rng_sample = excluded_df.sample(n=min(20, len(excluded_df)), random_state=RANDOM_SEED)
    sample_path = OUTPUT_DIR / "excluded_windows_sample_20.csv"
    rng_sample.to_csv(sample_path, index=False)
    print(f"육안 확인용 무작위 20건 -> {sample_path}")


if __name__ == "__main__":
    main()
