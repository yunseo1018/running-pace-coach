# -*- coding: utf-8 -*-
"""
리샘플링(포인트 솎아내기)이 haversine 거리에 주는 영향을 직접 측정.

원본 샘플링 간격이 작은(자연스럽게 촘촘하게 기록된) 비500 세션들을 골라,
포인트를 2배/4배/8배 간격으로 균등하게 솎아낸(decimate) 뒤 같은 GPS 점프
필터(01단계와 동일, 시속 25km 초과 구간 제외)를 적용해 haversine 거리를
다시 계산한다. 원본(촘촘한) 거리를 "참값"으로 보고, 솎아낸 거리의 상대
오차(부호 포함)를 구한다.

대상 세션 선정 기준: sport=='run', n_points<500, n_points>=80(8배
솎아내도 10개 이상 남게), 평균 간격(duration/(n-1)) <= 1.5초(솎아낸
뒤 간격이 500군의 평균 간격대(3~11초)와 겹치도록).
"""

import ast
import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RAW_JSON_PATH = Path("endomondoHR.json")
OUTPUT_PATH = Path("data") / "resampling_distance_impact.pkl"

SPORT_RUN_MARKER = "'sport': 'run'"
MAX_SESSIONS = 400
MIN_N_POINTS = 80
MAX_MEAN_INTERVAL_SEC = 1.5
MAX_PLAUSIBLE_SPEED_MPS = 25 / 3.6  # 01단계와 동일한 GPS 점프 필터
EARTH_RADIUS_KM = 6371.0088
DECIMATION_FACTORS = [2, 4, 8]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def filtered_distance(lats, lons, timestamps):
    """01단계와 동일한 GPS 점프 필터를 적용한 누적 haversine 거리."""
    if len(lats) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(lats)):
        seg_km = haversine_km(lats[i - 1], lons[i - 1], lats[i], lons[i])
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            continue
        implied_speed = (seg_km * 1000) / dt
        if implied_speed > MAX_PLAUSIBLE_SPEED_MPS:
            continue
        dist += seg_km
    return dist


def decimate_keep_last(arr, k):
    idx = list(range(0, len(arr), k))
    if idx[-1] != len(arr) - 1:
        idx.append(len(arr) - 1)
    return [arr[i] for i in idx]


def main():
    rows = []
    total_lines = 0
    n_selected = 0

    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            if n_selected >= MAX_SESSIONS:
                break
            if SPORT_RUN_MARKER not in line:
                continue
            record = ast.literal_eval(line)
            if record.get("sport") != "run":
                continue
            timestamps = record.get("timestamp") or []
            n_points = len(timestamps)
            if n_points < MIN_N_POINTS or n_points >= 500:
                continue
            duration = timestamps[-1] - timestamps[0]
            if duration <= 0:
                continue
            mean_interval = duration / (n_points - 1)
            if mean_interval > MAX_MEAN_INTERVAL_SEC:
                continue

            lats = record.get("latitude") or []
            lons = record.get("longitude") or []
            if len(lats) != n_points or len(lons) != n_points:
                continue

            original_dist = filtered_distance(lats, lons, timestamps)
            if original_dist <= 0:
                continue

            row = {
                "session_id": record["id"],
                "n_points": n_points,
                "mean_interval_sec": mean_interval,
                "original_distance_km": original_dist,
            }
            for k in DECIMATION_FACTORS:
                dec_lats = decimate_keep_last(lats, k)
                dec_lons = decimate_keep_last(lons, k)
                dec_ts = decimate_keep_last(timestamps, k)
                dec_dist = filtered_distance(dec_lats, dec_lons, dec_ts)
                row[f"decimated_{k}x_distance_km"] = dec_dist
                row[f"decimated_{k}x_mean_interval_sec"] = duration / (len(dec_ts) - 1)
                row[f"relative_error_{k}x"] = (dec_dist - original_dist) / original_dist
            rows.append(row)
            n_selected += 1

            if total_lines % 50000 == 0:
                print(f"  ...{total_lines}줄 처리, 선정된 세션 {n_selected}개")

    result_df = pd.DataFrame(rows)
    result_df.to_pickle(OUTPUT_PATH)

    print("=" * 60)
    print(f"선정된 세션 수: {len(result_df)}")
    for k in DECIMATION_FACTORS:
        col = f"relative_error_{k}x"
        interval_col = f"decimated_{k}x_mean_interval_sec"
        print(f"\n--- {k}배 솎아내기 ---")
        print(f"솎아낸 뒤 평균 간격: {result_df[interval_col].mean():.2f}초 "
              f"(중앙값 {result_df[interval_col].median():.2f}초)")
        print(f"상대 오차 분포: 중앙값 {result_df[col].median():.4%}, "
              f"5백분위 {result_df[col].quantile(0.05):.4%}, "
              f"95백분위 {result_df[col].quantile(0.95):.4%}")
        print(f"음수(거리 과소추정) 비율: {(result_df[col] < 0).mean():.2%}")
    print(f"\n저장 위치: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
