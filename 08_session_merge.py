# -*- coding: utf-8 -*-
"""
세션 병합(merge) 검토 (CLAUDE.md 5주차 신규 섹션).

배경: 2단계에서 "한 번의 실제 운동이 여러 세션으로 쪼개져 기록된 것으로 보이는
사례"를 발견했다 (예: 사용자 69의 세션 2~4번이 각각 12분, 17분 간격으로 붙어있었음).
이번 단계에서는 "같은 사용자의 연속된 두 세션이 사실 하나의 운동이었다"를
판단하는 규칙을 정의하고, 그 규칙으로 세션을 병합한 새 데이터셋을 만든다.

=== 병합 규칙 정의 (세 가지 조건을 모두 만족해야 병합) ===

(a) 시간 간격: 이전 세션이 끝난 시각(start_time + duration_sec)부터
    다음 세션이 시작한 시각까지의 간격이 MAX_GAP_SEC(180초=3분) 이내.
    근거: GPS/심박 센서가 신호를 놓쳐서 기록이 몇 초~몇 분간 끊기는 경우가
    흔하다. 반면 3분을 넘기면 "잠깐 멈췄다가 다시 시작" 수준을 넘어 실제로
    휴식을 취했거나 다른 용무를 본 것으로 보는 게 더 합리적이라고 판단했다.

(b) GPS 위치 근접성: 이전 세션의 마지막 GPS 포인트와 다음 세션의 첫 GPS
    포인트 사이 거리가 MAX_GAP_DISTANCE_M(200m) 이내.
    근거: 일반적인 GPS 오차/드리프트는 수십m 수준이라, 200m면 "그 자리에서
    쉬다가 다시 뛴 것"을 충분히 포함하면서도 "다른 장소로 이동했다가 새로
    시작한 것"은 걸러낼 수 있다고 판단했다.

(c) GPS 연속성(암시된 속도): (b)의 거리를 (a)의 시간으로 나눈 "암시된 이동
    속도"가 MAX_IMPLIED_SPEED_MPS(3m/s, 사람이 걷는 속도보다 약간 빠른 수준)
    이내. 근거: (a)와 (b)를 각각 넉넉하게 잡았기 때문에, 두 조건을 모두
    만족해도 "매우 짧은 시간에 경계값 수준의 거리를 이동한" 경계 사례가
    남을 수 있다. 이 경우 "같은 자리에서 쉬었다"기엔 이동 속도가 과하므로,
    셀째 조건으로 한 번 더 걸러낸다.

세 조건을 모두 만족하는 "연속된" 세션들은 체인(chain)으로 묶어 하나로 합친다
(예: A-B가 만족, B-C도 만족하면 A-B-C를 전부 하나로 합침).

=== 병합된 세션의 피처 재계산 ===
- start_time: 체인의 첫 세션 start_time
- duration_sec: (마지막 세션의 start_time+duration_sec) - 첫 세션의 start_time
  (중간의 "끊긴 시간"도 실제로 흐른 시간이므로 duration에 포함)
- distance_km: 체인에 속한 세션들의 distance_km 합산 (서로 겹치지 않는 구간이므로
  단순 합산이 타당)
- avg_heart_rate: duration_sec으로 가중평균 (raw 심박 배열은 1단계 이후 버렸기
  때문에, 가진 정보(세션별 평균 심박 + 세션별 길이)로 재구성할 수 있는 가장
  합리적인 근사치. 세션 길이가 비슷하면 단순평균과 거의 같아진다.)
- avg_pace_sec_per_km: duration_sec / distance_km (반드시 합산된 총량으로
  다시 계산 — 개별 페이스를 평균 내면 안 됨, 페이스는 거리/시간에 대해
  비선형이라 평균의 평균이 실제 페이스와 다르다.)
"""

import ast
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RAW_JSON_PATH = Path("endomondoHR.json")
RUN_SESSIONS_PATH = Path("data") / "run_sessions.pkl"
GPS_ENDPOINTS_CACHE_PATH = Path("data") / "session_gps_endpoints.pkl"
MERGED_OUTPUT_PATH = Path("data") / "run_sessions_merged.pkl"

SPORT_RUN_MARKER = "'sport': 'run'"

MAX_GAP_SEC = 180
MAX_GAP_DISTANCE_M = 200
MAX_IMPLIED_SPEED_MPS = 3.0

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def extract_gps_endpoints():
    """
    01단계에서는 거리 계산에만 쓰고 버렸던 GPS 시작/종료 좌표를, 병합 판단을
    위해 raw 파일에서 다시 뽑아 session_id -> (start_lat, start_lon, end_lat, end_lon)
    형태로 캐싱한다. (data/에 캐시가 있으면 재사용해서 6GB 파일을 또 긁지 않는다.)
    """
    if GPS_ENDPOINTS_CACHE_PATH.exists():
        print(f"[캐시 사용] {GPS_ENDPOINTS_CACHE_PATH} 에서 GPS 시작/종료 좌표 로드")
        return pd.read_pickle(GPS_ENDPOINTS_CACHE_PATH)

    print("[신규 추출] raw 파일에서 run 세션의 GPS 시작/종료 좌표 추출 중...")
    rows = []
    total_lines = 0
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            if SPORT_RUN_MARKER not in line:
                continue
            record = ast.literal_eval(line)
            if record.get("sport") != "run":
                continue
            lat = record.get("latitude") or []
            lon = record.get("longitude") or []
            if len(lat) < 2 or len(lat) != len(lon):
                continue
            rows.append(
                {
                    "session_id": record["id"],
                    "start_lat": lat[0],
                    "start_lon": lon[0],
                    "end_lat": lat[-1],
                    "end_lon": lon[-1],
                }
            )
            if total_lines % 50000 == 0:
                print(f"  ...{total_lines}줄 처리")

    gps_df = pd.DataFrame(rows)
    gps_df.to_pickle(GPS_ENDPOINTS_CACHE_PATH)
    print(f"[완료] GPS 시작/종료 좌표 {len(gps_df)}건 캐싱 완료 -> {GPS_ENDPOINTS_CACHE_PATH}")
    return gps_df


def should_merge(prev_row, next_row):
    """prev_row(이전 세션), next_row(다음 세션)가 병합 조건 (a)(b)(c)를 모두 만족하는지."""
    gap_sec = next_row["start_time"] - (prev_row["start_time"] + prev_row["duration_sec"])
    if gap_sec < 0 or gap_sec > MAX_GAP_SEC:
        return False

    gap_distance_km = haversine_km(
        prev_row["end_lat"], prev_row["end_lon"], next_row["start_lat"], next_row["start_lon"]
    )
    gap_distance_m = gap_distance_km * 1000
    if gap_distance_m > MAX_GAP_DISTANCE_M:
        return False

    if gap_sec == 0:
        implied_speed_mps = float("inf") if gap_distance_m > 0 else 0.0
    else:
        implied_speed_mps = gap_distance_m / gap_sec
    if implied_speed_mps > MAX_IMPLIED_SPEED_MPS:
        return False

    return True


def merge_chain(chain_rows):
    """같은 체인(연속으로 병합 대상인 세션들)을 하나의 세션 행으로 합친다."""
    first = chain_rows[0]
    last = chain_rows[-1]

    total_distance_km = sum(r["distance_km"] for r in chain_rows)
    merged_duration_sec = (last["start_time"] + last["duration_sec"]) - first["start_time"]
    total_duration_for_hr = sum(r["duration_sec"] for r in chain_rows)
    weighted_hr = sum(r["avg_heart_rate"] * r["duration_sec"] for r in chain_rows) / total_duration_for_hr

    return {
        "user_id": first["user_id"],
        "session_id": first["session_id"],
        "start_time": first["start_time"],
        "distance_km": total_distance_km,
        "duration_sec": merged_duration_sec,
        "avg_pace_sec_per_km": merged_duration_sec / total_distance_km,
        "avg_heart_rate": weighted_hr,
        "gender": first["gender"],
        "merged_count": len(chain_rows),
        "merged_session_ids": [r["session_id"] for r in chain_rows],
    }


def merge_user_sessions(user_df):
    """한 사용자의 시간순 세션들에서 연속 병합 체인을 찾아 전부 합친다."""
    rows = user_df.to_dict("records")
    merged_rows = []
    current_chain = [rows[0]]

    for row in rows[1:]:
        if should_merge(current_chain[-1], row):
            current_chain.append(row)
        else:
            merged_rows.append(merge_chain(current_chain))
            current_chain = [row]
    merged_rows.append(merge_chain(current_chain))
    return merged_rows


def main():
    run_sessions_df = pd.read_pickle(RUN_SESSIONS_PATH)
    gps_df = extract_gps_endpoints()

    df = run_sessions_df.merge(gps_df, on="session_id", how="inner")
    n_dropped_no_gps = len(run_sessions_df) - len(df)

    df = df.sort_values(["user_id", "start_time"]).reset_index(drop=True)

    all_merged_rows = []
    for user_id, user_df in df.groupby("user_id"):
        all_merged_rows.extend(merge_user_sessions(user_df))

    merged_df = pd.DataFrame(all_merged_rows)
    merged_df.to_pickle(MERGED_OUTPUT_PATH)

    n_before = len(df)
    n_after = len(merged_df)
    n_merge_events = (merged_df["merged_count"] > 1).sum()
    chain_size_dist = merged_df["merged_count"].value_counts().sort_index()

    print("=" * 60)
    print(f"GPS 좌표 없어 이번 단계에서 제외된 세션: {n_dropped_no_gps}")
    print(f"병합 전 세션 수: {n_before}  (사용자 {df['user_id'].nunique()}명)")
    print(f"병합 후 세션 수: {n_after}  (사용자 {merged_df['user_id'].nunique()}명)")
    print(f"실제로 2개 이상이 합쳐진 '병합 결과' 세션 수: {n_merge_events}")
    print(f"줄어든 세션 수: {n_before - n_after} ({(n_before - n_after) / n_before:.1%})")
    print("-" * 60)
    print("병합 체인 크기 분포 (merged_count=1은 병합 안 된 단독 세션):")
    print(chain_size_dist)
    print("-" * 60)

    example = merged_df[merged_df["merged_count"] >= 3].iloc[0] if (merged_df["merged_count"] >= 3).any() else None
    if example is not None:
        print(f"[예시] user_id={example['user_id']}, {example['merged_count']}개 세션이 하나로 병합됨 "
              f"(원본 session_id: {example['merged_session_ids']})")
        print(f"  합산 거리={example['distance_km']:.3f}km, 합산 시간={example['duration_sec']:.0f}초, "
              f"재계산 페이스={example['avg_pace_sec_per_km']:.3f}초/km, 가중평균 심박={example['avg_heart_rate']:.1f}")
    print("-" * 60)
    print(f"저장 위치: {MERGED_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
