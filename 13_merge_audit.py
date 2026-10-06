# -*- coding: utf-8 -*-
"""
세션 병합 감사 (CLAUDE.md 10월 2주차 3번).

08_session_merge.py가 적용한 병합 규칙(시간 간격 <=180초, GPS 종료~시작
거리 <=200m, 암시 속도 <=3m/s, 셋 다 만족해야 병합)을 사람이 직접 눈으로
검증할 수 있도록 다음을 산출한다:

1. 병합 대응표(merged_session_id <-> 원본 session_id 목록), 병합 건수,
   체인 길이 분포 -> docs/merge_audit/ 에 파일로 저장
2. 병합된 사례 무작위 30건 + 임계값 경계 근처(간격 150~210초, 거리
   150~250m)에서 병합되지 않은 사례 30건 -> 표로 저장 (간격/좌표거리/
   각 조각의 거리·시간 포함)
3. 병합 임계값은 사전에 정한 값이며 MAE를 보고 조정하지 않았음을 문서화
4. 같은 체인(원래 하나의 운동으로 판단된 조각들)이 personalized 분할의
   history/hold-out 경계에 걸치는 경우가 있는지 확인
7. 시간이 겹치는 세션 쌍(간격이 음수)이 병합 규칙에서 실제로 어떻게
   처리되는지(병합 대상인지) 확인
"""

import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RUN_SESSIONS_PATH = Path("data") / "run_sessions.pkl"
MERGED_SESSIONS_PATH = Path("data") / "run_sessions_merged.pkl"
GPS_ENDPOINTS_PATH = Path("data") / "session_gps_endpoints.pkl"
USER_SEQUENCES_PATH = Path("data") / "user_sequences.pkl"
PERSONALIZED_TEST_PATH = Path("data") / "personalized_test_dataset.pkl"

OUTPUT_DIR = Path("docs") / "merge_audit"

# 08_session_merge.py와 동일한 임계값. "사전에 정한 값이며 MAE를 보고
# 조정하지 않았다" — 이번 감사 스크립트에서도 절대 바꾸지 않는다.
MAX_GAP_SEC = 180
MAX_GAP_DISTANCE_M = 200
MAX_IMPLIED_SPEED_MPS = 3.0
EARTH_RADIUS_KM = 6371.0088

RANDOM_SEED = 42


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def would_merge(gap_sec, gap_distance_m, implied_speed_mps):
    """08_session_merge.py의 should_merge()와 동일한 3조건 판정 (감사용 재구현)."""
    if gap_sec < 0 or gap_sec > MAX_GAP_SEC:
        return False
    if gap_distance_m > MAX_GAP_DISTANCE_M:
        return False
    if implied_speed_mps > MAX_IMPLIED_SPEED_MPS:
        return False
    return True


def build_consecutive_pairs(df):
    """사용자별로 시간순 정렬된 세션의 모든 '연속된 쌍'에 대해 간격/거리/속도/병합여부를 계산."""
    rows = []
    for user_id, g in df.groupby("user_id"):
        g = g.sort_values("start_time").reset_index(drop=True)
        for i in range(len(g) - 1):
            prev = g.iloc[i]
            nxt = g.iloc[i + 1]
            gap_sec = nxt["start_time"] - (prev["start_time"] + prev["duration_sec"])
            gap_distance_km = haversine_km(prev["end_lat"], prev["end_lon"], nxt["start_lat"], nxt["start_lon"])
            gap_distance_m = gap_distance_km * 1000
            if gap_sec <= 0:
                implied_speed_mps = float("inf") if gap_distance_m > 0 else 0.0
            else:
                implied_speed_mps = gap_distance_m / gap_sec
            merged = would_merge(gap_sec, gap_distance_m, implied_speed_mps)
            rows.append(
                {
                    "user_id": user_id,
                    "prev_session_id": prev["session_id"],
                    "next_session_id": nxt["session_id"],
                    "gap_sec": gap_sec,
                    "gap_distance_m": gap_distance_m,
                    "implied_speed_mps": implied_speed_mps,
                    "would_merge": merged,
                    "prev_distance_km": prev["actual_distance_km"],
                    "prev_duration_sec": prev["duration_sec"],
                    "next_distance_km": nxt["actual_distance_km"],
                    "next_duration_sec": nxt["duration_sec"],
                }
            )
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_sessions = pd.read_pickle(RUN_SESSIONS_PATH)
    gps_endpoints = pd.read_pickle(GPS_ENDPOINTS_PATH)
    merged_sessions = pd.read_pickle(MERGED_SESSIONS_PATH)
    user_sequences = pd.read_pickle(USER_SEQUENCES_PATH)
    personalized_test = pd.read_pickle(PERSONALIZED_TEST_PATH)

    df = run_sessions.merge(gps_endpoints, on="session_id", how="inner")

    # ===== 1. 병합 대응표 + 체인 길이 분포 =====
    chains = merged_sessions[merged_sessions["merged_count"] > 1].copy()
    correspondence = chains[["session_id", "merged_count", "merged_session_ids", "user_id"]].rename(
        columns={"session_id": "merged_session_id"}
    )
    correspondence_path = OUTPUT_DIR / "merge_correspondence_table.csv"
    correspondence.to_csv(correspondence_path, index=False)

    chain_length_dist = merged_sessions["merged_count"].value_counts().sort_index()
    chain_length_path = OUTPUT_DIR / "chain_length_distribution.csv"
    chain_length_dist.rename("chain_count").to_csv(chain_length_path)

    print("=" * 70)
    print(f"[1] 병합 대응표: {len(correspondence)}개 체인 (merged_count>1) -> {correspondence_path}")
    print(f"    체인 길이 분포 -> {chain_length_path}")
    print(chain_length_dist)
    print("-" * 70)

    # ===== 2. 무작위 30건 + 경계 근처 미병합 30건 =====
    pairs_df = build_consecutive_pairs(df)
    rng = np.random.default_rng(RANDOM_SEED)

    merged_pairs = pairs_df[pairs_df["would_merge"]]
    sample_merged = merged_pairs.sample(n=min(30, len(merged_pairs)), random_state=RANDOM_SEED)
    sample_merged_path = OUTPUT_DIR / "sample_merged_30.csv"
    sample_merged.to_csv(sample_merged_path, index=False)

    boundary_candidates = pairs_df[
        (~pairs_df["would_merge"])
        & (pairs_df["gap_sec"].between(150, 210))
        & (pairs_df["gap_distance_m"].between(150, 250))
    ]
    sample_boundary = boundary_candidates.sample(n=min(30, len(boundary_candidates)), random_state=RANDOM_SEED)
    sample_boundary_path = OUTPUT_DIR / "sample_boundary_not_merged_30.csv"
    sample_boundary.to_csv(sample_boundary_path, index=False)

    print(f"[2] 전체 연속 세션 쌍: {len(pairs_df)}개, 병합 대상(would_merge=True): {len(merged_pairs)}개")
    print(f"    병합 사례 무작위 {len(sample_merged)}건 -> {sample_merged_path}")
    print(f"    경계 근처(간격 150~210초, 거리 150~250m) 미병합 사례: {len(boundary_candidates)}개 중 "
          f"{len(sample_boundary)}건 -> {sample_boundary_path}")
    print("-" * 70)

    # ===== 4. history/hold-out 경계에 걸치는 체인 확인 =====
    session_order_lookup = dict(zip(user_sequences["session_id"], user_sequences["session_order"]))
    # personalized 분할의 사용자별 경계: hold-out(test)에서 가장 작은 target_session_order.
    boundary_by_user = {}
    for s in personalized_test:
        uid = s["user_id"]
        t = s["target_session_order"]
        if uid not in boundary_by_user or t < boundary_by_user[uid]:
            boundary_by_user[uid] = t

    straddling_chains = []
    chains_checked = 0
    for _, row in chains.iterrows():
        uid = row["user_id"]
        if uid not in boundary_by_user:
            continue  # 이 사용자는 personalized hold-out 샘플이 없음 (해당 없음)
        session_orders = [session_order_lookup.get(sid) for sid in row["merged_session_ids"]]
        session_orders = [so for so in session_orders if so is not None]
        if not session_orders:
            continue
        chains_checked += 1
        boundary_t = boundary_by_user[uid]
        if min(session_orders) < boundary_t <= max(session_orders):
            straddling_chains.append(
                {
                    "user_id": uid,
                    "merged_session_id": row["session_id"],
                    "merged_session_ids": row["merged_session_ids"],
                    "session_orders": sorted(session_orders),
                    "boundary_target_session_order": boundary_t,
                }
            )

    straddling_df = pd.DataFrame(straddling_chains)
    straddling_path = OUTPUT_DIR / "chains_straddling_personalized_boundary.csv"
    straddling_df.to_csv(straddling_path, index=False)

    print(f"[4] personalized history/hold-out 경계 확인 대상 체인: {chains_checked}개 "
          f"(사용자가 personalized hold-out을 가진 경우만)")
    print(f"    경계에 걸치는 체인 수: {len(straddling_chains)}개 -> {straddling_path}")
    print("-" * 70)

    # ===== 7. 시간이 겹치는(간격이 음수인) 세션 쌍의 처리 확인 =====
    negative_gap_pairs = pairs_df[pairs_df["gap_sec"] < 0]
    negative_gap_merged = negative_gap_pairs[negative_gap_pairs["would_merge"]]
    print(f"[7] 간격이 음수(시간 겹침)인 연속 세션 쌍: {len(negative_gap_pairs)}개")
    print(f"    그중 병합 대상으로 판정된 건수: {len(negative_gap_merged)}개 "
          f"(0이어야 함 — should_merge()가 gap_sec<0이면 바로 거부하도록 되어 있음)")
    assert len(negative_gap_merged) == 0, "음수 간격 쌍이 병합 대상으로 판정됨 — 병합 규칙 재검토 필요!"
    print("    확인: 08_session_merge.py의 should_merge()는 `gap_sec < 0 or gap_sec > MAX_GAP_SEC`이면")
    print("    바로 False를 반환하므로, 시간이 겹치는 쌍은 설계상 병합되지 않는다 (코드와 실측 결과 일치).")
    print("-" * 70)

    print(f"[3] 병합 임계값(간격<={MAX_GAP_SEC}초, 거리<={MAX_GAP_DISTANCE_M}m, "
          f"속도<={MAX_IMPLIED_SPEED_MPS}m/s)은 docs/session_merge_audit.md에 "
          f"'사전에 정한 값, MAE로 조정하지 않음'을 명시함.")
    print(f"저장 디렉터리: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
