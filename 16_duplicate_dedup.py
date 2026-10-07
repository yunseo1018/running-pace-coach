# -*- coding: utf-8 -*-
"""
중복 기록 처리 (CLAUDE.md 10월 2주차 3-2번).

배경: 과거 정보 감사(04/06단계)에서 제외한 2,700개 윈도우를 다시 들여다본
결과(14_overlap_exclusion_audit.py), 96.8%가 "시작 시각·위치가 거의 동일한
중복 기록"으로 추정되는 패턴이었다. 지금까지는 "윈도우가 경계에 걸릴 때만"
제외했기 때문에, 중복 세션이 입력(과거 5개) 중간에 끼어 있는 윈도우는
그대로 남아 같은 운동이 입력에 두 번 반영되는 문제가 있다. 이 스크립트는
세션 자체를 정리해서(중복 쌍 중 하나를 제거) 이 문제를 근본적으로 해결한다.

=== 1. 중복 판정 규칙 (14_overlap_exclusion_audit.py에서 쓴 값 그대로 재사용,
MAE를 보고 조정하지 않음) ===
같은 사용자의 두 세션이 아래 세 조건을 모두 만족하면 "중복 기록"으로 판정:
(a) 시작 시각 차이 <= 60초 (NEAR_SIMULTANEOUS_START_SEC, 14단계와 동일)
(b) 시작 좌표 거리 <= 50m (NEAR_SAME_LOCATION_M, 14단계와 동일)
(c) 두 세션의 시간 구간 [start, start+duration]이 실제로 겹침
    (max(start) < min(end) — 그냥 시작이 가깝다고 다 중복은 아니고,
    실제로 같은 시간대를 차지해야 "중복 기록"이라 할 수 있으므로 추가)

=== 2. 어느 쪽을 남길지 규칙 (v2, 채택 — "의미" 기준으로 선택) ===
같은 중복 클러스터(2개 이상이 서로 중복 관계로 묶인 그룹) 안에서:
1순위: GPS 포인트 개수(session_point_counts.pkl, timestamp 배열 길이)가
       더 많은 쪽을 남긴다 — 포인트가 많을수록 더 상세하게 기록된 트랙이라고
       보는 것이 합리적 (기기/신호 문제로 샘플링이 거칠게 끊긴 쪽보다 낫다).
2순위(동률): **duration_sec이 더 긴 쪽**을 남긴다. 중복 쌍의 상당수가
       "같은 운동을 두 기록이 서로 다른 길이로 담아낸 경우"로 보이므로
       (docs/target_pace_definition.md 6번 참고), 더 길게/완전하게
       기록된 쪽이 그 운동을 더 온전히 담았다고 보는 것이 의미상
       타당하다고 판단해 이 규칙을 선택했다. (이전 버전은 start_time이
       빠른 쪽을 썼는데, duration을 전혀 보지 않아 "더 짧게 끝난 기록"이
       남는 경우가 있었다 — 동률 쌍 2,687건 중 206건에서 결과가 달라짐.
       MAE/RMSE에 대한 영향은 0.0004~0.0074로 거의 없음을 확인했으나,
       이건 규칙을 고른 "이유"가 아니라 "성능을 해치지 않는다"는 부수적
       확인일 뿐이다 — 선택 기준은 끝까지 "의미"였다.)
3순위(1,2순위 모두 동률): session_id가 더 작은 쪽을 남긴다. 이 역시
       특별한 의미는 없는 순수 결정적 tie-break다. **참고**: 맨 처음
       작성했던 버전(2순위=start_time)에서는 이 3순위가 없었고, 1·2순위가
       모두 동률일 때 Python `set`의 내부(해시 기반) 순회 순서로
       결정됐다 — 실제로 확인해보니 전체 제거 2,687건 중 **2,268건
       (84.4%)**이 그 동률 케이스였다. session_id를 3순위로 추가해
       "결정적이고 재현 가능한" 규칙으로 바로잡았다
       (docs/duplicate_audit.md "3. 동점 처리" 참고 — 그 문서는 구버전
       (2순위=start_time) 기준으로 작성되어 있다).
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
RUN_SESSIONS_PATH = Path("data") / "run_sessions.pkl"
GPS_ENDPOINTS_PATH = Path("data") / "session_gps_endpoints.pkl"
POINT_COUNTS_PATH = Path("data") / "session_point_counts.pkl"

OUTPUT_DIR = Path("docs") / "merge_audit"
DEDUP_CORRESPONDENCE_PATH = OUTPUT_DIR / "dedup_correspondence_table.csv"
DEDUP_SAMPLE_PATH = OUTPUT_DIR / "dedup_sample_20.csv"
DEDUPED_SESSIONS_CACHE = Path("data") / "run_sessions_deduped.pkl"

SPORT_RUN_MARKER = "'sport': 'run'"

# 14_overlap_exclusion_audit.py와 동일한 값. 사전에 정한 값이며 MAE를 보고
# 조정하지 않는다 (CLAUDE.md 병합 규칙과 같은 원칙을 중복 판정에도 적용).
NEAR_SIMULTANEOUS_START_SEC = 60
NEAR_SAME_LOCATION_M = 50
EARTH_RADIUS_KM = 6371.0088
RANDOM_SEED = 42


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def extract_point_counts():
    """session_id -> GPS/타임스탬프 포인트 개수. 캐시 있으면 재사용."""
    if POINT_COUNTS_PATH.exists():
        print(f"[캐시 사용] {POINT_COUNTS_PATH}")
        return pd.read_pickle(POINT_COUNTS_PATH)

    print("[신규 추출] raw 파일에서 run 세션의 포인트 개수 추출 중...")
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
            timestamps = record.get("timestamp") or []
            rows.append({"session_id": record["id"], "n_points": len(timestamps)})
            if total_lines % 50000 == 0:
                print(f"  ...{total_lines}줄 처리")

    point_counts_df = pd.DataFrame(rows)
    point_counts_df.to_pickle(POINT_COUNTS_PATH)
    print(f"[완료] {len(point_counts_df)}건 캐싱 -> {POINT_COUNTS_PATH}")
    return point_counts_df


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def find_duplicate_pairs(df):
    """같은 사용자 안에서 모든 세션 쌍(시간순으로 가까운 것들)을 검사해
    중복 판정 3조건을 모두 만족하는 쌍을 찾는다. 세션 수가 많은 사용자를
    고려해 start_time 기준으로 정렬 후 윈도우를 좁혀 탐색(인접 + 가까운
    범위만 비교)하여 전체 쌍(O(n^2))을 다 보지 않고도 놓치지 않게 한다."""
    duplicate_edges = []
    for user_id, g in df.groupby("user_id"):
        g = g.sort_values("start_time").reset_index(drop=True)
        n = len(g)
        for i in range(n):
            # start_time이 가까운 몇 개만 비교 (60초 이내 조건이라, start_time
            # 차이가 그보다 훨�씬 커지는 순간 더 볼 필요 없음).
            for j in range(i + 1, n):
                dt = g.iloc[j]["start_time"] - g.iloc[i]["start_time"]
                if dt > NEAR_SIMULTANEOUS_START_SEC:
                    break
                a, b = g.iloc[i], g.iloc[j]
                start_distance_m = haversine_km(a["start_lat"], a["start_lon"], b["start_lat"], b["start_lon"]) * 1000
                if start_distance_m > NEAR_SAME_LOCATION_M:
                    continue
                a_end = a["start_time"] + a["duration_sec"]
                b_end = b["start_time"] + b["duration_sec"]
                overlap = max(a["start_time"], b["start_time"]) < min(a_end, b_end)
                if not overlap:
                    continue
                duplicate_edges.append((user_id, a["session_id"], b["session_id"], dt, start_distance_m))
    return duplicate_edges


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_sessions = pd.read_pickle(RUN_SESSIONS_PATH)
    gps_endpoints = pd.read_pickle(GPS_ENDPOINTS_PATH)
    point_counts = extract_point_counts()

    df = run_sessions.merge(gps_endpoints, on="session_id", how="inner").merge(
        point_counts, on="session_id", how="left"
    )
    df["n_points"] = df["n_points"].fillna(0)

    duplicate_edges = find_duplicate_pairs(df)
    print("=" * 70)
    print(f"[1] 중복 판정 규칙(시작시각차<=60초, 시작좌표거리<=50m, 시간구간겹침)을 만족하는 쌍: "
          f"{len(duplicate_edges)}건")

    # 클러스터링 (union-find): 서로 연쇄적으로 중복인 세션들을 하나의 그룹으로.
    involved_ids = set()
    for _, a, b, _, _ in duplicate_edges:
        involved_ids.add(a)
        involved_ids.add(b)
    uf = UnionFind(involved_ids)
    for _, a, b, _, _ in duplicate_edges:
        uf.union(a, b)

    clusters = {}
    for sid in involved_ids:
        clusters.setdefault(uf.find(sid), []).append(sid)

    point_count_lookup = dict(zip(df["session_id"], df["n_points"]))
    start_time_lookup = dict(zip(df["session_id"], df["start_time"]))
    user_lookup = dict(zip(df["session_id"], df["user_id"]))
    distance_lookup = dict(zip(df["session_id"], df["actual_distance_km"]))
    duration_lookup = dict(zip(df["session_id"], df["duration_sec"]))
    start_lat_lookup = dict(zip(df["session_id"], df["start_lat"]))
    start_lon_lookup = dict(zip(df["session_id"], df["start_lon"]))

    # 동점 처리 집계용: 1·2순위(n_points, start_time)까지 전부 동률인 쌍이
    # 실제로 얼마나 있는지 (3순위 session_id가 실질적으로 결정하는 비율).
    n_tied_on_points_and_time = 0

    correspondence_rows = []
    removed_ids = set()
    for root, members in clusters.items():
        # 2. 어느 쪽을 남길지 (v2, 채택): 포인트 개수 내림차순 -> duration_sec
        #    내림차순(더 길게 기록된 쪽 우선) -> session_id 오름차순(그래도
        #    동률이면 결정적 tie-break). "의미" 기준으로 골랐다: 중복 쌍은
        #    대개 같은 운동을 두 기록이 서로 다른 길이로 담아낸 경우라
        #    (docs/target_pace_definition.md 6번 참고), 더 길게/완전하게
        #    기록된 쪽을 남기는 것이 "그 운동을 더 온전히 담은 기록을
        #    남긴다"는 의미에서 더 타당하다. MAE/RMSE에 미치는 영향은
        #    거의 없음(0.0004~0.0074)을 확인했지만, 그건 이 규칙을 고른
        #    이유가 아니라 "성능을 해치지 않는다"는 부수적 확인일 뿐이다.
        members_sorted = sorted(
            members, key=lambda sid: (-point_count_lookup.get(sid, 0), -duration_lookup[sid], sid)
        )
        kept = members_sorted[0]
        for removed in members_sorted[1:]:
            removed_ids.add(removed)
            if (
                point_count_lookup.get(kept) == point_count_lookup.get(removed)
                and duration_lookup[kept] == duration_lookup[removed]
            ):
                n_tied_on_points_and_time += 1

            start_distance_m = (
                haversine_km(
                    start_lat_lookup[kept], start_lon_lookup[kept],
                    start_lat_lookup[removed], start_lon_lookup[removed],
                )
                * 1000
            )
            kept_end = start_time_lookup[kept] + duration_lookup[kept]
            removed_end = start_time_lookup[removed] + duration_lookup[removed]
            overlap_sec = max(
                0, min(kept_end, removed_end) - max(start_time_lookup[kept], start_time_lookup[removed])
            )
            shorter_duration = min(duration_lookup[kept], duration_lookup[removed])
            overlap_ratio = overlap_sec / shorter_duration if shorter_duration > 0 else 0.0

            correspondence_rows.append(
                {
                    "user_id": user_lookup[kept],
                    "cluster_size": len(members),
                    "kept_session_id": kept,
                    "removed_session_id": removed,
                    "kept_n_points": point_count_lookup.get(kept, 0),
                    "removed_n_points": point_count_lookup.get(removed, 0),
                    "kept_start_time": start_time_lookup[kept],
                    "removed_start_time": start_time_lookup[removed],
                    "kept_distance_km": distance_lookup[kept],
                    "removed_distance_km": distance_lookup[removed],
                    "kept_duration_sec": duration_lookup[kept],
                    "removed_duration_sec": duration_lookup[removed],
                    "start_distance_m": start_distance_m,
                    "overlap_ratio": overlap_ratio,
                    "distance_diff_km": abs(distance_lookup[kept] - distance_lookup[removed]),
                    "duration_diff_sec": abs(duration_lookup[kept] - duration_lookup[removed]),
                }
            )

    correspondence_df = pd.DataFrame(correspondence_rows)
    # 거리 차이가 0.5km 넘거나 시간 차이가 300초 넘으면 "중복이 아닐 수도 있음"으로 표시.
    correspondence_df["large_mismatch_flag"] = (
        (correspondence_df["distance_diff_km"] > 0.5) | (correspondence_df["duration_diff_sec"] > 300)
    )
    correspondence_df.to_csv(DEDUP_CORRESPONDENCE_PATH, index=False)

    print(f"[1] 중복 클러스터 수(2개 이상 묶인 그룹): {len(clusters)}개, "
          f"제거 대상 세션 수: {len(removed_ids)}개")
    print(f"    대응표 -> {DEDUP_CORRESPONDENCE_PATH}")
    print(f"[3] n_points와 duration_sec이 모두 동률이라 session_id(3순위)로 결정된 쌍: "
          f"{n_tied_on_points_and_time} / {len(removed_ids)} "
          f"({n_tied_on_points_and_time / len(removed_ids):.1%})")
    print(f"[2] 거리차 0.5km 초과 또는 시간차 300초 초과로 '큰 차이' 플래그 걸린 쌍(전체): "
          f"{correspondence_df['large_mismatch_flag'].sum()} / {len(correspondence_df)}")

    # 육안 확인용 20건: 무작위 17건 + "큰 차이" 플래그 쌍 중 최대 3건을 섞어서,
    # 샘플 안에 이상 사례가 안 뽑힐 가능성을 줄인다.
    rng = np.random.default_rng(RANDOM_SEED)
    flagged = correspondence_df[correspondence_df["large_mismatch_flag"]]
    n_flagged_in_sample = min(3, len(flagged))
    flagged_sample = flagged.sample(n=n_flagged_in_sample, random_state=RANDOM_SEED) if n_flagged_in_sample else flagged
    remaining_pool = correspondence_df.drop(flagged_sample.index)
    random_sample = remaining_pool.sample(n=min(20 - n_flagged_in_sample, len(remaining_pool)), random_state=RANDOM_SEED)
    sample = pd.concat([flagged_sample, random_sample]).sort_index()
    sample.to_csv(DEDUP_SAMPLE_PATH, index=False)
    print(f"    육안 확인용 20건(큰 차이 플래그 {n_flagged_in_sample}건 포함) -> {DEDUP_SAMPLE_PATH}")
    print("-" * 70)

    print("[4] n_points 분포 (전체 run 세션 기준):")
    print(point_counts["n_points"].describe())
    print(f"    n_points == 500 인 세션 비율: {(point_counts['n_points'] == 500).mean():.2%}")
    print(f"    max n_points: {point_counts['n_points'].max()} "
          "(500이 상한(cap)인지는 미확인 — 다수가 정확히 500에 몰려있다는 정황만 있음)")
    print("-" * 70)

    cluster_sizes = correspondence_df.groupby("kept_session_id")["cluster_size"].first()
    triple_plus = cluster_sizes[cluster_sizes >= 3]
    print(f"[5] 3개 이상의 세션이 하나로 묶인 중복 클러스터 수: {len(triple_plus)}개 "
          f"(여기 속한 세션 총 {triple_plus.sum()}개 — kept 1개 + removed 나머지, "
          f"클러스터별 평균 크기 {triple_plus.mean():.1f}개)" if len(triple_plus) else
          "[5] 3개 이상의 세션이 하나로 묶인 중복 클러스터: 0개 (전부 2개짜리 쌍)")
    print("-" * 70)

    # 중복 제거 후 세션 테이블 저장 (다음 단계인 17_dedup_merge_rebuild.py에서 사용).
    deduped = run_sessions[~run_sessions["session_id"].isin(removed_ids)].reset_index(drop=True)
    deduped.to_pickle(DEDUPED_SESSIONS_CACHE)
    print(f"중복 제거 후 세션 수: {len(deduped)} (제거 전 {len(run_sessions)}, {len(removed_ids)}건 제거)")
    print(f"저장 위치: {DEDUPED_SESSIONS_CACHE.resolve()}")


if __name__ == "__main__":
    main()
