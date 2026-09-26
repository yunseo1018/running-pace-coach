# -*- coding: utf-8 -*-
"""
1단계: endomondoHR.json에서 러닝(run) 세션만 뽑아 세션 단위 표(DataFrame)로 만든다.

중요한 전제(탐색으로 확인한 사실들, README에도 기록):
- 이 파일은 진짜 JSON이 아니라, 한 줄에 파이썬 dict 리터럴이 하나씩 있는 형태다.
  (따옴표가 '싱글쿼트'이고, json.load()/json.loads()로는 파싱이 안 된다.)
  그래서 json 모듈이 아니라 ast.literal_eval()로 한 줄씩 읽는다.
- 파일에 실제로 존재하는 필드는 다음 뿐이다:
  speed, altitude, gender, heart_rate, id, url, userId, timestamp,
  longitude, latitude, sport
  => distance(거리), weather(날씨) 필드는 이 파일에 전혀 없다.
     CLAUDE.md는 이 필드들이 있다고 가정했지만, 실제로 없으므로
     - 거리는 latitude/longitude로 haversine 누적 거리를 계산해서 만든다.
     - 날씨는 이번 파이프라인에서는 사용하지 않는다 (사용자와 상의 후 결정).
- speed 필드는 run 세션의 약 81%에서 비어 있어(신뢰 불가) 페이스 계산에 쓰지 않는다.
  대신 (누적 거리) / (세션 총 시간)으로 평균 페이스를 직접 계산한다.
- sport 필드의 실제 값은 'run' 117,902건, 'bike' 98,001건 등 49종류였다.
  이번 파이프라인은 정확히 sport == 'run' 인 것만 쓴다
  ('treadmill running'은 GPS 기반 거리 계산이 안 되므로 제외).
"""

import ast
import io
import math
import sys
from pathlib import Path

import pandas as pd

# 콘솔 코드페이지(cp949 등)에서 한글 print가 깨지는 것을 방지하기 위해
# 표준출력을 UTF-8로 강제한다.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RAW_JSON_PATH = Path("endomondoHR.json")
OUTPUT_DIR = Path("data")
OUTPUT_PATH = OUTPUT_DIR / "run_sessions.pkl"

# 이상치(비정상 세션) 필터 기준값. 근거:
# - MIN_DISTANCE_KM: 0.3km 미만이면 GPS 기록 실수/워밍업 정지 등으로 보고 제외
# - MIN_DURATION_SEC / MAX_DURATION_SEC: 1분 미만이거나 5시간(18000초) 초과인
#   세션은 센서 오류나 기록 누락으로 보고 제외
MIN_DISTANCE_KM = 0.3
MIN_DURATION_SEC = 60
MAX_DURATION_SEC = 5 * 60 * 60

# GPS 포인트가 순간적으로 튀는("GPS 점프") 오류 방지용 기준.
# 두 연속 포인트 사이의 암시된 속도가 이 값(m/s)을 넘으면 GPS 오류로 보고
# 그 구간의 거리는 누적하지 않는다. 25km/h ~= 6.94m/s. 러닝 세션 기준으로
# 넉넉하게 잡은 값이라, 정상적인 러닝 구간을 잘못 버리는 경우는 거의 없다.
MAX_PLAUSIBLE_SPEED_MPS = 25 / 3.6

# 심박수 정제 범위(생리학적으로 가능한 범위 밖의 값은 센서 오류로 보고 평균에서 제외)
MIN_PLAUSIBLE_HR = 30
MAX_PLAUSIBLE_HR = 230

# 줄 단위 사전 필터에 쓰는 문자열. ast.literal_eval은 비용이 크므로,
# 이 값이 줄에 없으면 파싱 자체를 시도하지 않는다.
SPORT_RUN_MARKER = "'sport': 'run'"


def haversine_km(lat1, lon1, lat2, lon2):
    """두 위경도 좌표 사이의 거리(km)를 haversine 공식으로 계산."""
    earth_radius_km = 6371.0088
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def total_distance_km(latitudes, longitudes, timestamps):
    """
    위경도 포인트 배열을 순서대로 이어서 누적 거리(km)를 계산.
    단, 두 포인트 사이 암시된 속도가 MAX_PLAUSIBLE_SPEED_MPS를 넘으면
    GPS 오류(점프)로 보고 그 구간의 거리는 누적하지 않는다.
    반환값: (누적 거리 km, 점프로 스킵된 구간 수)
    """
    if len(latitudes) < 2:
        return 0.0, 0
    dist = 0.0
    skipped_segments = 0
    for i in range(1, len(latitudes)):
        segment_km = haversine_km(latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i])
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            # 타임스탬프가 중복/역전된 구간은 속도를 정의할 수 없으므로 스킵
            skipped_segments += 1
            continue
        implied_speed_mps = (segment_km * 1000) / dt
        if implied_speed_mps > MAX_PLAUSIBLE_SPEED_MPS:
            skipped_segments += 1
            continue
        dist += segment_km
    return dist, skipped_segments


def clean_heart_rates(heart_rates):
    """생리학적으로 불가능한 심박수 값을 제거하고 남은 리스트를 반환."""
    return [hr for hr in heart_rates if MIN_PLAUSIBLE_HR <= hr <= MAX_PLAUSIBLE_HR]


def build_session_row(record, stats):
    """
    파싱된 dict(record) 하나를 세션 단위 피처 dict로 변환. 유효하지 않으면 None.
    stats: 호출 쪽에서 누적 집계용으로 넘기는 dict (gps_jump_segments, hr_cleaned_sessions 등)
    """
    latitudes = record.get("latitude") or []
    longitudes = record.get("longitude") or []
    timestamps = record.get("timestamp") or []
    heart_rates = record.get("heart_rate") or []

    if len(timestamps) < 2 or len(latitudes) != len(longitudes) or len(latitudes) < 2:
        return None
    if not heart_rates:
        return None

    duration_sec = timestamps[-1] - timestamps[0]
    if duration_sec < MIN_DURATION_SEC or duration_sec > MAX_DURATION_SEC:
        return None

    distance_km, skipped_segments = total_distance_km(latitudes, longitudes, timestamps)
    stats["gps_jump_segments"] += skipped_segments
    if distance_km < MIN_DISTANCE_KM:
        return None
    # GPS 점프 필터링 이후에도 남을 수 있는 극단값에 대한 안전망.
    # 한 세션 100km는 러닝 기록으로는 사실상 상한선(울트라마라톤 수준도 넘김).
    if distance_km > 100:
        return None

    valid_heart_rates = clean_heart_rates(heart_rates)
    if not valid_heart_rates:
        return None
    if len(valid_heart_rates) < len(heart_rates):
        stats["hr_cleaned_sessions"] += 1

    avg_pace_sec_per_km = duration_sec / distance_km
    avg_heart_rate = sum(valid_heart_rates) / len(valid_heart_rates)

    return {
        "user_id": record["userId"],
        "session_id": record["id"],
        "start_time": timestamps[0],
        "distance_km": distance_km,
        "duration_sec": duration_sec,
        "avg_pace_sec_per_km": avg_pace_sec_per_km,
        "avg_heart_rate": avg_heart_rate,
        "gender": record.get("gender"),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_lines = 0
    run_line_count = 0
    parsed_count = 0
    dropped_invalid = 0
    rows = []
    stats = {"gps_jump_segments": 0, "hr_cleaned_sessions": 0}

    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1

            # 값비싼 literal_eval 전에 문자열 포함 여부로 먼저 걸러낸다.
            if SPORT_RUN_MARKER not in line:
                continue
            run_line_count += 1

            record = ast.literal_eval(line)
            # 문자열 포함 검사는 부분 일치라서, sport 값이 정확히 'run'인지 다시 확인.
            if record.get("sport") != "run":
                continue

            row = build_session_row(record, stats)
            if row is None:
                dropped_invalid += 1
                continue

            rows.append(row)
            parsed_count += 1

            if total_lines % 20000 == 0:
                print(f"  ...{total_lines}줄 처리, 현재까지 유효 run 세션 {parsed_count}건")

    df = pd.DataFrame(rows)
    df.to_pickle(OUTPUT_PATH)

    print("=" * 60)
    print(f"전체 라인 수: {total_lines}")
    print(f"sport == 'run' 매칭 라인 수: {run_line_count}")
    print(f"이상치 필터로 제외된 건수: {dropped_invalid}")
    print(f"GPS 점프로 스킵된 구간 수(세션 누적 거리 계산 중): {stats['gps_jump_segments']}")
    print(f"심박수 이상값이 있어 일부 값을 제외하고 평균 낸 세션 수: {stats['hr_cleaned_sessions']}")
    print(f"최종 유효 run 세션 수: {len(df)}")
    print(f"고유 사용자 수: {df['user_id'].nunique()}")
    print("-" * 60)
    print("컬럼 통계 (describe):")
    print(df[["distance_km", "duration_sec", "avg_pace_sec_per_km", "avg_heart_rate"]].describe())
    print("-" * 60)
    print("상위 5개 행 샘플:")
    print(df.head())
    print("-" * 60)
    print(f"저장 위치: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
