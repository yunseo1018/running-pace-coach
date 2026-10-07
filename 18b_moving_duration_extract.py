# -*- coding: utf-8 -*-
"""
타깃 페이스 정의 확정 - 2단계: 정지 판정 규칙 적용 + moving_duration_sec 계산.

=== 정지 판정 규칙 (18a 샘플 분석 결과에 근거) ===
연속 포인트 간 시간 간격이 **60초를 초과**하면 그 구간을 "정지"로 보고
moving time에서 제외한다.

근거 (sampled_intervals.pkl, 500포인트군 1,458세션/727,542개 간격,
비500포인트군 1,557세션/424,248개 간격 기준):
- 정상 간격은 압도적으로 짧다: 500군 중앙값 7초, 95백분위 19초, 99백분위
  32초. 비500군은 중앙값 5초, 95백분위 9초, 99백분위 14초.
- 60초를 넘는 간격은 500군 0.26%, 비500군 0.15%뿐이다 — 명백한 이례적
  구간(정지/GPS 신호 끊김)으로 볼 수 있다.
- 60초는 이 프로젝트에서 이미 "중복 판정"(시작 시각차 임계값)에 쓴 값과
  같아 일관성도 있다.
- 음수 간격(타임스탬프 역전)은 500군에서 단 1건(0.0001%)뿐이라 무시할
  수준의 글리치로 보고, 0으로 clip해서 moving time 계산에서 음수가
  누적되지 않게 한다.

이 스크립트는 전체 117,902개 run 세션에 대해 moving_duration_sec(정지
구간 제외 이동 시간), n_pause_segments(정지로 판정된 구간 수),
max_gap_sec을 계산해 캐싱한다.
"""

import ast
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RAW_JSON_PATH = Path("endomondoHR.json")
OUTPUT_PATH = Path("data") / "session_moving_duration.pkl"

SPORT_RUN_MARKER = "'sport': 'run'"
PAUSE_THRESHOLD_SEC = 60


def main():
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
            if len(timestamps) < 2:
                continue

            deltas = np.diff(timestamps).astype(float)
            deltas = np.clip(deltas, 0, None)  # 음수(타임스탬프 역전) 글리치 방지

            is_pause = deltas > PAUSE_THRESHOLD_SEC
            moving_duration_sec = deltas[~is_pause].sum()
            n_pause_segments = int(is_pause.sum())
            max_gap_sec = float(deltas.max())

            rows.append(
                {
                    "session_id": record["id"],
                    "moving_duration_sec": moving_duration_sec,
                    "n_pause_segments": n_pause_segments,
                    "max_gap_sec": max_gap_sec,
                    "elapsed_duration_sec": timestamps[-1] - timestamps[0],
                }
            )

            if total_lines % 50000 == 0:
                print(f"  ...{total_lines}줄 처리")

    out_df = pd.DataFrame(rows)
    out_df.to_pickle(OUTPUT_PATH)

    print("=" * 60)
    print(f"처리 세션 수: {len(out_df)}")
    print(f"정지 구간이 1개 이상 있는 세션 비율: {(out_df['n_pause_segments'] > 0).mean():.2%}")
    print(f"elapsed - moving 차이(초) 분포:")
    gap = out_df["elapsed_duration_sec"] - out_df["moving_duration_sec"]
    print(gap.describe(percentiles=[.5, .9, .95, .99]))
    print(f"저장 위치: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
