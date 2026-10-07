# -*- coding: utf-8 -*-
"""
타깃 페이스 정의 확정 - 1단계: 세션 내 샘플링 간격(연속 timestamp 차이) 추출.

전체 117,902개 run 세션의 원시 timestamp 배열을 전부 메모리에 들고 있기엔
부담이 커서, n_points==500 / n_points<500 두 그룹에서 각각 약 1,500개씩
층화 무작위 샘플링(고정 시드)해 그 세션들의 '모든 연속 구간 간격'을 풀어서
저장한다. 이 샘플로 간격의 분포(중앙값/분산/꼬리)를 보고, 정지 판정
임계값을 정하는 근거로 쓴다.
"""

import ast
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RAW_JSON_PATH = Path("endomondoHR.json")
OUTPUT_PATH = Path("data") / "sampled_intervals.pkl"

SPORT_RUN_MARKER = "'sport': 'run'"
TARGET_SAMPLE_PER_GROUP = 1500
RANDOM_SEED = 42

# 01단계에서 이미 확인한 전체 run 세션 중 500포인트/비500포인트 대략적 비율
# (session_point_counts.pkl 기준 75,180 vs 42,722) -> 샘플링 확률 계산에 사용.
APPROX_TOTAL_500 = 75180
APPROX_TOTAL_OTHER = 42722


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    prob_500 = min(1.0, TARGET_SAMPLE_PER_GROUP / APPROX_TOTAL_500)
    prob_other = min(1.0, TARGET_SAMPLE_PER_GROUP / APPROX_TOTAL_OTHER)

    rows = []
    total_lines = 0
    n_sampled_500 = 0
    n_sampled_other = 0

    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            if SPORT_RUN_MARKER not in line:
                continue
            record = ast.literal_eval(line)
            if record.get("sport") != "run":
                continue
            timestamps = record.get("timestamp") or []
            n_points = len(timestamps)
            if n_points < 2:
                continue

            is_500 = n_points == 500
            accept_prob = prob_500 if is_500 else prob_other
            if rng.random() > accept_prob:
                continue

            deltas = np.diff(timestamps)
            rows.append(
                {
                    "session_id": record["id"],
                    "n_points": n_points,
                    "is_500": is_500,
                    "deltas": deltas,
                }
            )
            if is_500:
                n_sampled_500 += 1
            else:
                n_sampled_other += 1

            if total_lines % 50000 == 0:
                print(f"  ...{total_lines}줄 처리, 샘플: 500군 {n_sampled_500}, 비500군 {n_sampled_other}")

    sample_df = pd.DataFrame(rows)
    sample_df.to_pickle(OUTPUT_PATH)

    print("=" * 60)
    print(f"500군 샘플 세션 수: {n_sampled_500} (목표 {TARGET_SAMPLE_PER_GROUP})")
    print(f"비500군 샘플 세션 수: {n_sampled_other} (목표 {TARGET_SAMPLE_PER_GROUP})")
    print(f"저장 위치: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
