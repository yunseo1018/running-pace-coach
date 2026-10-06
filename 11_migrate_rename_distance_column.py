# -*- coding: utf-8 -*-
"""
마이그레이션: 캐시된 pkl의 distance_km 컬럼을 actual_distance_km로 리네임.

배경: docs/feature_availability.md에서 이 값을 actual_distance_km(GPS로
계산한 실제 완주 거리, 러닝 후 확정)라고 부르기로 했다. 01단계를 처음부터
다시 돌리면 6GB 원본을 또 긁어야 해서 시간이 오래 걸리므로, 이미 저장된
pkl 파일의 컬럼명만 바꿔서 저장한다 (값 자체는 전혀 바뀌지 않음, 이름만 변경).

리네임 대상 (distance_km 컬럼을 가진 DataFrame 캐시):
- data/run_sessions.pkl          (01단계 출력)
- data/user_sequences.pkl        (02단계 출력)
- data/run_sessions_merged.pkl   (08단계 출력)

(train_dataset.pkl/test_dataset.pkl/personalized_*/posthoc_* 등은 X가 이미
flatten된 numpy 배열이라 컬럼명 자체가 없음 — 02~09 재실행으로 자연히
새 이름 기준으로 재생성된다.)
"""

import io
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TARGET_PICKLES = [
    Path("data") / "run_sessions.pkl",
    Path("data") / "user_sequences.pkl",
    Path("data") / "run_sessions_merged.pkl",
]

OLD_NAME = "distance_km"
NEW_NAME = "actual_distance_km"


def migrate(path):
    df = pd.read_pickle(path)
    if OLD_NAME not in df.columns:
        print(f"[스킵] {path}: '{OLD_NAME}' 컬럼이 없음 (이미 리네임됐거나 해당 없음)")
        return

    sample_before = df[OLD_NAME].iloc[0]
    n_rows_before = len(df)

    df = df.rename(columns={OLD_NAME: NEW_NAME})
    df.to_pickle(path)

    sample_after = df[NEW_NAME].iloc[0]
    print(f"[완료] {path}")
    print(f"  행 수: {n_rows_before} (변경 전) -> {len(df)} (변경 후, 동일해야 정상)")
    print(f"  첫 행 값: {sample_before} (변경 전 '{OLD_NAME}') -> {sample_after} (변경 후 '{NEW_NAME}') "
          f"-> 값 동일 여부: {sample_before == sample_after}")
    print(f"  현재 컬럼 목록: {list(df.columns)}")


def main():
    for path in TARGET_PICKLES:
        migrate(path)
        print("-" * 60)


if __name__ == "__main__":
    main()
