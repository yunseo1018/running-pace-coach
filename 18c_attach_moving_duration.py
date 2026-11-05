# -*- coding: utf-8 -*-
"""
18c단계 (P1-3에서 추가): 메인 세션 표(중복 제거 → 병합)에 이동 시간 열을 붙여 `data/run_sessions_deduped_merged_with_moving.pkl`을 만든다.

이 파일은 2026-10-07에 임시 실행으로 만들어졌고 만든 코드가 저장소에 없었다(docs/split_provenance.md 4절).
조사에서 찾은 규칙을 코드로 옮긴 것이며 규칙은 다음과 같다.
  moving_duration_sec = min( merged_session_ids에 속한 조각들의 session_moving_duration.pkl `moving_duration_sec` 합 , duration_sec )
  moving_pace         = moving_duration_sec / actual_distance_km
  gap_sec             = duration_sec - moving_duration_sec
  gap_ratio           = gap_sec / duration_sec
조각 합이 duration_sec(세션 표의 경과 시간)보다 커서 **잘리는 세션이 42개**(전체 105,709건의 0.04%, 모두 병합되지 않은 단일 조각 세션)다. 자르지 않으면 gap_sec가 음수가 된다.

입력: data/run_sessions_deduped_merged.pkl (17번 산출물), data/session_moving_duration.pkl (18b 산출물)
출력: data/run_sessions_deduped_merged_with_moving.pkl. 24·33번이 읽는다.
검증(assert): 잘린 세션 수 42, 열 순서, 내용 해시가 고정본과 같음, 출력 파일이 이미 있으면 DataFrame 내용이 정확히 같음(같으면 덮어쓰지 않는다. 다르면 중단).
"""
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MOVING_PIECES_PATH = Path("data") / "session_moving_duration.pkl"     # 18b 산출물(원본 세션 id별 이동 시간)
OUTPUT_PATH = config.MAIN_DATA_PATH.parent / "run_sessions_deduped_merged_with_moving.pkl"
EXPECTED_ROWS = 105709
EXPECTED_CLIPPED = 42
NEW_COLS = ["moving_duration_sec", "moving_pace", "gap_sec", "gap_ratio"]
# 고정본(현재 파일)의 내용 해시: 병합 id 목록을 쉼표 문자열로 바꾼 CSV 텍스트의 sha256(플랫폼·pickle 형식과 무관)
EXPECTED_CONTENT_SHA256 = "9e5c1bdd69bc32abf9606673a7242839577da339487cf9b4c2a5f64bb19d78a2"


def content_sha256(df):
    d = df.copy()
    d["merged_session_ids"] = d["merged_session_ids"].map(lambda x: ",".join(str(i) for i in x))
    return hashlib.sha256(d.to_csv(index=False).encode("utf-8")).hexdigest()


def build(main, pieces):
    ids = main["merged_session_ids"]
    summed = ids.map(lambda x: float(pieces.loc[list(x), "moving_duration_sec"].sum()))
    clipped = summed > main["duration_sec"]
    moving = summed.clip(upper=main["duration_sec"])
    out = main.copy()
    out["moving_duration_sec"] = moving
    out["moving_pace"] = moving / out["actual_distance_km"]
    out["gap_sec"] = out["duration_sec"] - moving
    out["gap_ratio"] = out["gap_sec"] / out["duration_sec"]
    return out, int(clipped.sum()), int((clipped & (main["merged_count"] == 1)).sum())


def main():
    main_df = config.load_main_sessions()                     # 메인 세션 표(세션 수·사용자 수 assert 포함)
    assert len(main_df) == EXPECTED_ROWS, len(main_df)
    pieces = pd.read_pickle(MOVING_PIECES_PATH).set_index("session_id")
    out, n_clipped, n_clipped_single = build(main_df, pieces)
    assert n_clipped == EXPECTED_CLIPPED, f"duration_sec로 잘린 세션 수가 다름: {n_clipped} != {EXPECTED_CLIPPED}"
    assert n_clipped_single == n_clipped, "잘린 세션이 모두 단일 조각이어야 함"
    assert list(out.columns) == list(main_df.columns) + NEW_COLS
    assert (out["gap_sec"] >= 0).all(), "gap_sec가 음수인 세션이 있음"
    sha = content_sha256(out)
    assert sha == EXPECTED_CONTENT_SHA256, f"내용 해시가 고정본과 다름: {sha}"
    print(f"내용 해시가 고정본과 일치: {sha}")
    print(f"조각 이동 시간 합이 duration_sec보다 커서 잘린 세션: {n_clipped}개 / {len(out):,}건 ({100 * n_clipped / len(out):.2f}%), 모두 단일 조각 세션")
    if OUTPUT_PATH.exists():
        existing = pd.read_pickle(OUTPUT_PATH)
        pd.testing.assert_frame_equal(out, existing, check_exact=True)
        assert content_sha256(existing) == sha
        print(f"{OUTPUT_PATH}: 이미 있음, DataFrame 내용이 정확히 같음(덮어쓰지 않음)")
    else:
        out.to_pickle(OUTPUT_PATH)
        print(f"{OUTPUT_PATH}: 생성")


if __name__ == "__main__":
    main()
