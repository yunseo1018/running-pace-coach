# -*- coding: utf-8 -*-
"""
프로젝트 공통 설정 (10월 2주차 5번, "재발 방지").

배경: 지난 단계에서 n_points/elapsed-moving/극단비율 분석을 전부
`run_sessions.pkl`(중복제거·병합 **전**)로 계산했다가 뒤늦게 발견한 적이
있다. 같은 실수(어느 파일이 "메인 데이터"인지 스크립트마다 다르게
가정하는 것)가 재발하지 않도록, 메인 데이터 경로를 이 파일 **한 곳**
에서만 정의하고 모든 스크립트가 여기서 import해서 쓴다.

사용법: 각 스크립트는 `load_main_sessions()`를 호출해 메인 데이터를
읽는다. 이 함수가 (1) 파일명 (2) 세션 수 (3) 고유 사용자 수를 항상
먼저 출력하고, 세션 수가 `MAIN_EXPECTED_SESSIONS`와 다르면 즉시
`AssertionError`로 중단한다 — "내가 지금 보고 있는 게 진짜 메인
데이터가 맞는지"를 매번 스크립트 시작 시점에 강제로 확인시킨다.
"""

import io
import sys
from pathlib import Path

import pandas as pd

# 표준출력을 UTF-8로 강제 (콘솔 cp949 깨짐 방지). 이 모듈을 import하는
# 모든 스크립트에 공통 적용된다.
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ============================================================
# 메인 데이터 (이 프로젝트의 "메인"은 항상 이 파일이다 — 중복 제거 ->
# 병합까지 끝난 세션 테이블. 다른 경로를 쓰고 싶으면 이 상수를 바꾸는
# 것이 아니라, 왜 메인이 바뀌는지 CLAUDE.md에 먼저 기록해야 한다.)
# ============================================================
MAIN_DATA_PATH = Path("data") / "run_sessions_deduped_merged.pkl"
MAIN_EXPECTED_SESSIONS = 105709
MAIN_EXPECTED_ELIGIBLE_USERS = 742  # run 세션 6개 이상인 사용자 수 (02단계 필터 후)

# ============================================================
# 예외: 원본(중복제거·병합 전) 비교용. "메인"이 아니므로 이름에
# RAW/REFERENCE를 명시해 혼동을 막는다. 09_merge_ablation.py 등
# 전/후 비교 스크립트에서만 의도적으로 사용한다.
# ============================================================
RAW_REFERENCE_DATA_PATH = Path("data") / "run_sessions.pkl"
RAW_REFERENCE_EXPECTED_SESSIONS = 114425

# ============================================================
# 공통 파이프라인 상수
# ============================================================
N_PAST_SESSIONS = 5  # CLAUDE.md: N=5로 고정
MIN_SESSIONS_REQUIRED = 6  # 과거 5개 + 예측 대상 1개
EXTREME_PACE_THRESHOLD_SEC_PER_KM = 1000

USER_SEQUENCES_PATH = Path("data") / "user_sequences.pkl"
TRAIN_USERS_PATH = Path("data") / "train_user_ids.txt"
TEST_USERS_PATH = Path("data") / "test_user_ids.txt"
TRAIN_DATASET_PATH = Path("data") / "train_dataset.pkl"
TEST_DATASET_PATH = Path("data") / "test_dataset.pkl"
PERSONALIZED_TRAIN_DATASET_PATH = Path("data") / "personalized_train_dataset.pkl"
PERSONALIZED_TEST_DATASET_PATH = Path("data") / "personalized_test_dataset.pkl"


def _report_and_check(df, label, path, expected_sessions):
    n_sessions = len(df)
    n_users = df["user_id"].nunique()
    print(f"[입력 확인] {label}: {path} (세션 {n_sessions}건, 사용자 {n_users}명)")
    assert n_sessions == expected_sessions, (
        f"'{label}' 데이터의 세션 수가 예상과 다릅니다! "
        f"기대값={expected_sessions}, 실제={n_sessions} (경로: {path}) "
        f"-- 메인 데이터 파일이 바뀌었거나 잘못된 파일을 읽고 있을 수 있습니다."
    )
    return n_sessions, n_users


def load_main_sessions():
    """메인 데이터(중복제거->병합 완료)를 읽고, 파일명/세션수/사용자수를
    출력한 뒤 세션 수가 MAIN_EXPECTED_SESSIONS와 다르면 즉시 assert로 중단한다."""
    df = pd.read_pickle(MAIN_DATA_PATH)
    _report_and_check(df, "메인 데이터", MAIN_DATA_PATH, MAIN_EXPECTED_SESSIONS)
    return df


def load_raw_reference_sessions():
    """예외: 원본(중복제거·병합 전) 비교용 데이터. 메인이 아님을 로그에 명시한다."""
    df = pd.read_pickle(RAW_REFERENCE_DATA_PATH)
    _report_and_check(df, "원본 비교용(메인 아님)", RAW_REFERENCE_DATA_PATH, RAW_REFERENCE_EXPECTED_SESSIONS)
    return df


def load_user_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def build_eligible_sequences(session_df, check_expected=True):
    """02단계 로직: user_id별 시간순 정렬 + session_order 부여 + 6세션 미만 제외.
    check_expected=True이면 결과 사용자 수가 MAIN_EXPECTED_ELIGIBLE_USERS와
    다를 때 assert로 중단한다 (메인 데이터에 적용할 때만 의미 있는 체크이므로,
    원본 비교용 데이터에 쓸 때는 check_expected=False로 호출해야 한다)."""
    df = session_df.sort_values(["user_id", "start_time"]).reset_index(drop=True)
    df["session_order"] = df.groupby("user_id").cumcount()
    counts = df.groupby("user_id").size()
    eligible_ids = set(counts[counts >= MIN_SESSIONS_REQUIRED].index)
    df_eligible = df[df["user_id"].isin(eligible_ids)].reset_index(drop=True)
    if check_expected:
        n_users = len(eligible_ids)
        assert n_users == MAIN_EXPECTED_ELIGIBLE_USERS, (
            f"자격 사용자 수가 예상과 다릅니다! 기대값={MAIN_EXPECTED_ELIGIBLE_USERS}, "
            f"실제={n_users} -- 메인 데이터가 바뀌었을 수 있습니다."
        )
    return df_eligible, eligible_ids


def intersect_train_test_with_eligible(eligible_ids):
    """기존 03단계 train/test 배정(748명 기준)을 새 eligible 집합과 교집합해,
    재분할 없이 기존 배정을 유지한 채 탈락한 사용자만 제외한다.
    반환: (train_ids, test_ids, excluded_user_ids)"""
    original_train = load_user_ids(TRAIN_USERS_PATH)
    original_test = load_user_ids(TEST_USERS_PATH)
    train_ids = original_train & eligible_ids
    test_ids = original_test & eligible_ids
    excluded = (original_train | original_test) - eligible_ids
    return train_ids, test_ids, excluded
