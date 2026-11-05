# -*- coding: utf-8 -*-
"""
3단계: 사용자 단위로 train/test를 80/20으로 분할한다.

CLAUDE.md 규칙: 같은 사용자의 세션이 train/test 양쪽에 걸치면 안 됨.
그래서 세션 단위가 아니라 user_id 단위로 셔플/분할한다.

[재현 정정, P1-3] 입력은 **정제 전 세션 표 `data/run_sessions.pkl`(01번 산출물)에서 세션 6개 이상인 사용자**다.
고정 분할 목록(train 598명, test 150명)은 처음부터 이 입력에서 만들어진 것이다(처음 구현에서는 02번 산출물이 정제 전 데이터 기준이었다).
10월 7일 메인 데이터를 중복 제거→병합으로 바꾼 뒤 02번 입력이 메인 데이터가 되면서 03이 17번 산출물에 의존하게 되어(17은 이 ID 파일을 읽는다) 순환이 생겼고,
메인 데이터 기준으로 다시 돌리면 742명이 되어 다른 배정(593/149)이 나왔다(docs/split_provenance.md).
이 스크립트는 01번 산출물만 읽으므로 17보다 먼저 실행된다(순환 없음). 사용자 순서는 (user_id, start_time) 정렬 뒤 unique 순서다.

고정본 보호: 생성한 목록의 sha256이 고정본(아래 EXPECTED)과 같아야 하며(assert), 같지 않으면 아무것도 쓰지 않고 중단한다.
이미 ID 파일이 있으면 바이트가 같은지 확인만 하고 덮어쓰지 않는다. 없으면 고정본과 같은 바이트(CRLF 줄바꿈)로 쓴다.
ID 파일은 data/ 아래(git 제외)에만 둔다.
"""

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TRAIN_USERS_PATH = Path("data") / "train_user_ids.txt"
TEST_USERS_PATH = Path("data") / "test_user_ids.txt"

TRAIN_RATIO = 0.8
# 재현성을 위한 고정 시드. CLAUDE.md에 명시되진 않았지만, 같은 결과를
# 재현할 수 있어야 보고서/실험 비교가 의미 있어서 고정값으로 둔다.
RANDOM_SEED = 42

# 고정본(2026-09-26에 03이 만든 파일)의 sha256. 파일은 Windows 텍스트 모드로 쓰여 줄바꿈이 CRLF다.
EXPECTED_N = {"train": 598, "test": 150}
EXPECTED_SHA256_CRLF = {"train": "4e9c1f40680f3f24da0f7a5ffbb3658486620ea16db5f269edd364ee69292f9a",
                        "test": "57d4c9ad592b48e720c8cf08bd6cd27003ad34ccf0af53f063e92370b5e7ff79"}
EXPECTED_SHA256_LF = {"train": "68e0b05882a4f82d",   # 줄바꿈을 LF로 바꾼 내용의 sha256 앞 16자리(플랫폼 독립 확인용)
                      "test": "b4b4a61e89739c88"}


def to_bytes(ids):
    return "".join(f"{uid}\n" for uid in ids).replace("\n", "\r\n").encode("utf-8")


def main():
    df = config.load_raw_reference_sessions()          # data/run_sessions.pkl (01번 산출물, 정제 전)
    counts = df.groupby("user_id").size()
    eligible = set(counts[counts >= config.MIN_SESSIONS_REQUIRED].index)
    ordered = df[df["user_id"].isin(eligible)].sort_values(["user_id", "start_time"], kind="stable")
    user_ids = ordered["user_id"].unique()

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(user_ids)

    n_train = int(len(shuffled) * TRAIN_RATIO)
    train_user_ids = shuffled[:n_train]
    test_user_ids = shuffled[n_train:]

    # 두 집합이 겹치지 않는지, 그리고 합쳤을 때 전체 사용자를 다 포함하는지 검증.
    train_set = set(train_user_ids)
    test_set = set(test_user_ids)
    assert train_set.isdisjoint(test_set), "train/test 사용자 집합이 겹칩니다!"
    assert train_set | test_set == set(user_ids), "일부 사용자가 분할에서 빠졌습니다!"

    # ---- 고정본과 sha256까지 같은지 확인(다르면 쓰지 않고 중단) ----
    out = {}
    for name, ids in (("train", train_user_ids), ("test", test_user_ids)):
        b = to_bytes(ids)
        sha = hashlib.sha256(b).hexdigest()
        sha_lf = hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()
        assert len(ids) == EXPECTED_N[name], f"{name} 사용자 수가 고정본과 다름: {len(ids)} != {EXPECTED_N[name]}"
        assert sha == EXPECTED_SHA256_CRLF[name], f"{name} 목록의 sha256이 고정본과 다름: {sha}"
        assert sha_lf.startswith(EXPECTED_SHA256_LF[name]), f"{name} 목록(LF 정규화)의 sha256이 고정본과 다름"
        out[name] = (b, sha)
    print(f"고정본과 sha256 일치: train {out['train'][1]}, test {out['test'][1]}")

    for name, path in (("train", TRAIN_USERS_PATH), ("test", TEST_USERS_PATH)):
        b = out[name][0]
        if path.exists():
            assert path.read_bytes() == b, f"기존 {path} 와 바이트가 다름: 덮어쓰지 않고 중단"
            print(f"{path}: 이미 있음, 생성한 목록과 바이트 동일(덮어쓰지 않음)")
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b)
            print(f"{path}: 생성")

    train_session_count = df[df["user_id"].isin(train_set)].shape[0]
    test_session_count = df[df["user_id"].isin(test_set)].shape[0]

    print("=" * 60)
    print(f"전체 사용자 수(세션 {config.MIN_SESSIONS_REQUIRED}개 이상, 정제 전 기준): {len(user_ids)}")
    print(f"train 사용자 수: {len(train_user_ids)} ({len(train_user_ids) / len(user_ids):.1%})")
    print(f"test 사용자 수: {len(test_user_ids)} ({len(test_user_ids) / len(user_ids):.1%})")
    print("-" * 60)
    print(f"train 세션 수(정제 전): {train_session_count}")
    print(f"test 세션 수(정제 전): {test_session_count}")
    print("-" * 60)
    print(f"train/test 사용자 교집합 크기: {len(train_set & test_set)} (0이어야 정상)")
    print(f"저장 위치: {TRAIN_USERS_PATH.resolve()}, {TEST_USERS_PATH.resolve()}")


if __name__ == "__main__":
    main()
