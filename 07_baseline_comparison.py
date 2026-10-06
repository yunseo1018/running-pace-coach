# -*- coding: utf-8 -*-
"""
7단계 (v2, 5주차): Baseline 5종을 cold-start/personalized 두 평가 방식으로 비교.

CLAUDE.md "비교 모델 및 Baseline" 순서대로 5개 baseline을 구현한다.
전부 "과거 5개 세션의 페이스만 보고 단순하게 예측하면 얼마나 틀리는지"를 보는
목적이라, X(과거 5세션 x [actual_distance_km, avg_pace_sec_per_km, avg_heart_rate]
= 15개, 10월 2주차부터 타깃 세션 값은 X에 전혀 없음) 중 페이스 값만 사용한다.
(참고: test_dataset.pkl이 10_posthoc_distance_condition.py가 만든 posthoc_*
데이터셋으로 바뀌어도 앞 15개만 읽는 구조라 안전하게 동작한다.)

1. recent_avg   : 과거 5개 세션 페이스의 평균
2. last_session : 과거 5개 중 가장 최근(5번째) 세션의 페이스 그대로
3. median5      : 과거 5개 세션 페이스의 중앙값 (극단치에 덜 흔들림)
4. user_overall_avg : 그 사용자의 "지금까지의 모든 세션"(과거 5개보다 더 이전 것까지
   전부) 페이스 평균 — 과거 5개만 보는 1번과 달리 전체 이력을 사용하므로
   data/user_sequences.pkl에서 사용자별 전체 이력을 따로 불러와야 한다.
5. linear_trend : 과거 5개 세션의 (순서, 페이스)에 단순 선형회귀(최소제곱)를
   적용해 바로 다음 시점의 페이스를 추정 (빨라지는/느려지는 추세 반영).

10월 2주차 과거 정보 감사 (추가): 1/2/3/5번은 X(과거 5개 세션)만 쓰므로
04/06단계에서 이미 저장한 last_input_session_time < target_start_time
검증을 그대로 재사용한다. 4번(user_overall_avg)은 과거 5개보다 더 이전
세션까지 참조하므로, build_user_overall_avg_lookup()에서 "참조한 세션 중
가장 늦은 start_time"도 같이 계산해 evaluate_split()에서 target_start_time
보다 이른지 별도로 assert한다 (타깃 및 그 이후 세션, 평가 구간 세션이
이력 피처에 섞이지 않았는지 확인).

허용 학습 자료: cold-start는 03단계가 고정한 test_user_ids의 기록만
평가에 쓰고(이 baseline들은 fit이 없어 train_dataset.pkl은 쓰지 않음),
personalized는 각 사용자의 history 구간(=personalized_train_dataset.pkl이
만들어진 세션들)만 user_overall_avg 계산에 참조한다 — 둘 다
build_user_overall_avg_lookup()이 session_order 기준 expanding().shift(1)로
"현재 이전"만 보게 만들어져 있어 자동으로 지켜진다.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

USER_SEQUENCES_PATH = Path("data") / "user_sequences.pkl"
COLD_START_TEST_PATH = Path("data") / "test_dataset.pkl"
PERSONALIZED_TEST_PATH = Path("data") / "personalized_test_dataset.pkl"

N_PAST_SESSIONS = 5
N_FEATURES_PER_PAST_SESSION = 3
PACE_FEATURE_INDEX = 1  # [distance, pace, heart_rate] 중 pace의 위치


def extract_past_paces(x_flat):
    """flatten된 X 벡터(길이 15)에서 과거 5개 세션의 avg_pace_sec_per_km만 뽑는다."""
    past_part = np.array(x_flat[: N_PAST_SESSIONS * N_FEATURES_PER_PAST_SESSION])
    return past_part.reshape(N_PAST_SESSIONS, N_FEATURES_PER_PAST_SESSION)[:, PACE_FEATURE_INDEX]


def build_user_overall_avg_lookup(user_sequences_df):
    """
    (user_id, target_session_order) ->
        (그 세션 이전까지의 모든 세션 페이스 평균, 그 중 가장 늦은 start_time)
    을 조회할 수 있는 dict를 만든다.

    expanding()은 현재 행까지 포함한 누적값이므로, shift(1)로 한 칸 밀어서
    "현재 행 제외, 그 이전까지의 누적값"으로 만든다 (타겟 세션 자신의 값이
    섞이면 leakage이므로 반드시 shift 필요). mean과 함께 max(start_time)도
    같은 방식으로 계산해, 이 baseline이 참조한 세션 중 가장 늦은 시각이
    실제로 타깃보다 이른지 감사할 수 있게 한다.
    """
    df = user_sequences_df.sort_values(["user_id", "session_order"]).copy()
    df["cum_mean_before"] = (
        df.groupby("user_id")["avg_pace_sec_per_km"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    df["cum_max_time_before"] = (
        df.groupby("user_id")["start_time"]
        .transform(lambda s: s.expanding().max().shift(1))
    )
    lookup = {
        (row.user_id, row.session_order): (row.cum_mean_before, row.cum_max_time_before)
        for row in df.itertuples()
    }
    return lookup


def predict_all_baselines(sample, overall_avg_lookup):
    past_paces = extract_past_paces(sample["X"])

    x_idx = np.arange(N_PAST_SESSIONS)
    slope, intercept = np.polyfit(x_idx, past_paces, 1)
    linear_trend_pred = slope * N_PAST_SESSIONS + intercept

    overall_avg_pred, overall_avg_max_time_before = overall_avg_lookup.get(
        (sample["user_id"], sample["target_session_order"])
    )

    return {
        "recent_avg": past_paces.mean(),
        "last_session": past_paces[-1],
        "median5": np.median(past_paces),
        "user_overall_avg": overall_avg_pred,
        "linear_trend": linear_trend_pred,
    }, overall_avg_max_time_before


BASELINE_NAMES = ["recent_avg", "last_session", "median5", "user_overall_avg", "linear_trend"]
BASELINE_LABELS = {
    "recent_avg": "1. 최근 5회 평균",
    "last_session": "2. last-session (가장 최근 세션)",
    "median5": "3. median-5 (최근 5개 중앙값)",
    "user_overall_avg": "4. 사용자 전체 평균",
    "linear_trend": "5. 단순 선형 추세",
}


def evaluate_split(dataset_path, overall_avg_lookup, label):
    samples = pd.read_pickle(dataset_path)

    rows = []
    for sample in samples:
        preds, overall_avg_max_time_before = predict_all_baselines(sample, overall_avg_lookup)
        # target_session_order는 항상 5 이상(과거 5개 세션이 보장됨)이므로
        # user_overall_avg는 항상 최소 5개 이전 세션의 평균을 가진다 (None/NaN 불가능).
        assert pd.notna(preds["user_overall_avg"]), \
            f"user_overall_avg 조회 실패: user_id={sample['user_id']}, target_session_order={sample['target_session_order']}"

        # 과거 정보 감사 1: 1/2/3/5번 baseline은 X(과거 5개 세션)만 쓰므로,
        # 04/06단계가 저장해둔 last_input_session_time < target_start_time을 재검증.
        assert sample["last_input_session_time"] < sample["target_start_time"], (
            f"과거 정보 감사 실패(과거 5개 세션 baseline): user_id={sample['user_id']}, "
            f"target_session_order={sample['target_session_order']}"
        )
        # 과거 정보 감사 2: user_overall_avg(4번)가 참조한 세션 중 가장 늦은 시각도
        # target_start_time보다 이른지 별도로 확인 (과거 5개보다 더 이전 세션까지
        # 참조하므로 위 검증과는 독립적으로 봐야 함).
        assert overall_avg_max_time_before < sample["target_start_time"], (
            f"과거 정보 감사 실패(user_overall_avg): user_id={sample['user_id']}, "
            f"target_session_order={sample['target_session_order']}, "
            f"overall_avg_max_time_before={overall_avg_max_time_before} >= "
            f"target_start_time={sample['target_start_time']}"
        )

        for name in BASELINE_NAMES:
            y_pred = preds[name]
            y_true = sample["y"]
            rows.append(
                {
                    "baseline": name,
                    "user_id": sample["user_id"],
                    "abs_error": abs(y_true - y_pred),
                    "sq_error": (y_true - y_pred) ** 2,
                }
            )

    result_df = pd.DataFrame(rows)

    print(f"[과거 정보 감사] {len(samples)}개 샘플 전부에서 last_input_session_time < "
          f"target_start_time, user_overall_avg 참조 세션의 최대 시각 < target_start_time "
          f"검증 통과 (assert 전부 통과).")

    summary_rows = []
    for name in BASELINE_NAMES:
        sub = result_df[result_df["baseline"] == name]
        per_user_mae = sub.groupby("user_id")["abs_error"].mean()
        summary_rows.append(
            {
                "baseline": BASELINE_LABELS[name],
                "overall_mae": sub["abs_error"].mean(),
                "overall_rmse": np.sqrt(sub["sq_error"].mean()),
                "per_user_mae_mean": per_user_mae.mean(),
                "per_user_mae_median": per_user_mae.median(),
                "per_user_mae_std": per_user_mae.std(),
            }
        )
    summary_df = pd.DataFrame(summary_rows).set_index("baseline")

    print("=" * 70)
    print(f"[{label}] baseline 5종 비교 (샘플 {len(samples)}건, "
          f"사용자 {result_df['user_id'].nunique()}명)")
    print(summary_df.to_string())
    print()

    return summary_df


def main():
    user_sequences_df = pd.read_pickle(USER_SEQUENCES_PATH)
    overall_avg_lookup = build_user_overall_avg_lookup(user_sequences_df)

    cold_start_summary = evaluate_split(COLD_START_TEST_PATH, overall_avg_lookup,
                                         "cold-start (사용자 단위 80/20)")
    personalized_summary = evaluate_split(PERSONALIZED_TEST_PATH, overall_avg_lookup,
                                           "personalized (사용자 내 시간순 temporal hold-out)")

    print("=" * 70)
    print("[최종 비교] 두 평가 방식에서 각 baseline의 전체 MAE/RMSE:")
    combined = pd.DataFrame(
        {
            "cold_start_MAE": cold_start_summary["overall_mae"],
            "cold_start_RMSE": cold_start_summary["overall_rmse"],
            "personalized_MAE": personalized_summary["overall_mae"],
            "personalized_RMSE": personalized_summary["overall_rmse"],
        }
    )
    print(combined.to_string())


if __name__ == "__main__":
    main()
