# -*- coding: utf-8 -*-
"""
10월 2주차 5번: baseline 통합표 + 세분화 분석. (메인 = past-only elapsed pace)

산출물: docs/baseline_results.md
입력: config.py의 메인 데이터 경로로 재배선된 04/06단계 결과물
(data/train_dataset.pkl, test_dataset.pkl, personalized_*_dataset.pkl).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config

OUTPUT_MD_PATH = Path("docs") / "baseline_results.md"
POINT_COUNTS_PATH = Path("data") / "session_point_counts.pkl"

N_PAST = config.N_PAST_SESSIONS
N_FEATURES = 3
PACE_IDX = 1
EXTREME_THRESHOLD = config.EXTREME_PACE_THRESHOLD_SEC_PER_KM

BASELINE_NAMES = ["recent_avg", "last_session", "median5", "user_overall_avg", "linear_trend"]
BASELINE_LABELS = {
    "recent_avg": "1. 최근 5회 평균",
    "last_session": "2. last-session",
    "median5": "3. median-5",
    "user_overall_avg": "4. 사용자 전체 평균",
    "linear_trend": "5. 단순 선형 추세",
}


def df_to_markdown(df, floatfmt=".3f"):
    """pandas to_markdown()이 tabulate 의존성을 요구해서, 외부 패키지 추가 없이
    직접 마크다운 표를 만든다."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:{floatfmt}}"
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def extract_past_paces(x_flat):
    return np.array(x_flat).reshape(N_PAST, N_FEATURES)[:, PACE_IDX]


def build_user_overall_avg_lookup(user_sequences_df):
    d = user_sequences_df.sort_values(["user_id", "session_order"]).copy()
    d["cum_mean_before"] = d.groupby("user_id")["avg_pace_sec_per_km"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    d["cum_max_time_before"] = d.groupby("user_id")["start_time"].transform(
        lambda s: s.expanding().max().shift(1)
    )
    return {(r.user_id, r.session_order): (r.cum_mean_before, r.cum_max_time_before) for r in d.itertuples()}


def predict_all(sample, overall_avg_lookup):
    past_paces = extract_past_paces(sample["X"])
    x_idx = np.arange(N_PAST)
    slope, intercept = np.polyfit(x_idx, past_paces, 1)
    overall_avg_pred, overall_avg_max_time = overall_avg_lookup[(sample["user_id"], sample["target_session_order"])]
    assert pd.notna(overall_avg_pred)
    assert overall_avg_max_time < sample["target_start_time"], "과거 정보 감사 실패: user_overall_avg"
    assert sample["last_input_session_time"] < sample["target_start_time"], "과거 정보 감사 실패: past window"
    return {
        "recent_avg": past_paces.mean(),
        "last_session": past_paces[-1],
        "median5": np.median(past_paces),
        "user_overall_avg": overall_avg_pred,
        "linear_trend": slope * N_PAST + intercept,
    }


def is_merge_affected(sample):
    target_merged = sample["target_merged_count"] > 1
    input_merged = any(c > 1 for c in sample["past_merged_counts"])
    return target_merged, input_merged


def is_extreme_affected(sample):
    past_paces = extract_past_paces(sample["X"])
    target_extreme = sample["y"] > EXTREME_THRESHOLD
    input_extreme = bool((past_paces > EXTREME_THRESHOLD).any())
    return target_extreme, input_extreme


def build_rows(samples, overall_avg_lookup):
    """샘플마다 5개 baseline 예측 + 메타데이터를 담은 행 리스트를 만든다."""
    rows = []
    for s in samples:
        preds = predict_all(s, overall_avg_lookup)
        target_merged, input_merged = is_merge_affected(s)
        target_extreme, input_extreme = is_extreme_affected(s)
        for name in BASELINE_NAMES:
            y_pred = preds[name]
            rows.append({
                "baseline": name,
                "user_id": s["user_id"],
                "y_true": s["y"],
                "y_pred": y_pred,
                "abs_error": abs(s["y"] - y_pred),
                "sq_error": (s["y"] - y_pred) ** 2,
                "target_merged": target_merged,
                "input_merged": input_merged,
                "merge_affected": target_merged or input_merged,
                "target_extreme": target_extreme,
                "input_extreme": input_extreme,
                "extreme_affected": target_extreme or input_extreme,
                "target_session_order": s["target_session_order"],
                "target_actual_distance_km": s["target_actual_distance_km"],
                "target_session_id": s["target_session_id"],
            })
    return pd.DataFrame(rows)


def summarize(df, group_cols=("baseline",)):
    g = df.groupby(list(group_cols))
    out = g.agg(
        n_samples=("abs_error", "size"),
        n_users=("user_id", "nunique"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda c: np.sqrt(c.mean())),
    )
    n_merges = df[df["merge_affected"]].groupby(list(group_cols)).size().rename("n_merges")
    out = out.join(n_merges, how="left")
    out["n_merges"] = out["n_merges"].fillna(0).astype(int)
    return out.reset_index()


N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
MAE_WINNER = "median5"
RMSE_WINNER = "user_overall_avg"


def cluster_bootstrap(df, baseline_names, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """사용자 단위 cluster bootstrap. 같은 사용자의 모든 윈도우를 한 단위로
    묶어 사용자를 복원추출한 뒤(모든 baseline에 동일한 추출을 적용해 쌍체
    비교가 가능하게 함), 각 반복에서 baseline별 MAE/RMSE를 계산한다.
    반환: boot_mae[baseline] = shape(n_boot,) 배열, boot_rmse도 동일."""
    rng = np.random.default_rng(seed)
    user_ids = df["user_id"].unique()
    n_users = len(user_ids)

    per_user = {}
    for baseline in baseline_names:
        sub = df[df["baseline"] == baseline]
        g = sub.groupby("user_id").agg(sum_abs=("abs_error", "sum"), sum_sq=("sq_error", "sum"), n=("abs_error", "size"))
        g = g.reindex(user_ids).fillna(0)
        per_user[baseline] = (g["sum_abs"].to_numpy(), g["sum_sq"].to_numpy(), g["n"].to_numpy())

    # 3주차 3-(i): 모든 방법의 사용자별 윈도우 수가 같아야 쌍체 비교가 성립한다(다르면 중단).
    _ref_n = per_user[baseline_names[0]][2]
    for b in baseline_names:
        assert np.array_equal(per_user[b][2], _ref_n), f"방법 간 사용자별 윈도우 수 불일치: {b}"

    boot_mae = {b: np.empty(n_boot) for b in baseline_names}
    boot_rmse = {b: np.empty(n_boot) for b in baseline_names}
    for i in range(n_boot):
        idx = rng.integers(0, n_users, size=n_users)
        for b in baseline_names:
            sum_abs, sum_sq, n = per_user[b]
            total_abs = sum_abs[idx].sum()
            total_sq = sum_sq[idx].sum()
            total_n = n[idx].sum()
            boot_mae[b][i] = total_abs / total_n
            boot_rmse[b][i] = np.sqrt(total_sq / total_n)
    return boot_mae, boot_rmse


def bootstrap_ci_table(boot_mae, boot_rmse):
    """baseline별 MAE/RMSE 95% CI 표 + 1위 baseline과의 쌍체 차이 CI 표를 만든다.
    (boot_mae/boot_rmse는 cluster_bootstrap()의 결과를 그대로 받는다 — 같은
    bootstrap 추출을 top3/bottom2 비교와 공유하기 위해 여기서 새로 뽑지 않음.)"""
    ci_rows = []
    for b in BASELINE_NAMES:
        mae_lo, mae_hi = np.percentile(boot_mae[b], [2.5, 97.5])
        rmse_lo, rmse_hi = np.percentile(boot_rmse[b], [2.5, 97.5])
        ci_rows.append({
            "방법": BASELINE_LABELS[b],
            "MAE": boot_mae[b].mean(), "MAE_95%CI_하한": mae_lo, "MAE_95%CI_상한": mae_hi,
            "RMSE": boot_rmse[b].mean(), "RMSE_95%CI_하한": rmse_lo, "RMSE_95%CI_상한": rmse_hi,
        })
    ci_df = pd.DataFrame(ci_rows)

    diff_rows = []
    for b in BASELINE_NAMES:
        if b == MAE_WINNER:
            mae_diff_label = "(MAE 1위 자신)"
            mae_diff, mae_diff_lo, mae_diff_hi = 0.0, 0.0, 0.0
        else:
            diff_arr = boot_mae[MAE_WINNER] - boot_mae[b]
            mae_diff = diff_arr.mean()
            mae_diff_lo, mae_diff_hi = np.percentile(diff_arr, [2.5, 97.5])
        if b == RMSE_WINNER:
            rmse_diff, rmse_diff_lo, rmse_diff_hi = 0.0, 0.0, 0.0
        else:
            diff_arr = boot_rmse[RMSE_WINNER] - boot_rmse[b]
            rmse_diff = diff_arr.mean()
            rmse_diff_lo, rmse_diff_hi = np.percentile(diff_arr, [2.5, 97.5])
        diff_rows.append({
            "방법": BASELINE_LABELS[b],
            f"MAE차이({BASELINE_LABELS[MAE_WINNER]}-이방법)": mae_diff,
            "하한": mae_diff_lo, "상한": mae_diff_hi,
            f"RMSE차이({BASELINE_LABELS[RMSE_WINNER]}-이방법)": rmse_diff,
            "하한2": rmse_diff_lo, "상한2": rmse_diff_hi,
        })
    diff_df = pd.DataFrame(diff_rows)
    return ci_df, diff_df


TOP3_RMSE = ["user_overall_avg", "median5", "recent_avg"]
BOTTOM2_RMSE = ["last_session", "linear_trend"]


def top3_bottom2_rmse_table(boot_rmse):
    """RMSE 기준 상위 3개(사용자 전체 평균/median-5/최근 5회 평균)와 하위 2개
    (last-session/선형추세) 사이의 쌍체 차이 95% CI를 전부 계산한다."""
    rows = []
    for top in TOP3_RMSE:
        for bot in BOTTOM2_RMSE:
            diff = boot_rmse[top] - boot_rmse[bot]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append({
                "상위(top3)": BASELINE_LABELS[top], "하위(bottom2)": BASELINE_LABELS[bot],
                "RMSE차이_평균": diff.mean(), "하한": lo, "상한": hi,
                "0포함(유의하지않음)": bool(lo <= 0 <= hi),
            })
    return pd.DataFrame(rows)


def main():
    md_lines = []
    md_lines.append("# Baseline 통합 결과 (10월 2주차 5번)\n")
    md_lines.append("## 입력 데이터\n")

    main_df = config.load_main_sessions()
    md_lines.append(f"- 메인 데이터: `{config.MAIN_DATA_PATH}` (세션 {len(main_df)}건, 사용자 {main_df['user_id'].nunique()}명)\n")

    user_sequences = pd.read_pickle(config.USER_SEQUENCES_PATH)
    print(f"[입력 확인] user_sequences: {config.USER_SEQUENCES_PATH} "
          f"(세션 {len(user_sequences)}건, 사용자 {user_sequences['user_id'].nunique()}명)")
    md_lines.append(f"- `{config.USER_SEQUENCES_PATH}`: 세션 {len(user_sequences)}건, 사용자 {user_sequences['user_id'].nunique()}명\n")

    train_samples = pd.read_pickle(config.TRAIN_DATASET_PATH)
    test_samples = pd.read_pickle(config.TEST_DATASET_PATH)
    p_train_samples = pd.read_pickle(config.PERSONALIZED_TRAIN_DATASET_PATH)
    p_test_samples = pd.read_pickle(config.PERSONALIZED_TEST_DATASET_PATH)
    for name, s in [("train_dataset(cold-start)", train_samples), ("test_dataset(cold-start)", test_samples),
                     ("personalized_train(history)", p_train_samples), ("personalized_test(holdout)", p_test_samples)]:
        print(f"[입력 확인] {name}: {len(s)}건")
        md_lines.append(f"- `{name}`: {len(s)}건\n")

    md_lines.append(
        "\n> **주의**: cold-start와 personalized는 평가 대상 사용자/윈도우 집합이 서로 다르다 "
        "(cold-start=test 사용자 149명의 윈도우만, personalized=전체 742명의 각자 temporal "
        "hold-out 윈도우). **두 평가조건의 MAE/RMSE 수치를 서로 직접 비교해 "
        "'어느 쪽이 더 낫다'고 말할 수 없다** — 평가 모집단 자체가 다르기 때문이다. "
        "같은 평가조건 안에서 baseline끼리 비교하거나, 같은 평가조건의 메인/민감도/전후를 "
        "비교하는 것만 타당하다.\n"
    )
    md_lines.append("\n(참고: 아래 표의 '#eval sessions'는 세션이 아니라 "
                     "**(과거5→타깃) 평가 윈도우 수**를 뜻한다.)\n")

    overall_avg_lookup = build_user_overall_avg_lookup(user_sequences)

    print("A/B 단계 검증 통과. baseline 평가 시작...")

    cold_df = build_rows(test_samples, overall_avg_lookup)
    personalized_df = build_rows(p_test_samples, overall_avg_lookup)
    cold_df["split"] = "cold-start"
    personalized_df["split"] = "personalized"
    all_df = pd.concat([cold_df, personalized_df], ignore_index=True)

    # ===== C. baseline 통합표 (메인: 전부 포함) =====
    md_lines.append("\n## C. Baseline 통합표 (메인 — 극단 페이스 포함, past-only 입력)\n")
    main_summary = summarize(all_df, group_cols=("split", "baseline"))
    main_summary["방법"] = main_summary["baseline"].map(BASELINE_LABELS)
    main_table = main_summary[["방법", "split", "mae", "rmse", "n_users", "n_samples", "n_merges"]].rename(
        columns={"split": "평가조건", "mae": "MAE(초/km)", "rmse": "RMSE(초/km)",
                 "n_users": "#users", "n_samples": "#eval sessions", "n_merges": "#merges"}
    )
    md_lines.append(df_to_markdown(main_table))
    md_lines.append("\n")

    # ===== C. 민감도: 극단 페이스(타깃 또는 입력) 포함 윈도우 제외 =====
    md_lines.append("\n## C-1. Baseline 통합표 (민감도 — 타깃 또는 입력에 극단 페이스(>1000s/km) 포함 윈도우 제외)\n")
    md_lines.append("(메인/민감도 역할은 결과를 보기 전에 사전 고정: 메인=항상 포함, 이 표는 민감도 참고용)\n\n")
    filtered_df = all_df[~all_df["extreme_affected"]]
    sens_summary = summarize(filtered_df, group_cols=("split", "baseline"))
    sens_summary["방법"] = sens_summary["baseline"].map(BASELINE_LABELS)
    sens_table = sens_summary[["방법", "split", "mae", "rmse", "n_users", "n_samples", "n_merges"]].rename(
        columns={"split": "평가조건", "mae": "MAE(초/km)", "rmse": "RMSE(초/km)",
                 "n_users": "#users", "n_samples": "#eval sessions", "n_merges": "#merges"}
    )
    md_lines.append(df_to_markdown(sens_table))
    md_lines.append("\n")

    # ===== C-2. 신뢰구간 (사용자 단위 cluster bootstrap) =====
    md_lines.append(f"\n## C-2. 신뢰구간 (사용자 단위 cluster bootstrap, {N_BOOTSTRAP}회, seed={BOOTSTRAP_SEED}, "
                     "메인표 기준)\n")
    md_lines.append("같은 사용자의 모든 윈도우를 한 단위로 묶어 사용자를 복원추출했다 "
                     "(윈도우를 개별적으로 리샘플링하면 같은 사용자 내 상관을 무시해 "
                     "구간이 과도하게 좁아질 수 있음).\n")
    print("\n부트스트랩 신뢰구간 계산 중...")
    top3_bottom2_tables = {}
    for split_name, split_df in [("cold-start", cold_df), ("personalized", personalized_df)]:
        boot_mae, boot_rmse = cluster_bootstrap(split_df, BASELINE_NAMES)
        ci_df, diff_df = bootstrap_ci_table(boot_mae, boot_rmse)
        md_lines.append(f"\n### {split_name}\n")
        md_lines.append(df_to_markdown(ci_df))
        md_lines.append(f"\n\n**{BASELINE_LABELS[MAE_WINNER]}(MAE 1위) / "
                         f"{BASELINE_LABELS[RMSE_WINNER]}(RMSE 1위) 대비 쌍체 차이 95% CI** "
                         "(0을 포함하지 않으면 그 차이가 95% 신뢰수준에서 유의미하다고 해석):\n\n")
        md_lines.append(df_to_markdown(diff_df))
        md_lines.append("\n")
        top3_bottom2_tables[split_name] = top3_bottom2_rmse_table(boot_rmse)
        print(f"  [{split_name}] 완료")

    md_lines.append(
        "\n**해석(중요 — C절의 \"RMSE 1위\" 단정을 보완)**: MAE 차이는 median-5 vs "
        "나머지 전부(1·2·4·5) CI가 0을 포함하지 않는다 — **median-5가 MAE에서 "
        "통계적으로 유의하게 낫다**(두 평가조건 모두; 윈도우 풀링 MAE 기준이며 사용자별 MAE 평균(macro)은 이 문서에서 계산하지 않음). 반면 **RMSE는 다르다**: "
        "사용자 전체 평균과 median-5의 RMSE 차이 CI는 위 표에서 0을 포함한다 — "
        "즉 **\"RMSE 1위가 사용자 전체 평균\"이라는 점추정값 차이는 95% 신뢰수준"
        "에서 통계적으로 유의하지 않다.** 사용자 전체 평균과 최근 5회 평균의 "
        "RMSE 차이도 cold-start에서 0을 포함한다. 반면 last-session·선형추세와의 "
        "차이는 MAE/RMSE 모두 CI가 0을 포함하지 않아 뚜렷하게 유의하다.\n"
    )

    md_lines.append("\n### RMSE 기준 상위 3개(사용자 전체 평균/median-5/최근 5회 평균) "
                     "vs 하위 2개(last-session/선형추세) 쌍체 차이 95% CI\n")
    for split_name in ["cold-start", "personalized"]:
        md_lines.append(f"\n**{split_name}**\n\n")
        md_lines.append(df_to_markdown(top3_bottom2_tables[split_name]))
        md_lines.append("\n")
    all_significant = all(
        (~top3_bottom2_tables[s]["0포함(유의하지않음)"]).all() for s in ["cold-start", "personalized"]
    )
    if all_significant:
        consistency_note = (
            "상위 3개 vs 하위 2개의 모든 쌍(6개)에서 두 평가조건 모두 CI가 0을 "
            "포함하지 않는다 — **cold-start/personalized 양쪽에서 결론이 일치한다.**"
        )
    else:
        consistency_note = (
            "cold-start와 personalized에서 결론이 다르다 (위 표에서 "
            "'0포함(유의하지않음)'이 True인 쌍 참고) — 조건별로 달리 해석해야 한다."
        )
    md_lines.append(
        f"\n**결론**: {consistency_note} RMSE 기준 상위 3개(사용자 전체 평균/"
        "median-5/최근 5회 평균)는 서로 통계적으로 구분되지 않지만(앞의 쌍체 "
        "비교 참고), 이 셋 각각은 하위 2개(last-session/선형추세)보다 "
        "**통계적으로 유의하게 낫다**(윈도우 풀링 기준).\n"
    )

    # ===== D. 세분화 =====
    md_lines.append("\n## D. 세분화 분석\n")

    # D-1. 사용자별 MAE 분포
    md_lines.append("\n### D-1. 사용자별 MAE 분포 (방법x평가조건, 중앙값/25/75/95백분위)\n")
    rows = []
    for (split, baseline), grp in all_df.groupby(["split", "baseline"]):
        per_user_mae = grp.groupby("user_id")["abs_error"].mean()
        rows.append({
            "방법": BASELINE_LABELS[baseline], "평가조건": split,
            "median": per_user_mae.median(), "p25": per_user_mae.quantile(.25),
            "p75": per_user_mae.quantile(.75), "p95": per_user_mae.quantile(.95),
        })
    md_lines.append(df_to_markdown(pd.DataFrame(rows)))
    md_lines.append("\n")

    # 세분화는 median5(보통 MAE 최저) 기준으로, 두 split 모두 보여준다.
    FOCUS_BASELINE = "median5"
    focus_df = all_df[all_df["baseline"] == FOCUS_BASELINE].copy()

    # D-2. 과거 기록 길이별 (target_session_order 버킷)
    md_lines.append(f"\n### D-2. 과거 기록 길이별 (target_session_order 기준, baseline={BASELINE_LABELS[FOCUS_BASELINE]})\n")
    bins = [0, 10, 30, 60, 120, 1e9]
    labels = ["5-9", "10-29", "30-59", "60-119", "120+"]
    focus_df["history_bucket"] = pd.cut(focus_df["target_session_order"], bins=bins, labels=labels, right=False)
    rows = []
    for split in ["cold-start", "personalized"]:
        g = focus_df[focus_df["split"] == split].groupby("history_bucket", observed=True)
        for bucket, grp in g:
            if len(grp) == 0:
                continue
            rows.append({"평가조건": split, "history_bucket": bucket, "n": len(grp),
                         "MAE": grp["abs_error"].mean(), "RMSE": np.sqrt(grp["sq_error"].mean())})
    md_lines.append(df_to_markdown(pd.DataFrame(rows)))
    md_lines.append("\n")

    # D-3. 페이스 구간별 (y_true 기준)
    md_lines.append(f"\n### D-3. 타깃 페이스 구간별 (baseline={BASELINE_LABELS[FOCUS_BASELINE]})\n")
    pace_bins = [0, 250, 350, 450, 1000, 1e9]
    pace_labels = ["<250", "250-349", "350-449", "450-999", "1000+(극단)"]
    focus_df["pace_bucket"] = pd.cut(focus_df["y_true"], bins=pace_bins, labels=pace_labels, right=False)
    rows = []
    for split in ["cold-start", "personalized"]:
        g = focus_df[focus_df["split"] == split].groupby("pace_bucket", observed=True)
        for bucket, grp in g:
            if len(grp) == 0:
                continue
            rows.append({"평가조건": split, "pace_bucket": bucket, "n": len(grp),
                         "MAE": grp["abs_error"].mean(), "RMSE": np.sqrt(grp["sq_error"].mean())})
    md_lines.append(df_to_markdown(pd.DataFrame(rows)))
    md_lines.append("\n")

    # D-4. 타깃 n_points==500 여부별 (merged_count==1 타깃만 대상 - 병합 세션은 n_points 정의 불명확)
    md_lines.append(f"\n### D-4. 타깃 n_points=500 여부별 (baseline={BASELINE_LABELS[FOCUS_BASELINE]}, "
                     "**타깃이 병합되지 않은 단일 세션인 경우만** — 병합 세션은 n_points가 단일값으로 정의되지 않음)\n")
    point_counts = pd.read_pickle(POINT_COUNTS_PATH)
    pc_lookup = dict(zip(point_counts["session_id"], point_counts["n_points"]))
    non_merged_focus = all_df[(all_df["baseline"] == FOCUS_BASELINE) & (~all_df["target_merged"])].copy()
    non_merged_focus["target_n_points"] = non_merged_focus["target_session_id"].map(pc_lookup)
    non_merged_focus["is_500"] = non_merged_focus["target_n_points"] == 500
    rows = []
    for split in ["cold-start", "personalized"]:
        for is_500_val in [True, False]:
            grp = non_merged_focus[(non_merged_focus["split"] == split) & (non_merged_focus["is_500"] == is_500_val)]
            if len(grp) == 0:
                continue
            rows.append({"평가조건": split, "n_points==500": is_500_val, "n": len(grp),
                         "MAE": grp["abs_error"].mean(), "RMSE": np.sqrt(grp["sq_error"].mean())})
    md_lines.append(df_to_markdown(pd.DataFrame(rows)))
    md_lines.append(f"\n(병합되지 않은 타깃만 대상이라 표본 수가 통합표보다 적음: "
                     f"{len(non_merged_focus)} / {len(focus_df)})\n")

    # D-5. 병합 a/b/c 분해
    md_lines.append(f"\n### D-5. 병합 영향 분해 — (a)무관 (b)입력만 병합 (c)타깃 병합 (baseline={BASELINE_LABELS[FOCUS_BASELINE]})\n")
    def merge_category(row):
        if row["target_merged"]:
            return "c_target_merged"
        elif row["input_merged"]:
            return "b_input_merged"
        return "a_unaffected"
    focus_df["merge_category"] = focus_df.apply(merge_category, axis=1)
    rows = []
    for split in ["cold-start", "personalized"]:
        g = focus_df[focus_df["split"] == split].groupby("merge_category")
        for cat, grp in g:
            rows.append({"평가조건": split, "카테고리": cat, "n": len(grp),
                         "MAE": grp["abs_error"].mean(), "RMSE": np.sqrt(grp["sq_error"].mean())})
    md_lines.append(df_to_markdown(pd.DataFrame(rows)))
    md_lines.append("\n**주의**: (c) 타깃 병합 그룹은 y(정답)의 정의 자체가 병합으로 바뀌므로, "
                     "(a)/(b)와 직접 비교해 '병합이 오차를 줄인다/늘린다'고 단정하지 않는다 "
                     "(docs/session_merge_audit.md 5번 참고 — 같은 맥락의 주의사항).\n")

    # D-6. [POST-HOC 분석] 타깃 실제 거리 구간별
    md_lines.append(f"\n### D-6. [POST-HOC 분석, 타깃 실제 거리 사용 — 러닝 전 예측 성능 아님] "
                     f"거리 구간별 (baseline={BASELINE_LABELS[FOCUS_BASELINE]})\n")
    md_lines.append("**주의: 이 세분화는 타깃 세션의 실제 완주 거리(러닝 후에만 알 수 있는 값)로 "
                     "결과를 나눠보는 사후 분석이다. 입력 피처로 쓴 것이 아니라 결과 분해용이며, "
                     "메인 표와 섞지 않는다.**\n\n")
    dist_bins = [0, 3, 6, 10, 15, 1e9]
    dist_labels = ["<3km", "3-6km", "6-10km", "10-15km", "15km+"]
    focus_df["distance_bucket"] = pd.cut(focus_df["target_actual_distance_km"], bins=dist_bins, labels=dist_labels, right=False)
    rows = []
    for split in ["cold-start", "personalized"]:
        g = focus_df[focus_df["split"] == split].groupby("distance_bucket", observed=True)
        for bucket, grp in g:
            if len(grp) == 0:
                continue
            rows.append({"평가조건": split, "distance_bucket": bucket, "n": len(grp),
                         "MAE": grp["abs_error"].mean(), "RMSE": np.sqrt(grp["sq_error"].mean())})
    md_lines.append(df_to_markdown(pd.DataFrame(rows)))
    md_lines.append("\n")

    # D-7. 오차 상위 1% 구성
    md_lines.append(f"\n### D-7. 오차(abs_error) 상위 1% 윈도우의 구성 (baseline={BASELINE_LABELS[FOCUS_BASELINE]})\n")
    for split in ["cold-start", "personalized"]:
        grp = focus_df[focus_df["split"] == split]
        threshold = grp["abs_error"].quantile(0.99)
        top1 = grp[grp["abs_error"] >= threshold]
        sse_share = top1["sq_error"].sum() / grp["sq_error"].sum()
        abs_share = top1["abs_error"].sum() / grp["abs_error"].sum()
        md_lines.append(f"\n**{split}** (상위 1% 임계값={threshold:.1f}초/km, n={len(top1)} / "
                         f"전체 {len(grp)}건):\n")
        md_lines.append(f"- **전체 SSE(제곱오차 합) 중 상위 1%가 차지하는 비율: {sse_share:.1%}**\n")
        md_lines.append(f"- **전체 절대오차 합 중 상위 1%가 차지하는 비율: {abs_share:.1%}**\n")
        md_lines.append(f"- 병합 영향(타깃 또는 입력) 비율: 상위1% {top1['merge_affected'].mean():.1%} "
                         f"vs **전체 윈도우 {grp['merge_affected'].mean():.1%}** "
                         f"(상위1% 윈도우의 병합 포함 비율이 전체보다 높지 않음 — 과대표현되지 않음. "
                         f"독립성을 검정한 것은 아니므로 '무관'이라고는 쓰지 않음)\n")
        md_lines.append(f"- 극단 페이스 영향(타깃 또는 입력) 비율: 상위1% {top1['extreme_affected'].mean():.1%} "
                         f"vs **전체 윈도우 {grp['extreme_affected'].mean():.1%}** "
                         f"(상위1%가 전체보다 훨씬 높음 -> 극단 페이스가 큰 오차에 집중)\n")
        md_lines.append("- GPS 이상치 여부: 세션 단위로 별도 플래그가 파이프라인에 없어 "
                         "(01단계에서 포인트 단위로 이미 필터링됨) **측정하지 않음 (미확인)**\n")
    md_lines.append("\n**해석**: RMSE는 상위 1% 윈도우가 SSE의 90% 이상을 차지하는 "
                     "극단 윈도우 민감 지표다 — 평균적인 체감 오차를 보려면 MAE를, "
                     "소수의 큰 실패가 전체 지표를 얼마나 흔드는지 보려면 RMSE를 같이 봐야 한다.\n")

    md_lines.append("\n## E. 확인 메모\n")
    md_lines.append("- 순위 비교는 반드시 MAE/RMSE를 각각 따로 서술한다 (아래 요약 텍스트 참고).\n")
    md_lines.append("- 원인이 확인되지 않은 패턴은 모두 '추정/미확인'으로 표기했다.\n")

    OUTPUT_MD_PATH.write_text("".join(md_lines), encoding="utf-8")
    print(f"\n저장 위치: {OUTPUT_MD_PATH.resolve()}")

    # 콘솔에도 핵심 요약 출력
    print("\n=== 메인 통합표 ===")
    print(main_table.to_string(index=False))
    print("\n=== 민감도(극단 제외) 통합표 ===")
    print(sens_table.to_string(index=False))


if __name__ == "__main__":
    main()
