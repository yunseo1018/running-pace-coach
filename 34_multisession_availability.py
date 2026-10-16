# -*- coding: utf-8 -*-
"""
10월 3주차 7번: 다중 세션 집계 코칭 평가 — 가용성 점검(기술통계). **모델 학습·평가·baseline 오차 계산은 하지 않는다.**
설계안(앵커 격자, 입력 28일, 후보 조합, 충분성 기준, 선택 규칙)은 실행 전에 docs/week3_predeclaration.md 10절에 고정했다.
정제 세션 데이터(data/user_sequences.pkl, 적격 사용자)에서 (사용자, 앵커 주) 표본 수와 사용자 수를 센다.

앵커 t = (epoch 초를 UTC로 해석한) 월요일 00:00의 7일 격자(+ 격자 시작 요일 오프셋 0~6일 민감도).
입력 세션: t-28일 <= end_time < t (end_time = start_time + duration_sec), t 시점에 끝나지 않은 세션은 제외.
타깃 주: 시작 시각이 [t, t+7일)인 세션.

출력: docs/multisession_coaching_design.md 안의 <!-- AVAILABILITY_START --> ~ <!-- AVAILABILITY_END --> 블록만 갱신한다.
sklearn 의존 없음(numpy/pandas).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config

DOC_PATH = Path("docs") / "multisession_coaching_design.md"
DAY = 86400
WEEK = 7 * DAY
WIN = 28 * DAY
MONDAY0 = 4 * DAY  # epoch 0(1970-01-01)은 목요일 -> 첫 월요일 00:00 UTC = 4일 후
COMBOS = [(3, 1), (3, 2), (4, 1), (4, 2)]  # (최소 입력 세션 수, 타깃 주 최소 세션 수) — 사전 고정
MIN_TEST_USERS, MIN_TEST_ANCHORS = 100, 1000        # cold-start test 충분성 기준(사전 고정)
MIN_HOLD_USERS, MIN_HOLD_ANCHORS = 300, 1000        # personalized hold-out 충분성 기준(사전 고정)
HISTORY_RATIO = 0.8


def log(*a):
    print(*a, flush=True)


def md_table(rows):
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def user_anchor_counts(start, end, offset_days):
    """한 사용자의 격자 앵커별 (입력 세션 수, 타깃 주 세션 수, t 시점에 진행 중인 세션 수, 앵커 t 배열)."""
    base = MONDAY0 + offset_days * DAY
    k0 = int(np.ceil((start.min() - base) / WEEK))
    k1 = int(np.floor((start.max() - base) / WEEK))
    if k1 < k0:
        return None
    t = base + np.arange(k0, k1 + 1) * WEEK
    e_sorted = np.sort(end)
    s_sorted = np.sort(start)
    n_in = np.searchsorted(e_sorted, t, "left") - np.searchsorted(e_sorted, t - WIN, "left")
    n_tg = np.searchsorted(s_sorted, t + WEEK, "left") - np.searchsorted(s_sorted, t, "left")
    n_straddle = np.searchsorted(s_sorted, t, "left") - np.searchsorted(e_sorted, t, "left")  # start < t 이고 end >= t
    return t, n_in, n_tg, n_straddle, e_sorted, s_sorted


def main():
    df = pd.read_pickle(config.USER_SEQUENCES_PATH)
    assert df["user_id"].nunique() == config.MAIN_EXPECTED_ELIGIBLE_USERS
    df = df.sort_values(["user_id", "start_time"])
    users = {}
    for uid, g in df.groupby("user_id", sort=True):
        s = g["start_time"].to_numpy(dtype=np.int64)
        users[uid] = (s, s + g["duration_sec"].to_numpy(dtype=np.int64))
    train_ids = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip()} & set(users)
    test_ids = {int(x) for x in Path(config.TEST_USERS_PATH).read_text(encoding="utf-8").split() if x.strip()} & set(users)
    assert train_ids.isdisjoint(test_ids)
    n_users_all = len(users)

    def collect(offset_days):
        """반환: {combo: {uid: 앵커 t 배열(시간순)}}, 후보(입력 조건만 충족) 통계, 진행 중 세션 통계."""
        res = {c: {} for c in COMBOS}
        cand = {mi: np.zeros(3, dtype=np.int64) for mi in (3, 4)}  # [후보 앵커 수, 타깃 주 0세션, 타깃 주 >=2세션]
        cand_ge1 = {mi: 0 for mi in (3, 4)}
        straddle = {c: [0, 0] for c in COMBOS}  # [유효 앵커 수, 진행 중 세션 포함 앵커 수]
        for uid, (s, e) in users.items():
            r = user_anchor_counts(s, e, offset_days)
            if r is None:
                continue
            t, n_in, n_tg, n_st, e_sorted, s_sorted = r
            for mi in (3, 4):
                m = n_in >= mi
                cand[mi][0] += int(m.sum())
                cand[mi][1] += int((m & (n_tg == 0)).sum())
                cand[mi][2] += int((m & (n_tg >= 2)).sum())
                cand_ge1[mi] += int((m & (n_tg >= 1)).sum())
            for (mi, mt) in COMBOS:
                m = (n_in >= mi) & (n_tg >= mt)
                if m.any():
                    tt = t[m]
                    # 누수 규칙 assert: 입력의 마지막 종료 시각 < t, 타깃 첫 시작 시각 >= t
                    idx_in = np.searchsorted(e_sorted, tt, "left") - 1
                    assert (idx_in >= 0).all() and (e_sorted[idx_in] < tt).all(), "입력 세션 end_time < t 위반"
                    idx_tg = np.searchsorted(s_sorted, tt, "left")
                    assert (s_sorted[idx_tg] >= tt).all(), "타깃 세션 start_time >= t 위반"
                    res[(mi, mt)][uid] = tt
                    straddle[(mi, mt)][0] += int(m.sum())
                    straddle[(mi, mt)][1] += int((m & (n_st > 0)).sum())
        return res, cand, cand_ge1, straddle

    def split_counts(anchors):
        """cold-start train/test, personalized history/hold-out 사용자·앵커 수."""
        tr_u = [u for u in anchors if u in train_ids]
        te_u = [u for u in anchors if u in test_ids]
        n_tr = sum(len(anchors[u]) for u in tr_u)
        n_te = sum(len(anchors[u]) for u in te_u)
        h_users = h_anch = ho_users = ho_anch = 0
        for u, t in anchors.items():
            n = len(t)
            n_hist = int(n * HISTORY_RATIO)
            n_hist = min(max(n_hist, 0), n - 1) if n >= 2 else 0   # 24단계 split_history_and_holdout와 동일 규칙
            if n_hist > 0:
                h_users += 1
                h_anch += n_hist
            if n - n_hist > 0:
                ho_users += 1
                ho_anch += n - n_hist
        return {"train": (len(tr_u), n_tr), "test": (len(te_u), n_te), "hist": (h_users, h_anch), "hold": (ho_users, ho_anch)}

    log("오프셋 0(월요일 00:00 UTC 격자) 집계...")
    res0, cand0, cand_ge1_0, strad0 = collect(0)
    rows = []
    passing = []
    for c in COMBOS:
        a = res0[c]
        sc = split_counts(a)
        n_anchor = sum(len(v) for v in a.values())
        ok = (sc["test"][0] >= MIN_TEST_USERS and sc["test"][1] >= MIN_TEST_ANCHORS and sc["hold"][0] >= MIN_HOLD_USERS and sc["hold"][1] >= MIN_HOLD_ANCHORS)
        if ok:
            passing.append(c)
        rows.append({"입력 28일 최소 세션": c[0], "타깃 주 최소 세션": c[1], "(사용자,앵커) 표본 수": f"{n_anchor:,}", "사용자 수(앵커 ≥1)": f"{len(a):,} / {n_users_all}",
                     "cold-start train 사용자/앵커": f"{sc['train'][0]:,} / {sc['train'][1]:,}", "cold-start test 사용자/앵커": f"{sc['test'][0]:,} / {sc['test'][1]:,}",
                     "personalized history 사용자/앵커": f"{sc['hist'][0]:,} / {sc['hist'][1]:,}", "personalized hold-out 사용자/앵커": f"{sc['hold'][0]:,} / {sc['hold'][1]:,}",
                     "충분성 기준": "통과" if ok else "미달"})
    block = ["### 5-1. 후보 조합별 (사용자, 앵커 주) 표본 수와 사용자 수 (월요일 00:00 UTC 격자)\n",
             f"정제 세션 데이터의 적격 사용자 {n_users_all}명(세션 6개 이상) 기준. 앵커 = 입력·타깃 조건을 모두 만족하는 격자 시점. "
             f"cold-start는 기존 사용자 분할(train/test), personalized는 사용자별 앵커를 시간순으로 앞 80%(history)/뒤 20%(hold-out)로 나눈 값(`int(0.8n)` 규칙, 앵커가 1개인 사용자는 전부 hold-out). "
             f"**충분성 기준(사전 고정)**: cold-start test 사용자 ≥ {MIN_TEST_USERS} & 앵커 ≥ {MIN_TEST_ANCHORS:,}, personalized hold-out 사용자 ≥ {MIN_HOLD_USERS} & 앵커 ≥ {MIN_HOLD_ANCHORS:,}.\n\n",
             md_table(rows), "\n\n"]
    if passing:
        prop = sorted(passing)[-1]
        block.append(f"- **선택 규칙(사전 고정: 기준 통과 조합 중 (최소 입력 세션 수, 최소 타깃 세션 수)가 사전식으로 가장 큰 조합)에 따른 주 설정 제안**: 입력 28일 최소 세션 {prop[0]}, 타깃 주 최소 세션 {prop[1]}. "
                     f"기준을 통과한 조합: {', '.join(f'({a},{b})' for a, b in sorted(passing))}. 미달 조합은 배제 근거로 사용한다.\n\n")
    else:
        block.append("- **기준을 통과한 조합이 없다** — 사전 선언에 따라 설계를 확정하지 않고 보고한다.\n\n")

    # 후보 앵커(입력 조건만 충족) 중 타깃 주 구성
    rows = []
    for mi in (3, 4):
        n_c, n0, n2 = (int(x) for x in cand0[mi])
        g1 = cand_ge1_0[mi]
        rows.append({"최소 입력 세션": mi, "후보 앵커(입력 조건만 충족)": f"{n_c:,}", "타깃 주 세션 0개": f"{n0:,} ({100 * n0 / n_c:.1f}%)",
                     "타깃 주 세션 ≥1": f"{g1:,} ({100 * g1 / n_c:.1f}%)", "타깃 주 세션 ≥2": f"{n2:,} ({100 * n2 / n_c:.1f}%)"})
    block += ["### 5-2. 조건부 평가의 선택 편향 가늠: 입력 조건은 만족하지만 타깃 주에 러닝이 없는 앵커 비율\n",
              "평가 표본은 '타깃 주에 러닝이 있었던 앵커'로 조건부이므로(미래 활동에 의존한 선택), 입력 조건을 만족한 앵커 중 타깃 주에 러닝이 있었던 비율을 기록한다. 이 비율은 '다음 주에 달릴지'를 예측하지 않는다는 한계의 크기를 보여준다(원인 해석 없음).\n\n",
              md_table(rows), "\n\n"]

    # 사용자별 앵커 수 분포
    rows = []
    for c in COMBOS:
        v = np.array([len(x) for x in res0[c].values()])
        rows.append({"입력 최소 세션": c[0], "타깃 주 최소 세션": c[1], "앵커 ≥1 사용자": len(v), "사용자당 앵커 수 p25": round(float(np.quantile(v, .25)), 1), "중앙값": round(float(np.median(v)), 1),
                     "p75": round(float(np.quantile(v, .75)), 1), "p90": round(float(np.quantile(v, .9)), 1), "최대": int(v.max()), "평균": round(float(v.mean()), 1)})
    block += ["### 5-3. 사용자당 앵커 수 분포 (앵커 ≥ 1인 사용자 기준)\n", "같은 사용자의 앵커들은 입력 창이 겹치므로 독립이 아니다(3절). 소수 사용자가 표본 수를 좌우할 수 있음을 보기 위한 기술통계다.\n\n", md_table(rows), "\n\n"]

    # 격자 시작 요일 민감도
    log("격자 오프셋 1~6 집계...")
    tot = {c: [sum(len(v) for v in res0[c].values())] for c in COMBOS}
    for off in range(1, 7):
        r_off, *_ = collect(off)
        for c in COMBOS:
            tot[c].append(sum(len(v) for v in r_off[c].values()))
    rows = []
    for c in COMBOS:
        t = tot[c]
        rows.append({"입력 최소 세션": c[0], "타깃 주 최소 세션": c[1], "오프셋 0(월)": f"{t[0]:,}", **{f"+{i}일": f"{t[i]:,}" for i in range(1, 7)},
                     "최소~최대": f"{min(t):,}~{max(t):,}", "최대/최소 − 1": f"{100 * (max(t) / min(t) - 1):.1f}%"})
    block += ["### 5-4. 타임존 한계 확인: 격자 시작 요일(UTC 월요일 00:00 + 오프셋 0~6일)에 따른 전체 (사용자, 앵커) 표본 수\n",
              "타임존 정보가 없어 UTC 격자를 쓴다. 격자 경계를 하루씩 옮겨도 표본 수가 얼마나 달라지는지만 본다(경계 위치에 대한 민감도의 기술통계; 현지 시각과의 어긋남 자체를 검증한 것은 아님).\n\n", md_table(rows), "\n\n"]

    # 진행 중 세션
    rows = []
    for c in COMBOS:
        n, st = strad0[c]
        rows.append({"입력 최소 세션": c[0], "타깃 주 최소 세션": c[1], "유효 앵커": f"{n:,}", "t 시점에 진행 중인 세션이 있는 앵커": f"{st:,} ({100 * st / n:.2f}%)"})
    block += ["### 5-5. 누수 규칙 점검: t 시점에 끝나지 않은 세션(진행 중)이 있는 앵커\n",
              "이런 세션은 입력에도 타깃에도 넣지 않는다(2-1절). 모든 유효 앵커에서 `최대 입력 end_time < t`, `최소 타깃 start_time ≥ t`를 assert로 검증했다(통과).\n\n", md_table(rows), "\n"]

    text = DOC_PATH.read_text(encoding="utf-8")
    a, b = "<!-- AVAILABILITY_START -->", "<!-- AVAILABILITY_END -->"
    assert a in text and b in text
    i, j = text.index(a), text.index(b)
    new = text[:i] + a + "\n" + "".join(block) + text[j:]
    DOC_PATH.write_text(new, encoding="utf-8")
    log("갱신:", DOC_PATH)


if __name__ == "__main__":
    main()
