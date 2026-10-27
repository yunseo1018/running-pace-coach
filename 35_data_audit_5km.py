# -*- coding: utf-8 -*-
"""
10월 4주차(8주차) 1번: 이번 주 채팅 보고 값을 재현 가능하게 다시 계산해 docs/week4_data_audit.md에 저장한다. (읽기 전용 점검)

- 세션 표 컬럼·자료형·결측, 평균 심박 결측(전체/5km), 성별 분포, 원본 파일 줄 수·크기(버전 판단 근거)
- 5km 세션(4.5~5.5km, 1회) 수·사용자 수, 기준 시점 개수표(전체/train/test)
- 2번(개수만): 같은 사람 비교 가능성, 다음 7일 5km 2회 이상, 5km 쌍 사이 쉰 날수 구간별 쌍 수·사용자 수

**기록 변화(초) 같은 결과값의 분포·평균은 계산하지 않는다.** 개수만 센다(사전 선언 전 결과 열람 금지).
실제 행 값, 사용자/세션 ID, 좌표는 출력·저장하지 않는다.

기준 시점 t = start_time(epoch 초)을 UTC로 가정한 월요일 00:00의 7일 격자(UTC 여부는 미확인, 36번 참고).
입력: t-28일 <= end_time < t (end_time = start_time + duration_sec). 타깃: 시작 시각 ∈ [t, t+7일).

출력: docs/week4_data_audit.md (36번이 쓰는 <!-- TZ_START --> ~ <!-- TZ_END --> 블록은 보존).
sklearn 의존 없음(numpy/pandas).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config

DOC_PATH = Path("docs") / "week4_data_audit.md"
RAW_PATH = Path("endomondoHR.json")
TZ_START, TZ_END = "<!-- TZ_START -->", "<!-- TZ_END -->"
DAY = 86400
WEEK = 7 * DAY
WIN = 28 * DAY
MONDAY0 = 4 * DAY  # epoch 0(1970-01-01)은 목요일 -> 첫 월요일 00:00 UTC = 4일 후
LO, HI = 4.5, 5.5
SPORT_RUN_MARKER = b"'sport': 'run'"
GAP_BINS = [(0, 7), (7, 14), (14, 28), (28, 56), (56, None)]
GAP_LABELS = ["[0,7)", "[7,14)", "[14,28)", "[28,56)", "56일 이상"]


def md_table(rows):
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def n(x):
    return f"{int(x):,}"


def raw_file_facts():
    size = RAW_PATH.stat().st_size
    lines = runs = 0
    with open(RAW_PATH, "rb") as f:
        for line in f:
            lines += 1
            if SPORT_RUN_MARKER in line:
                runs += 1
    return size, lines, runs


def user_groups(d):
    tr = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    te = {int(x) for x in Path(config.TEST_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    assert tr.isdisjoint(te)
    users = d["user_id"].unique()
    grp = {}
    for u in users:
        grp[int(u)] = "train" if int(u) in tr else ("test" if int(u) in te else "분할 밖(적격 아님)")
    return grp, len(tr), len(te)


def anchor_table(d):
    """사용자별 앵커 개수. 반환: DataFrame(uid, anchors, in4, A, B, A2, B2)."""
    is5_all = ((d["actual_distance_km"] >= LO) & (d["actual_distance_km"] <= HI)).to_numpy()
    d = d.assign(_is5=is5_all).sort_values(["user_id", "start_time"])
    res = []
    for uid, g in d.groupby("user_id", sort=True):
        s = g["start_time"].to_numpy(np.int64)
        e = s + g["duration_sec"].to_numpy(np.int64)
        is5 = g["_is5"].to_numpy()
        k0 = int(np.ceil((s.min() - MONDAY0) / WEEK))
        k1 = int(np.floor((s.max() - MONDAY0) / WEEK))
        if k1 < k0:
            res.append((int(uid), 0, 0, 0, 0, 0, 0))
            continue
        t = MONDAY0 + np.arange(k0, k1 + 1) * WEEK
        eo = np.argsort(e, kind="stable")
        es = e[eo]
        lo_i, hi_i = np.searchsorted(es, t - WIN, "left"), np.searchsorted(es, t, "left")
        n_in = hi_i - lo_i
        c5e = np.concatenate([[0], np.cumsum(is5[eo])])
        n5_in = c5e[hi_i] - c5e[lo_i]
        so = np.argsort(s, kind="stable")
        ss = s[so]
        a, b = np.searchsorted(ss, t, "left"), np.searchsorted(ss, t + WEEK, "left")
        c5s = np.concatenate([[0], np.cumsum(is5[so])])
        n5_tg = c5s[b] - c5s[a]
        A = (n_in >= 4) & (n5_tg >= 1)
        B = A & (n5_in >= 1)
        res.append((int(uid), len(t), int((n_in >= 4).sum()), int(A.sum()), int(B.sum()),
                    int((B & (n5_tg >= 2)).sum()), int((A & (n5_tg >= 2)).sum())))
    return pd.DataFrame(res, columns=["uid", "anchors", "in4", "A", "B", "B_tg2", "A_tg2"])


def dist5(v):
    v = np.asarray(v)
    return "n/a" if len(v) == 0 else f"{int(v.min())} / {np.percentile(v, 25):g} / {np.median(v):g} / {np.percentile(v, 75):g} / {int(v.max())}"


def rest_pairs(d):
    """같은 사용자의 연속된 5km 쌍 중 사이에 다른 달리기 세션이 없는 쌍의 쉰 날수(일). 기록 차이는 계산하지 않는다."""
    d = d.sort_values(["user_id", "start_time"])
    out = []
    for uid, g in d.groupby("user_id", sort=True):
        s = g["start_time"].to_numpy(np.int64)
        e = s + g["duration_sec"].to_numpy(np.int64)
        idx5 = np.flatnonzero(((g["actual_distance_km"] >= LO) & (g["actual_distance_km"] <= HI)).to_numpy())
        for i, j in zip(idx5[:-1], idx5[1:]):
            if j == i + 1:  # 시작 시각 정렬상 바로 이웃 = 사이에 다른 세션 없음
                out.append((int(uid), (s[j] - e[i]) / DAY))
    return pd.DataFrame(out, columns=["uid", "gap_days"])


def main():
    d = config.load_main_sessions()
    grp, n_tr_ids, n_te_ids = user_groups(d)
    d["grp"] = d["user_id"].map(lambda u: grp[int(u)])
    is5 = (d["actual_distance_km"] >= LO) & (d["actual_distance_km"] <= HI)
    d5 = d[is5]
    out = ["# 10월 4주차(8주차) 데이터 점검 기록 (5km 과제 준비)", "",
           "생성: `35_data_audit_5km.py` (읽기 전용·개수만). 시각 기준 점검은 `36_timezone_check.py`가 아래 TZ 블록에 기록한다.",
           "**기록 변화(초) 등 결과값의 분포·평균은 계산하지 않았다(사전 선언 전 결과 열람 금지). 실제 행 값·ID·좌표는 싣지 않았다.**", "",
           "기준 시점 t는 `start_time`(epoch 초)을 UTC로 가정한 월요일 00:00의 7일 격자이며, 이 가정은 미확인이다(TZ 블록 참고). "
           "입력 구간은 t−28일 ≤ end_time < t(end_time = start_time + duration_sec), 타깃 구간은 시작 시각 ∈ [t, t+7일)이다. "
           "5km 세션은 한 번 달린 거리가 4.5~5.5km인 세션 1회다.", ""]

    # 1) 세션 표
    out += ["## 1. 세션 표 (`data/run_sessions_deduped_merged.pkl`)", "",
            f"{n(len(d))}행, 사용자 {n(d['user_id'].nunique())}명.", ""]
    meaning = {
        "user_id": "원본 `userId`", "session_id": "원본 `id`(병합 세션의 부여 규칙은 확인 불가)",
        "start_time": "원본 `timestamp` 배열의 첫 값(epoch 초; UTC 여부 미확인)",
        "actual_distance_km": "GPS haversine 누적 거리(km), 러닝 후 확정값", "duration_sec": "경과 시간(초, 정지 포함)",
        "avg_pace_sec_per_km": "duration_sec / 거리(초/km)", "avg_heart_rate": "유효 범위 심박 평균(bpm)",
        "gender": "원본 `gender`(범주)", "merged_count": "병합된 조각 수", "merged_session_ids": "병합에 쓰인 원본 session id 목록",
    }
    rows = []
    for c in [c for c in d.columns if c != "grp"]:
        rows.append({"컬럼": c, "자료형": str(d[c].dtype), "결측": f"{d[c].isna().sum()} ({100 * d[c].isna().mean():.2f}%)",
                     "의미(단위)": meaning.get(c, "확인 불가")})
    out += [md_table(rows), "",
            "고도(누적 상승)는 세션 표에 없다. 원본에 `altitude` 배열이 있어 만들 수는 있으나 만들지 않았다(값 품질 확인 불가).", ""]

    # 2) 심박·성별
    out += ["## 2. 평균 심박 결측과 성별", "",
            md_table([
                {"대상": "전체 세션", "세션 수": n(len(d)), "평균 심박 결측": n(d["avg_heart_rate"].isna().sum())},
                {"대상": "5km 세션", "세션 수": n(len(d5)), "평균 심박 결측": n(d5["avg_heart_rate"].isna().sum())},
            ]), "",
            "결측이 0인 것은 파싱 단계(`01_parse_run_sessions.py:build_session_row`)에서 유효 심박이 없는 세션을 제외했기 때문이다. "
            "그 이유로 제외된 세션 수는 단계별로 집계되지 않아 **확인 불가**다.", "",
            md_table([{"성별": k, "세션 수": n(v), "비율": f"{100 * v / len(d):.2f}%"} for k, v in d["gender"].value_counts().items()]), "",
            "`unknown`은 결측이 아니라 원본의 범주 값이다.", ""]

    # 3) 원본 버전
    size, lines, runs = raw_file_facts()
    out += ["## 3. 원본 파일과 버전 판단 근거", "",
            md_table([
                {"항목": "파일", "값": "`endomondoHR.json`"},
                {"항목": "크기", "값": f"{size / 2**30:.2f} GiB"},
                {"항목": "줄 수(= 운동 건수, 한 줄에 한 건)", "값": n(lines)},
                {"항목": "`'sport': 'run'` 문자열을 포함한 줄 수", "값": n(runs)},
            ]), "",
            "공식 FitRec 배포본 건수는 raw 253,020 / filtered 167,373 / resampled 102,343으로 알려져 있다(출처: 사용자 제공 값, 이 저장소에서 대조 불가). "
            f"줄 수 {n(lines)}가 raw 건수와 같고 다른 두 버전과 다르므로 **raw 버전**으로 판단한다. 공식 파일 해시와의 대조는 **확인 불가**다.", ""]

    # 4) 5km와 기준 시점
    cnt = d5.groupby("user_id").size()
    out += ["## 4. 5km 세션과 기준 시점 개수표", "",
            f"5km 세션 {n(len(d5))}건({100 * len(d5) / len(d):.2f}%), 사용자 {n(d5['user_id'].nunique())}명. "
            f"사용자당 5km 세션 수(5km가 1회 이상인 사용자, 최소/25%/중앙값/75%/최대): {dist5(cnt.values)}.", ""]
    at = anchor_table(d)
    at["grp"] = at["uid"].map(grp)
    sets = [("전체", at), ("train", at[at.grp == "train"]), ("test", at[at.grp == "test"]), ("분할 밖(적격 아님)", at[at.grp == "분할 밖(적격 아님)"])]
    d5g = d5.groupby("grp").agg(s=("user_id", "size"), u=("user_id", "nunique"))
    rows = []
    for name, df in sets:
        g5 = (len(d5), d5["user_id"].nunique()) if name == "전체" else (
            (int(d5g.loc[name, "s"]), int(d5g.loc[name, "u"])) if name in d5g.index else (0, 0))
        rows.append({"집단": name, "사용자(표 안)": n(len(df)), "5km 세션": n(g5[0]), "5km 사용자": n(g5[1]),
                     "전체 기준 시점": n(df.anchors.sum()),
                     "① 직전28일 4회↑": f"{n(df.in4.sum())} / {n((df.in4 > 0).sum())}명",
                     "② ①+다음7일 5km≥1": f"{n(df.A.sum())} / {n((df.A > 0).sum())}명",
                     "③ ②+직전28일 5km≥1 (평가 대상)": f"{n(df.B.sum())} / {n((df.B > 0).sum())}명"})
    out += [md_table(rows), "",
            f"분할 파일의 사용자 수는 train {n_tr_ids}명, test {n_te_ids}명이다(세션 6개 미만 사용자는 표에 없을 수 있음). "
            "③이 평가 대상 자격(직전 28일 4회 이상 + 그중 5km 1회 이상 + 다음 7일 5km 1회 이상)이다. 셀은 `기준 시점 수 / 사용자 수`.", ""]

    # 2a
    out += ["## 5. 같은 사람 비교 가능성 (개수만)", "",
            "### 5-a. 평가 대상 기준 시점(③)의 사용자당 개수", ""]
    rows = []
    for name, df in sets[:3]:
        v = df[df.B > 0].B.to_numpy()
        rows.append({"집단": name, "③ 있는 사용자": n(len(v)), "사용자당 개수(최소/25%/중앙값/75%/최대)": dist5(v),
                     "2개 이상": n((v >= 2).sum()), "3개 이상": n((v >= 3).sum()), "5개 이상": n((v >= 5).sum())})
    out += [md_table(rows), "",
            "분포의 분모는 ③이 1개 이상인 사용자다. 같은 사용자의 기준 시점들은 28일 입력 창이 서로 겹칠 수 있어 독립이 아니다.", ""]

    # 2b
    out += ["### 5-b. 다음 7일 안 5km가 2회 이상인 기준 시점 (\"첫 번째 5km를 쓴다\" 규칙의 영향 확인용)", ""]
    rows = []
    for name, df in sets[:3]:
        rows.append({"집단": name, "③ 기준 시점": n(df.B.sum()), "③ 중 다음 7일 5km≥2": f"{n(df.B_tg2.sum())} ({100 * df.B_tg2.sum() / max(df.B.sum(), 1):.1f}%)",
                     "② 기준 시점": n(df.A.sum()), "② 중 다음 7일 5km≥2": f"{n(df.A_tg2.sum())} ({100 * df.A_tg2.sum() / max(df.A.sum(), 1):.1f}%)"})
    out += [md_table(rows), ""]

    # 2c
    out += ["### 5-c. 연속된 5km 쌍 사이의 쉰 날수 (사이에 다른 달리기 세션이 없는 쌍만; 기록 차이는 계산하지 않음)", "",
            "쉰 날수 = (두 번째 5km 시작 − 첫 번째 5km 종료) / 86,400초. 시작 시각 순서에서 두 5km 세션이 바로 이웃인 쌍만 쓴다.", ""]
    pr = rest_pairs(d)
    pr["grp"] = pr["uid"].map(grp)
    neg = int((pr.gap_days < 0).sum())

    def gbin(x):
        for (lo, hi), lab in zip(GAP_BINS, GAP_LABELS):
            if x >= lo and (hi is None or x < hi):
                return lab
        return "<0(겹침)"
    pr["bin"] = pr.gap_days.map(gbin)
    rows = []
    multi = {}
    for name in ["전체", "train", "test"]:
        sub = pr if name == "전체" else pr[pr.grp == name]
        for lab in GAP_LABELS + ["<0(겹침)"]:
            s2 = sub[sub.bin == lab]
            if lab == "<0(겹침)" and len(s2) == 0:
                continue
            rows.append({"집단": name, "쉰 날수 구간": lab, "쌍 수": n(len(s2)), "사용자 수": n(s2.uid.nunique())})
        nb = sub.groupby("uid").bin.nunique()
        multi[name] = (int((nb >= 2).sum()), sub.uid.nunique(), len(sub))
    out += [md_table(rows), "",
            md_table([{"집단": k, "쌍이 있는 사용자": n(v[1]), "전체 쌍 수": n(v[2]), "서로 다른 구간 2개 이상에 쌍이 있는 사용자": n(v[0])} for k, v in multi.items()]), "",
            f"겹침(쉰 날수 < 0) 쌍은 {neg}건이며 구간별 표에서 별도 행으로 둔다(0건이면 행 없음). 사용자 수 합은 구간마다 중복 집계되므로 더하지 않는다.", ""]

    # TZ 블록 보존
    tz_block = f"{TZ_START}\n## 6. 시각 기준 점검\n\n`36_timezone_check.py`를 아직 실행하지 않았다(`python run_all.py --only 36`; 선택 의존성 timezonefinder 필요).\n{TZ_END}"
    if DOC_PATH.exists():
        old = DOC_PATH.read_text(encoding="utf-8")
        if TZ_START in old and TZ_END in old:
            tz_block = old[old.index(TZ_START):old.index(TZ_END) + len(TZ_END)]
    out += [tz_block, ""]
    DOC_PATH.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC_PATH)


if __name__ == "__main__":
    main()
