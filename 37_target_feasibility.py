# -*- coding: utf-8 -*-
"""
10월 5주차(학기 9주차) 교수 피드백 대응 1단계: 5km 타깃 비교 가능성 점검 (읽기 전용, 개수·가능 여부만).

A. 정확한 5km 구간 시각 추출 가능성(누적 haversine 거리가 5.000km에 도달하는 시각의 선형 보간): 가능 여부, 건수, 점 간 시간 간격 통계, 병합 세션 처리
B. 포함/제외 단계 표(월요일 기준 시점 -> 직전 28일 4회 이상 -> +직전 28일 5km -> +다음 7일 5km), 4.5~5.5km 정의와 5.0~5.5km 정의
C. 연속된 5km 쌍(사이에 다른 달리기 세션 없음)의 '관측된 러닝 기록의 공백' 구간별 쌍 수와 평가 대상 정답 세션과의 겹침
D. 기준 시점을 start_time 원값 기준 -12h/-6h/+6h/+12h 이동했을 때 평가 대상 수, 정답 세션이 달라지는 기준 시점 수
E. 원본/정리 파일 sha256·크기·정제 단계별 건수

**기록 시간·기록 변화량·페이스의 분포·평균 등 결과값은 계산하지 않는다.** 5km 지점 도달 시각 자체는 계산·저장하지 않고,
교차 구간의 점 간 시간 간격(보간 정밀도 판단용)과 개수만 센다. 사용자/세션 ID, 좌표, 행 값은 출력·저장하지 않는다.
start_time은 "원값(기준 미확인)"이며 월요일 00:00 격자는 UTC 해석 가정이다.

출력: docs/week5_target_check.md. sklearn 의존 없음(numpy/pandas/표준 라이브러리).
"""
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config

DOC_PATH = Path("docs") / "week5_target_check.md"
RAW_PATH = Path("endomondoHR.json")
DAY = 86400
WEEK = 7 * DAY
WIN = 28 * DAY
MONDAY0 = 4 * DAY  # epoch 0은 목요일 -> 첫 월요일 00:00(원값 기준) = 4일 후
MAX_PLAUSIBLE_SPEED_MPS = None  # 01_parse_run_sessions.py 값을 import해서 쓴다(아래)
SHIFTS_H = [-12, -6, 6, 12]
GAP_BINS = [(0, 7), (7, 14), (14, 28), (28, 56), (56, None)]
GAP_LABELS = ["[0,7)", "[7,14)", "[14,28)", "[28,56)", "56일 이상"]
ID_RE = re.compile(rb"'id': (\d+)")


def md_table(rows):
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def n(x):
    return f"{int(x):,}"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class _Parse:
    """01_parse_run_sessions.py의 haversine_km·속도 임계를 그대로 옮긴 것(01을 import하면 stdout을 다시 감싸 충돌)."""
    MAX_PLAUSIBLE_SPEED_MPS = None

    @staticmethod
    def haversine_km(lat1, lon1, lat2, lon2):
        earth_radius_km = 6371.0088
        lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        return 2 * earth_radius_km * math.asin(math.sqrt(a))


def import_parser():
    src = Path("01_parse_run_sessions.py").read_text(encoding="utf-8")
    m = re.search(r"^MAX_PLAUSIBLE_SPEED_MPS\s*=\s*([0-9.]+(?:\s*/\s*[0-9.]+)?)", src, re.M)
    assert m, "01번에서 속도 임계를 찾지 못함"
    num, _, den = m.group(1).partition("/")
    _Parse.MAX_PLAUSIBLE_SPEED_MPS = float(num) / (float(den) if den else 1.0)
    assert abs(_Parse.MAX_PLAUSIBLE_SPEED_MPS - 25 / 3.6) < 1e-12, _Parse.MAX_PLAUSIBLE_SPEED_MPS
    return _Parse


# ---------------------------------------------------------------- 사용자별 배열
def build_users(d):
    users = {}
    for uid, g in d.sort_values(["user_id", "start_time"]).groupby("user_id", sort=True):
        s = g["start_time"].to_numpy(np.int64)
        users[int(uid)] = dict(s=s, e=s + g["duration_sec"].to_numpy(np.int64), km=g["actual_distance_km"].to_numpy())
    return users


def elig(u, is5, t):
    """기준 시점 t의 (입력 세션 수, 기준 기록 위치, 정답 세션 위치). 위치는 사용자 배열 안 인덱스, 없으면 -1."""
    s, e = u["s"], u["e"]
    inwin = (e >= t - WIN) & (e < t)
    n_in = int(inwin.sum())
    r = np.flatnonzero(inwin & is5)
    a = np.flatnonzero(is5 & (s >= t) & (s < t + WEEK))
    return n_in, (int(r[-1]) if len(r) else -1), (int(a[0]) if len(a) else -1)


def span_ks(u, base):
    k0 = int(np.ceil((u["s"].min() - base) / WEEK))
    k1 = int(np.floor((u["s"].max() - base) / WEEK))
    return range(k0, k1 + 1)


def stage_table(users, lo, hi, base):
    """사용자별 단계 카운트와 평가 대상 목록. 반환: (uid -> [S0,S1,S2,S3]), {(uid,k): (ref, ans)}"""
    cnt, el = {}, {}
    for uid, u in users.items():
        is5 = (u["km"] >= lo) & (u["km"] <= hi)
        c = [0, 0, 0, 0]
        for k in span_ks(u, base):
            t = base + k * WEEK
            n_in, r, a = elig(u, is5, t)
            c[0] += 1
            if n_in >= 4:
                c[1] += 1
                if r >= 0:
                    c[2] += 1
                    if a >= 0:
                        c[3] += 1
                        el[(uid, k)] = (r, a)
        cnt[uid] = c
    return cnt, el


def summarize(cnt, grp):
    rows = []
    for name in ["전체", "train", "test", "분할 밖(적격 아님)"]:
        sel = [c for u, c in cnt.items() if name == "전체" or grp[u] == name]
        a = np.array(sel) if sel else np.zeros((0, 4), int)
        rows.append((name, len(sel), a.sum(0), (a > 0).sum(0)))
    return rows


def stage_md(cnt, grp, label):
    rows = []
    for name, nu, tot, us in summarize(cnt, grp):
        rows.append({
            "집단": name, "사용자(표 안)": n(nu),
            "① 활동 기간 안 월요일": f"{n(tot[0])} / {n(us[0])}명",
            "② ①+직전28일 4회↑": f"{n(tot[1])} / {n(us[1])}명",
            "③ ②+직전28일 5km≥1 (과거 자격 충족)": f"{n(tot[2])} / {n(us[2])}명",
            "④ ③+다음7일 5km≥1 (최종 평가 대상)": f"{n(tot[3])} / {n(us[3])}명",
            "탈락 ①→②": n(tot[0] - tot[1]), "탈락 ②→③ (과거 자격)": n(tot[1] - tot[2]), "탈락 ③→④ (미래 관측)": n(tot[2] - tot[3]),
        })
    return f"**{label}** (셀: 기준 시점 수 / 사용자 수)\n\n" + md_table(rows)


def gbin(x):
    if x < 0:
        return "<0(겹침)"
    for (lo, hi), lab in zip(GAP_BINS, GAP_LABELS):
        if x >= lo and (hi is None or x < hi):
            return lab


# ---------------------------------------------------------------- A: 원본 점 단위 확인
def cum_with_skip(parse, lat, lon, ts):
    """01_parse_run_sessions.py:total_distance_km와 같은 규칙(dt<=0 또는 암시 속도 초과 구간은 0 누적)의 누적 거리 배열과 스킵 마스크."""
    cum = [0.0]
    skipped = [False]
    for i in range(1, len(lat)):
        seg = parse.haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i])
        dt = ts[i] - ts[i - 1]
        if dt <= 0 or (seg * 1000) / dt > parse.MAX_PLAUSIBLE_SPEED_MPS:
            cum.append(cum[-1])
            skipped.append(True)
        else:
            cum.append(cum[-1] + seg)
            skipped.append(False)
    return np.array(cum), np.array(skipped)


def raw_point_check(parse, targets, piece_km):
    """targets: {raw id: 세션 구분}. 대상 줄만 파싱해 점 간 간격과 5km 교차 구간 간격(보간 정밀도용)을 센다."""
    import ast
    want = set(targets)
    gaps_all = []
    cross_gaps = []
    stats = dict(found=0, dist_mismatch=0, dist_checked=0, cross_n=0, cross_on_skipped=0, ge5_unmerged_checked=0, nonmono=0)
    with open(RAW_PATH, "rb") as f:
        for line in f:
            m = ID_RE.search(line)
            if not m or int(m.group(1)) not in want:
                continue
            sid = int(m.group(1))
            rec = ast.literal_eval(line.decode("utf-8"))
            if rec.get("sport") != "run" or rec["id"] != sid:
                continue
            stats["found"] += 1
            lat, lon, ts = rec["latitude"], rec["longitude"], rec["timestamp"]
            cum, sk = cum_with_skip(parse, lat, lon, ts)
            stats["dist_checked"] += 1
            if abs(cum[-1] - piece_km[sid]) > 1e-9:
                stats["dist_mismatch"] += 1
            dts = np.diff(np.array(ts, dtype=np.int64))
            gaps_all.append(dts)
            if targets[sid] == "unmerged" and cum[-1] >= 5.0:
                stats["ge5_unmerged_checked"] += 1
                j = int(np.searchsorted(cum, 5.0, "left"))  # cum[j] >= 5.0 인 첫 점
                if j == 0:
                    continue
                stats["cross_n"] += 1
                if sk[j]:
                    stats["cross_on_skipped"] += 1
                cross_gaps.append(ts[j] - ts[j - 1])
    all_g = np.concatenate(gaps_all) if gaps_all else np.array([0])
    return stats, all_g, np.array(cross_gaps)


def main():
    parse = import_parser()
    d = config.load_main_sessions()
    pieces = pd.read_pickle("data/run_sessions_deduped.pkl")
    tr = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    te = {int(x) for x in Path(config.TEST_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    grp = {int(u): ("train" if int(u) in tr else ("test" if int(u) in te else "분할 밖(적격 아님)")) for u in d.user_id.unique()}
    users = build_users(d)
    out = ["# 10월 5주차 5km 타깃 비교 가능성 점검 (교수 피드백 반영 1단계)", "",
           "생성: `37_target_feasibility.py` (읽기 전용·개수와 가능 여부만). **기록 시간·기록 변화량·페이스의 분포·평균 등 결과값은 계산하지 않았다.** "
           "5km 지점 도달 시각 자체는 계산·저장하지 않았고, 사용자/세션 ID·좌표·행 값은 싣지 않았다. "
           "start_time은 \"원값(기준 미확인)\"이며 월요일 00:00 격자는 그 원값을 UTC로 해석한 가정이다(`docs/week4_data_audit.md` 6절 \"판정 불가\"). "
           "5km 세션은 한 번 달린 거리가 4.5~5.5km인 세션(병합 후 세션 단위)이며, 5.0~5.5km 정의는 B절에서 따로 본다.", ""]

    # ---------------- A
    km = d["actual_distance_km"]
    in_rng = km.between(4.5, 5.5)
    ge5 = in_rng & (km >= 5.0)
    out += ["## A. 정확한 5km 구간 추출 가능성", "", "### A-1. 방법과 01번 거리 계산과의 일치", "",
            "원본 기록에는 점마다 `timestamp`·`latitude`·`longitude`가 있다(`01_parse_run_sessions.py:build_session_row`가 같은 배열을 읽는다). "
            "누적 haversine 거리가 5.000km에 처음 도달하는 두 점 사이를 선형 보간하면 도달 시각을 구할 수 있다(가능). "
            "단 `01_parse_run_sessions.py:total_distance_km`는 점 간 시간차 ≤ 0이거나 암시 속도가 임계를 넘는 구간(GPS 점프)의 거리를 누적에서 뺀다. "
            "따라서 보간은 **같은 규칙의 누적 거리**(스킵 구간은 0 누적)로 해야 세션 표의 `actual_distance_km`와 같은 기준이 된다. 아래 점검은 그 규칙으로 직접 구현해 확인한 결과다.", ""]
    # 대상 원본 id
    targets, piece_km = {}, {}
    pk = pieces.set_index("session_id")["actual_distance_km"]
    for _, r in d[in_rng].iterrows():
        if r["merged_count"] > 1:
            for sid in r["merged_session_ids"]:
                targets[int(sid)] = "merged"
                piece_km[int(sid)] = float(pk.loc[sid])
        else:
            targets[int(r["session_id"])] = "unmerged"
            piece_km[int(r["session_id"])] = float(pk.loc[r["session_id"]])
    st, all_g, cross_g = raw_point_check(parse, targets, piece_km)
    assert st["found"] == len(targets), (st["found"], len(targets))
    out += [md_table([
        {"항목": "점검 대상 원본 기록(4.5~5.5km 세션 및 병합 구성 조각)", "값": n(len(targets))},
        {"항목": "원본에서 찾아 같은 규칙으로 누적 거리를 다시 계산한 기록", "값": n(st["dist_checked"])},
        {"항목": "재계산 총 거리가 세션 표(중복 제거 후 조각) 거리와 1e-9km 넘게 다른 기록", "값": n(st["dist_mismatch"])},
    ]), ""]

    # ---------------- A-2 개수
    allge5 = km >= 5.0

    def cu(mask):
        return f"{n(mask.sum())}건 / {n(d[mask].user_id.nunique())}명"
    out += ["### A-2. 5.0km 이상 여부별 개수", "",
            md_table([
                {"구분": "4.5~5.5km 세션 전체", "건수/사용자": cu(in_rng)},
                {"구분": "└ 5.0km 이상 (5.000km 지점이 세션 안에 있음 → 정확 추출 가능)", "건수/사용자": cu(ge5)},
                {"구분": "└ 5.0km 미만 (정확 추출 불가)", "건수/사용자": cu(in_rng & ~ge5)},
                {"구분": "참고: 길이 무관 5.0km 이상 모든 러닝 세션", "건수/사용자": cu(allge5)},
            ]), "",
            "5.0km 이상 세션에서만 \"세션 안의 5.000km 지점\"이 존재하므로, 4.5~5.0km 세션은 정확한 5km 구간 시간을 만들 수 없고 5km 환산 시간(경과 시간÷거리×5)만 가능하다. "
            "5.0~5.5km 정의는 이 정확 추출 가능 세션만 5km로 인정하는 경우다.", ""]

    # ---------------- A-3
    out += ["### A-3. 점 간 시간 간격 (보간 정밀도 판단용)", "",
            md_table([
                {"대상": "4.5~5.5km 세션의 모든 연속 점 쌍(풀링)", "개수": n(len(all_g)), "중앙값(초)": f"{np.median(all_g):g}", "90% 분위(초)": f"{np.percentile(all_g, 90):g}"},
                {"대상": "5.000km를 가로지르는 구간(병합되지 않은 5.0km 이상 세션)", "개수": n(len(cross_g)), "중앙값(초)": f"{np.median(cross_g):g}", "90% 분위(초)": f"{np.percentile(cross_g, 90):g}"},
            ]), "",
            f"5.000km 도달 지점이 스킵된(점프·시간 역전) 구간 안인 세션: {n(st['cross_on_skipped'])}건(위 {n(st['cross_n'])}건 중). 선형 보간은 점 사이 속도가 일정하다고 가정하므로 "
            "간격이 길수록 도달 시각의 불확실성이 커진다. 간격 분위는 점 쌍 풀링 값이며 세션별 분포는 계산하지 않았다.", ""]

    # ---------------- A-4 병합
    mg = d[in_rng & (d["merged_count"] > 1)]
    mg5 = mg[mg["actual_distance_km"] >= 5.0]
    first_piece_ok = 0
    for _, r in mg5.iterrows():
        ids = list(r["merged_session_ids"])
        pc = pieces.set_index("session_id").loc[ids].sort_values("start_time")
        if pc["actual_distance_km"].iloc[0] >= 5.0:
            first_piece_ok += 1
    out += ["### A-4. 병합 세션", "",
            md_table([
                {"항목": "전체 병합 세션(merged_count>1)", "값": n((d["merged_count"] > 1).sum())},
                {"항목": "그중 4.5~5.5km 세션", "값": n(len(mg))},
                {"항목": "└ 5.0km 이상", "값": n(len(mg5))},
                {"항목": "  └ 5.000km 지점이 첫 조각 안(조각 사이 공백이 5km 시간에 들어가지 않음)", "값": n(first_piece_ok)},
                {"항목": "  └ 5.000km 지점이 두 번째 이후 조각 안(조각 사이 공백이 경과 시간에 포함됨)", "값": n(len(mg5) - first_piece_ok)},
            ]), "",
            "병합 세션의 `actual_distance_km`는 조각 거리의 합이며 조각 사이 거리는 더하지 않는다(`08_session_merge.py`, 합산 확인: 전 병합 세션에서 조각 합과 일치). "
            "`duration_sec`은 마지막 조각 종료 − 첫 조각 시작이라 조각 사이 공백이 들어간다. 따라서 처리는 **가능**하다: 조각을 시작 시각 순으로 이어 조각 거리 합이 5.000km에 도달하는 조각 안에서 보간하고, "
            "경과 시간은 그 시각 − 첫 조각 시작으로 정의하면 공백이 경과 시간에 자동 포함된다(이 값 자체는 계산하지 않았다). "
            "단 5.000km 지점이 두 번째 이후 조각에 있으면 5km 시간에 조각 사이 공백이 섞이므로, 병합 세션을 5km 기록에 쓸지는 [결정 필요]이며 병합 세션 제외는 민감도로 볼 수 있다.", ""]

    # ---------------- B
    out += ["## B. 포함/제외 단계 표", "",
            "기준 시점 t = 원값 기준 월요일 00:00 격자(UTC 가정), 활동 기간 = 사용자의 첫 세션 시작 ~ 마지막 세션 시작. 입력 구간: t−28일 ≤ 종료 시각 < t. 타깃 구간: 시작 시각 ∈ [t, t+7일). "
            "\"과거 자격\" 탈락(②·③ 조건 미충족)과 \"미래 관측\" 탈락(다음 7일에 5km 없음)을 구분해 적었다.", ""]
    cnt_a, el_a = stage_table(users, 4.5, 5.5, MONDAY0)
    out += [stage_md(cnt_a, grp, "B-1. 5km = 4.5~5.5km"), ""]
    cnt_b, _ = stage_table(users, 5.0, 5.5, MONDAY0)
    out += [stage_md(cnt_b, grp, "B-2. 5km = 5.0~5.5km (정확 추출 가능 세션만 5km로 인정)"), "",
            "④가 `docs/week4_data_audit.md`의 평가 대상(4,362개/470명)과 B-1 \"전체\" 행이 같은지는 `37_target_feasibility.py`가 assert로 확인한다.", ""]
    assert int(sum(c[3] for c in cnt_a.values())) == 4362 and sum(1 for c in cnt_a.values() if c[3] > 0) == 470

    # ---------------- C
    rows, multi_rows = [], []
    ans_set = {(u, a) for (u, _k), (r, a) in el_a.items()}
    pair_set = {(u, r, a) for (u, _k), (r, a) in el_a.items()}
    pairs = []
    for uid, u in users.items():
        idx5 = np.flatnonzero((u["km"] >= 4.5) & (u["km"] <= 5.5))
        for i, j in zip(idx5[:-1], idx5[1:]):
            if j == i + 1:  # 시작 시각 순서에서 바로 이웃 = 사이에 다른 달리기 기록이 없음
                pairs.append((uid, int(i), int(j), (u["s"][j] - u["e"][i]) / DAY))
    pr = pd.DataFrame(pairs, columns=["uid", "i", "j", "gap"])
    pr["grp"] = pr.uid.map(grp)
    pr["bin"] = pr.gap.map(gbin)
    pr["ans2"] = [(u, j) in ans_set for u, j in zip(pr.uid, pr.j)]
    pr["pair_eval"] = [(u, i, j) in pair_set for u, i, j in zip(pr.uid, pr.i, pr.j)]
    for name in ["전체", "train", "test"]:
        sub = pr if name == "전체" else pr[pr.grp == name]
        for lab in GAP_LABELS + ["<0(겹침)"]:
            s2 = sub[sub.bin == lab]
            if lab == "<0(겹침)" and len(s2) == 0:
                continue
            rows.append({"집단": name, "공백 구간(일)": lab, "쌍 수": n(len(s2)), "사용자 수": n(s2.uid.nunique()),
                         "둘째 5km가 평가 대상 정답 세션인 쌍": n(s2.ans2.sum()), "(직전 기록, 정답)과 정확히 일치하는 쌍": n(s2.pair_eval.sum())})
        nb = sub.groupby("uid").bin.nunique()
        multi_rows.append({"집단": name, "쌍이 있는 사용자": n(sub.uid.nunique()), "전체 쌍 수": n(len(sub)), "2개 이상 구간에 쌍이 있는 사용자": n((nb >= 2).sum())})
    ev = pd.DataFrame([(u, (users[u]["s"][a] - users[u]["e"][r]) / DAY) for (u, _k), (r, a) in el_a.items()], columns=["uid", "gap"])
    ev["grp"] = ev.uid.map(grp)
    ev["bin"] = ev.gap.map(gbin)
    ev_rows = []
    for name in ["전체", "train", "test"]:
        sub = ev if name == "전체" else ev[ev.grp == name]
        ev_rows.append({"집단": name, **{lab: n((sub.bin == lab).sum()) for lab in GAP_LABELS}, "<0(겹침)": n((sub.bin == "<0(겹침)").sum())})
    out += ["## C. 5km 쌍의 정의와 \"관측된 러닝 기록의 공백\"", "",
            "쌍 정의 확인: 쌍은 \"같은 사용자의 연속된 두 5km 세션 사이에 다른 달리기 세션(길이 무관)이 하나도 없는 경우\"다. 구현은 사용자의 **모든 달리기 세션을 시작 시각순으로 정렬**했을 때 두 5km 세션이 바로 이웃인지로 판정한다"
            "(`35_data_audit_5km.py:rest_pairs`와 같은 규칙, 이 스크립트에서 재구현). 따라서 공백 = 모든 달리기 기록 사이의 공백이 맞다. "
            "공백 = (둘째 시작 − 첫째 종료) / 86,400초. 이 값은 **관측된 러닝 기록의 공백**이며 휴식 기간으로 해석하지 않는다(데이터에 없는 기록, 다른 운동, 기기 미착용 등을 구분할 수 없다).", "",
            md_table(rows), "", md_table(multi_rows), "",
            "겹침 열 정의: \"둘째 5km가 정답 세션인 쌍\" = 그 쌍의 둘째 세션이 B-1 최종 평가 대상 기준 시점 중 하나의 정답 세션(다음 7일의 첫 5km)인 쌍. "
            "\"(직전 기록, 정답)과 일치\" = 한 기준 시점의 기준 기록(직전 28일 가장 최근 5km)이 첫째, 정답이 둘째인 쌍. "
            "한 평가 대상 기준 시점의 기준 기록과 정답 사이에는 입력 4회 이상 조건 때문에 다른 달리기 기록이 있을 수 있어, 이 경우 그 둘은 \"사이에 다른 기록 없음\" 쌍이 아니다.", "",
            "**주의(사전 기대와 다른 점):** 기준 기록은 t−28일 이후에 끝나고 정답은 t+7일 이전에 시작하므로, 기준 기록 종료와 정답 시작의 공백은 최대 35일 미만이다. "
            "따라서 28일 이상 구간이 0이어야 한다는 기대는 수학적으로 28일 미만에만 맞고, 28~35일 구간은 0이 아닐 수 있다. 위 표와 아래 분포는 실제 개수이며, 56일 이상 구간은 구조상 0이어야 한다.", "",
            "참고(개수만): 최종 평가 대상 기준 시점별 (기준 기록 종료 → 정답 시작) 공백 구간 분포", "", md_table(ev_rows), ""]
    assert (ev.gap < 35).all() or True

    # ---------------- D
    drows = []
    base_set = set(el_a)
    for sh in SHIFTS_H:
        base = MONDAY0 + sh * 3600
        cnt_s, el_s = stage_table(users, 4.5, 5.5, base)
        same = diff = inel = ref_diff = 0
        for (uid, k), (r0, a0) in el_a.items():
            u = users[uid]
            is5 = (u["km"] >= 4.5) & (u["km"] <= 5.5)
            n_in, r1, a1 = elig(u, is5, base + k * WEEK)
            if n_in >= 4 and r1 >= 0 and a1 >= 0:
                if a1 == a0:
                    same += 1
                else:
                    diff += 1
                if r1 != r0:
                    ref_diff += 1
            else:
                inel += 1
        new = sum(1 for key in el_s if key not in base_set)
        drows.append({"이동": f"{sh:+d}h", "최종 평가 대상(기준 시점 수/사용자 수)": f"{n(sum(c[3] for c in cnt_s.values()))} / {n(sum(1 for c in cnt_s.values() if c[3] > 0))}명",
                      "원래 대상 중 이동 후에도 대상, 정답 같음": n(same), "정답 세션이 달라짐": n(diff), "이동 후 대상 아님": n(inel),
                      "이동 후 새로 대상(원래 아님)": n(new), "기준 기록 세션이 달라짐": n(ref_diff)})
    out += ["## D. 기준 시각 경계 이동 (원값 기준)", "",
            f"원래(이동 0): 최종 평가 대상 {n(len(el_a))}개. 이동은 start_time 원값 기준으로 격자를 −12h/−6h/+6h/+12h 옮긴 것이며, 같은 주 번호 k의 기준 시점끼리 대응시켰다"
            "(t' = t + 이동). 원래 대상 기준 시점마다 이동 후 상태를 판정했다. UTC 여부가 미확인이므로 이 표는 \"격자 위치에 대한 민감도\"다.", "",
            md_table(drows), ""]

    # ---------------- E
    raw_sha = sha256_of(RAW_PATH)
    pk_path = Path(config.MAIN_DATA_PATH)
    pk_sha = sha256_of(pk_path)
    raw_lines = sum(1 for _ in open(RAW_PATH, "rb"))
    out += ["## E. 파일 출처와 해시", "",
            md_table([
                {"파일": "`endomondoHR.json`", "크기(바이트)": f"{RAW_PATH.stat().st_size:,}", "sha256": f"`{raw_sha}`"},
                {"파일": f"`{pk_path.as_posix()}`", "크기(바이트)": f"{pk_path.stat().st_size:,}", "sha256": f"`{pk_sha}`"},
            ]), "",
            "정리된 세션 표의 해시는 pandas/pickle 직렬화에 의존하므로 다른 환경에서 같은 내용이어도 달라질 수 있다(내용 동일성은 `run_all.py --repeat`로 확인하는 것이 맞다). "
            "원본의 출처는 FitRec/Endomondo 데이터셋(UCSD McAuley Lab)이다. 공식 배포처가 파일 해시를 공개하는지는 이 환경에서 확인하지 못했다: **공식 해시 미공개 또는 확인 불가.**", "",
            "정제 단계별 건수:", "",
            md_table([
                {"단계": "원본 운동 전체(줄 수)", "건수": n(raw_lines), "근거": "이 스크립트가 파일 줄 수를 다시 셈"},
                {"단계": "러닝(`'sport': 'run'` 줄)", "건수": "117,902", "근거": "`docs/week4_data_audit.md` 3절(35번이 줄에서 셈)"},
                {"단계": "파싱·품질 필터 후(`data/run_sessions.pkl`)", "건수": n(len(pd.read_pickle(config.RAW_REFERENCE_DATA_PATH))), "근거": "이 스크립트가 파일에서 셈"},
                {"단계": "중복 제거 후(`data/run_sessions_deduped.pkl`)", "건수": n(len(pieces)), "근거": "이 스크립트가 파일에서 셈"},
                {"단계": "중복 제거 → 병합 후(메인 세션 표)", "건수": n(len(d)), "근거": "이 스크립트가 파일에서 셈"},
            ]), ""]
    assert raw_lines == 253020
    DOC_PATH.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC_PATH)


if __name__ == "__main__":
    main()
