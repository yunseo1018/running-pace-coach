# -*- coding: utf-8 -*-
"""
5km 과제 데이터셋 구축과 개수 재계산 (`docs/predeclaration_5km_final.md` 확정 정의, 커밋 3fad402 이후).

- 5km 후보: merged_count == 1 이고 actual_distance_km ∈ [5.0, 5.5]. T5 = 누적 haversine(01번과 같은 규칙)이 5.000km에 처음 도달하는 시각을
  인접 두 점 선형 보간으로 구한 (그 시각 − 세션 첫 timestamp) 초. 극단 = T5 > 5,000초.
- 기준 시점 t: 사용자별 첫 세션 시작 ~ 마지막 세션 시작 사이의 모든 월요일 00:00(start_time 원값 기준, 기준 미확인).
- 단계: ① 월요일 → ② 직전 28일(종료 ∈ [t-28일, t)) 달리기 4회 이상 → ③ + 극단 아닌 5km 후보(기준 기록) → ④ + 다음 7일 5km 후보(정답 세션).

**test 사용자(및 분할 밖 사용자)는 T5 값과 정답 y를 계산·저장·출력하지 않는다.** 이들의 개수에 필요한 것은 "극단 여부"(T5 > 5,000초) 불리언뿐이라
그 불리언만 메모리에서 만들고 값은 즉시 버린다(해석 지점으로 문서에 기록). 데이터셋(`data/5km_dataset_train.pkl`)은 train 사용자만 담으며 git에 올리지 않는다.
사용자/세션 ID, 좌표, 행 값은 문서에 싣지 않는다.

출력: docs/dataset_counts_5km.md, data/5km_dataset_train.pkl. sklearn 의존 없음.
"""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import config

spec = importlib.util.spec_from_file_location("feas37", "37_target_feasibility.py")
feas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feas)

DOC_PATH = Path("docs") / "dataset_counts_5km.md"
DATA_OUT = Path("data") / "5km_dataset_train.pkl"
RAW_PATH = Path("endomondoHR.json")
DAY, WEEK, WIN = feas.DAY, feas.WEEK, feas.WIN
MONDAY0 = feas.MONDAY0
EXTREME_SEC = 5000.0
LO, HI = 5.0, 5.5
GROUPS = ["전체", "train", "test", "분할 밖(적격 아님)"]
md_table, n = feas.md_table, feas.n


def compute_t5(parse, rec, start_time):
    lat, lon, ts = rec["latitude"], rec["longitude"], rec["timestamp"]
    assert ts[0] == start_time
    cum, _ = feas.cum_with_skip(parse, lat, lon, ts)
    j = int(np.searchsorted(cum, 5.0, "left"))
    assert 1 <= j < len(cum), "5.000km에 도달하지 않음(후보 정의와 불일치)"
    frac = (5.0 - cum[j - 1]) / (cum[j] - cum[j - 1])
    return (ts[j - 1] + frac * (ts[j] - ts[j - 1])) - ts[0]


def scan_raw(parse, cand_ids, train_ids_by_sid):
    """후보 세션의 극단 여부(전원)와 T5(train 사용자만)."""
    want = set(cand_ids)
    ext, t5 = {}, {}
    start_of = cand_ids
    with open(RAW_PATH, "rb") as f:
        for line in f:
            m = feas.ID_RE.search(line)
            if not m or int(m.group(1)) not in want:
                continue
            sid = int(m.group(1))
            rec = ast.literal_eval(line.decode("utf-8"))
            if rec.get("sport") != "run" or rec["id"] != sid:
                continue
            v = compute_t5(parse, rec, start_of[sid])
            ext[sid] = bool(v > EXTREME_SEC)
            if train_ids_by_sid[sid]:
                t5[sid] = float(v)
            del v  # test/분할 밖 사용자의 T5 값은 보관하지 않는다
    assert len(ext) == len(want), (len(ext), len(want))
    return ext, t5


def main():
    parse = feas.import_parser()
    d = config.load_main_sessions()
    tr = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    te = {int(x) for x in Path(config.TEST_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    grp = {int(u): ("train" if int(u) in tr else ("test" if int(u) in te else "분할 밖(적격 아님)")) for u in d.user_id.unique()}
    d = d.sort_values(["user_id", "start_time"]).reset_index(drop=True)
    d["cand"] = (d["merged_count"] == 1) & d["actual_distance_km"].between(LO, HI)
    c = d[d.cand]
    cand_start = {int(s): int(t) for s, t in zip(c.session_id, c.start_time)}
    is_train_sid = {int(s): grp[int(u)] == "train" for s, u in zip(c.session_id, c.user_id)}
    ext_by_sid, t5_by_sid = scan_raw(parse, cand_start, is_train_sid)
    assert all((sid in t5_by_sid) == is_train_sid[sid] for sid in cand_start)

    stage = {g: np.zeros(4, int) for g in GROUPS}
    stage_users = {g: [set(), set(), set(), set()] for g in GROUPS}
    extra = {g: dict(ref_skipped=0, ref_skipped_replaced=0, ref_skipped_none=0, ans_ext=0, one_cand=0, one_cand_in3=0) for g in GROUPS}
    el_by_user = {}
    rows = []
    for uid, g in d.groupby("user_id", sort=True):
        uid = int(uid)
        gname = grp[uid]
        s = g["start_time"].to_numpy(np.int64)
        e = s + g["duration_sec"].to_numpy(np.int64)
        km = g["actual_distance_km"].to_numpy()
        hr = g["avg_heart_rate"].to_numpy()
        pace = g["avg_pace_sec_per_km"].to_numpy()
        iscand = g["cand"].to_numpy()
        sids = g["session_id"].to_numpy()
        ext = np.array([ext_by_sid.get(int(x), False) for x in sids])
        t5 = np.array([t5_by_sid.get(int(x), np.nan) for x in sids])
        gender = g["gender"].iloc[0]
        for k in feas.span_ks({"s": s}, MONDAY0):
            t = MONDAY0 + k * WEEK
            inwin = (e >= t - WIN) & (e < t)
            n_in = int(inwin.sum())
            tags = [gname, "전체"]
            for tg in tags:
                stage[tg][0] += 1
                stage_users[tg][0].add(uid)
            if n_in < 4:
                continue
            for tg in tags:
                stage[tg][1] += 1
                stage_users[tg][1].add(uid)
            cw = np.flatnonzero(inwin & iscand)
            valid = cw[~ext[cw]]
            if len(cw) and ext[cw[-1]]:
                for tg in tags:
                    extra[tg]["ref_skipped"] += 1
                    extra[tg]["ref_skipped_replaced" if len(valid) else "ref_skipped_none"] += 1
            if len(valid) == 0:
                continue
            ref = int(valid[-1])
            past = np.flatnonzero(iscand & (e < t) & ~ext)
            for tg in tags:
                stage[tg][2] += 1
                stage_users[tg][2].add(uid)
                if len(past) == 1:
                    extra[tg]["one_cand_in3"] += 1
            fut = np.flatnonzero(iscand & (s >= t) & (s < t + WEEK))
            if len(fut) == 0:
                continue
            ans = int(fut[0])
            for tg in tags:
                stage[tg][3] += 1
                stage_users[tg][3].add(uid)
                extra[tg]["ans_ext"] += int(ext[ans])
                extra[tg]["one_cand"] += int(len(past) == 1)
            el_by_user.setdefault(uid, []).append(t)
            if gname == "train":
                m = float(np.median(t5[past]))
                if len(past) == 1:
                    m = float(t5[ref])
                w = np.flatnonzero(inwin)
                wk = [float(km[w][(e[w] >= t - WIN + q * WEEK) & (e[w] < t - WIN + (q + 1) * WEEK)].sum()) for q in range(4)]
                rows.append(dict(
                    uid=uid, k=k, t=t, n_runs=n_in, total_km=float(km[w].sum()), wk1_km=wk[0], wk2_km=wk[1], wk3_km=wk[2], wk4_km=wk[3],
                    longest_km=float(km[w].max()), days_since_last=float((t - e[w].max()) / DAY), mean_hr=float(hr[w].mean()),
                    t5_ref=float(t5[ref]), t5_ref_minus_m=float(t5[ref] - m), m=m, n_past=int(len(past)),
                    mean_pace=float(pace[w].mean()), gender=gender, y=float(t5[ans] - t5[ref]), ans_extreme=bool(ext[ans])))
    ds = pd.DataFrame(rows)
    assert set(ds.uid.unique()) <= tr
    ds.to_pickle(DATA_OUT)

    out = ["# 5km 데이터셋 구축과 개수 재계산 (확정 정의)", "",
           "생성: `38_build_5km_dataset.py`. 정의는 `docs/predeclaration_5km_final.md`(커밋 3fad402)를 그대로 따랐다. 시각은 start_time 원값(기준 미확인). "
           "**test 사용자의 T5 값과 정답 y는 계산·저장·출력하지 않았다**(극단 여부 불리언만 개수 산출에 사용). 사용자/세션 ID·좌표·행 값은 싣지 않았다.", "",
           "## 1. 단계별 개수 (셀: 기준 시점 수 / 사용자 수)", ""]
    rows_t = []
    for gname in GROUPS:
        a = stage[gname]
        us = [len(x) for x in stage_users[gname]]
        rows_t.append({"집단": gname,
                       "① 월요일": f"{n(a[0])} / {n(us[0])}명", "② 직전28일 4회↑": f"{n(a[1])} / {n(us[1])}명",
                       "③ +유효 5km 기준 기록 (과거 자격)": f"{n(a[2])} / {n(us[2])}명", "④ +다음7일 5km (평가 대상)": f"{n(a[3])} / {n(us[3])}명",
                       "탈락 ①→②": n(a[0] - a[1]), "과거 자격 탈락 ②→③": n(a[1] - a[2]), "미래 관측 탈락 ③→④": n(a[2] - a[3])})
    out += [md_table(rows_t), ""]

    out += ["## 2. 극단 후보와 m 대체", ""]
    ncand = d[d.cand].assign(ext=lambda x: x.session_id.map(ext_by_sid), g=lambda x: x.user_id.map(grp))
    rows_t = []
    for gname in GROUPS:
        sub = ncand if gname == "전체" else ncand[ncand.g == gname]
        ex = extra[gname]
        rows_t.append({"집단": gname, "5km 후보 세션": n(len(sub)), "└ 극단(T5>5,000초)": n(sub.ext.sum()),
                       "② 중 직전 28일 가장 늦은 후보가 극단이라 건너뜀": n(ex["ref_skipped"]),
                       "└ 더 이른 비극단 후보로 대체(③ 통과)": n(ex["ref_skipped_replaced"]), "└ 대체 후보 없음(과거 자격 탈락)": n(ex["ref_skipped_none"]),
                       "④ 중 정답 세션이 극단": n(ex["ans_ext"]),
                       "③ 중 과거 후보 1개뿐이라 m=기준 기록": n(ex["one_cand_in3"]), "④ 중 m=기준 기록": n(ex["one_cand"])})
    out += [md_table(rows_t), "",
            "극단 판정은 5km 후보 세션 단위다. \"건너뜀\" 행은 ②를 통과한 기준 시점 중 직전 28일 창의 가장 늦은 5km 후보가 극단인 경우를 센 것이다.", ""]

    # 평가 대상 2개 이상 사용자, personalized 분할
    out += ["## 3. 평가 대상 ≥2개 사용자와 personalized 분할", ""]
    rows_t = []
    for gname in GROUPS[:3]:
        us = [u for u in el_by_user if gname == "전체" or grp[u] == gname]
        cnts = np.array([len(el_by_user[u]) for u in us]) if us else np.array([0])
        n_tr = n_ev = n_gap = 0
        n_incl = 0
        for u in us:
            ts_ = np.sort(np.array(el_by_user[u]))
            nn = len(ts_)
            if nn < 2:
                continue
            n_incl += 1
            ne = max(1, int(round(0.2 * nn)))
            first_eval = ts_[nn - ne]
            pre = ts_[: nn - ne]
            ok = (pre + WEEK) <= first_eval
            n_tr += int(ok.sum())
            n_gap += int((~ok).sum())
            n_ev += ne
        rows_t.append({"집단": gname, "평가 대상 있는 사용자": n(len(us)), "평가 대상 ≥2개 사용자": n((cnts >= 2).sum()),
                       "n=1 사용자(personalized 제외)": n((cnts == 1).sum()), "personalized 포함 사용자": n(n_incl),
                       "학습 기준 시점": n(n_tr), "평가 기준 시점": n(n_ev), "학습·평가 모두에서 제외된 기준 시점(t+7일 > 첫 평가 시점)": n(n_gap)})
    out += [md_table(rows_t), "",
            "personalized: 사용자별 평가 대상(④) 기준 시점을 시간순 정렬해 뒤쪽 max(1, round(0.2n))개를 평가(n ≥ 2), 나머지 중 t+7일 ≤ 첫 평가 기준 시점인 것만 학습. 이 단계에서는 개수만 센다. 기준 시점이 7일 간격 격자라 첫 평가 시점 이전의 모든 기준 시점은 t+7일 ≤ 첫 평가 시점을 만족하므로 마지막 열은 구조상 0이다(규칙이 실제로 걸러내는 경우 없음).", ""]

    # 보조 코호트
    out += ["## 4. 보조 코호트: 관측된 러닝 기록의 공백 (시작 시각순으로 이웃한 두 달리기 세션이 모두 5km 후보인 쌍)", ""]
    pairs = []
    for uid, g in d.groupby("user_id", sort=True):
        s = g["start_time"].to_numpy(np.int64)
        e = s + g["duration_sec"].to_numpy(np.int64)
        cd = g["cand"].to_numpy()
        for i in range(len(g) - 1):
            if cd[i] and cd[i + 1]:
                pairs.append((int(uid), (s[i + 1] - e[i]) / DAY))
    pr = pd.DataFrame(pairs, columns=["uid", "gap"])
    pr["g"] = pr.uid.map(grp)
    pr["bin"] = pr.gap.map(feas.gbin)
    rows_t = []
    for gname in GROUPS[:3]:
        sub = pr if gname == "전체" else pr[pr.g == gname]
        for lab in feas.GAP_LABELS + ["<0(겹침)"]:
            s2 = sub[sub.bin == lab]
            if lab == "<0(겹침)" and len(s2) == 0:
                continue
            rows_t.append({"집단": gname, "공백 구간(일)": lab, "쌍 수": n(len(s2)), "사용자 수": n(s2.uid.nunique())})
    out += [md_table(rows_t), "",
            "공백 = 둘째 시작 − 첫째 종료(일). 주 분석과 별개 코호트이며 쌍의 기록 차이는 이 문서에서 계산하지 않았다.", ""]

    out += ["## 5. 구현 중 해석한 지점 (정의가 문자 그대로 정하지 않은 부분)", "",
            "1. **test 개수 산출과 T5**: ③·④ 개수와 \"건너뜀\" 개수에는 극단 여부(T5 > 5,000초)가 필요하다. test·분할 밖 사용자는 T5를 일시적으로 계산해 극단 불리언만 남기고 값은 버렸다(저장·출력 없음, y는 계산하지 않음).",
            "2. **평균 심박**: 직전 28일 창 안 세션의 `avg_heart_rate` 단순 평균(시간 가중 아님). 평균 페이스(추가 비교용)도 같은 방식.",
            "3. **주별 거리 4개**: 창 안 세션의 종료 시각 기준으로 t−28~−21, −21~−14, −14~−7, −7~0일 구간별 거리 합(오래된 구간부터).",
            "4. **마지막 달리기 종료 후 일수**: 창 안 세션 중 가장 늦은 종료 시각에서 t까지.",
            "5. **달리기 횟수**: 병합 세션은 세션 표 단위로 1회로 센다(정의 4: 병합 세션은 횟수·입력값 계산에 포함).",
            "6. **m의 \"과거 후보\"**: 종료 시각 < t인 비병합·비극단 5km 후보 전체(기준 기록 포함). 1개뿐이면 m = T5(기준 기록).",
            "7. **월요일 범위**: 첫 세션 시작 ≤ t ≤ 마지막 세션 시작(양 끝 포함), 사용자별 달력은 원값 기준 월요일 00:00 격자.",
            "8. **성별**: male/female/unknown 범주 그대로 저장(추가 비교에서 쓸 때 범주 부호화 방법은 그때 정의).",
            "9. **\"(그 시각 − 세션 첫 timestamp)\"**: 세션 첫 timestamp = 원본 `timestamp[0]`이며 세션 표의 start_time과 같음을 assert로 확인."]
    DOC_PATH.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC_PATH, DATA_OUT, "rows", len(ds))


if __name__ == "__main__":
    main()
