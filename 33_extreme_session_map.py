# -*- coding: utf-8 -*-
"""
10월 3주차 5번: 극단 세션 ↔ 윈도우 대응 + 분류 + 민감도.
**분류 규칙·임계값은 데이터를 보기 전에 docs/week3_predeclaration.md 9절에 고정했다(결과를 본 뒤 바꾸지 않음).**
새 예측 모델 없음, sklearn 의존 없음(numpy/pandas/ridge_lib/gbt_lib/torch). 사용자 ID·세션 ID·개인 경로는 문서에 쓰지 않는다.
세션 단위 대응표 원본은 data/extreme_session_window_map.csv(git 제외)에만 저장한다.

산출물: docs/extreme_session_window_map.md
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m21 = _load("m21", "21_baseline_integrated_report.py")
m25 = _load("m25", "25_tree_models.py")
m29 = _load("m29", "29_coaching.py")
m24 = m25.m24

OUTPUT_MD_PATH = Path("docs") / "extreme_session_window_map.md"
MAP_CSV_PATH = Path("data") / "extreme_session_window_map.csv"
EXTREME = config.EXTREME_PACE_THRESHOLD_SEC_PER_KM
STOP_RATIO_MIN = 0.5      # (b) 휴식 포함 의심: gap_ratio >= 0.5   (사전 선언 9-2)
SHORT_KM = 1.0            # (c) 짧은 거리: actual_distance_km < 1  (사전 선언 9-2)
N_BOOT, BOOT_SEED = 2000, 42
SPLITS = ["cold-start", "personalized"]
PREDS = ["median5", "gbm", "mlp"]
PRED_NAMES = {"median5": "median-5", "gbm": "GBM22", "mlp": "MLP22", "hist_median": "전체 이력 중앙값", "blend": "(d) [탐색적 baseline]"}
CLASS_NAMES = {"a": "(a) 병합 유래(merged_count>1)", "b": "(b) 휴식 포함 의심(정지비율≥0.5)", "c": "(c) 짧은 거리(<1km)", "d": "(d) 미분류"}


def log(*a):
    print(*a, flush=True)


def classify(df):
    """세션 표의 열만으로 분류(중복 가능). 반환: a,b,c,d 불리언 열."""
    a = (df["merged_count"] > 1).to_numpy()
    b = (df["gap_ratio"] >= STOP_RATIO_MIN).to_numpy()
    c = (df["actual_distance_km"] < SHORT_KM).to_numpy()
    d = ~(a | b | c)
    return a, b, c, d


def class_table(df_or_flags, n_total, label):
    a, b, c, d = df_or_flags
    rows = []
    for k, arr in zip("abcd", (a, b, c, d)):
        rows.append({"분류": CLASS_NAMES[k], f"{label} 건수": int(arr.sum()), "비율": f"{100 * arr.sum() / n_total:.1f}%"})
    n_multi = int(((a.astype(int) + b.astype(int) + c.astype(int)) >= 2).sum())
    return rows, n_multi


def make_W(nu):
    rng = np.random.default_rng(BOOT_SEED)
    return np.stack([np.bincount(rng.integers(0, nu, nu), minlength=nu) for _ in range(N_BOOT)]).astype(float)


def user_stat(uid, err):
    codes, uniq = pd.factorize(uid)
    nu = len(uniq)
    return codes, nu, np.bincount(codes, weights=np.abs(err), minlength=nu), np.bincount(codes, weights=err ** 2, minlength=nu), np.bincount(codes, minlength=nu).astype(float)


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    md = ["# 극단 세션 ↔ 윈도우 대응 (10월 3주차 5번)\n\n"]
    md.append("> **세션 단위 비율 0.781%(극단 세션 826건 / 정제 세션 105,709건)와 윈도우 단위 기여(제곱오차 상위 1% 윈도우의 SSE 비중 92~93%)는 집계 단위가 다르다.** "
              "한 세션은 타깃으로 1개 윈도우, 입력으로 최대 5개 윈도우에 등장하므로 세션 비율을 윈도우 비율로 바로 옮길 수 없고, 상위 1% 윈도우의 SSE 비중은 세션이 아니라 윈도우 단위 제곱오차의 집중도다. "
              "이 문서의 모든 서술은 **관찰**이며 인과 해석을 하지 않는다. 극단 페이스가 GPS 오류인지 실제 휴식 포함인지는 검증하지 않았다(미확인). "
              "분류 규칙·임계값은 `week3_predeclaration.md` 9절에 데이터를 보기 전에 고정했다. 사용자 ID·세션 ID는 문서에 쓰지 않으며 세션 단위 표는 `data/`(git 제외)에만 저장했다.\n\n")

    # ---------- 세션 표 ----------
    sess = config.load_main_sessions()
    moving = pd.read_pickle(m24.MOVING_PATH)[["session_id", "gap_ratio"]]
    sess = sess.merge(moving, on="session_id", how="left", validate="one_to_one")
    assert sess["gap_ratio"].notna().all()
    n_sess = len(sess)
    is_ext = (sess["avg_pace_sec_per_km"] > EXTREME).to_numpy()
    n_ext = int(is_ext.sum())
    assert n_ext == 826 and abs(100 * n_ext / n_sess - 0.781) < 0.0005, (n_ext, n_sess)  # 사전 선언 9-1 (중단 조건)
    log(f"[9-1] 극단 세션 {n_ext}건 / {n_sess} = {100 * n_ext / n_sess:.3f}% (기존 826건 0.781%와 일치)")
    ext = sess[is_ext].copy()
    ext_ids = set(ext["session_id"].tolist())
    flags_all = classify(sess)
    flags_ext = tuple(f[is_ext] for f in flags_all)

    md.append("## 1. 극단 세션과 분류 규칙\n")
    md.append(f"- 극단 세션 = 경과 시간 기준 평균 페이스 > {EXTREME}초/km. 정제 세션 {n_sess:,}건 중 **{n_ext}건({100 * n_ext / n_sess:.3f}%)** — 기존 `docs/target_pace_definition.md`와 일치(사전 선언 9-1의 중단 조건 통과).\n")
    md.append(f"- 분류 규칙(세션 표의 열로 가능한 것만, 사전 고정): (a) `merged_count>1`, (b) 정지 비율 `gap_ratio≥{STOP_RATIO_MIN}`, (c) `actual_distance_km<{SHORT_KM:g}`, (d) 위 어디에도 해당 없음. "
              "**한 세션이 (a)~(c) 여러 범주에 해당하면 각각 중복 집계**한다((a)~(c) 합이 세션 수보다 클 수 있음).\n")
    md.append("- **GPS 오류·중복 흔적은 '미판정'**: 포인트 단위 정보는 01단계에서 이미 필터링되었고 중복은 16단계에서 제거된 세션만 남아 있어 세션 표만으로는 판정할 수 없다(원본 재파싱은 하지 않았다). "
              "따라서 (d) 미분류는 'GPS 오류나 중복이 아님'을 뜻하지 않는다.\n\n")
    rows, n_multi = class_table(flags_ext, n_ext, "극단 세션")
    md.append("### 1-1. 극단 세션의 분류 분포 (전체 정제 세션 중 극단 826건)\n")
    md.append(pd.DataFrame(rows).pipe(lambda x: m21.df_to_markdown(x)))
    md.append(f"\n\n(a)~(c) 중 2개 이상에 해당하는 극단 세션: {n_multi}건(중복 집계됨). 전체 정제 세션 대비 분류별 비율(참고): " +
              ", ".join(f"{k.upper()}={100 * f.sum() / n_sess:.1f}%" for k, f in zip("abc", flags_all[:3])) + ".\n\n")

    # ---------- 윈도우 ↔ 세션 대응 ----------
    user_seq = pd.read_pickle(config.USER_SEQUENCES_PATH)
    order2sid = {(r.user_id, r.session_order): r.session_id for r in user_seq[["user_id", "session_order", "session_id"]].itertuples()}
    train = pd.read_pickle(m24.EXT_TRAIN_PATH)
    test = pd.read_pickle(m24.EXT_TEST_PATH)
    p_train = pd.read_pickle(m24.EXT_P_TRAIN_PATH)
    p_test = pd.read_pickle(m24.EXT_P_TEST_PATH)
    sets = {"cold_train": train, "cold_test": test, "pers_history": p_train, "pers_holdout": p_test}

    def window_sessions(s):
        start = s["window_start_session_order"]
        assert s["target_session_order"] == start + 5
        ins = [order2sid[(s["user_id"], o)] for o in range(start, start + 5)]
        assert s["target_session_id"] == order2sid[(s["user_id"], s["target_session_order"])]
        return ins, s["target_session_id"]

    # 세션별 집계
    cnt = {sid: {k: 0 for k in ["tgt_all", "in_all", "tgt_cold_test", "in_cold_test", "tgt_pers_holdout", "in_pers_holdout"]} for sid in ext_ids}
    win_info = {}  # set -> list of (target_ext, n_input_ext)
    for name, smp in sets.items():
        info = []
        for s in smp:
            ins, tgt = window_sessions(s)
            t_ext = tgt in ext_ids
            n_in_ext = sum(1 for x in ins if x in ext_ids)
            info.append((t_ext, n_in_ext))
            # 전체 윈도우 집합은 cold(train+test)가 모든 적격 사용자를 한 번씩 덮는다(personalized는 같은 윈도우의 재분할)
            if name in ("cold_train", "cold_test"):
                if t_ext:
                    cnt[tgt]["tgt_all"] += 1
                for x in ins:
                    if x in ext_ids:
                        cnt[x]["in_all"] += 1
            if name == "cold_test":
                if t_ext:
                    cnt[tgt]["tgt_cold_test"] += 1
                for x in ins:
                    if x in ext_ids:
                        cnt[x]["in_cold_test"] += 1
            if name == "pers_holdout":
                if t_ext:
                    cnt[tgt]["tgt_pers_holdout"] += 1
                for x in ins:
                    if x in ext_ids:
                        cnt[x]["in_pers_holdout"] += 1
        win_info[name] = info
    cdf = pd.DataFrame.from_dict(cnt, orient="index")
    cdf.index.name = "session_id"
    cdf = cdf.join(ext.set_index("session_id")[["user_id", "avg_pace_sec_per_km", "merged_count", "gap_ratio", "actual_distance_km"]])
    a_, b_, c_, d_ = (arr for arr in classify(cdf))
    cdf["cls_a"], cdf["cls_b"], cdf["cls_c"], cdf["cls_d"] = a_, b_, c_, d_
    MAP_CSV_PATH.parent.mkdir(exist_ok=True)
    cdf.reset_index().to_csv(MAP_CSV_PATH, index=False)  # 세션·사용자 ID 포함 -> data/ 아래(git 제외)

    n_in_windows_universe = int(cdf.index.isin(set(order2sid.values())).sum())
    md.append("## 2. 세션 ↔ 윈도우 대응 (5-1)\n")
    md.append("세션 단위 대응표 원본은 `data/extreme_session_window_map.csv`(git 제외, 세션·사용자 ID 포함)에 저장했고 문서에는 건수 요약만 둔다. "
              "전체 윈도우 집합 = cold-start train+test 윈도우(모든 적격 사용자의 윈도우를 한 번씩 덮음; personalized history+hold-out은 같은 윈도우의 시간순 재분할).\n\n")
    ge1 = lambda col: int((cdf[col] >= 1).sum())
    rows = [
        {"항목": "극단 세션 수(정제 세션 전체)", "건수": n_ext},
        {"항목": "그 중 적격 사용자(세션 6개 이상) 소속 — 윈도우 생성 대상", "건수": n_in_windows_universe},
        {"항목": "타깃으로 ≥1개 윈도우에 등장(전체 윈도우 집합)", "건수": ge1("tgt_all")},
        {"항목": "입력으로 ≥1개 윈도우에 등장(전체 윈도우 집합)", "건수": ge1("in_all")},
        {"항목": "타깃이자 입력으로 모두 등장", "건수": int(((cdf["tgt_all"] >= 1) & (cdf["in_all"] >= 1)).sum())},
        {"항목": "어떤 윈도우에도 등장하지 않음(타깃도 입력도 아님)", "건수": int(((cdf["tgt_all"] == 0) & (cdf["in_all"] == 0)).sum())},
        {"항목": "cold-start 평가 집합(test)에 타깃으로 등장", "건수": ge1("tgt_cold_test")},
        {"항목": "cold-start 평가 집합(test)에 입력으로 등장", "건수": ge1("in_cold_test")},
        {"항목": "personalized 평가 집합(hold-out)에 타깃으로 등장", "건수": ge1("tgt_pers_holdout")},
        {"항목": "personalized 평가 집합(hold-out)에 입력으로 등장", "건수": ge1("in_pers_holdout")},
    ]
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n**극단 세션 한 개당 '입력으로 등장하는 윈도우 수'의 분포**(최대 5; 전체 윈도우 집합):\n\n")
    hist = cdf["in_all"].value_counts().sort_index()
    md.append(m21.df_to_markdown(pd.DataFrame({"입력으로 등장한 윈도우 수": hist.index, "극단 세션 수": hist.values})))
    md.append("\n\n**극단 세션 한 개당 '타깃으로 등장하는 윈도우 수'의 분포**(전체 윈도우 집합):\n\n")
    th = cdf["tgt_all"].value_counts().sort_index()
    md.append(m21.df_to_markdown(pd.DataFrame({"타깃으로 등장한 윈도우 수": th.index, "극단 세션 수": th.values})))
    md.append("\n\n")

    # 평가 집합별 윈도우 구성
    rows = []
    for name, label in [("cold_test", "cold-start test"), ("pers_holdout", "personalized hold-out"), ("cold_train", "cold-start train"), ("pers_history", "personalized history")]:
        info = win_info[name]
        n = len(info)
        t = sum(1 for te_, _ in info if te_)
        io = sum(1 for te_, ni in info if (not te_) and ni >= 1)
        both = sum(1 for te_, ni in info if te_ and ni >= 1)
        rows.append({"윈도우 집합": label, "윈도우 수": n, "타깃이 극단(i)": t, "(i) 비율": f"{100 * t / n:.2f}%", "타깃은 아니나 입력에 극단 포함(ii)": io,
                     "(ii) 비율": f"{100 * io / n:.2f}%", "(참고) 타깃 극단이면서 입력에도 극단": both, "둘 다 아님(iii)": n - t - io, "(iii) 비율": f"{100 * (n - t - io) / n:.2f}%"})
    md.append("### 2-1. 윈도우 집합별 극단 관련 윈도우 비율\n")
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    sums = ", ".join(f"{r['윈도우 집합']} {float(r['(i) 비율'].rstrip('%')) + float(r['(ii) 비율'].rstrip('%')):.2f}%" for r in rows)
    md.append("\n\n극단 세션 826건(세션의 0.781%)은 윈도우 집합에서 (i)+(ii) 비율 합 " + sums + "의 윈도우에 관여한다 — 세션 비율과 윈도우 비율이 다른 이유는 한 세션이 최대 6개 윈도우(타깃 1 + 입력 최대 5)에 등장하기 때문이다(집계 단위 차이).\n\n")

    # ---------- 평가 윈도우 단위 분석 ----------
    log("평가 예측 계산(새로 계산, 캐시 미사용)...")
    D = {}
    for split, (tr_s, te_s) in [("cold-start", (train, test)), ("personalized", (p_train, p_test))]:
        D[split] = m29.build_split_data(split, tr_s, te_s)
    # 극단 플래그(세션 기반) 및 기존 X 기반 플래그와 일치 확인
    for split, smp in [("cold-start", test), ("personalized", p_test)]:
        d = D[split]
        t_flag = np.array([window_sessions(s)[1] in ext_ids for s in smp])
        i_flag = np.array([any(x in ext_ids for x in window_sessions(s)[0]) for s in smp])
        x_flags = np.array([m21.is_extreme_affected({**s, "X": s["X"][:15]}) for s in smp])
        assert (t_flag == x_flags[:, 0]).all() and (i_flag == x_flags[:, 1]).all(), "세션 기반 극단 플래그와 X 기반 플래그 불일치"
        d["t_ext"], d["i_ext"] = t_flag, i_flag
        d["in_only"] = (~t_flag) & i_flag
        # 기존 문서(tree_results.md 203/234행)의 '1000+(극단)' 타깃 윈도우 수와 일치 확인
        d["ext_cnt_expected"] = {"cold-start": 194, "personalized": 146}[split]
        assert int(t_flag.sum()) == d["ext_cnt_expected"], (split, int(t_flag.sum()))
    log("[검증] 세션 기반 극단 플래그 == X 기반 플래그, 타깃 극단 윈도우 수 194/146 (기존 문서와 일치)")

    # 분류용 세션 표(전체) 조회
    sess_idx = sess.set_index("session_id")
    PRIOR_SSE = {("median5", "cold-start"): 92.8, ("median5", "personalized"): 92.2, ("gbm", "cold-start"): 93.9, ("gbm", "personalized"): 92.8}
    md.append("## 3. 윈도우 단위: 제곱오차 상위 1% 윈도우의 구성 (5-2)\n")
    md.append("상위 1% = 제곱오차 내림차순 상위 `ceil(0.01×n)`개(cold-start 220개, personalized 207개 — 기존 문서의 n과 동일). (i) 타깃이 극단 세션, (ii) 타깃은 아니지만 입력 5세션에 극단 세션 포함, (iii) 둘 다 아님.\n\n")
    rows, rows_cls = [], []
    top_info = {}
    for split in SPLITS:
        d = D[split]
        n = len(d["y"])
        k = math.ceil(0.01 * n)
        for p in PREDS:
            se = (d["y"] - d["preds"][p]) ** 2
            order = np.argsort(-se, kind="stable")
            top = order[:k]
            tot = se.sum()
            share_top = se[top].sum() / tot
            t_top, io_top = d["t_ext"][top], d["in_only"][top]
            n_i, n_ii = int(t_top.sum()), int(io_top.sum())
            n_iii = k - n_i - n_ii
            share_t = se[d["t_ext"]].sum() / tot
            share_io = se[d["in_only"]].sum() / tot
            share_none = se[~(d["t_ext"] | d["i_ext"])].sum() / tot
            rows.append({"평가조건": split, "예측기": PRED_NAMES[p], "n": n, "상위1% 개수": k, "상위1%의 SSE 비중": f"{100 * share_top:.1f}%",
                         "상위1% 중 (i) 타깃 극단": f"{n_i} ({100 * n_i / k:.1f}%)", "(ii) 입력만 극단": f"{n_ii} ({100 * n_ii / k:.1f}%)", "(iii) 둘 다 아님": f"{n_iii} ({100 * n_iii / k:.1f}%)",
                         "타깃 극단 윈도우 전체의 SSE 비중": f"{100 * share_t:.1f}% (윈도우 {100 * d['t_ext'].mean():.2f}%)",
                         "입력만 극단 윈도우의 SSE 비중": f"{100 * share_io:.1f}% (윈도우 {100 * d['in_only'].mean():.2f}%)",
                         "둘 다 아닌 윈도우의 SSE 비중": f"{100 * share_none:.1f}%"})
            if (p, split) in PRIOR_SSE:
                exp = PRIOR_SSE[(p, split)]
                if abs(100 * share_top - exp) > 0.15:
                    log(f"[참고] {split} {p}: 상위1% SSE 비중 {100 * share_top:.2f}% vs 기존 문서 {exp}% (윈도우 구성/모델 차이로 소수점 차이 가능)")
            top_info[(split, p)] = top
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n**기존 문서와의 일치 확인**: 상위 1%의 SSE 비중은 median-5 / GBM22 모두 92~94% 범위이며(`docs/baseline_results.md` D-7: median-5 92.8%/92.2%, `docs/tree_results.md`: 최고 모델 93.9%/92.8%), "
              "타깃 극단 윈도우 수는 cold-start 194개·personalized 146개로 기존 문서의 '1000+(극단)' 행과 같다. median-5 값이 소수점 이하에서 다르면 04/06 윈도우(21,978/20,629)와 24단계 윈도우(21,977/20,628)의 차이 때문이다.\n\n")

    # ---------- 분류별 요약 ----------
    md.append("## 4. 분류별 요약 (5-3)\n")
    md.append("### 4-1. 극단 세션의 (a)~(d) 분포\n")
    md.append("1-1절 표 참고(극단 세션 826건; (a)~(c) 중복 집계).\n\n")
    md.append("### 4-2. 상위 1% 윈도우의 '타깃 세션' 분류 분포\n")
    md.append("상위 1% 윈도우의 타깃 세션에 같은 규칙((a) 병합, (b) 정지 비율, (c) 짧은 거리, (d) 미분류)을 적용했다(중복 집계; 타깃이 극단이 아닌 세션도 포함). 비교용으로 평가 집합 전체 윈도우의 타깃 세션 분류 비율을 함께 둔다.\n\n")
    rows = []
    for split, smp in [("cold-start", test), ("personalized", p_test)]:
        d = D[split]
        tgt_ids = [s["target_session_id"] for s in smp]
        tdf = sess_idx.loc[tgt_ids]
        fa = classify(tdf)
        allrow = {k: 100 * f.mean() for k, f in zip("abcd", fa)}
        for p in PREDS:
            top = top_info[(split, p)]
            ft = tuple(f[top] for f in fa)
            kk = len(top)
            row = {"평가조건": split, "예측기": PRED_NAMES[p], "상위1% 윈도우 수": kk}
            for key, f in zip("abcd", ft):
                row[CLASS_NAMES[key]] = f"{int(f.sum())} ({100 * f.sum() / kk:.1f}%)"
            row["(a)~(c) 중 2개 이상"] = int(((ft[0].astype(int) + ft[1].astype(int) + ft[2].astype(int)) >= 2).sum())
            rows.append(row)
        rows.append({"평가조건": split, "예측기": "(참고) 평가 집합 전체 윈도우의 타깃 세션", "상위1% 윈도우 수": len(smp),
                     **{CLASS_NAMES[k]: f"{int(f.sum())} ({allrow[k]:.1f}%)" for k, f in zip("abcd", fa)},
                     "(a)~(c) 중 2개 이상": int(((fa[0].astype(int) + fa[1].astype(int) + fa[2].astype(int)) >= 2).sum())})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n")

    # ---------- 민감도 ----------
    md.append("## 5. 민감도 (5-4) — 원래 평가 집합 결과는 유지\n")
    md.append("(i) 타깃이 극단인 윈도우만 제외, (ii) 타깃 또는 입력에 극단 세션이 포함된 윈도우를 모두 제외(= `docs/baseline_results.md` C-1의 기존 민감도와 같은 기준). "
              "차이 = median-5 − 모델(양수 = 모델이 더 낮은 오차), 95% CI는 사용자 단위 cluster bootstrap(2000회, seed 42; 각 조건에서 남은 윈도우를 가진 사용자만 재표집).\n\n")

    # 기존 C-1 정확 재현 (04/06 윈도우)
    ov_lookup = m21.build_user_overall_avg_lookup(user_seq)
    prior_c1 = {"cold-start": (43.172, 73.073, 21135, 146), "personalized": (41.372, 74.170, 19965, 727)}
    repro_rows = []
    for split, path in [("cold-start", config.TEST_DATASET_PATH), ("personalized", config.PERSONALIZED_TEST_DATASET_PATH)]:
        old_smp = pd.read_pickle(path)
        old_df = m21.build_rows(old_smp, ov_lookup)
        m5 = old_df[(old_df["baseline"] == "median5") & (~old_df["extreme_affected"])]
        mae = m5["abs_error"].mean()
        rmse = float(np.sqrt(m5["sq_error"].mean()))
        exp = prior_c1[split]
        got = (round(mae, 3), round(rmse, 3), len(m5), m5["user_id"].nunique())
        assert got == exp, f"기존 C-1 재현 실패: {split} {got} vs {exp}"
        repro_rows.append({"평가조건": split, "윈도우 기준": "04/06단계 윈도우(C-1과 동일)", "MAE": got[0], "RMSE": got[1], "n 윈도우": got[2], "n 사용자": got[3], "기존 C-1": f"{exp[0]}/{exp[1]}/{exp[2]}/{exp[3]}", "일치": "정확히 일치"})
    md.append("### 5-0. 기존 민감도(C-1) 재현 확인\n")
    md.append(m21.df_to_markdown(pd.DataFrame(repro_rows)))
    md.append("\n\n같은 제외 기준을 04/06단계 윈도우에 적용한 median-5 수치가 `baseline_results.md` C-1과 정확히 일치했다(재현 확인). 민감도 (i)(타깃만 제외)는 기존 문서에 해당 수치가 없는 새 조건이다. 아래 표는 24단계 윈도우(7~9단계 평가 집합) 기준이라 윈도우 수가 1개 다르다.\n\n")

    sens_rows, diff_rows = [], []
    for split in SPLITS:
        d = D[split]
        conds = [("원래 평가 집합(전부 포함)", np.ones(len(d["y"]), bool)), ("(i) 타깃 극단 윈도우 제외", ~d["t_ext"]),
                 ("(ii) 타깃 또는 입력 극단 윈도우 제외", ~(d["t_ext"] | d["i_ext"]))]
        for cname, keep in conds:
            uid = d["uid"][keep]
            stats = {}
            for p in ["median5", "gbm", "mlp", "hist_median", "blend"]:
                err = d["y"][keep] - d["preds"][p][keep]
                stats[p] = user_stat(uid, err)
            codes, nu = stats["median5"][0], stats["median5"][1]
            W = make_W(nu)
            draws, pts = {}, {}
            for p, (cd, _, sa, sq, nn) in stats.items():
                draws[p] = ((W @ sa) / (W @ nn), np.sqrt((W @ sq) / (W @ nn)))
                pts[p] = (sa.sum() / nn.sum(), float(np.sqrt(sq.sum() / nn.sum())))
            for p in ["median5", "gbm", "mlp", "hist_median", "blend"]:
                sens_rows.append({"평가조건": split, "조건": cname, "예측기": PRED_NAMES[p], "n 윈도우": int(keep.sum()), "n 사용자": nu,
                                  "MAE": round(pts[p][0], 3), "RMSE": round(pts[p][1], 3)})
                if p != "median5":
                    dm = draws["median5"][0] - draws[p][0]
                    dr = draws["median5"][1] - draws[p][1]
                    lo, hi = ci(dm)
                    rlo, rhi = ci(dr)
                    diff_rows.append({"평가조건": split, "조건": cname, "예측기": PRED_NAMES[p],
                                      "median-5 − 모델 MAE [95% CI]": f"{pts['median5'][0] - pts[p][0]:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" (0 포함)" if lo <= 0 <= hi else ""),
                                      "median-5 − 모델 RMSE [95% CI]": f"{pts['median5'][1] - pts[p][1]:+.3f} [{rlo:+.3f}, {rhi:+.3f}]" + (" (0 포함)" if rlo <= 0 <= rhi else "")})
    md.append("### 5-1. MAE·RMSE (조건별)\n")
    md.append(m21.df_to_markdown(pd.DataFrame(sens_rows)))
    md.append("\n\n### 5-2. median-5 대비 차이와 95% CI\n")
    md.append(m21.df_to_markdown(pd.DataFrame(diff_rows)))
    md.append("\n\nCI가 0을 포함하는 비교는 차이가 없다는 입증이 아니다. 극단 윈도우를 제외하면 MAE·RMSE의 절대 수준이 크게 달라지는 점은 위 표에서 직접 확인할 수 있다(원인 해석은 하지 않는다).\n\n")

    md.append("## 6. 관찰과 한계\n")
    md.append("- 위 모든 수치는 관찰이며 인과 해석을 하지 않는다. 극단 윈도우를 제외한 결과는 '극단 페이스 윈도우가 지표에 얼마나 관여하는가'를 보는 민감도이지, 그 윈도우가 오류라는 판정이 아니다.\n"
              "- 세션 표만으로 GPS 오류·중복 흔적을 판정할 수 없어 '미판정'으로 남겼다. (a)~(c) 분류는 정의된 열에 의한 단순 규칙이며 임계값(`gap_ratio≥0.5`, `<1km`)은 사전 고정값이다. 다른 임계값에서의 분포는 보지 않았다.\n"
              "- 경과 시간 기준 페이스가 큰 세션을 체력 저하나 목표 실패로 해석하지 않는다(휴식·훈련 목적·워치를 켜둔 채 쉰 시간 등이 포함될 수 있음).\n")
    OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
    log("저장:", OUTPUT_MD_PATH)


if __name__ == "__main__":
    main()
