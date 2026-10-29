# -*- coding: utf-8 -*-
"""
사후 분석: 같은 사람 안 관계(A)와 공백 코호트(B). 사양: docs/posthoc_spec_5km.md (커밋 49b5f2b).
**사후 분석(test 정답을 본 뒤의 분석)이며 주 결론(49c02f8)을 바꾸지 않는다.** 상관관계까지만 주장한다. 인과·처방 표현을 쓰지 않는다.

- T5(5km 후보 전체)는 38번과 같은 코드로 다시 계산해 data/t5_main_all.pkl(ID 포함, git 제외)에 캐시한다. 44번이 재사용한다.
- 주 분석 데이터셋을 42번과 같은 함수로 만들고 49c02f8의 평가 행(cold test 352개, personalized 442개)과 일치하는지 assert로 확인한다.
- ID·좌표·행 단위 값은 출력·저장하지 않는다(문서에는 집계만).
출력: docs/within_person_results_5km.md. sklearn 의존 없음(numpy/pandas).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import config


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m42 = _load("42_final_eval_5km.py", "m42")
m38, feas = m42.m38, m42.feas
md_table = m42.md_table
DOC_PATH = Path("docs") / "within_person_results_5km.md"
T5_CACHE = Path("data") / "t5_main_all.pkl"
N_BOOT, SEED, BAND = 2000, 42, 10.0
BINS = [(0, 7), (7, 14), (14, 28), (28, 56), (56, None)]
BIN_LABELS = ["[0,7)", "[7,14)", "[14,28)", "[28,56)", "56일 이상"]


def load_base():
    """세션 표, 분할 집합, 후보 마스크, (극단 여부, T5) 캐시를 반환."""
    d = config.load_main_sessions().sort_values(["user_id", "start_time"]).reset_index(drop=True)
    tr = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    te = {int(x) for x in Path(config.TEST_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    grp = {int(u): ("train" if int(u) in tr else ("test" if int(u) in te else "other")) for u in d.user_id.unique()}
    d["cand"] = (d["merged_count"] == 1) & d["actual_distance_km"].between(m38.LO, m38.HI)
    c = d[d.cand]
    if T5_CACHE.exists():
        cache = pd.read_pickle(T5_CACHE)
    else:
        parse = feas.import_parser()
        cand_start = {int(s): int(t) for s, t in zip(c.session_id, c.start_time)}
        ext, t5 = m38.scan_raw(parse, cand_start, {s: True for s in cand_start})
        cache = {"ext": ext, "t5": t5}
        pd.to_pickle(cache, T5_CACHE)
    assert len(cache["t5"]) == len(c)
    return d, tr, te, grp, cache


def main_dataset(d, grp, cache):
    ds = m42.build_dataset(set(grp), grp, d, cache["ext"], cache["t5"])
    return ds


def verify_against_final(ds, tr, te):
    """49c02f8 평가 행과 일치 확인(uid, t, y)."""
    saved = pd.read_pickle(m42.PRED_OUT)
    cold = ds[ds.uid.isin(te)].reset_index(drop=True)
    _p_tr, p_ev, _u = m42.personalized_split(ds)
    for nm, ev in (("cold", cold), ("pers", p_ev)):
        s = pd.DataFrame({"uid": saved[nm]["uid"], "t": saved[nm]["t"], "y": saved[nm]["y"]}).sort_values(["uid", "t"]).reset_index(drop=True)
        e = ev[["uid", "t", "y"]].sort_values(["uid", "t"]).reset_index(drop=True)
        assert len(s) == len(e) and np.array_equal(s.to_numpy(), e.to_numpy()), nm
    return len(cold), len(p_ev)


def ci(a):
    lo, hi = np.nanpercentile(a, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def within_ols(groups, X, y, idx=None, names=None):
    """사용자별 평균을 뺀 within OLS. 점추정과 사용자 단위 bootstrap(사용자 충분통계 합산)."""
    users, inv = np.unique(groups, return_inverse=True)
    U = len(users)
    cnt = np.bincount(inv, minlength=U).astype(float)
    k = X.shape[1]
    Xd = X - (np.stack([np.bincount(inv, weights=X[:, j], minlength=U) for j in range(k)], 1) / cnt[:, None])[inv]
    yd = y - (np.bincount(inv, weights=y, minlength=U) / cnt)[inv]
    Sxx = np.zeros((U, k, k))
    Sxy = np.zeros((U, k))
    for u in range(U):
        m = inv == u
        Sxx[u] = Xd[m].T @ Xd[m]
        Sxy[u] = Xd[m].T @ yd[m]
    beta = np.linalg.pinv(Sxx.sum(0)) @ Sxy.sum(0)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, U, size=(N_BOOT, U))
    bs = np.einsum("bij,bj->bi", np.linalg.pinv(Sxx[idx].sum(1)), Sxy[idx].sum(1))
    return beta, bs, Xd, U


def main():
    d, tr, te, grp, cache = load_base()
    ds = main_dataset(d, grp, cache)
    n_cold, n_pers = verify_against_final(ds, tr, te)
    assert len(ds) == 2072 and (n_cold, n_pers) == (352, 442), (len(ds), n_cold, n_pers)
    out = ["# 사후 분석: 같은 사람 안 관계와 공백 코호트 (5km 과제)", "",
           "생성: `43_within_person_5km.py`. 사양: `docs/posthoc_spec_5km.md`(커밋 49b5f2b). **사후 분석**(test 정답을 본 뒤의 분석)이며 주 결론(49c02f8)을 바꾸지 않는다. "
           "**상관관계까지만 주장하며 인과·처방 해석이 아니다.** 시각은 start_time 원값(기준 미확인), 러닝 기록의 빈 구간은 \"관측된 러닝 기록의 공백\"이다. ID·행 값은 싣지 않았다. "
           "주 분석 데이터셋은 49c02f8의 평가 행(cold test 352개, personalized 442개)과 (사용자, 시점, 정답) 일치를 assert로 확인했다.", ""]

    # ------------------------------------------------------------- A
    sizes = ds.groupby("uid").size()
    keep = sizes[sizes >= 2].index
    a = ds[ds.uid.isin(keep)].reset_index(drop=True)
    names = ["β1 직전 28일 총 거리 (10km당)", "β2 직전 28일 달리기 횟수 (1회당)", "β3 가장 긴 거리 (1km당)", "β4 마지막 달리기 후 일수 (1일당)", "γ T5(기준 기록) − m (1초당)"]
    X = np.column_stack([a.total_km / 10.0, a.n_runs, a.longest_km, a.days_since_last, a.t5_ref_minus_m]).astype(float)
    beta, bs, Xd, U = within_ols(a.uid.to_numpy(), X, a.y.to_numpy(float))
    corr = np.corrcoef(Xd.T)
    short = ["총거리/10km", "횟수", "최장거리", "마지막후일수", "T5(기준)−m"]
    out += ["## A. 같은 사람 안 분석 (사후 분석)", "",
            f"- 자료: 평가 대상 {len(ds):,}개 중 평가 대상이 2개 이상인 사용자 {U}명의 모든 평가 대상 {len(a):,}개(train·test 모두).",
            "- 식: y = 사용자 고정효과 + β1·(직전 28일 총 거리/10km) + β2·(직전 28일 달리기 횟수) + β3·(가장 긴 거리 km) + β4·(마지막 달리기 후 일수) + γ·(T5(기준 기록) − m) + 오차. 사용자별 평균을 뺀 within OLS.",
            f"- CI: 사용자 단위 bootstrap {N_BOOT:,}회(seed {SEED}), 사용자를 복원추출하고 뽑힌 사용자의 모든 기준 시점 유지, 백분위 95%. 단위는 초(y의 단위).", "",
            md_table([{"계수": nm, "점추정(초)": f"{beta[j]:+.3f}", "95% CI": ci(bs[:, j]), "CI가 0 포함": "예" if np.percentile(bs[:, j], 2.5) <= 0 <= np.percentile(bs[:, j], 97.5) else "아니오"} for j, nm in enumerate(names)]), "",
            "해석(상관, 인과 아님): " + " ".join(
                f"같은 사람 안에서 {lab} 1단위({unit}) 큰 기준 시점은 다음 5km 기록 변화가 평균 {beta[j]:+.2f}초 달랐다(상관, 인과 아님)."
                for j, (lab, unit) in enumerate([("직전 28일 총 거리가", "10km"), ("직전 28일 달리기 횟수가", "1회"), ("가장 긴 거리가", "1km"),
                                                 ("마지막 달리기 후 일수가", "1일"), ("T5(기준 기록) − m이", "1초")])), "",
            "γ는 기준 기록의 상대 위치와 변화량의 관계이며 훈련 효과로 해석하지 않는다. 총 거리·횟수·가장 긴 거리는 서로 강하게 관련될 수 있어 개별 β의 해석이 불안정할 수 있다(아래 상관행렬). CI가 0을 포함한 계수는 \"관계가 없다\"는 증명이 아니다.", "",
            "사용자 평균을 뺀 설명 변수 간 상관행렬:", "",
            md_table([{"": short[i], **{short[j]: f"{corr[i, j]:.2f}" for j in range(5)}} for i in range(5)]), ""]

    # ------------------------------------------------------------- B
    pairs = []
    ext_cnt = 0
    neg = 0
    for uid, g in d.groupby("user_id", sort=True):
        s = g.start_time.to_numpy(np.int64)
        e = s + g.duration_sec.to_numpy(np.int64)
        cd = g.cand.to_numpy()
        sids = g.session_id.to_numpy()
        for i in range(len(g) - 1):
            if cd[i] and cd[i + 1]:
                gap = (s[i + 1] - e[i]) / 86400.0
                if gap < 0:
                    neg += 1
                    continue
                t1, t2 = cache["t5"][int(sids[i])], cache["t5"][int(sids[i + 1])]
                prior = [cache["t5"][int(sids[j])] for j in np.flatnonzero(cd[:i] & (e[:i] < e[i])) if not cache["ext"][int(sids[j])]]
                m_first = float(np.median(prior)) if prior else float(t1)
                ext_cnt += int(cache["ext"][int(sids[i])] or cache["ext"][int(sids[i + 1])])
                pairs.append((int(uid), gap, t2 - t1, t1 - m_first))
    pr = pd.DataFrame(pairs, columns=["uid", "gap", "delta", "rel"])

    def gbin(x):
        for j, (lo, hi) in enumerate(BINS):
            if x >= lo and (hi is None or x < hi):
                return j
    pr["b"] = pr.gap.map(gbin)
    up = np.unique(pr.uid)
    Uq = len(up)
    uidx = np.searchsorted(up, pr.uid.to_numpy())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, Uq, size=(N_BOOT, Uq))
    C = np.stack([np.bincount(r, minlength=Uq) for r in idx]).astype(float)  # (B, Uq) 뽑힌 횟수

    def boot_median(mask):
        v = pr.delta.to_numpy()[mask]
        uu = uidx[mask]
        order = np.argsort(v, kind="stable")
        v, uu = v[order], uu[order]
        W = C[:, uu]
        cum = np.cumsum(W, axis=1)
        tot = cum[:, -1]
        ok = tot > 0
        k1 = (tot + 1) // 2
        k2 = (tot + 2) // 2
        i1 = (cum >= k1[:, None]).argmax(1)
        i2 = (cum >= k2[:, None]).argmax(1)
        med = (v[i1] + v[i2]) / 2.0
        med[~ok] = np.nan
        return np.median(v) if len(v) else np.nan, med, 1 - ok.mean()
    rows = []
    for j, lab in enumerate(BIN_LABELS):
        mk = (pr.b == j).to_numpy()
        n = int(mk.sum())
        if n == 0:
            rows.append({"공백 구간(일)": lab, "쌍 수": 0, "사용자 수": 0, "Δ 중앙값(초)": "-", "95% CI": "-", "개선": "-", "변화 없음": "-", "악화": "-", "비고": "쌍 없음"})
            continue
        pt, med, dropped = boot_median(mk)
        dl = pr.delta.to_numpy()[mk]
        rows.append({"공백 구간(일)": lab, "쌍 수": n, "사용자 수": int(pr[mk].uid.nunique()), "Δ 중앙값(초)": f"{pt:+.1f}", "95% CI": ci(med),
                     "개선": f"{100 * (dl <= -BAND).mean():.1f}%", "변화 없음": f"{100 * ((dl > -BAND) & (dl < BAND)).mean():.1f}%", "악화": f"{100 * (dl >= BAND).mean():.1f}%",
                     "비고": "참고(표본 부족)" if n < 30 else ""})
    allmk = np.ones(len(pr), bool)
    pt, med, dropped = boot_median(allmk)
    dl = pr.delta.to_numpy()
    rows.append({"공백 구간(일)": "전체(구간 합산)", "쌍 수": len(pr), "사용자 수": Uq, "Δ 중앙값(초)": f"{pt:+.1f}", "95% CI": ci(med),
                 "개선": f"{100 * (dl <= -BAND).mean():.1f}%", "변화 없음": f"{100 * ((dl > -BAND) & (dl < BAND)).mean():.1f}%", "악화": f"{100 * (dl >= BAND).mean():.1f}%", "비고": ""})
    # within OLS: 구간 더미(기준 [0,7)) + γ(T5(첫째) − m_first)
    D = np.column_stack([(pr.b == j).astype(float) for j in range(1, 5)] + [pr.rel.to_numpy()])
    beta_b, bs_b, _, _ = within_ols(pr.uid.to_numpy(), D, pr.delta.to_numpy(float))
    nb = [int((pr.b == j).sum()) for j in range(5)]
    lab_b = [f"구간 {BIN_LABELS[j]} (기준 [0,7) 대비)" + (" — 참고(표본 부족)" if nb[j] < 30 else "") for j in range(1, 5)] + ["γ T5(첫째) − m_first (1초당)"]
    out += ["## B. 공백 코호트: 관측된 러닝 기록의 공백 (사후 분석)", "",
            f"- 쌍: 시작 시각순으로 이웃한 두 달리기 세션이 모두 5km 후보(비병합 5.0~5.5km). 겹침(공백 음수) 쌍은 제외했다: **{neg}쌍**. 남은 쌍 {len(pr):,}개, 사용자 {Uq}명. "
            f"첫째·둘째 중 극단(T5 > 5,000초)이 있는 쌍은 제외하지 않고 포함했다: {ext_cnt}쌍.",
            "- Δ = T5(둘째) − T5(첫째), 음수 = 개선. m_first = 첫째 종료 이전에 종료된 그 사용자의 비극단 5km 후보 T5 중앙값(첫째 제외, 없으면 T5(첫째)).",
            f"- Δ 중앙값 CI: 쌍이 있는 사용자 {Uq}명을 복원추출({N_BOOT:,}회, seed {SEED})하고 뽑힌 사용자의 해당 구간 쌍 전체의 중앙값을 반복 계산한 백분위 95%. 방향 밴드 ±{BAND:g}초(Δ ≤ −10 개선, −10 < Δ < 10 변화 없음, Δ ≥ 10 악화).",
            f"- 쌍이 30개 미만인 구간은 \"참고(표본 부족)\". 한 사용자의 쌍들은 독립이 아니다.", "",
            md_table(rows), "",
            "같은 사람 안 비교(사후 분석): Δ = 사용자 고정효과 + 공백 구간 더미(기준 = [0,7)) + γ·(T5(첫째) − m_first) + 오차. 사용자별 평균을 뺀 within OLS, 같은 사용자 단위 bootstrap.", "",
            md_table([{"계수": lab_b[j], "점추정(초)": f"{beta_b[j]:+.3f}", "95% CI": ci(bs_b[:, j]), "CI가 0 포함": "예" if np.percentile(bs_b[:, j], 2.5) <= 0 <= np.percentile(bs_b[:, j], 97.5) else "아니오"} for j in range(5)]), "",
            "해석(상관, 인과 아님): 구간 계수는 같은 사람 안에서 첫째와 둘째 5km 사이의 관측된 러닝 기록의 공백이 [0,7)일이 아닌 해당 구간이었던 쌍의 Δ가 평균 몇 초 달랐는지(상관)를 뜻한다. "
            "공백 구간 안에 관측되지 않은 운동·기기 미착용 등이 있을 수 있으며 데이터에서 구분할 수 없다. 쌍이 적은 구간과 사용자 내 구간 변동이 작은 경우 추정이 불안정하다(CI 참고).", ""]
    DOC_PATH.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC_PATH)


if __name__ == "__main__":
    main()
