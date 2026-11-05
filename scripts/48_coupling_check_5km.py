# -*- coding: utf-8 -*-
"""
사후 탐색: γ의 수학적 결합 점검. 사양: docs/posthoc_spec_coupling_5km.md (커밋 c7e9a33, 분석 실행 전 고정).
**사후 탐색(test 정답을 본 뒤의 분석)이며 주 결론(49c02f8)을 바꾸지 않는다. 어떤 결과가 나와도 γ를 훈련·회복 효과로 해석하지 않는다.**
(a) m_ex(기준 기록을 뺀 t 이전 유효 5km 중앙값, 없으면 제외)로 43번 within OLS의 γ를 재추정(+ 같은 표본에서 원래 x로 추정한 대조).
(b) 귀무: 사용자 안에서 정답 T5(= y + T5(기준))를 섞어 y* = T5*(정답) - T5(기준)으로 γ를 다시 추정, 1,000회, seed 42.
검산: 43번의 γ(-0.845)를 같은 코드로 재현(assert). ID·행 값은 출력·저장하지 않는다.
사용법(저장소 루트): python scripts/48_coupling_check_5km.py   (run_all에는 넣지 않음)
출력: docs/coupling_check_5km.md
"""
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m43 = _load("43_within_person_5km.py", "m43")
m42, m38, feas = m43.m42, m43.m38, m43.feas
md_table = m42.md_table
DOC = Path("docs") / "coupling_check_5km.md"
SPEC_COMMIT = "c7e9a33"
N_PERM, PERM_SEED = 1000, 42
WEEK = feas.WEEK
NAMES = ["β1 직전 28일 총 거리 (10km당)", "β2 직전 28일 달리기 횟수 (1회당)", "β3 가장 긴 거리 (1km당)", "β4 마지막 달리기 후 일수 (1일당)", "γ x (1초당)"]


def m_ex_table(d, grp, cache):
    """평가 대상(uid, t)별 m_ex와 m_ex 계산에 쓴 세션 수. 42번 build_dataset와 같은 논리, 기준 기록 제외."""
    ext_by_sid, t5_by_sid = cache["ext"], cache["t5"]
    rows = []
    for uid, g in d.groupby("user_id", sort=True):
        uid = int(uid)
        if uid not in grp:
            continue
        s = g["start_time"].to_numpy(np.int64)
        e = s + g["duration_sec"].to_numpy(np.int64)
        iscand = g["cand"].to_numpy()
        sids = g["session_id"].to_numpy()
        ext = np.array([ext_by_sid.get(int(x), False) for x in sids])
        t5 = np.array([t5_by_sid.get(int(x), np.nan) for x in sids])
        for k in feas.span_ks({"s": s}, m38.MONDAY0):
            t = m38.MONDAY0 + k * WEEK
            inwin = (e >= t - m38.WIN) & (e < t)
            if int(inwin.sum()) < 4:
                continue
            cw = np.flatnonzero(inwin & iscand)
            valid = cw[~ext[cw]]
            if len(valid) == 0:
                continue
            fut = np.flatnonzero(iscand & (s >= t) & (s < t + WEEK))
            if len(fut) == 0:
                continue
            ref = int(valid[-1])
            past = np.flatnonzero(iscand & (e < t) & ~ext)
            others = past[past != ref]
            rows.append(dict(uid=uid, t=t, t5_ref_chk=float(t5[ref]), m_ex=float(np.median(t5[others])) if len(others) else np.nan, n_others=int(len(others))))
    return pd.DataFrame(rows)


def design(df, xcol):
    return np.column_stack([df.total_km / 10.0, df.n_runs, df.longest_km, df.days_since_last, df[xcol]]).astype(float)


def ci_str(col):
    lo, hi = np.percentile(col, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]", lo <= 0 <= hi


def main():
    d, tr, te, grp, cache = m43.load_base()
    ds = m43.main_dataset(d, grp, cache)
    n_cold, n_pers = m43.verify_against_final(ds, tr, te)
    assert len(ds) == 2072 and (n_cold, n_pers) == (352, 442)
    tbl = m_ex_table(d, grp, cache)
    j = ds.merge(tbl, on=["uid", "t"], how="left", validate="one_to_one")
    assert len(j) == 2072 and np.array_equal(j.t5_ref.to_numpy(), j.t5_ref_chk.to_numpy())

    sizes = j.groupby("uid").size()
    pop = j[j.uid.isin(sizes[sizes >= 2].index)].sort_values(["uid", "t"]).reset_index(drop=True)
    assert (len(pop), pop.uid.nunique()) == (1973, 250)
    y = pop.y.to_numpy(float)
    g = pop.uid.to_numpy()

    # ---- 검산: 43번 γ 재현
    X0 = design(pop, "t5_ref_minus_m")
    beta0, bs0, Xd0, U0 = m43.within_ols(g, X0, y)
    gamma0 = float(beta0[4])
    txt43 = Path("docs/within_person_results_5km.md").read_text(encoding="utf-8")
    mm = re.search(r"\| γ T5\(기준 기록\) − m \(1초당\) \| ([+-][\d.]+) \|", txt43)
    assert abs(gamma0 - float(mm.group(1))) < 5.1e-4, (gamma0, mm.group(1))
    ci0, zero0 = ci_str(bs0[:, 4])

    # ---- (a) m_ex
    has = pop.m_ex.notna()
    sub = pop[has].reset_index(drop=True)
    n_ex_drop = int((~has).sum())
    u_sub = sub.uid.nunique()
    sub = sub.assign(x_ex=sub.t5_ref - sub.m_ex)
    ys, gs = sub.y.to_numpy(float), sub.uid.to_numpy()
    b_same, bs_same, _x1, U_same = m43.within_ols(gs, design(sub, "t5_ref_minus_m"), ys)
    b_ex, bs_ex, _x2, U_ex = m43.within_ols(gs, design(sub, "x_ex"), ys)
    ci_same, zero_same = ci_str(bs_same[:, 4])
    ci_ex, zero_ex = ci_str(bs_ex[:, 4])
    assert U_same == U_ex == u_sub

    # ---- (b) 귀무: 사용자 안에서 정답 T5 섞기
    order = np.argsort(g, kind="stable")
    assert np.array_equal(order, np.arange(len(g)))              # pop이 uid, t 순으로 정렬돼 있음
    users, start_idx = np.unique(g, return_index=True)
    ends = list(start_idx[1:]) + [len(g)]
    R = pop.t5_ref.to_numpy(float)
    A = y + R                                                     # 정답 T5
    inv = np.searchsorted(users, g)
    cnt = np.bincount(inv).astype(float)
    Xd = Xd0                                                      # within 변환된 X (43번과 같음)
    P = np.linalg.pinv(Xd.T @ Xd) @ Xd.T                          # (k, n)
    rng = np.random.default_rng(PERM_SEED)
    gam = np.empty(N_PERM)
    for b in range(N_PERM):
        As = A.copy()
        for a0, a1 in zip(start_idx, ends):
            if a1 - a0 > 1:
                As[a0:a1] = A[a0:a1][rng.permutation(a1 - a0)]
        ystar = As - R
        yd = ystar - (np.bincount(inv, weights=ystar) / cnt)[inv]
        gam[b] = (P @ yd)[4]
    # 섞지 않은 y로 같은 식을 풀면 원래 γ가 나오는지(식 검산)
    yd0 = y - (np.bincount(inv, weights=y) / cnt)[inv]
    assert abs((P @ yd0)[4] - gamma0) < 1e-9
    q_lo, q_med, q_hi = np.percentile(gam, [2.5, 50, 97.5])
    pos = float((gam <= gamma0).mean())
    big_neg = bool(q_hi < 0 and q_med <= 0.5 * gamma0)

    if big_neg:
        sent_null = "**γ의 상당 부분은 정의상 결합으로 설명될 수 있음.** 귀무 분포(사용자 안에서 정답 T5를 섞어도 남는 부분)의 2.5%~97.5% 구간이 모두 0보다 작고 중앙값이 원래 γ의 절반 이하다."
    else:
        sent_null = ("귀무 분포에서는 원래 γ와 같은 크기의 음의 값이 나타나지 않았다(정의상 결합만으로는 γ 전부가 설명되지 않는 것으로 보임). "
                     "이것이 훈련·회복 효과의 증거라는 뜻은 아니며, 귀무가 모든 연관(예: 평균 회귀)을 없애지는 않는다.")
    if gamma0 < q_lo:
        where = "원래 γ는 귀무 2.5%~97.5% 구간 아래(더 음수)에 있다"
    elif gamma0 > q_hi:
        where = "원래 γ는 귀무 2.5%~97.5% 구간 위에 있다"
    else:
        where = "원래 γ는 귀무 2.5%~97.5% 구간 안으로, 귀무에서 흔히 나오는 범위다"

    out = ["# 사후 탐색 결과: γ의 수학적 결합 점검 (5km 과제)", "",
           f"생성: `scripts/48_coupling_check_5km.py`. 사양: `docs/posthoc_spec_coupling_5km.md`(커밋 {SPEC_COMMIT}, 분석 실행 전 고정). **사후 탐색**(test 정답을 본 뒤의 분석)이며 상관관계까지만 주장한다. "
           "**어떤 결과가 나와도 γ를 훈련·회복 효과로 해석하지 않으며 인과·처방 표현을 쓰지 않는다.** 주 결론(49c02f8)·모델·정의·범위·시연 예측식은 바꾸지 않았다. 모든 평가는 \"다음 7일 안에 적격 5km 기록이 관측된 경우\"에 조건부다. ID·행 값은 싣지 않았다.", "",
           "배경: y = T5(정답) − T5(기준)과 x = T5(기준) − m에 같은 T5(기준)이 **반대 부호로** 들어가므로, 정답 T5가 기준 기록과 무관해도 γ는 이 결합과 측정 오차만으로 음수가 될 수 있다.", "",
           "## 1. 표본과 검산", "",
           md_table([{"항목": "43번 대상 범위(평가 대상 2개 이상 사용자의 모든 평가 대상)", "평가 대상": "1,973", "사용자": "250"},
                     {"항목": f"(a)에서 제외: 기준 기록을 뺀 t 이전 유효 5km 세션이 없음(m_ex 없음)", "평가 대상": f"{n_ex_drop:,}", "사용자": f"{pop[~has].uid.nunique():,} (제외된 평가 대상이 1개 이상인 사용자)"},
                     {"항목": "(a) 분석 표본", "평가 대상": f"{len(sub):,}", "사용자": f"{u_sub:,}"}]), "",
           f"검산: 43번과 같은 코드·표본으로 γ를 다시 추정한 값 {gamma0:+.3f} {ci0}가 `docs/within_person_results_5km.md`의 γ와 같음을 assert로 확인했다. 모형·bootstrap(사용자 단위 {m43.N_BOOT:,}회, seed {m43.SEED}, 백분위 95% CI)은 43번과 같다. 단위는 초(y의 단위).", "",
           "## 2. (a) m_ex로 γ 재추정", "",
           md_table([{"설정": "원래 γ (43번, x = T5(기준) − m, 43번 표본)", "표본": "1,973개 / 250명", "γ (1초당)": f"{gamma0:+.3f}", "95% CI": ci0, "CI가 0 포함": "예" if zero0 else "아니오"},
                     {"설정": "대조: 같은 모형, (a) 표본에서 원래 x", "표본": f"{len(sub):,}개 / {u_sub}명", "γ (1초당)": f"{b_same[4]:+.3f}", "95% CI": ci_same, "CI가 0 포함": "예" if zero_same else "아니오"},
                     {"설정": "(a) γ_ex (x_ex = T5(기준) − m_ex)", "표본": f"{len(sub):,}개 / {u_sub}명", "γ (1초당)": f"{b_ex[4]:+.3f}", "95% CI": ci_ex, "CI가 0 포함": "예" if zero_ex else "아니오"}]), "",
           "m_ex = 기준 기록을 뺀 t 이전 유효 5km(비병합 5.0~5.5km, T5 ≤ 5,000초)의 T5 중앙값. 다른 설명 변수는 원래 모형 그대로다.", "",
           "## 3. (b) 귀무 점검: 사용자 안에서 정답 T5 섞기", "",
           f"사용자별로 그 사용자의 평가 대상들의 정답 T5(= y + T5(기준))를 섞어 다시 배정하고 y* = T5*(정답) − T5(기준)으로 같은 모형의 γ*를 점추정했다({N_PERM:,}회, seed {PERM_SEED}; 평가 대상이 1개뿐인 사용자는 그대로, 표본은 1,973개 / 250명). x와 다른 설명 변수는 원래 값이다.", "",
           md_table([{"항목": "원래 γ", "값": f"{gamma0:+.3f}"},
                     {"항목": "귀무 γ* 중앙값", "값": f"{q_med:+.3f}"},
                     {"항목": "귀무 γ* 2.5% 분위수", "값": f"{q_lo:+.3f}"},
                     {"항목": "귀무 γ* 97.5% 분위수", "값": f"{q_hi:+.3f}"},
                     {"항목": "γ* ≤ 원래 γ 인 비율(경험적 위치)", "값": f"{pos:.3f} ({int((gam <= gamma0).sum())} / {N_PERM:,})"}]), "",
           "## 4. 해석 (사양 4절의 규칙)", "",
           f"- (b) {sent_null}",
           f"- 위치: {where}(원래 γ {gamma0:+.3f}, 귀무 중앙값 {q_med:+.3f}, 2.5%~97.5% [{q_lo:+.3f}, {q_hi:+.3f}]).",
           f"- (a) γ_ex = {b_ex[4]:+.3f} {ci_ex}: CI가 0을 {'포함하므로 **관계를 확인하지 못함**' if zero_ex else '포함하지 않지만 **상관이며 인과가 아니다**'}. 같은 표본 대조 γ {b_same[4]:+.3f}와의 차이는 점추정만 기술한다(m에서 기준 기록 자신이 빠져도 T5(기준)이 x와 y에 반대 부호로 들어가는 구조는 그대로다).",
           "- **어떤 결과가 나와도** γ를 훈련·회복 효과로 해석하지 않는다. 이 결과로 주 결론·모델·정의·범위·필터·시연 예측식을 바꾸지 않는다.", "",
           "한계: 이 점검은 정답 T5를 사용자 안에서 섞는 한 가지 귀무만 본다. 섞으면 사용자별 수준 외의 시점·기준 기록과의 연관(실제 평균 회귀 포함)이 모두 사라진다. "
           "m이 사용자 안에서 시간에 따라 변하고 평가 대상 수가 사용자마다 달라(1개뿐인 사용자 포함) γ*의 값은 이 표본의 구조에 의존한다. 사용자 단위 불확실성은 귀무 분포 폭(반복 간 변동)만 보여 준다.", ""]
    DOC.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC, "| 원래 γ %+.3f | (a) %+.3f %s (same-sample %+.3f) | null med %+.3f [%+.3f, %+.3f] pos %.3f big_neg %s" % (gamma0, b_ex[4], ci_ex, b_same[4], q_med, q_lo, q_hi, pos, big_neg))


if __name__ == "__main__":
    main()
