# -*- coding: utf-8 -*-
"""
사후 탐색: 기준 기록 세션의 상대 심박 r과 5km 기록 변화. 사양: docs/posthoc_spec_hr_5km.md (커밋 9d2efda, 분석 실행 전 고정).
**사후 탐색(test 정답을 본 뒤의 분석)이며 주 결론(49c02f8)과 시연 예측식을 바꾸지 않는다.** 상관관계까지만 주장한다.

- h_base = 기준 기록 세션의 avg_heart_rate, h_ref = t 이전에 끝난 본인의 유효 5km 세션(기준 기록 제외) avg_heart_rate의 중앙값(없으면 제외),
  r = h_base - h_ref (bpm), 모형에는 r10 = r / 10을 넣는다.
- 모형: 43번과 같은 고정효과 within OLS(`within_ols`), 사용자 단위 bootstrap 2,000회·seed 42·백분위 95% CI.
  주 모형: y ~ (T5(기준)-m) + r10, 보조 모형: 위 + (T5(기준)-m) x r10.
- 대상 범위는 43번과 동일(평가 대상이 2개 이상인 사용자의 모든 평가 대상)에서 h_ref가 없는 평가 대상을 제외.
- ID·좌표·행 단위 값은 출력·저장하지 않는다. 사용법(저장소 루트에서): python scripts/47_hr_posthoc_5km.py  (run_all에는 넣지 않음)
출력: docs/hr_posthoc_results_5km.md
"""
import importlib.util
import os
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
DOC_PATH = Path("docs") / "hr_posthoc_results_5km.md"
SPEC_COMMIT = "9d2efda"
WEEK = feas.WEEK


def build_targets(d, grp, cache):
    """42번 build_dataset와 같은 논리로 각 평가 대상의 기준 기록 세션·h_base·h_ref를 모은다(ID는 저장하지 않음)."""
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
        hr = g["avg_heart_rate"].to_numpy(float)
        ext = np.array([ext_by_sid.get(int(x), False) for x in sids])
        for k in feas.span_ks({"s": s}, m38.MONDAY0):
            t = m38.MONDAY0 + k * WEEK
            inwin = (e >= t - m38.WIN) & (e < t)
            if int(inwin.sum()) < 4:
                continue
            cw = np.flatnonzero(inwin & iscand)
            valid = cw[~ext[cw]]
            if len(valid) == 0:
                continue
            ref = int(valid[-1])
            fut = np.flatnonzero(iscand & (s >= t) & (s < t + WEEK))
            if len(fut) == 0:
                continue
            past = np.flatnonzero(iscand & (e < t) & ~ext)       # m을 만드는 집합과 같음(사양 6.1~6.2)
            past = past[past != ref]                               # 기준 기록 제외
            hv = hr[past]
            hv = hv[np.isfinite(hv)]                               # 사양 6.5: 결측 세션은 h_ref 계산에서 제외
            rows.append(dict(uid=uid, t=t, t5_ref=float(t5_by_sid[int(sids[ref])]), h_base=float(hr[ref]),
                             h_ref=float(np.median(hv)) if len(hv) else np.nan, n_h_ref=int(len(hv)), n_past_nan=int(len(past) - len(hv))))
    return pd.DataFrame(rows)


def main():
    d, tr, te, grp, cache = m43.load_base()
    ds = m43.main_dataset(d, grp, cache)
    n_cold, n_pers = m43.verify_against_final(ds, tr, te)
    assert len(ds) == 2072 and (n_cold, n_pers) == (352, 442), (len(ds), n_cold, n_pers)

    tg = build_targets(d, grp, cache)
    assert len(tg) == len(ds)
    j = ds.merge(tg, on=["uid", "t"], how="left", validate="one_to_one")
    assert len(j) == 2072 and np.array_equal(j.t5_ref_x.to_numpy(), j.t5_ref_y.to_numpy()), "기준 기록 정합 실패"
    j = j.rename(columns={"t5_ref_x": "t5_ref"}).drop(columns=["t5_ref_y"])

    # ---- 43번 대상 범위: 평가 대상이 2개 이상인 사용자의 모든 평가 대상
    sizes = j.groupby("uid").size()
    pop = j[j.uid.isin(sizes[sizes >= 2].index)].reset_index(drop=True)
    n_pop, u_pop = len(pop), pop.uid.nunique()
    assert (n_pop, u_pop) == (1973, 250), (n_pop, u_pop)
    # ---- 제외: h_base 결측, h_ref 없음(사양 1절·6.5)
    miss_base = pop.h_base.isna()
    no_ref = pop.h_ref.isna()
    excl = miss_base | no_ref
    an = pop[~excl].reset_index(drop=True)
    an["r"] = an.h_base - an.h_ref
    an["r10"] = an.r / 10.0
    cnt_an = an.groupby("uid").size()
    n_an, u_an = len(an), an.uid.nunique()
    u_an2 = int((cnt_an >= 2).sum())
    n_excl = int(excl.sum())
    u_excl = pop[excl].uid.nunique()
    u_lost = u_pop - u_an

    # ---- 모형
    y = an.y.to_numpy(float)
    g = an.uid.to_numpy()
    x1 = an.t5_ref_minus_m.to_numpy(float)
    r10 = an.r10.to_numpy(float)
    names_main = ["(T5(기준) − m) 1초당", "r10: r이 +10bpm 클 때"]
    beta1, bs1, _xd, U1 = m43.within_ols(g, np.column_stack([x1, r10]), y)
    names_aux = names_main + ["(T5(기준) − m) × r10: r이 +10bpm 클 때 γ의 변화(1초당)"]
    beta2, bs2, _xd2, U2 = m43.within_ols(g, np.column_stack([x1, r10, x1 * r10]), y)
    assert U1 == U2 == u_an

    def zero_in(col):
        lo, hi = np.percentile(col, [2.5, 97.5])
        return lo <= 0 <= hi

    def coef_table(names, beta, bs):
        return md_table([{"계수": nm, "점추정(초)": f"{beta[i]:+.3f}", "95% CI": m43.ci(bs[:, i]), "CI가 0 포함": "예" if zero_in(bs[:, i]) else "아니오"}
                         for i, nm in enumerate(names)])

    def reading(names, beta, bs):
        out = []
        for i, nm in enumerate(names):
            if zero_in(bs[:, i]):
                out.append(f"- {nm}: CI가 0을 포함하므로 **관계를 확인하지 못함**(관계가 없다는 증명이 아니다).")
            else:
                out.append(f"- {nm}: CI가 0을 포함하지 않지만 **상관이며 인과가 아니다**.")
        return out

    q1, med, q3 = np.percentile(an.r, [25, 50, 75])
    lines = ["# 사후 탐색 결과: 기준 기록 세션의 상대 심박 (5km 과제)", "",
             f"생성: `scripts/47_hr_posthoc_5km.py`. 사양: `docs/posthoc_spec_hr_5km.md`(커밋 {SPEC_COMMIT}, 분석 실행 전 고정). 이 문서의 구성은 사양 4절의 보고 항목 순서를 따른다. "
             "**사후 탐색**(test 정답을 본 뒤의 분석)이며 상관관계까지만 주장하고 인과·처방 해석을 하지 않는다. "
             "**freeze-5km-v1 결과(`docs/final_results_5km.md`, 49c02f8)와 시연 예측식(`demo/b2_params.json`, `demo/coach_demo.py`)은 바꾸지 않았고, 이 결과를 시연이나 최종 평가에 반영하지 않는다.** "
             "시각은 start_time 원값(기준 미확인)이다. ID·행 단위 값은 싣지 않았다. 주 분석 데이터셋이 49c02f8의 평가 행(cold test 352개, personalized 442개)과 일치함을 assert로 확인했다.", "",
             "정의(사양 1·6절): h_base = 기준 기록 세션 `avg_heart_rate`(30~230bpm 유효 점 평균, 세션 전체), h_ref = t 이전에 끝난 본인의 유효 5km 세션(비병합 5.0~5.5km, T5 ≤ 5,000초, 기준 기록 제외) `avg_heart_rate`의 중앙값, "
             "r = h_base − h_ref(bpm). 모형에는 r10 = r/10을 넣어 계수를 +10bpm당 값으로 보고한다.", "",
             "## 1. 포함/제외 개수와 사람 수", "",
             md_table([
                 {"단계": "평가 대상 전체(참고)", "평가 대상": f"{len(j):,}", "사용자": f"{j.uid.nunique():,}"},
                 {"단계": "43번 대상 범위(평가 대상 2개 이상 사용자의 모든 평가 대상)", "평가 대상": f"{n_pop:,}", "사용자": f"{u_pop:,}"},
                 {"단계": "제외: h_ref 없음(기준 기록 외 t 이전 유효 5km 세션이 없음)", "평가 대상": f"{int(no_ref.sum()):,}", "사용자": f"{pop[no_ref].uid.nunique():,}"},
                 {"단계": "제외: h_base 결측(세션 표 avg_heart_rate)", "평가 대상": f"{int(miss_base.sum()):,}", "사용자": f"{pop[miss_base].uid.nunique():,}"},
                 {"단계": "제외 합계", "평가 대상": f"{n_excl:,}", "사용자": f"{u_excl:,} (제외된 평가 대상이 1개 이상인 사용자)"},
                 {"단계": "분석 표본", "평가 대상": f"{n_an:,}", "사용자": f"{u_an:,}"},
                 {"단계": "└ 분석 표본에서 평가 대상이 2개 이상 남은 사용자", "평가 대상": "-", "사용자": f"{u_an2:,}"},
                 {"단계": "└ 분석 표본에서 사라진 사용자(43번 대상 범위 대비)", "평가 대상": "-", "사용자": f"{u_lost:,}"},
             ]), "",
             "h_ref 계산에서 avg_heart_rate 결측으로 제외된 세션 수: " + f"{int(tg.n_past_nan.sum()):,}개(평가 대상 {len(tg):,}개 전체 기준 합계). "
             "남은 평가 대상이 1개뿐인 사용자는 within 변환에서 기여가 0이며 추가로 제외하지 않았다(사양 3절).", "",
             "## 2. r의 분포 (분석 표본)", "",
             md_table([{"통계": "중앙값", "r (bpm)": f"{med:+.2f}", "r10 (10bpm 단위)": f"{med / 10:+.3f}"},
                       {"통계": "25% 분위수", "r (bpm)": f"{q1:+.2f}", "r10 (10bpm 단위)": f"{q1 / 10:+.3f}"},
                       {"통계": "75% 분위수", "r (bpm)": f"{q3:+.2f}", "r10 (10bpm 단위)": f"{q3 / 10:+.3f}"},
                       {"통계": "IQR(75% − 25%)", "r (bpm)": f"{q3 - q1:.2f}", "r10 (10bpm 단위)": f"{(q3 - q1) / 10:.3f}"}]), "",
             f"분석 표본 평가 대상 {n_an:,}개 기준. r > 0은 기준 기록 세션의 평균 심박이 본인의 다른 유효 5km 세션 중앙값보다 높았다는 뜻이다.", "",
             "## 3. 모형 계수와 95% CI", "",
             f"- 추정: 43번과 같은 고정효과 within OLS(사용자별 평균을 뺌), 사용자 단위 bootstrap {m43.N_BOOT:,}회(seed {m43.SEED}), 사용자를 복원추출하고 뽑힌 사용자의 모든 평가 대상을 유지, 백분위 95% CI. 단위는 초(y의 단위).",
             f"- 표본: 평가 대상 {n_an:,}개, 사용자 {u_an:,}명.", "",
             "### 3-1. 주 모형: y ~ (T5(기준) − m) + r10", "",
             coef_table(names_main, beta1, bs1), "",
             "### 3-2. 보조 모형: 주 모형 + (T5(기준) − m) × r10", "",
             coef_table(names_aux, beta2, bs2), "",
             "## 4. CI가 0을 포함하는지와 읽는 법 (사양 5절의 해석 규칙)", "",
             "- CI가 0을 포함하면 **\"관계를 확인하지 못함\"**이라고 쓴다. 관계가 없다는 증명(동등성)이 아니다.",
             "- CI가 0을 포함하지 않아도 **상관이며 인과가 아니다**. 평균 심박은 기온·컨디션·센서·훈련 강도 등과 얽힐 수 있으며 이 중 어느 것도 확인하지 않았다(미확인).",
             "- 모델이 둘이고 계수가 여럿이므로 우연히 CI가 0을 벗어나는 계수가 생길 수 있다. 결과를 보고 모형·정의·범위를 바꾸지 않았다.", "",
             "주 모형:", *reading(names_main, beta1, bs1), "", "보조 모형:", *reading(names_aux, beta2, bs2), "",
             "한계: h_base는 세션 전체 평균 심박이며 처음 5km 구간의 평균이 아니다(사양 6.4). T5(기준) − m의 계수 γ는 기준 기록의 상대 위치와 변화량의 관계이며 훈련 효과로 해석하지 않는다.", ""]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("saved", DOC_PATH, "| 대상", n_pop, "-> 분석 표본", n_an, "/ 사용자", u_an)


if __name__ == "__main__":
    main()
