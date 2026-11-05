# -*- coding: utf-8 -*-
"""
사후 탐색: 사용자별 짝지은 오차 차이 d_u = (모델 사용자 MAE) - (B2 사용자 MAE). 사양: docs/posthoc_spec_paired_5km.md (커밋 99950af, 분석 실행 전 고정).
49c02f8이 저장한 평가 행별 예측값(data/final_preds_5km.pkl)만 읽고 새로 학습하지 않는다. 문서에는 집계값만(ID·행 값 없음).
검산: d_u의 평균이 docs/final_results_5km.md (c)표의 '사용자 평균 MAE 차이'와 같아야 한다(assert).
사용법(저장소 루트): python scripts/49_paired_user_diff_5km.py   (run_all에는 넣지 않음)
출력: docs/paired_user_diff_5km.md
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PRED = Path("data") / "final_preds_5km.pkl"
FINAL = Path("docs") / "final_results_5km.md"
DOC = Path("docs") / "paired_user_diff_5km.md"
N_BOOT, SEED = 2000, 42
MODELS = ["Ridge", "GBM", "MLP"]
COND = [("cold", "cold-start test", "## A.", "## B."), ("pers", "personalized", "## B.", None)]


def md_table(rows):
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def final_means():
    txt = FINAL.read_text(encoding="utf-8")
    out = {}
    for key, _n, a, b in COND:
        sec = txt[txt.index(a): txt.index(b) if b else len(txt)]
        sec = sec[sec.index("### (c)"): sec.index("### (d)")]
        for m in MODELS:
            mm = re.search(r"\| " + m + r" \| ([+-][\d.]+) \[", sec)
            out[(key, m)] = float(mm.group(1))
    return out


def main():
    P = pd.read_pickle(PRED)
    fm = final_means()
    out = ["# 사후 탐색 결과: 사용자별 짝지은 오차 차이 (모델 − B2, 5km 과제)", "",
           "생성: `scripts/49_paired_user_diff_5km.py`. 사양: `docs/posthoc_spec_paired_5km.md`(커밋 99950af, 분석 실행 전 고정). 구성은 사양 2절의 보고 항목 순서를 따른다. "
           "**사후 탐색**(test 정답을 본 뒤의 분석)이며 상관관계까지만 주장한다. 49c02f8이 저장한 평가 행별 예측값만 읽었고 새로 학습하지 않았다. "
           "**주 결론(49c02f8), 모델, 정의, 범위, 시연 예측식은 바꾸지 않았다.** 모든 평가는 \"다음 7일 안에 적격 5km 기록이 관측된 경우\"에 조건부다. ID·행 값은 싣지 않았다.", "",
           "d_u = (모델의 사용자 u MAE) − (B2의 사용자 u MAE), 단위 초, 음수 = 모델 쪽 오차가 작음. 기본 입력(11) Ridge·GBM·MLP.", ""]
    sec1, sec2, sec3, sec4 = [], [], [], []
    worse = []
    for key, name, _a, _b in COND:
        c = P[key]
        uid, y = np.asarray(c["uid"]), np.asarray(c["y"], float)
        users, inv = np.unique(uid, return_inverse=True)
        U = len(users)
        cnt = np.bincount(inv, minlength=U).astype(float)
        err_b2 = np.abs(y - np.asarray(c["preds"]["B2"], float))
        mae_b2 = np.bincount(inv, weights=err_b2, minlength=U) / cnt
        idx = np.random.default_rng(SEED).integers(0, U, size=(N_BOOT, U))
        q1, med, q3 = np.percentile(cnt, [25, 50, 75])
        sec1.append({"평가 조건": name, "평가 대상": f"{len(y):,}", "사용자": f"{U:,}", "사용자당 평가 대상 중앙값 [25%, 75%]": f"{med:.0f} [{q1:.0f}, {q3:.0f}]",
                     "평가 대상이 1개뿐인 사용자": f"{int((cnt == 1).sum()):,}"})
        for m in MODELS:
            pred = np.asarray(c["preds"][str((m, "기본(11)"))], float)   # 저장된 키는 튜플의 문자열 표현
            mae_m = np.bincount(inv, weights=np.abs(y - pred), minlength=U) / cnt
            d = mae_m - mae_b2
            assert abs(d.mean() - fm[(key, m)]) < 5.1e-4, (key, m, d.mean(), fm[(key, m)])   # 49c02f8 (c)표와 일치
            pc = np.percentile(d, [5, 25, 50, 75, 95])
            sec2.append({"평가 조건": name, "모델 − B2": m, "d_u 평균(= 49c02f8 사용자 평균 MAE 차이)": f"{d.mean():+.3f}", "중앙값": f"{pc[2]:+.3f}",
                         "25%·75%": f"{pc[1]:+.3f}, {pc[3]:+.3f}", "5%·95%": f"{pc[0]:+.3f}, {pc[4]:+.3f}"})
            neg, zero, pos = int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())
            sec3.append({"평가 조건": name, "모델 − B2": m, "d_u < 0 (모델이 낮음)": f"{neg} ({100 * neg / U:.1f}%)", "d_u = 0": f"{zero} ({100 * zero / U:.1f}%)",
                         "d_u > 0 (모델이 높음)": f"{pos} ({100 * pos / U:.1f}%)"})
            med_b = np.median(d[idx], axis=1)
            if np.percentile(med_b, 2.5) > 0:
                worse.append(f"{name} {m}")
            pos_b = (d[idx] > 0).mean(axis=1)
            sec4.append({"평가 조건": name, "모델 − B2": m, "d_u 중앙값 [95% CI]": f"{pc[2]:+.3f} {ci(med_b)}", "CI가 0 포함": "예" if np.percentile(med_b, 2.5) <= 0 <= np.percentile(med_b, 97.5) else "아니오",
                         "d_u > 0 비율 [95% CI]": f"{100 * pos / U:.1f}% [{100 * np.percentile(pos_b, 2.5):.1f}%, {100 * np.percentile(pos_b, 97.5):.1f}%]",
                         "CI가 50% 포함": "예" if np.percentile(pos_b, 2.5) <= 0.5 <= np.percentile(pos_b, 97.5) else "아니오"})
    out += ["## 1. 사용자 수와 사용자당 평가 대상 수", "", md_table(sec1), "",
            "## 2. d_u의 분포 (초)", "", md_table(sec2), "",
            "d_u의 평균은 `docs/final_results_5km.md` (c)표의 \"사용자 평균 MAE 차이\"와 같은 값임을 assert로 확인했다(그 문서의 CI를 인용한다).", "",
            "## 3. 모델이 낮은·같은·높은 사용자 수와 비율 (분모 = 사용자 수)", "", md_table(sec3), "",
            f"## 4. 사용자 단위 bootstrap 95% CI (사용자 복원추출 {N_BOOT:,}회, seed {SEED}, 모든 모델·조건에 같은 추출)", "", md_table(sec4), "",
            "## 5. 읽는 법 (사양 3절의 해석 규칙)", "",
            "- 중앙값의 CI가 0을 포함하면 \"점추정은 컸으나(또는 작았으나) 우열을 확정할 수 없음\"이며 \"현재 자료로 뚜렷한 연관성을 확인하지 못함\"이다. 비유의는 동등성의 증명이 아니다.",
            "- CI 하한이 0보다 클 때만 \"B2보다 나쁨\"이라고 쓴다.",
            "- d_u > 0 비율의 CI가 50%를 포함하면 \"절반 이상의 사용자에서 모델이 높다(또는 낮다)고 확정할 수 없음\"이다.",
            "- 사용자별 오차는 평가 대상이 적은 사용자에서 변동이 크고(1개뿐인 사용자는 그 한 건의 오차), 사용자 능력의 차이로 해석하지 않는다. cold-start는 사용자가 71명뿐이라 불확실성이 크다.",
            "- 이 결과로 주 결론·모델·정의·범위·필터·시연 예측식을 바꾸지 않는다.", "",
            "## 6. 요약", "",
            ("- d_u 중앙값의 CI 하한이 0보다 큰(\"B2보다 나쁨\"을 쓸 수 있는) 경우: " + ", ".join(worse) + ". 이는 사용자 평균 MAE 차이 CI가 0을 제외한 49c02f8의 결과와 같은 방향이다(상관이며 인과가 아님).") if worse else "- d_u 중앙값의 CI 하한이 0보다 큰 경우는 없다.",
            "- 그 밖의 경우는 CI가 0을 포함해 \"점추정은 컸으나 우열을 확정할 수 없음\" 또는 \"현재 자료로 뚜렷한 연관성을 확인하지 못함\"이다.", ""]
    DOC.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC)


if __name__ == "__main__":
    main()
