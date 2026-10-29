# -*- coding: utf-8 -*-
"""
결과 그래프 4개 (docs/figures/). 숫자는 저장된 결과 문서(markdown 표)와 data/final_preds_5km.pkl에서 읽는다(손으로 옮기지 않음).
- fig1: 사용자 평균 MAE와 95% CI (docs/final_results_5km.md, 커밋 49c02f8)
- fig2: 정답 y 분포(히스토그램, 집계만; 행 단위 값·ID 없음). data/final_preds_5km.pkl의 y
- fig3: 공백 코호트(사후 분석) Δ 중앙값과 CI (docs/within_person_results_5km.md)
- fig4: 민감도 요약 (docs/sensitivity_results_5km.md)
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

OUT = Path("docs") / "figures"
FINAL = Path("docs") / "final_results_5km.md"
WITHIN = Path("docs") / "within_person_results_5km.md"
SENS = Path("docs") / "sensitivity_results_5km.md"
PREDS = Path("data") / "final_preds_5km.pkl"
BAND = 10.0

for fam in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
else:
    print("경고: 한글 글꼴을 찾지 못함 (글자가 깨질 수 있음)")
plt.rcParams["axes.unicode_minus"] = False

NUM = r"[-+]?\d[\d,]*\.?\d*"


def section(txt, start, end=None):
    i = txt.index(start)
    j = txt.index(end, i + len(start)) if end else len(txt)
    return txt[i:j]


def table_rows(block):
    rows = []
    for line in block.splitlines():
        if line.startswith("|") and not re.match(r"\|[-| ]+\|$", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def mae_with_ci(cell):
    m = re.match(rf"({NUM}) \[({NUM}), ({NUM})\]", cell)
    return float(m.group(1)), float(m.group(2).replace(",", "")), float(m.group(3).replace(",", ""))


def fig1():
    txt = FINAL.read_text(encoding="utf-8")
    names = [("B0 (ŷ=0)", "B0"), ("B1 (m−T5(기준))", "B1"), ("B2 (LAD)", "B2"), ("Ridge / 기본(11)", "Ridge"),
             ("GBM / 기본(11)", "GBM"), ("MLP / 기본(11)", "MLP(기본)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (title, a, b) in zip(axes, [("cold-start test", "## A.", "## B."), ("personalized", "## B.", None)]):
        sec = section(txt, a, b)
        sec = section(sec, "### (b)", "### (c)")
        d = {r[0]: r for r in table_rows(sec)[1:]}
        vals = [mae_with_ci(d[n][1]) for n, _ in names]
        x = np.arange(len(names))
        pt = np.array([v[0] for v in vals])
        lo = pt - np.array([v[1] for v in vals])
        hi = np.array([v[2] for v in vals]) - pt
        colors = ["#9aa0a6"] * 3 + ["#4c78a8"] * 3
        ax.bar(x, pt, color=colors)
        ax.errorbar(x, pt, yerr=[lo, hi], fmt="none", ecolor="black", capsize=4)
        for xi, v in zip(x, pt):
            ax.text(xi, 8, f"{v:.0f}", ha="center", color="white", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in names])
        ax.set_title(title)
        ax.set_ylim(0, 400)
    axes[0].set_ylabel("사용자 평균 MAE (초), 95% CI")
    fig.suptitle("5km 기록 변화 예측: 사용자 평균 MAE (49c02f8 최종 평가, 낮을수록 좋음)\n회색 = 기준치, 파랑 = 모델(기본 입력). 사용자 단위 cluster bootstrap 2000회")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_mae_ci.png", dpi=150)
    plt.close(fig)


def fig2():
    p = pd.read_pickle(PREDS)
    lim, step = 600, 20
    bins = np.arange(-lim, lim + step, step)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, key, title in zip(axes, ["cold", "pers"], ["cold-start test", "personalized 평가"]):
        y = np.asarray(p[key]["y"], float)
        out = int((np.abs(y) > lim).sum())
        ax.hist(np.clip(y, -lim, lim - 1e-9), bins=bins, color="#4c78a8")
        ax.axvspan(-BAND, BAND, color="#f58518", alpha=0.35, label=f"±{BAND:.0f}초 밴드(변화 없음)")
        ax.set_title(f"{title} (n={len(y)})")
        ax.set_xlabel("정답 y = 다음 5km 기록 - 기준 기록 (초, 음수 = 빨라짐)")
        ax.text(0.98, 0.95, f"|y|>{lim}초 {out}개는 양 끝 칸에 합산", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("기준 시점 수")
    fig.suptitle("정답 y의 분포 (집계만 표시)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_y_distribution.png", dpi=150)
    plt.close(fig)


def fig3():
    txt = WITHIN.read_text(encoding="utf-8")
    sec = section(txt, "| 공백 구간(일)", "같은 사람 안 비교")
    rows = [r for r in table_rows(sec)[1:] if not r[0].startswith("전체")]
    labels, med, lo, hi, n, small = [], [], [], [], [], []
    for r in rows:
        labels.append(r[0])
        med.append(float(r[3]))
        m = re.match(rf"\[({NUM}), ({NUM})\]", r[4])
        lo.append(float(m.group(1)))
        hi.append(float(m.group(2)))
        n.append(int(r[1]))
        small.append("표본 부족" in r[-1])
    med, lo, hi = map(np.array, (med, lo, hi))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i in range(len(labels)):
        a = 0.3 if small[i] else 1.0
        ax.errorbar(i, med[i], yerr=[[med[i] - lo[i]], [hi[i] - med[i]]], fmt="o", color="#4c78a8", ecolor="#4c78a8", capsize=5, alpha=a)
        ax.text(i + 0.08, hi[i] if hi[i] < 140 else 140, f"쌍 {n[i]}" + ("\n(참고)" if small[i] else ""), fontsize=8, alpha=a, va="bottom")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l + "일" if l != "56일 이상" else l for l in labels])
    ax.set_ylim(-270, 170)
    ax.set_xlabel("연속한 두 5km 사이의 관측된 러닝 기록의 공백")
    ax.set_ylabel("Δ 중앙값 (초) = 둘째 - 첫째, 음수 = 빨라짐")
    ax.set_title("공백 코호트 (사후 분석, 95% CI)\n흐린 점 = 쌍 30개 미만 '참고(표본 부족)'. 상관만 보며 인과 아님")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_gap_cohort.png", dpi=150)
    plt.close(fig)


def fig4():
    txt = SENS.read_text(encoding="utf-8")
    sec = section(txt, "| 민감도 | 평가", "## 민감도별 상세")
    rows = table_rows(sec)[1:]
    d = {}
    order = []
    for r in rows:
        key = r[0].split("(")[0].strip() if r[0].startswith("주 분석") else r[0]
        if key not in order:
            order.append(key)
        d[(key, r[1])] = (float(r[3]), int(r[4]))
    short = [re.sub(r"\(.*", "", k).strip() if not k.startswith("S3") else k for k in order]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(order))
    w = 0.38
    for off, cond, color in [(-w / 2, "cold-start test", "#4c78a8"), (w / 2, "personalized", "#f58518")]:
        v = [d[(k, cond)][0] for k in order]
        c = [d[(k, cond)][1] for k in order]
        ax.bar(x + off, v, w, label=cond, color=color)
        for xi, vi, ci in zip(x + off, v, c):
            ax.text(xi, vi + 4, f"{vi:.0f}\n넘음 {ci}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("S", "S", 1) for s in short], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("B2 사용자 평균 MAE (초)")
    ax.set_ylim(0, 320)
    ax.legend(loc="upper left")
    ax.set_title("민감도별 B2 MAE와 'B2를 통계적으로 넘은 모델 수(3개 중)'\n막대 위 숫자: MAE / 넘음 N. 주 분석은 참고(49c02f8), S1~S5는 사전 지정 민감도")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4):
        f()
        print("saved", f.__name__)
