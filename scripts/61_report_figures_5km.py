# -*- coding: utf-8 -*-
"""
보고·발표용 그림 3개 (docs/figures/). 새 계산 없음: 숫자는 저장된 결과 문서의 표에서 읽고, 같은 값이 다른 원 문서에도 있는지 assert로 확인한다(손으로 옮기지 않음).
- fig5_baseline_diff.png : 모델별 (모델 - B2) 사용자별 차이의 평균과 95% CI (평균 기준 = 주 지표; 중앙값은 그리지 않음). 출처 docs/baseline_comparison_table.md, 확인 docs/final_results_5km.md(49c02f8)
- fig6_coverage_by_condition.png : 조건별 명목 80% 범위 포함률과 95% CI (train 폴드 밖 개발 / test cold-start 탐색 나란히). 출처 docs/interval_by_condition.md, 확인 docs/b2_failure_5km.md
- fig7_sample_chain.png : 4,362 → 2,127 → 2,077 → 2,072 표본 감소 사슬. 출처 docs/baseline_comparison_table.md C절, 확인 docs/sample_reduction_5km.md
사용법(저장소 루트): python scripts/61_report_figures_5km.py   (data/ 불필요, run_all에는 넣지 않음). 같은 입력이면 PNG가 바이트까지 같다(Matplotlib 메타데이터 고정).
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "docs"
OUT = D / "figures"
BASE_DOC, FINAL_DOC, COND_DOC, B2F_DOC, SAMPLE_DOC = (D / n for n in ("baseline_comparison_table.md", "final_results_5km.md", "interval_by_condition.md", "b2_failure_5km.md", "sample_reduction_5km.md"))
META = {"Software": None}

for fam in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
else:
    sys.exit("한글 글꼴(Malgun Gothic/NanumGothic/AppleGothic)을 찾지 못했습니다. 글자가 깨지므로 중단합니다.")
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, GRAY, RED = "#2F6DB5", "#E08A1E", "#8A8A8A", "#C0392B"


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


def num(s):
    return float(s.replace(",", ""))


DIFF = re.compile(r"([+-]\d[\d,]*\.?\d*) \[(-?\d[\d,]*\.?\d*), (-?\d[\d,]*\.?\d*)\]")
COV = re.compile(r"([\d.]+)% \[([\d.]+)%, ([\d.]+)%\]")


# ---------------------------------------------------------------- fig5
def read_fig5():
    base = BASE_DOC.read_text(encoding="utf-8")
    final = FINAL_DOC.read_text(encoding="utf-8")
    out = {}
    for key, a, b in (("cold-start test", "## A.", "## B."), ("personalized", "## B.", None)):
        blk = section(section(base, a, b), "### 주 결론 3개", "### 보조")
        rows = [r for r in table_rows(blk) if r[0] != "모델"]
        got = []
        for r in rows:
            m = DIFF.match(r[3])
            got.append((r[0].split(" / ")[0], num(m.group(1)), num(m.group(2)), num(m.group(3)), r[4], int(r[7])))
        # 원 문서(49c02f8)의 (c) 주 결론 표와 같은 값인지 확인
        sec = section(final, a, b)
        c = section(sec, "### (c)", "### (d)")
        ref = {}
        for line in c.splitlines():
            m = re.match(r"\| (Ridge|GBM|MLP) \| ([+-][\d.]+) \[([-\d.]+), ([-\d.]+)\] \|", line)
            if m:
                ref[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        assert set(ref) == {"Ridge", "GBM", "MLP"} and [g[0] for g in got] == ["Ridge", "GBM", "MLP"], (key, ref, got)
        for g in got:
            assert all(abs(x - y) < 5e-4 for x, y in zip(g[1:4], ref[g[0]])), (key, g, ref[g[0]])
        out[key] = got
    return out


def fig5():
    data = read_fig5()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    lo_all = min(g[2] for d in data.values() for g in d)
    hi_all = max(g[3] for d in data.values() for g in d)
    pad = 0.12 * (hi_all - lo_all)
    for ax, (key, rows) in zip(axes, data.items()):
        for i, (name, mean, lo, hi, verdict, users) in enumerate(rows):
            excl = lo > 0 or hi < 0
            col = RED if excl else BLUE
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=col, ecolor=col, elinewidth=2.2, capsize=7, markersize=9)
            ax.text(i + 0.12, mean, f"{mean:+.1f}\n[{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=9, color=col)
        ax.axhline(0, color="black", lw=1.2)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows])
        ax.set_xlim(-0.6, len(rows) - 0.1)
        ax.set_ylim(lo_all - pad, hi_all + pad)
        ax.set_title(f"{key} (사용자 {rows[0][5]}명)", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("사용자별 (모델 MAE - B2 MAE)의 평균 (초)", fontsize=10.5)
    fig.suptitle("모델 - B2: 평균 기준(주 지표) 차이와 사용자 단위 bootstrap 95% CI", fontsize=13)
    fig.text(0.5, 0.012, "차이가 음수 = 모델이 B2보다 낮음, 0 기준선 = B2와 같음. CI가 0을 포함하면 우열 확정 불가(동등성 입증 아님); 빨강 = CI 하한 > 0(B2보다 나쁨).\n주 결론 3개(기본 입력) 기준. 출처 baseline_comparison_table.md(49c02f8 값과 일치).",
             ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "fig5_baseline_diff.png", dpi=150, metadata=META)
    plt.close(fig)
    return data


# ---------------------------------------------------------------- fig6
def read_fig6():
    txt = COND_DOC.read_text(encoding="utf-8")
    dims = []
    for head, nxt in (("## 1. (a)", "## 2. (b)"), ("## 2. (b)", "## 3. (e)"), ("## 3. (e)", "### 판독")):
        sec = section(txt, head, nxt)
        title = sec.splitlines()[0].split(". ", 1)[1]
        rows = [r for r in table_rows(sec) if r[0] != "조건"]
        items = []
        for r in rows:
            def obs(c):
                return int(num(c.split(" ")[0])), "표본 부족" in c
            tn, tl = obs(r[1])
            sn, sl = obs(r[5])
            tm, sm = COV.match(r[3]), COV.match(r[7])
            items.append(dict(cond=r[0], train=dict(n=tn, users=int(num(r[2])), cov=float(tm.group(1)), lo=float(tm.group(2)), hi=float(tm.group(3)), low=tl),
                              test=dict(n=sn, users=int(num(r[6])), cov=float(sm.group(1)), lo=float(sm.group(2)), hi=float(sm.group(3)), low=sl), width=float(r[9])))
        dims.append((title, items))
    # 원 문서(b2_failure_5km.md)의 같은 층 값과 비교
    b2f = B2F_DOC.read_text(encoding="utf-8")
    row_re = re.compile(r"^\| (train 폴드 밖|test cold-start[^|]*?) \| (.+?) \| ([\d,]+) \| ([\d,]+) \| [\d.]+ \| [\d.]+ \| [^|]+ \| ([\d.]+)% \[([\d.]+)%, ([\d.]+)%\] \|", re.M)
    for (title, items), head in zip(dims, ("## 1. (a)", "## 2. (b)", "## 5. (e)")):
        sec = section(b2f, head, "\n## ", ) if head != "## 5. (e)" else section(b2f, head, "## 6.")
        ref = {"train": [], "test": []}
        for m in row_re.finditer(sec):
            ref["train" if m.group(1).startswith("train") else "test"].append((int(num(m.group(3))), int(num(m.group(4))), float(m.group(5)), float(m.group(6)), float(m.group(7))))
        for ds in ("train", "test"):
            mine = [(it[ds]["n"], it[ds]["users"], it[ds]["cov"], it[ds]["lo"], it[ds]["hi"]) for it in items]
            assert mine == ref[ds], (title, ds, mine, ref[ds])
    for ds, total in (("train", 1720), ("test", 352)):
        for title, items in dims:
            assert sum(it[ds]["n"] for it in items) == total, (ds, title)
    assert all(abs(it["width"] - 480.0) < 1e-9 for _t, items in dims for it in items)             # 고정 폭
    return dims


def fig6():
    dims = read_fig6()
    flat = []
    for title, items in dims:
        for it in items:
            flat.append((title, it))
    n_rows = len(flat) + len(dims)
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    y = n_rows
    yt, yl = [], []
    h = 0.34
    for title, items in dims:
        y -= 1
        ax.text(1, y + 0.1, title.replace(chr(0x2212), "-"), fontsize=10.5, fontweight="bold", va="center", color="#222222")
        for it in items:
            y -= 1
            for k, (ds, col, dy, lab) in enumerate((("train", BLUE, h / 2 + 0.03, "개발"), ("test", ORANGE, -h / 2 - 0.03, "탐색"))):
                d = it[ds]
                alpha = 0.3 if d["n"] < 30 else 1.0
                ax.barh(y + dy, d["cov"], height=h, color=col, alpha=alpha * 0.85, edgecolor=col)
                ax.errorbar(d["cov"], y + dy, xerr=[[d["cov"] - d["lo"]], [d["hi"] - d["cov"]]], fmt="none", ecolor="black", elinewidth=1.3, capsize=3, alpha=max(alpha, 0.45))
                txt = f"{d['cov']:.1f}%  (관측 {d['n']:,} / 사용자 {d['users']:,}명)" + ("  표본 부족" if d["n"] < 30 else "")
                ax.text(max(d["hi"], d["cov"]) + 1.2, y + dy, txt, va="center", fontsize=8.4, color=("#777777" if d["n"] < 30 else "#222222"), alpha=1.0)
            yt.append(y)
            yl.append(it["cond"])
    ax.axvline(80, color=RED, ls="--", lw=1.4)
    ax.text(80.4, n_rows - 0.35, "명목 80%", color=RED, fontsize=9, va="center")
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=9.5)
    ax.set_xlim(0, 135)
    ax.set_xticks(range(0, 101, 20))
    ax.set_ylim(-0.7, n_rows - 0.2)
    ax.set_xlabel("명목 80% 범위 포함률 (%, 막대 = 점추정, 선 = 사용자 단위 bootstrap 95% CI)")
    ax.set_title("조건별 명목 80% 범위 포함률: train 폴드 밖(개발, 파랑) vs test cold-start(탐색·이미 열람, 주황)", fontsize=12.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, alpha=0.85, label="train 폴드 밖 (개발 결과)"), Patch(color=ORANGE, alpha=0.85, label="test cold-start (탐색 결과, 이미 열람)"),
                       Patch(color="#999999", alpha=0.3, label="관측 30건 미만 = 표본 부족(흐리게, 해석 안 함)")], loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.5, 0.012, "범위 폭은 모든 조건에서 480.0초로 같다(고정). 조건별 안내 4개 층은 train 폴드 밖 결과를 보고 고른 개발 결과이며 새 자료로 확인되지 않았다. 출처 interval_by_condition.md(= b2_failure_5km.md와 값 일치).",
             ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig6_coverage_by_condition.png", dpi=150, metadata=META)
    plt.close(fig)
    return dims


# ---------------------------------------------------------------- fig7
def read_fig7():
    base = BASE_DOC.read_text(encoding="utf-8")
    sec = section(base, "## C. 표본 감소 연결", None)
    steps = []
    for line in sec.splitlines():
        m = re.match(r"\| (S\d) → (S\d) \| (.+?) \| ([\d,]+) → ([\d,]+) \(([+-][\d,]+)\) \| ([\d,]+) → ([\d,]+) \(([+-][\d,]+)\) \| ([\d,]+) → ([\d,]+) \(([+-][\d,]+)\) \| ([+-][\d,]+) / ([+-][\d,]+) \|", line)
        if m:
            steps.append(dict(a=m.group(1), b=m.group(2), reason=m.group(3), n_from=int(num(m.group(10))), n_to=int(num(m.group(11))), d=int(num(m.group(12))),
                              past=int(num(m.group(13))), fut=int(num(m.group(14)))))
    assert len(steps) == 3
    counts = [steps[0]["n_from"]] + [s["n_to"] for s in steps]
    assert counts == [4362, 2127, 2077, 2072] and [s["d"] for s in steps] == [-2235, -50, -5]
    assert -sum(s["d"] for s in steps) == 2290 == 2235 + 50 + 5
    assert all(s["d"] == s["past"] + s["fut"] for s in steps)                                   # ④ 변화 = 과거 자격 쪽 + 미래 관측 쪽
    # 원 문서(sample_reduction_5km.md)의 '전체' 표와 같은 ④ 값
    sr = section(SAMPLE_DOC.read_text(encoding="utf-8"), "### 전체", "### train")
    ref = [int(num(m.group(1))) for m in re.finditer(r"^\| S\d [^|]+\| [\d,]+ / \d+명 \| [\d,]+ / \d+명 \| ([\d,]+) / \d+명 \|", sr, re.M)]
    assert ref == counts, (ref, counts)
    return counts, steps


def fig7():
    counts, steps = read_fig7()
    labels = ["초안 S0\n4.5~5.5km 후보\n(병합 포함, 극단 처리 없음)", "S1\n후보 길이 하한\n4.5 → 5.0km", "S2\n병합 세션을\n후보에서 제외", "S3 (확정 정의)\n극단(T5 > 5,000초)\n기준 기록 제외"]
    fig, ax = plt.subplots(figsize=(12, 6.4))
    cols = [GRAY, BLUE, BLUE, BLUE]
    ax.bar(range(4), counts, color=cols, width=0.55)
    for i, c in enumerate(counts):
        ax.text(i, c + 90, f"{c:,}", ha="center", fontsize=13, fontweight="bold")
    for i, st in enumerate(steps):
        ax.annotate("", xy=(i + 1, 5350), xytext=(i, 5350), arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
        ax.text(i + 0.5, 5120, "-%s" % format(abs(st["d"]), ","), ha="center", color=RED, fontsize=14, fontweight="bold")
        ax.text(i + 0.5, 4720, "과거 자격 쪽 %s" % format(st["past"], "+,") + chr(10) + "미래 관측 쪽 %s" % format(st["fut"], "+,"), ha="center", fontsize=8.8, color="#444444")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("평가 대상 기준 시점 수 (전체: train + test)")
    ax.set_ylim(0, 5700)
    ax.set_title("표본 감소 사슬: 초안 4,362개 → 확정 2,072개 (감소 2,290 = 2,235 + 50 + 5)", fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.012, "감소의 대부분(2,235)은 정의 변경(5.0km 미만 세션에는 5.000km 지점이 없어 T5를 만들 수 없음). '과거 자격 쪽/미래 관측 쪽' = ④ 변화를 두 단계 탈락의 변화로 나눈 값(합 = 감소). 출처 baseline_comparison_table.md C절 = sample_reduction_5km.md.",
             ha="center", fontsize=8.3, color="#444444", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig7_sample_chain.png", dpi=150, metadata=META)
    plt.close(fig)
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d5 = fig5()
    d6 = fig6()
    c7 = fig7()
    print("saved fig5_baseline_diff.png fig6_coverage_by_condition.png fig7_sample_chain.png")
    print("assert 통과: fig5 모델 3개 x 2조건의 평균·CI = 49c02f8 (c)표;", "fig6", sum(len(i) for _t, i in d6), "개 조건의 관측 수·사용자 수·포함률·CI = b2_failure_5km.md, 합 1,720/352;", "fig7 사슬", c7, "= 2,235+50+5 = sample_reduction_5km.md")


if __name__ == "__main__":
    main()
