# -*- coding: utf-8 -*-
"""
보고용 표 정리(교수 피드백 11/9 1·2번): docs/baseline_comparison_table.md. 새 학습 없음 — 최종 평가(49c02f8)가 저장한 예측값(data/final_preds_5km.pkl, git 제외)과 정답으로 집계만 한다.
- 모델별 한 행: 사용자 평균 MAE, B2 사용자 평균 MAE, [주 지표] 사용자별 (모델 MAE − B2 MAE)의 평균과 사용자 단위 bootstrap 95% CI,
  [보조] 같은 차이의 중앙값과 CI(사용자별 오차 차이 중앙값 기준), 모델이 더 나은 사용자 비율, 사용자 수.
- bootstrap은 42번과 같다: 사용자 복원추출 2,000회, default_rng(42), 모든 방법에 같은 추출, 백분위 95% CI. 평균 열·평균 차이·CI가 docs/final_results_5km.md와 같음을 assert한다.
- 표본 감소 사슬은 docs/sample_reduction_5km.md(48번)의 '전체' 표를 읽어 합을 assert한다(새로 세지 않음).
ID·행 값은 싣지 않는다(집계만). 사용법(저장소 루트): python scripts/57_baseline_table_5km.py   (run_all에는 넣지 않음)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FINAL_DOC = ROOT / "docs" / "final_results_5km.md"
SAMPLE_DOC = ROOT / "docs" / "sample_reduction_5km.md"
PRED = ROOT / "data" / "final_preds_5km.pkl"
OUT = ROOT / "docs" / "baseline_comparison_table.md"
N_BOOT, SEED = 2000, 42
SETS = ["기본(11)", "기본+평균 페이스(12)", "기본+성별(14)"]
MODELS = ["Ridge", "GBM", "MLP"]
FINAL_COMMIT = "49c02f8"


def md_table(rows):
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def key(model, sname):
    return str((model, sname))


def pct(a, q):
    return np.percentile(a, q)


def fmt_ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def verdict(lo, hi):
    if lo > 0:
        return "B2보다 나쁨(CI 하한 > 0)"
    if hi < 0:
        return "B2보다 좋음(CI 상한 < 0)"
    return "우열 확정 불가(동등성 입증 아님)"


def parse_final(doc):
    """final_results_5km.md에서 조건(A/B)별 (b) 사용자 평균 MAE 문자열과 (c) 차이 문자열을 읽는다."""
    res = {}
    for name, a, b in (("cold", "## A.", "## B."), ("pers", "## B.", None)):
        sec = doc[doc.index(a): doc.index(b) if b else len(doc)]
        mae = {}
        for line in sec[sec.index("### (b)"): sec.index("### (c)")].splitlines():
            m = re.match(r"\| (.+?) \| ([\d.]+) \[", line)
            if m and not line.startswith("| 방법"):
                lab = m.group(1).replace(" — 탐색적", "")
                mae[lab[:2] if lab[:2] in ("B0", "B1", "B2") else lab] = float(m.group(2))
        diff = {}
        for line in sec[sec.index("### (c)"): sec.index("### (d)")].splitlines():
            m = re.match(r"\| (Ridge|GBM|MLP) \| ([+-][\d.]+) \[([-\d.]+), ([-\d.]+)\] \|", line)
            if m:
                diff[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        emlp = {}
        for line in sec[sec.index("### (e)"):].splitlines():
            m = re.match(r"\| E-MLP / ([^|]+?) − B2 \| ([+-][\d.]+) \[([-\d.]+), ([-\d.]+)\] \|", line)
            if m:
                emlp[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        res[name] = (mae, diff, emlp)
    return res


def analyze(rec, final):
    uid, y = rec["uid"], rec["y"]
    users, inv = np.unique(uid, return_inverse=True)
    U = len(users)
    cnt = np.bincount(inv, minlength=U).astype(float)
    idx = np.random.default_rng(SEED).integers(0, U, size=(N_BOOT, U))
    um = {}
    for k, p in rec["preds"].items():
        um[k] = np.bincount(inv, weights=np.abs(y - p), minlength=U) / cnt
    b2 = um["B2"]
    rows_main, rows_extra, rows_emlp, rows_base = [], [], [], []
    mae_doc, diff_doc, emlp_doc = final

    def row(label, k, kind):
        d = um[k] - b2
        mean, med = d.mean(), float(np.median(d))
        bm, bmed = d[idx].mean(1), np.median(d[idx], axis=1)
        lo, hi = pct(bm, [2.5, 97.5])
        mlo, mhi = pct(bmed, [2.5, 97.5])
        better = int((d < 0).sum())
        ties = int((d == 0).sum())
        r = {"모델": label, "사용자 평균 MAE": f"{um[k].mean():.3f}", "B2 사용자 평균 MAE": f"{b2.mean():.3f}",
             "[주] 사용자별 (모델 − B2) 평균 [95% CI]": f"{mean:+.3f} {fmt_ci(lo, hi)}", "주 판정(평균 차이의 CI)": verdict(lo, hi),
             "[보조] 사용자별 오차 차이 중앙값 기준 [95% CI]": f"{med:+.3f} {fmt_ci(mlo, mhi)}",
             "모델이 더 나은 사용자 비율": f"{100 * better / U:.1f}% ({better}/{U}명)" + (f", 동률 {ties}명" if ties else ""), "사용자 수": U}
        return r, (mean, lo, hi)

    for m in MODELS:
        for s in SETS:
            lab = f"{m} / {s}"
            r, (mean, lo, hi) = row(lab, key(m, s), "x")
            assert abs(um[key(m, s)].mean() - mae_doc[lab]) < 5e-4, (lab, um[key(m, s)].mean(), mae_doc[lab])
            if s == "기본(11)":
                dd = diff_doc[m]
                assert abs(mean - dd[0]) < 5e-4 and abs(lo - dd[1]) < 5e-4 and abs(hi - dd[2]) < 5e-4, (m, mean, lo, hi, dd)
                rows_main.append(r)
            else:
                rows_extra.append(r)
    for s in SETS:
        lab = f"E-MLP / {s}"
        r, (mean, lo, hi) = row(lab, key("E-MLP", s), "x")
        dd = emlp_doc[s]
        assert abs(um[key("E-MLP", s)].mean() - mae_doc[lab]) < 5e-4
        assert abs(mean - dd[0]) < 5e-4 and abs(lo - dd[1]) < 5e-4 and abs(hi - dd[2]) < 5e-4, (lab, mean, lo, hi, dd)
        rows_emlp.append(r)
    for nm in ("B0", "B1"):
        r, _ = row(nm, nm, "x")
        assert abs(um[nm].mean() - mae_doc[nm]) < 5e-4
        rows_base.append(r)
    assert abs(b2.mean() - mae_doc["B2"]) < 5e-4
    return U, len(y), rows_main, rows_extra, rows_emlp, rows_base, b2.mean()


def num(s):
    return int(s.replace(",", ""))


def read_chain():
    doc = SAMPLE_DOC.read_text(encoding="utf-8")
    sec = doc[doc.index("### 전체"): doc.index("### train")]
    steps = []
    for line in sec.splitlines():
        m = re.match(r"\| (S\d) ([^|]+?) \| ([\d,]+) / (\d+)명 \| ([\d,]+) / (\d+)명 \| ([\d,]+) / (\d+)명 \|", line)
        if m:
            steps.append(dict(s=m.group(1), name=m.group(2).strip(), n2=num(m.group(3)), n3=num(m.group(5)), n4=num(m.group(7)), u4=int(m.group(8))))
    assert [x["s"] for x in steps] == ["S0", "S1", "S2", "S3"]
    return steps


def chain_section():
    st = read_chain()
    assert st[0]["n4"] == 4362 and st[3]["n4"] == 2072 and st[3]["n4"] == 2072
    total = st[0]["n4"] - st[3]["n4"]
    d = [st[i]["n4"] - st[i + 1]["n4"] for i in range(3)]
    assert total == 2290 and d == [2235, 50, 5] and sum(d) == total == 2235 + 50 + 5
    rows = []
    tot3 = totfd = 0
    for i in range(1, 4):
        a, b = st[i - 1], st[i]
        assert a["n2"] == b["n2"] == 38005                       # ②(달리기 4회 이상)는 모든 단계에서 같다
        d3 = b["n3"] - a["n3"]                                    # ③ 과거 자격 충족 변화
        fda, fdb = a["n3"] - a["n4"], b["n3"] - b["n4"]           # 미래 관측 탈락
        pa, pb = a["n2"] - a["n3"], b["n2"] - b["n3"]             # 과거 자격 탈락
        d4 = b["n4"] - a["n4"]
        assert d4 == d3 - (fdb - fda) and (pb - pa) == -d3, (i, d4, d3, fda, fdb)
        rows.append({"단계": f"{a['s']} → {b['s']}", "바꾼 정의": b["name"], "과거 자격 탈락 (②→③)": f"{pa:,} → {pb:,} ({pb - pa:+,})",
                     "미래 관측 탈락 (③→④)": f"{fda:,} → {fdb:,} ({fdb - fda:+,})", "④ 평가 대상": f"{a['n4']:,} → {b['n4']:,} ({d4:+,})",
                     "④ 변화의 구성: 과거 자격 쪽 / 미래 관측 쪽": f"{d3:+,} / {-(fdb - fda):+,}"})
        tot3 += d3
        totfd += -(fdb - fda)
    assert tot3 + totfd == -total
    chain = (f"{st[0]['n4']:,}개 →(S1 길이 하한 4.5→5.0km: −{d[0]:,})→ {st[1]['n4']:,} →(S2 병합 세션 제외: −{d[1]})→ {st[2]['n4']:,} →(S3 극단 기준 기록 제외: −{d[2]})→ {st[3]['n4']:,}개. "
             f"감소 {total:,} = {d[0]:,} + {d[1]} + {d[2]} (assert 확인).")
    return chain, rows, st, (tot3, totfd)


def main():
    final = parse_final(FINAL_DOC.read_text(encoding="utf-8"))
    pred = pd.read_pickle(PRED)
    out = ["# 보고용 정리 표 1: 기준치(B2) 대비 모델 비교와 표본 감소 연결", "",
           f"생성: `scripts/57_baseline_table_5km.py`. 새 학습 없이 최종 평가({FINAL_COMMIT})가 저장한 예측값(`data/final_preds_5km.pkl`, git 제외)과 정답으로 **집계만** 했다. 결과 숫자·정의·모형·범위는 바꾸지 않았다. "
           "시각은 start_time 원값(기준 미확인)이며 상관관계까지만 주장한다. 모든 평가는 \"다음 7일 안에 적격 5km 기록이 관측된 경우\"에 조건부다. ID·행 값은 싣지 않았다.", "",
           f"- 단위: 초. 오차 = |정답 y − 예측|. 사용자별 MAE = 그 사용자의 평가 기준 시점들의 평균 절대 오차. **차이 = 사용자별 (모델 MAE − B2 MAE)**이며 음수이면 모델 쪽이 낮다.",
           f"- **주 판정은 평균 차이의 95% CI로만 낸다.** CI가 0을 포함하면 \"우열 확정 불가(동등성 입증 아님)\", CI 하한이 0보다 크면 \"B2보다 나쁨\", 상한이 0보다 작으면 \"B2보다 좋음\"이다. 중앙값·비율은 보조이며 판정에 쓰지 않는다. 중앙값은 항상 \"사용자별 오차 차이 중앙값 기준\"으로 부른다.",
           f"- CI: 사용자 단위 cluster bootstrap({N_BOOT:,}회, seed {SEED}), 사용자를 복원추출하고 모든 방법에 같은 추출(쌍대), 백분위 95% CI. 42번(`42_final_eval_5km.py`)과 같은 추출이다.",
           f"- **검증(assert)**: 모든 모델의 사용자 평균 MAE, B2 사용자 평균 MAE, 기본 입력 3개와 E-MLP 3개의 (모델 − B2) 평균·CI가 `docs/final_results_5km.md`({FINAL_COMMIT})의 값과 소수 3자리까지 같다.",
           "- \"모델이 더 나은 사용자 비율\" = 차이 < 0인 사용자 수 / 사용자 수(동률은 더 나은 쪽에 넣지 않고 따로 표시). 평가 대상이 1개뿐인 사용자도 사용자 1명으로 센다.",
           "- 평균 차이의 CI와 사용자별 오차 차이 중앙값 기준 CI가 다르게 나올 수 있다(일부 행은 중앙값 쪽 CI만 0을 벗어난다). 판정은 평균 차이에 따르고, 중앙값은 극단 사용자의 영향을 줄여 보는 보조 확인이다. 행이 많아 우연히 CI가 0을 벗어나는 행이 있을 수 있다(다중 비교 보정 없음).",
           "- 판정 요약(49c02f8 결론 범위): 현재 입력·분할·평가 조건에서 B2의 통계적·실용적 통과 기준을 만족한 모델은 없었다. 아래 보조 행의 판정은 요약에 쓰지 않는다(다중 비교).", ""]
    for name, title in (("cold", "A. cold-start test"), ("pers", "B. personalized")):
        U, n, main_rows, extra_rows, emlp_rows, base_rows, b2m = analyze(pred[name], final[name])
        out += [f"## {title} (평가 기준 시점 {n}개, 사용자 {U}명, B2 사용자 평균 MAE {b2m:.3f})", "",
                "### 주 결론 3개: 기본 입력(11) Ridge·GBM·MLP", "", md_table(main_rows), "",
                "### 보조: 추가 입력 비교 (평균 페이스(12)·성별(14) 포함)", "", md_table(extra_rows), "",
                "### 탐색적: E-MLP (주 결론에 쓰지 않음)", "", md_table(emlp_rows), "",
                "### 참고: 단순 기준치 − B2", "", md_table(base_rows), ""]
    chain, rows, st, (tot3, totfd) = chain_section()
    out += ["## C. 표본 감소 연결: 초안 4,362개 → 확정 2,072개", "",
            f"사슬: {chain}", "",
            "단계별로 \"과거 자격 탈락\"(②→③: 직전 28일 안에 기준 기록이 될 5km 후보가 없음)과 \"미래 관측 탈락\"(③→④: 다음 7일 안에 5km 후보가 없음)을 나눴다. "
            "④ 평가 대상 = ② − 과거 자격 탈락 − 미래 관측 탈락이므로, ④의 변화 = −(과거 자격 탈락의 변화) − (미래 관측 탈락의 변화)이다(assert). ②(직전 28일 달리기 4회 이상)는 모든 단계에서 38,005개로 같다. 전체(train+test) 기준 기준 시점 수이며, 출처는 `docs/sample_reduction_5km.md` 1절(`scripts/48_sample_reduction_5km.py`)이다.", "",
            md_table(rows), "",
            f"- 합계: ④ 변화 {-(tot3 + totfd):,}개 감소 = 과거 자격 쪽 {tot3:+,} + 미래 관측 쪽 {totfd:+,}. 단계마다 두 쪽이 반대로 움직이므로(길이 하한을 올리면 과거 자격을 채우는 기준 시점이 크게 줄고, 같은 하한으로 미래 관측 탈락도 줄어든다) 어느 한쪽 숫자만으로 감소를 설명하지 않는다.",
            "- 표본이 줄면 CI가 넓어지므로 비유의는 동등성의 증명이 아니다. 감소의 대부분(2,235개)은 정의 변경(길이 하한: 5.0km 미만 세션에는 \"세션 안의 5.000km 지점\"이 없어 T5를 만들 수 없음)이다. S2·S3의 감소는 각각 50개·5개로 작다.", ""]
    OUT.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", OUT.name)


if __name__ == "__main__":
    main()
