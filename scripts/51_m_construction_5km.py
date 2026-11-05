# -*- coding: utf-8 -*-
"""
P2-1 m 구성 확인(읽기 전용 점검): 평가 대상의 m(= 과거 비극단 5km 후보 T5의 중앙값)에 쓰인 세션을 코드와 같은 논리로 모아
 (a) 기준 기록 세션이 들어가는지, (b) 정답 세션이나 t 이후에 끝난 세션이 들어갈 여지가 없는지(모든 평가 대상에서 assert),
 (c) 기준 기록이 유일한 과거 5km라서 x = T5(기준) - m = 0이 되는 평가 대상 수를 센다. 개수만 출력(ID·T5 값 없음).
출력: docs/m_construction_5km.md
사용법(저장소 루트): python scripts/51_m_construction_5km.py
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
DOC = Path("docs") / "m_construction_5km.md"
WEEK = feas.WEEK


def quote(path, a, b):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(f"{i}: {lines[i - 1].rstrip()}" for i in range(a, b + 1))


def main():
    d, tr, te, grp, cache = m43.load_base()
    ds = m43.main_dataset(d, grp, cache)
    m43.verify_against_final(ds, tr, te)
    assert len(ds) == 2072
    ext_by_sid, t5_by_sid = cache["ext"], cache["t5"]
    cnt = {g: dict(n=0, ref_in=0, ans_in=0, late_in=0, only_ref=0, x0=0, x0_multi=0) for g in ("전체", "train", "test")}
    n_past_dist = []
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
        grpname = "train" if uid in tr else "test"
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
            ref, ans = int(valid[-1]), int(fut[0])
            past = np.flatnonzero(iscand & (e < t) & ~ext)           # 42번 89행과 같음
            m = float(np.median(t5[past]))
            if len(past) == 1:
                m = float(t5[ref])
            for gname in ("전체", grpname):
                c = cnt[gname]
                c["n"] += 1
                c["ref_in"] += int(ref in set(past))                  # (a) 기준 기록이 m 계산에 들어감
                c["ans_in"] += int(ans in set(past))                  # (b) 정답 세션이 들어감(0이어야 함)
                c["late_in"] += int((e[past] >= t).any())             # (b) t 이후에 끝난 세션이 들어감(0이어야 함)
                c["only_ref"] += int(len(past) == 1)
                x0 = (t5[ref] - m) == 0.0
                c["x0"] += int(x0)
                c["x0_multi"] += int(x0 and len(past) > 1)
            n_past_dist.append(len(past))
    assert cnt["전체"]["n"] == 2072
    assert cnt["전체"]["ref_in"] == 2072, "기준 기록 세션은 항상 m 계산에 들어간다"
    assert cnt["전체"]["ans_in"] == 0 and cnt["전체"]["late_in"] == 0, "정답 세션 또는 t 이후 종료 세션이 m에 들어감"
    rows = [{"집단": g, "평가 대상": f"{c['n']:,}", "기준 기록이 m 계산에 들어감": f"{c['ref_in']:,}", "정답 세션이 들어감": f"{c['ans_in']:,}", "t 이후에 끝난 세션이 들어감": f"{c['late_in']:,}",
             "과거 5km가 기준 기록 1개뿐(m = 기준 기록)": f"{c['only_ref']:,}", "x = 0 (정확히 같음)": f"{c['x0']:,}", "└ 과거 5km가 2개 이상인데도 x = 0": f"{c['x0_multi']:,}"} for g, c in cnt.items()]
    npast = np.array(n_past_dist)
    out = ["# m 구성 확인 (P2-1, 읽기 전용 점검)", "",
           "생성: `scripts/51_m_construction_5km.py`. 평가 대상(2,072개)의 m을 만드는 코드(`42_final_eval_5km.py`, `38_build_5km_dataset.py`)를 읽고, 같은 논리로 m에 쓰인 세션을 모아 확인했다. 개수만 싣고 ID·T5 값은 싣지 않았다. 결과 숫자·정의는 바꾸지 않았다.", "",
           "## 1. m 계산 코드 (인용)", "",
           "`42_final_eval_5km.py` (최종 평가 데이터셋을 만드는 함수 `build_dataset`):", "", "```text", quote("42_final_eval_5km.py", 76, 76), "...", quote("42_final_eval_5km.py", 84, 92), "```", "",
           "`38_build_5km_dataset.py` (train 데이터셋, 같은 논리):", "", "```text", quote("38_build_5km_dataset.py", 127, 127), "...", quote("38_build_5km_dataset.py", 144, 146), "```", "",
           "- `past`(m 계산에 쓰는 과거 5km 후보)는 `iscand & (e < t) & ~ext`: 5km 후보이면서 **종료 시각이 t보다 이르고** 극단이 아닌 세션이다(사전 선언 문서의 정의: 종료 시각 < t인 비병합·비극단 5km 후보 전체, 기준 기록 포함).",
           "- **기준 기록 세션은 m에 들어간다.** 기준 기록은 입력 창 `[t−28일, t)`에 끝난 세션이라 `e < t`를 만족하고 극단이 아니기 때문이다(`ref = valid[-1]`, 84행). 그래서 x = T5(기준) − m은 기준 기록 자신을 포함한 중앙값에서 뺀 값이다(아래 `x = 0`이 생기는 이유).",
           "- **정답 세션이 들어갈 여지는 없다.** 정답 세션은 `s >= t`(85행)로 시작 시각이 t 이후이므로 종료 시각도 t 이후이고, `e < t` 조건에서 걸러진다. t 이후에 끝난 세션도 마찬가지다.",
           "- 위 두 가지(정답 세션·t 이후 종료 세션이 `past`에 없음)를 평가 대상 2,072개 모두에서 assert로 확인했다.",
           "- 과거 5km가 1개뿐이면(`len(past) == 1`) `m = T5(기준 기록)`으로 둔다(91~92행; 사전 선언 확정 사항). 이 경우 x = 0이다.", "",
           "## 2. 확인 결과 (개수)", "", md_table(rows), "",
           f"`past` 크기: 중앙값 {int(np.median(npast))}, 25%·75% 분위수 {int(np.percentile(npast, 25))}·{int(np.percentile(npast, 75))}, 1개뿐인 평가 대상 {int((npast == 1).sum()):,}개. "
           "\"과거 5km가 기준 기록 1개뿐\"은 `docs/dataset_counts_5km.md` 2절의 \"④ 중 m = 기준 기록\"(전체 131, train 109, test 22)과 같다. "
           "x = 0은 이 경우와, 과거 5km가 2개 이상이지만 중앙값이 우연히 기준 기록과 정확히 같은 경우(중앙값이 가운데 값 하나일 때)를 합한 수다.", "",
           "해석: 이 평가 대상들은 x = 0이므로 γ·x 항이 y에 기여하지 못한다(평균 회귀 규칙 B2의 보정 항이 0이 되어 예측 = a). 수학적 결합 점검은 `docs/coupling_check_5km.md`(사양 `docs/posthoc_spec_coupling_5km.md`)에서 다룬다.", ""]
    DOC.write_text("\n".join(out), encoding="utf-8", newline="\n")
    c = cnt["전체"]
    print("saved", DOC, "| 전체: only_ref", c["only_ref"], "x0", c["x0"], "x0_multi", c["x0_multi"], "| train", cnt["train"]["x0"], "test", cnt["test"]["x0"])


if __name__ == "__main__":
    main()
