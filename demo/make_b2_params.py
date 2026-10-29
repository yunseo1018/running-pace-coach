# -*- coding: utf-8 -*-
"""
demo/b2_params.json 생성 + docs/demo_interval_check.md(사후 점검) 생성. 저장소 루트에서 실행: python demo/make_b2_params.py
- a, b: cold-start 최종 평가(49c02f8)와 같은 학습 자료(train 1,720개)에서 LAD로 적합. docs/final_results_5km.md의 값(소수 2·3자리)과 일치를 assert.
- 80% 범위: train에서 B2 잔차(y - ŷ)의 10%·90% 분위수(학습 자료 안, 표본 내 잔차).
- 사후 점검: 이 범위의 cold-start test·personalized 평가(49c02f8 평가 행)에서의 실제 포함률. 범위는 점검 결과를 보고 고치지 않는다.
JSON·문서에는 집계값만 저장한다(ID·행 단위 값 없음).
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
import config  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m39 = _load("39_baselines.py", "m39")
m43 = _load("43_within_person_5km.py", "m43")
m42 = m43.m42
PARAMS = Path("demo") / "b2_params.json"
DOC = Path("docs") / "demo_interval_check.md"
BAND = 10.0


def cover(y, pred, q10, q90):
    lo, hi = pred + q10, pred + q90
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def main():
    tr = pd.read_pickle(Path("data") / "5km_dataset_train.pkl")
    assert len(tr) == 1720
    x, y = tr.t5_ref_minus_m.to_numpy(float), tr.y.to_numpy(float)
    a, b = m39.lad_fit(x, y)
    txt = (Path("docs") / "final_results_5km.md").read_text(encoding="utf-8")
    cold_sec = txt[txt.index("## A."): txt.index("## B.")]
    m = re.search(r"a = (-?\d+\.\d+)초, b = (-?\d+\.\d+)", cold_sec)
    assert round(a, 2) == float(m.group(1)) and round(b, 3) == float(m.group(2)), (a, b, m.groups())
    res = y - (a + b * x)
    q10, q90 = (float(v) for v in np.percentile(res, [10, 90]))
    params = {"model": "B2 (평균 회귀, LAD)", "source": "49c02f8 cold-start 최종 적합(train 평가 대상 1,720개)", "a": a, "b": b,
              "resid_q10": q10, "resid_q90": q90, "interval_label": "80% 범위(학습 자료 B2 잔차의 10%·90% 분위수)",
              "n_train": int(len(tr)), "band_sec": BAND}
    PARAMS.write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    # ---- 사후 점검: 49c02f8 평가 행
    d, _tr, te, grp, cache = m43.load_base()
    ds = m43.main_dataset(d, grp, cache)
    m43.verify_against_final(ds, _tr, te)
    cold = ds[ds.uid.isin(te)].reset_index(drop=True)
    _ptr, pers, _u = m42.personalized_split(ds)
    saved = pd.read_pickle(m42.PRED_OUT)
    rows = []
    for nm, ev, key in (("cold-start test", cold, "cold"), ("personalized 평가", pers, "pers")):
        pred = a + b * ev.t5_ref_minus_m.to_numpy(float)
        yy = ev.y.to_numpy(float)
        cov, width = cover(yy, pred, q10, q90)
        below = float(np.mean(yy < pred + q10))
        above = float(np.mean(yy > pred + q90))
        rows.append((nm, len(ev), ev.uid.nunique(), cov, below, above, width))
        if key == "cold":   # 저장된 49c02f8 B2 예측과 일치 확인
            s = pd.DataFrame({"uid": saved["cold"]["uid"], "t": saved["cold"]["t"], "p": saved["cold"]["preds"]["B2"]}).sort_values(["uid", "t"])
            e = pd.DataFrame({"uid": ev.uid, "t": ev.t, "p": pred}).sort_values(["uid", "t"])
            assert np.allclose(s.p.to_numpy(), e.p.to_numpy())
    lines = ["# 데모 80% 범위 사후 점검 (cold-start test · personalized)", "",
             "생성: `demo/make_b2_params.py`. **사후 점검**이다(범위는 학습 자료로만 정했고, 점검 결과를 보고 범위·모델을 바꾸지 않는다). 주 결론(49c02f8)을 바꾸지 않으며 상관 기반 예측의 불확실성 표시일 뿐 원인 분석이 아니다. ID·행 값은 싣지 않았다.", "",
             f"- B2: a = {a:.3f}초, b = {b:.4f} (cold-start 학습 자료 1,720개에서 LAD). 정답 y = 다음 5km 기록 − 기준 기록(초).",
             f"- 80% 범위 = ŷ + [{q10:.1f}, {q90:.1f}]초 (학습 자료 안 B2 잔차 y − ŷ의 10%·90% 분위수, 표본 내 잔차). 이론상 학습 자료 포함률은 약 80%.",
             "- cold-start 평가의 B2 예측은 49c02f8 저장 예측과 일치함을 assert로 확인했다. personalized 평가도 같은 파라미터(cold-start 학습 자료 기준)를 그대로 적용했다(49c02f8의 personalized B2는 자체 학습 분포에서 적합한 a·b가 다르므로 이 표와 B2 점추정이 다르다).",
             "- 주의: personalized 평가 행에는 학습 자료(train 사용자)에 속한 사용자의 뒤쪽 기준 시점이 포함되어 학습 사용자 분포와 완전히 독립이 아니다.", "",
             "| 평가 | 평가 대상 | 사용자 | 실제 포함률 | 범위 아래로 벗어남 | 위로 벗어남 | 범위 폭(초) |", "|---|---|---|---|---|---|---|"]
    for nm, n, u, c, lo, hi, w in rows:
        lines.append(f"| {nm} | {n} | {u} | {c*100:.1f}% | {lo*100:.1f}% | {hi*100:.1f}% | {w:.1f} |")
    lines += ["", "해석(상관·예측 정확도 점검): 포함률이 80%에서 벗어난 정도만 기술한다. 포함률이 80%에 가깝다고 개별 예측이 정확하다는 뜻은 아니며(폭이 크다), 80%보다 낮거나 높으면 학습 자료와 평가 자료의 분포 차이를 시사할 뿐 원인은 미확인이다. 사용자 단위 신뢰구간은 계산하지 않았다.", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(params, ensure_ascii=False))
    print("\n".join(lines[8:]))


if __name__ == "__main__":
    main()
