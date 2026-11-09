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
m55 = _load("scripts/55_interval_calibration_5km.py", "m55")   # 폴드 밖 잔차 분위수(보정 범위 규칙, docs/interval_calibration_spec_5km.md)
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
    # ---- 사후 점검: 49c02f8 평가 행
    d, _tr, te, grp, cache = m43.load_base()
    ds = m43.main_dataset(d, grp, cache)
    m43.verify_against_final(ds, _tr, te)
    cold = ds[ds.uid.isin(te)].reset_index(drop=True)
    _ptr, pers, _u = m42.personalized_split(ds)
    saved = pd.read_pickle(m42.PRED_OUT)
    q10o, q90o, _coefs, _res = m55.oof_quantiles(tr)          # 보정 범위 규칙(시연이 쓰는 범위): train 폴드 밖 잔차의 10%·90% 분위수
    rules = {"보정(폴드 밖 잔차, 시연에 사용)": (q10o, q90o), "기존(표본 내 잔차)": (q10, q90)}
    rows = {rn: [] for rn in rules}
    for nm, ev, key in (("cold-start test", cold, "cold"), ("personalized 평가", pers, "pers")):
        pred = a + b * ev.t5_ref_minus_m.to_numpy(float)
        yy = ev.y.to_numpy(float)
        for rn, (r10, r90) in rules.items():
            cov, width = cover(yy, pred, r10, r90)
            rows[rn].append((nm, len(ev), ev.uid.nunique(), cov, float(np.mean(yy < pred + r10)), float(np.mean(yy > pred + r90)), width))
        if key == "cold":   # 저장된 49c02f8 B2 예측과 일치 확인
            s_ = pd.DataFrame({"uid": saved["cold"]["uid"], "t": saved["cold"]["t"], "p": saved["cold"]["preds"]["B2"]}).sort_values(["uid", "t"])
            e_ = pd.DataFrame({"uid": ev.uid, "t": ev.t, "p": pred}).sort_values(["uid", "t"])
            assert np.allclose(s_.p.to_numpy(), e_.p.to_numpy())
    tr_keys = set(zip(tr.uid, tr.t))     # 분할 불일치(11/9 확인): personalized 평가 행 중 이 a·b와 범위를 만든 train 평가 대상에 그대로 들어 있는 행
    n_ov = sum((u, t) in tr_keys for u, t in zip(pers.uid, pers.t))
    n_pers = len(pers)
    main_rows = rows["보정(폴드 밖 잔차, 시연에 사용)"]
    params = {"model": "B2 (평균 회귀, LAD)", "source": "49c02f8 cold-start 최종 적합(train 평가 대상 1,720개)", "a": a, "b": b,
              "resid_q10": q10o, "resid_q90": q90o, "interval_label": "명목 80% 범위(train 사용자 단위 GroupKFold(5, seed 42) 폴드 밖 잔차의 10%·90% 분위수)",
              "resid_q10_insample": q10, "resid_q90_insample": q90,
              "n_train": int(len(tr)), "band_sec": BAND,
              "test_coverage_cold": main_rows[0][3], "n_test_cold": int(main_rows[0][1]), "test_coverage_pers": main_rows[1][3], "n_test_pers": int(main_rows[1][1])}
    PARAMS.write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    lines = ["# 데모 명목 80% 범위 점검 (cold-start test · personalized, 탐색 결과)", "",
             "생성: `demo/make_b2_params.py`. 시연의 범위 규칙은 **train 사용자 단위 GroupKFold(5, seed 42) 폴드 밖 잔차의 10%·90% 분위수**다(사양 `docs/interval_calibration_spec_5km.md`, 결과·CI는 `docs/interval_calibration_5km.md`). "
             "test는 이미 열람한 자료이므로 아래 포함률은 **탐색 결과**이며 \"검증됐다\"고 쓰지 않는다. 명목 80%는 개인에게 80%를 보장한다는 뜻이 아니다. 주 결론(49c02f8)·모델·정의·분할은 바꾸지 않았다.", "",
             f"- B2: a = {a:.3f}초, b = {b:.4f} (cold-start 학습 자료 1,720개에서 LAD). 정답 y = 다음 5km 기록 − 기준 기록(초).",
             f"- 시연 범위 = ŷ + [{q10o:.1f}, {q90o:.1f}]초 (폴드 밖 잔차 분위수). 기존 범위(참고) = ŷ + [{q10:.1f}, {q90:.1f}]초 (train 전체 적합의 표본 내 잔차 분위수; `demo/b2_params.json`의 `resid_q10_insample`·`resid_q90_insample`).",
             "- cold-start 평가의 B2 예측은 49c02f8 저장 예측과 일치함을 assert로 확인했다. personalized 평가 행에도 같은 파라미터(cold-start 학습 자료 기준)를 적용했다.",
             f"- **주의(분할 불일치): personalized 평가 행은 학습 자료와 겹쳐 독립 평가가 아니다.** {n_pers}건 중 {n_ov}건이 이 a·b와 범위를 만든 train 평가 대상 1,720건에 그대로 들어 있다(`docs/interval_by_condition.md` 4절). 그래서 personalized 포함률은 표에서 주 결과로 싣지 않는다(새로 계산하지도 않았다). 시연 화면은 cold-start 포함률만 쓴다. `demo/b2_params.json`의 `test_coverage_pers`·`n_test_pers`는 파라미터 파일에 남아 있으나 시연이 읽지 않으며 결과로 쓰지 않는다.", "",
             "| 범위 | 평가 | 평가 대상 | 사용자 | 실제 포함률 | 범위 아래로 벗어남 | 위로 벗어남 | 범위 폭(초) |", "|---|---|---|---|---|---|---|---|"]
    for rn, rr in rows.items():
        for nm, n, u, c, lo, hi, w in rr:
            if nm.startswith("personalized"):
                lines.append(f"| {rn} | {nm} | {n} | {u} | 독립 평가 아님(학습 자료와 겹침, {n_ov}/{n_pers}건 중복) | - | - | {w:.1f} |")
            else:
                lines.append(f"| {rn} | {nm} | {n} | {u} | {c*100:.1f}% | {lo*100:.1f}% | {hi*100:.1f}% | {w:.1f} |")
    lines += ["", "해석(상관·예측 정확도 점검): 포함률이 명목 80%에서 벗어난 정도만 기술한다. 포함률이 80%에 가깝다고 개별 예측이 정확하다는 뜻은 아니며(폭이 크다), 80%보다 낮거나 높으면 학습 자료와 평가 자료의 분포 차이를 시사할 뿐 원인은 미확인이다. 사용자 단위 신뢰구간은 `docs/interval_calibration_5km.md`에 있다.", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(params, ensure_ascii=False))
    print("\n".join(lines[8:]))


if __name__ == "__main__":
    main()
