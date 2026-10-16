# -*- coding: utf-8 -*-
"""
9단계 마무리 점검: 6단계 문서의 Ridge(MAE 69.928/62.054)와 9단계 종합표의 Ridge 15피처(70.274/62.270)가
다른 이유를 단계별로 분해한다. 새 모델/튜닝 없음 — 이미 쓴 설정을 하나씩 바꿔가며 숫자가 어디서 움직이는지만 본다.

변화 요인 후보(한 번에 하나씩 바꿈):
  W-test : 평가(test) 윈도우 구성 (04/06단계 윈도우 -> 24단계 윈도우)
  W-train: 학습 윈도우 구성
  alpha  : 0.0001 (6단계 선택값) -> 10000 (7단계 22피처 CV 선택값, 9단계 종합표가 15피처에도 그대로 씀)
  impl   : (참고) 과거 sklearn Pipeline 기록값과 ridge_lib 재현값의 비교 — sklearn은 호출하지 않는다

산출물: docs/ridge_discrepancy.md
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import config
import ridge_lib as rl

_spec = importlib.util.spec_from_file_location("m24", "24_build_xy_dataset_extended.py")
m24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m24)
_s21 = importlib.util.spec_from_file_location("m21", "21_baseline_integrated_report.py")
m21 = importlib.util.module_from_spec(_s21)
_s21.loader.exec_module(m21)

OUTPUT_MD_PATH = Path("docs") / "ridge_discrepancy.md"
ALPHA_6 = 0.0001   # 6단계 선택값 (cold/pers 둘 다, docs/ridge_results.md 3절)
ALPHA_9 = 10000.0  # 7단계/9단계 종합표 값


def xy(samples, n_feat=15):
    X = np.array([s["X"][:n_feat] for s in samples], dtype=float)
    y = np.array([s["y"] for s in samples], dtype=float)
    return X, y


def keys(samples):
    return {(s["user_id"], s["window_start_session_order"]) for s in samples}


def fit_pred_rl(tr, te, alpha):
    Xtr, ytr = xy(tr)
    Xte, yte = xy(te)
    sc = rl.ManualStandardScaler().fit(Xtr)
    beta, b0 = rl.ridge_fit(sc.transform(Xtr), ytr, alpha=alpha)[:2]
    p = rl.ridge_predict(sc.transform(Xte), beta, b0)
    return p, yte


KNOWN_6A = {"cold-start": (70.274, 262.419), "personalized": (62.270, 239.443)}  # docs/model_comparison.md 종합표 6a (과거 sklearn Pipeline으로 처음 산출한 기록값)


def metrics(p, y):
    return float(np.mean(np.abs(y - p))), float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    md = ["# Ridge 15피처 수치 차이 분해 (6단계 문서 vs 9단계 종합표)\n\n"]
    md.append("한 번에 한 요인만 바꿔 어디서 숫자가 움직이는지 본다. 새 튜닝 없음.\n\n")
    old = {"cold-start": (pd.read_pickle(config.TRAIN_DATASET_PATH), pd.read_pickle(config.TEST_DATASET_PATH)),
           "personalized": (pd.read_pickle(config.PERSONALIZED_TRAIN_DATASET_PATH),
                            pd.read_pickle(config.PERSONALIZED_TEST_DATASET_PATH))}
    new = {"cold-start": (pd.read_pickle(m24.EXT_TRAIN_PATH), pd.read_pickle(m24.EXT_TEST_PATH)),
           "personalized": (pd.read_pickle(m24.EXT_P_TRAIN_PATH), pd.read_pickle(m24.EXT_P_TEST_PATH))}

    md.append("## 0. 윈도우 구성 차이 (04/06단계 vs 24단계)\n")
    rows = []
    for sp in old:
        (otr, ote), (ntr, nte) = old[sp], new[sp]
        for nm, a, b in [("train/history", otr, ntr), ("test/holdout", ote, nte)]:
            ka, kb = keys(a), keys(b)
            rows.append({"평가조건": sp, "구간": nm, "04/06 윈도우 수": len(a), "24단계 윈도우 수": len(b),
                         "04/06에만 있음": len(ka - kb), "24단계에만 있음": len(kb - ka)})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)) + "\n\n")

    md.append("## 1. 단계별 분해 (Ridge, 15피처 = 6단계와 같은 피처)\n")
    md.append(f"alpha_6단계={ALPHA_6}, alpha_9단계종합표={ALPHA_9}. 학습/평가 윈도우 표기: old=04/06단계, new=24단계. "
              "모든 계산은 `ridge_lib`(numpy)로 이 스크립트가 직접 수행한다(sklearn 의존 없음).\n\n")
    results = {}
    out_rows = []
    for sp in old:
        (otr, ote), (ntr, nte) = old[sp], new[sp]
        steps = [
            ("S0 6단계 재현: ridge_lib, train old/test old, alpha 6단계", lambda: fit_pred_rl(otr, ote, ALPHA_6)),
            ("S1 +평가 윈도우만 new: ridge_lib, train old/test new", lambda: fit_pred_rl(otr, nte, ALPHA_6)),
            ("S2 +학습 윈도우만 new: ridge_lib, train new/test old", lambda: fit_pred_rl(ntr, ote, ALPHA_6)),
            ("S3 윈도우 둘 다 new: ridge_lib, train new/test new", lambda: fit_pred_rl(ntr, nte, ALPHA_6)),
            ("S4 +alpha 1e4: ridge_lib, new/new (= 현재 28번 종합표 6a)", lambda: fit_pred_rl(ntr, nte, ALPHA_9)),
            ("S7 old 윈도우에서 alpha만 1e4: ridge_lib, old/old", lambda: fit_pred_rl(otr, ote, ALPHA_9)),
        ]
        for name, fn in steps:
            p, y = fn()
            mae, rmse = metrics(p, y)
            results[(sp, name[:2])] = (mae, rmse)
            out_rows.append({"평가조건": sp, "단계": name, "n_eval": len(y), "MAE": mae, "RMSE": rmse})
        mae6a, rmse6a = KNOWN_6A[sp]
        results[(sp, "S5")] = (mae6a, rmse6a)
        out_rows.append({"평가조건": sp, "단계": "S5 (참고) 과거 sklearn Pipeline으로 처음 산출한 종합표 6a 기록값(소수 3자리)", "n_eval": len(nte),
                         "MAE": mae6a, "RMSE": rmse6a})
    df = pd.DataFrame(out_rows)
    md.append(m21.df_to_markdown(df) + "\n\n")
    md.append("**S6 행(sklearn으로 구현만 전환)을 제외한 이유**: 이 문서는 sklearn을 호출하지 않는다(sklearn import가 정책으로 간헐 차단되고 파이프라인이 sklearn에 의존하지 않으므로, 새로 계산하는 모든 단계를 ridge_lib로만 수행). 이전 버전에는 S6이 있었으나 sklearn import가 차단된 실행에서는 미실행이었고, S5는 과거 sklearn Pipeline으로 처음 산출해 기록한 값을 참고용으로만 남겼다.\n\n")

    md.append("## 2. 6단계 문서 수치 재현 여부 (S0)\n")
    doc = {"cold-start": (69.928493, 262.069142), "personalized": (62.053868, 239.761552)}
    for sp in old:
        mae, rmse = results[(sp, "S0")]
        md.append(f"- {sp}: 재계산 MAE={mae:.6f}/RMSE={rmse:.6f} vs 6단계 문서 {doc[sp][0]}/{doc[sp][1]} "
                  f"-> MAE 차이 {mae - doc[sp][0]:+.6f}, RMSE 차이 {rmse - doc[sp][1]:+.6f}\n")

    md.append("\n## 3. 요인별 이동폭 (MAE 기준)\n")
    md.append("| 평가조건 | 평가 윈도우 변경 | 학습 윈도우 변경 | 둘 다 | alpha 변경(ridge_lib, new) | 구현 전환 참고(S5 기록값 − S4) | 총 변화(S0→S4) |\n|---|---|---|---|---|---|---|\n")
    for sp in old:
        g = lambda k: results[(sp, k)][0]
        md.append(f"| {sp} | {g('S1')-g('S0'):+.4f} | {g('S2')-g('S0'):+.4f} | {g('S3')-g('S0'):+.4f} | "
                  f"{g('S4')-g('S3'):+.4f} | {g('S5')-g('S4'):+.4f} | {g('S4')-g('S0'):+.4f} |\n")
    md.append("\n**'총 변화' 기준 변경**: 이전 버전은 '총 변화'를 S5−S0(과거 sklearn 기록값 기준)으로 정의했으나, sklearn 호출 없이 이 스크립트가 직접 계산한 값만으로 정의하도록 S4−S0(ridge_lib 재현값 기준)으로 바꿨다. 변경 전후 값: cold-start 0.3455→0.3452, personalized 0.2161→0.2160 (차이는 S5가 소수 3자리로 반올림된 기록값이기 때문이다).\n")
    md.append("\n**읽는 법**: 어느 열의 이동폭이 총 변화의 대부분을 차지하는지가 원인이다. 요인 간 상호작용은 분리하지 않았다(순차 변경이라 순서에 의존). "
              "'구현 전환 참고'는 과거 sklearn Pipeline으로 처음 산출한 값(S5)과 ridge_lib로 재현한 값(S4)의 차이이며 소수 3자리 반올림 수준이다.\n\n")

    md.append("## 4. 사용 구현과 import 폴백 확인\n")
    md.append("- 이 스크립트의 모든 계산은 `ridge_lib`(numpy, ManualStandardScaler)로 수행한다. sklearn 의존이 없고 sklearn import 성공 여부가 문서 내용에 영향을 주지 않는다.\n")
    md.append("- 과거 이력: 9단계 종합표(28번)의 Ridge 6a/6b와 7단계(25번)의 Ridge는 처음에는 sklearn Pipeline(StandardScaler, Ridge)으로 산출했고, "
              "재현성 정리(커밋 a0bcfe8) 이후 `ridge_lib.RidgePipeline`으로 고정되어 같은 값을 재현한다. "
              "6단계 문서(23번)의 Ridge는 처음부터 ridge_lib였다.\n")
    md.append("- try/except import 폴백 확인(코드 검색 결과): 23/25/27/28/29/31/32 스크립트에는 Ridge·GroupKFold·scaler를 'sklearn 실패 시 ridge_lib로 대체'하는 폴백이 **없다**. "
              "sklearn import 진단은 `env_tools.py`가 `logs/sklearn_diagnostics.log`(git 제외)에만 기록하며 GBM 구현 선택에도 쓰이지 않는다(항상 gbt_lib).\n")
    md.append("- sklearn 로드 오류 기록: `docs/sklearn_import_issue.md`.\n")
    OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
    print(df.to_string())
    print("저장:", OUTPUT_MD_PATH)


if __name__ == "__main__":
    main()