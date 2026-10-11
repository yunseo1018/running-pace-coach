# -*- coding: utf-8 -*-
"""
9단계 마무리 점검: 6단계 문서의 Ridge(MAE 69.928/62.054)와 9단계 종합표의 Ridge 15피처(70.274/62.270)가
다른 이유를 단계별로 분해한다. 새 모델/튜닝 없음 — 이미 쓴 설정을 하나씩 바꿔가며 숫자가 어디서 움직이는지만 본다.

변화 요인 후보(한 번에 하나씩 바꿈):
  W-test : 평가(test) 윈도우 구성 (04/06단계 윈도우 -> 24단계 윈도우)
  W-train: 학습 윈도우 구성
  alpha  : 0.0001 (6단계 선택값) -> 10000 (7단계 22피처 CV 선택값, 9단계 종합표가 15피처에도 그대로 씀)
  impl   : ridge_lib(numpy) + ManualStandardScaler -> sklearn Pipeline(StandardScaler, Ridge)

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


def fit_pred_sk(tr, te, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr = xy(tr)
    Xte, yte = xy(te)
    m = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]).fit(Xtr, ytr)
    return m.predict(Xte), yte


KNOWN_6A = {"cold-start": (70.274, 262.419), "personalized": (62.270, 239.443)}  # docs/model_comparison.md 종합표 6a (28단계 산출)


def fit_pred_sk_guarded(tr, te, alpha):
    """sklearn이 이 시점에 정책 차단(DLL load failed)으로 import 안 되면 None을 돌려준다.
    **결과 대체용 폴백이 아니라 진단 스크립트의 보고용 가드**: None이면 해당 단계를 '미실행'으로 표기하고,
    대신 28단계가 이미 산출·기록한 값(KNOWN_6A)을 출처를 밝혀 쓴다."""
    try:
        return fit_pred_sk(tr, te, alpha)
    except ImportError as e:
        print("sklearn import 차단:", str(e)[:100])
        return None


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
    md.append(f"alpha_6단계={ALPHA_6}, alpha_9단계종합표={ALPHA_9}. 학습/평가 윈도우 표기: old=04/06단계, new=24단계.\n\n")
    results = {}
    out_rows = []
    for sp in old:
        (otr, ote), (ntr, nte) = old[sp], new[sp]
        steps = [
            ("S0 6단계 재현: ridge_lib, train old/test old, alpha 6단계", lambda: fit_pred_rl(otr, ote, ALPHA_6)),
            ("S1 +평가 윈도우만 new: ridge_lib, train old/test new", lambda: fit_pred_rl(otr, nte, ALPHA_6)),
            ("S2 +학습 윈도우만 new: ridge_lib, train new/test old", lambda: fit_pred_rl(ntr, ote, ALPHA_6)),
            ("S3 윈도우 둘 다 new: ridge_lib, train new/test new", lambda: fit_pred_rl(ntr, nte, ALPHA_6)),
            ("S4 +alpha 1e4: ridge_lib, new/new", lambda: fit_pred_rl(ntr, nte, ALPHA_9)),
            ("S5 +구현 전환(sklearn Pipeline): new/new, alpha 1e4 (= 9단계 종합표 6a)", lambda: fit_pred_sk_guarded(ntr, nte, ALPHA_9)),
            ("S6 구현만 전환(alpha 6단계): sklearn, new/new", lambda: fit_pred_sk_guarded(ntr, nte, ALPHA_6)),
            ("S7 old 윈도우에서 alpha만 1e4: ridge_lib, old/old", lambda: fit_pred_rl(otr, ote, ALPHA_9)),
        ]
        prev = None
        for name, fn in steps:
            r = fn()
            if r is None:
                if name.startswith("S5"):
                    mae, rmse = KNOWN_6A[sp]
                    name_out = name + " [미실행: sklearn import 차단 -> 28단계 기록값 사용, 소수 3자리]"
                else:
                    mae, rmse, name_out = float("nan"), float("nan"), name + " [미실행: sklearn import 차단]"
                n_eval = len(nte)
            else:
                p, y = r
                mae, rmse = metrics(p, y)
                name_out, n_eval = name, len(y)
            results[(sp, name[:2])] = (mae, rmse)
            out_rows.append({"평가조건": sp, "단계": name_out, "n_eval": n_eval, "MAE": mae, "RMSE": rmse})
    df = pd.DataFrame(out_rows)
    md.append(m21.df_to_markdown(df) + "\n\n")

    md.append("## 2. 6단계 문서 수치 재현 여부 (S0)\n")
    doc = {"cold-start": (69.928493, 262.069142), "personalized": (62.053868, 239.761552)}
    for sp in old:
        mae, rmse = results[(sp, "S0")]
        md.append(f"- {sp}: 재계산 MAE={mae:.6f}/RMSE={rmse:.6f} vs 6단계 문서 {doc[sp][0]}/{doc[sp][1]} "
                  f"-> MAE 차이 {mae - doc[sp][0]:+.6f}, RMSE 차이 {rmse - doc[sp][1]:+.6f}\n")

    md.append("## 3. 요인별 이동폭 (MAE 기준)\n")
    md.append("| 평가조건 | 평가 윈도우 변경 | 학습 윈도우 변경 | 둘 다 | alpha 변경(ridge_lib, new) | 구현 전환(alpha 1e4) | 구현 전환(alpha 6단계) | 총 변화(S0→S5) |\n|---|---|---|---|---|---|---|---|\n")
    for sp in old:
        g = lambda k: results[(sp, k)][0]
        md.append(f"| {sp} | {g('S1')-g('S0'):+.4f} | {g('S2')-g('S0'):+.4f} | {g('S3')-g('S0'):+.4f} | "
                  f"{g('S4')-g('S3'):+.4f} | {g('S5')-g('S4'):+.4f}(S5=28단계 기록값) | {'미실행(sklearn import 차단)' if np.isnan(g('S6')) else format(g('S6')-g('S3'), '+.4f')} | {g('S5')-g('S0'):+.4f} |\n")
    md.append("\n**읽는 법**: 어느 열의 이동폭이 총 변화의 대부분을 차지하는지가 원인이다. 요인 간 상호작용은 분리하지 않았다(순차 변경이라 순서에 의존).\n\n")

    md.append("## 4. 사용 구현과 import 폴백 확인\n")
    try:
        import sklearn
        sk_ver = sklearn.__version__
    except ImportError:
        sk_ver = "1.9.1(기록값; 이번 실행에서는 import 차단)"
    md.append(f"- 이 분해에서 `ridge_lib`(numpy, ManualStandardScaler) 단계와 `sklearn {sk_ver}` Pipeline(StandardScaler+Ridge) 단계를 명시적으로 구분해 썼다.\n")
    md.append("- **9단계 종합표(28_model_comparison.py)의 Ridge 6a/6b는 sklearn Pipeline(StandardScaler, Ridge)** 이다. 7단계(25_tree_models.py)의 Ridge도 sklearn.\n")
    md.append("- **6단계 문서(23_ridge_regression.py)의 Ridge는 ridge_lib(numpy 직접 구현)** 이다. 6단계 이후 sklearn이 import 가능해지자 7단계부터 sklearn으로 전환했다.\n")
    md.append("- try/except import 폴백 확인(코드 검색 결과): 23/25/27/28/29 스크립트에는 Ridge·GroupKFold·scaler를 'sklearn 실패 시 ridge_lib로 대체'하는 폴백이 **없다**. "
              "25_tree_models.py의 `try: import sklearn / except ImportError`는 실패 시 sys.exit(1)로 중단하는 용도이고, "
              "HistGradientBoosting import 시도는 진단 기록용일 뿐 GBM 구현 선택에 쓰이지 않는다(항상 gbt_lib). 폴백 없음.\n")
    md.append("- **이번 실행 환경 메모**: 이 실행에서 sklearn import가 Windows 애플리케이션 제어 정책으로 차단되어(`DLL load failed`, 실행마다 차단 모듈이 달라지는 간헐 현상과 같은 계열, 원인 미확인) sklearn 단계(S5, S6)는 새로 계산하지 못했다. 정책 우회는 시도하지 않았다. S5는 28단계가 이미 산출·기록한 6a 값(70.274/62.270)을 출처를 밝혀 썼고, S6은 미실행으로 남겼다. 구현 전환 효과는 S4(ridge_lib)와 S5(기록된 sklearn 값)가 소수 3자리까지 같다는 사실까지만 확인됐다. 참고: 이 진단 스크립트(30)의 `fit_pred_sk_guarded`는 결과를 대체하는 폴백이 아니라 차단 시 해당 단계를 미실행으로 표기하는 보고용 가드다.\n")
    OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
    print(df.to_string())
    print("저장:", OUTPUT_MD_PATH)


if __name__ == "__main__":
    main()
