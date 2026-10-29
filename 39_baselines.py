# -*- coding: utf-8 -*-
"""
5km 과제 기준치 B0~B2의 **train 사용자 내부** 결과 (`docs/predeclaration_5km_final.md` 11~14, 15절 개발 단계 평가).

- 입력 데이터셋: data/5km_dataset_train.pkl (38번이 만든 train 사용자 평가 대상만; test 사용자 행·정답은 없음).
- B0: ŷ = 0. B1: ŷ = m − T5(기준 기록). B2: ŷ = a + b × (T5(기준 기록) − m), a·b는 학습 폴드에서 절대오차 합 최소(LAD, scipy.optimize.linprog/HiGHS).
- 개발 단계 평가: train 사용자 GroupKFold(5, seed 42, `ridge_lib.group_kfold_indices`). 각 기준 시점은 자기 사용자가 속하지 않은 폴드로 적합한 B2로 예측(out-of-fold).
- 지표: 사용자 평균 MAE(주), 풀링 MAE, 방향 일치율(풀링·사용자 평균; 밴드 ±10초). 사용자 단위 bootstrap 2,000회(seed 42), 모든 방법에 같은 추출(쌍대), 백분위 95% CI.
- 정답 y의 분포는 개수·중앙값·25/75% 분위와 개선/변화 없음/악화 비율까지만 보고한다. **test 사용자는 계산하지 않는다.**

출력: docs/baseline_results_train.md. sklearn 의존 없음(numpy/pandas/scipy).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import linprog

import config
import ridge_lib

DATA_IN = Path("data") / "5km_dataset_train.pkl"
DOC_PATH = Path("docs") / "baseline_results_train.md"
BAND = 10.0
N_BOOT, SEED, N_FOLDS = 2000, 42, 5


def md_table(rows):
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def direction(v):
    v = np.asarray(v, float)
    return np.where(v <= -BAND, -1, np.where(v >= BAND, 1, 0))  # -1 개선, 0 변화 없음, +1 악화


def lad_fit(x, y):
    """min_a,b Σ|y - a - b x| 를 선형계획으로 푼다(HiGHS). 반환 (a, b)."""
    nobs = len(y)
    c = np.concatenate([[0.0, 0.0], np.ones(2 * nobs)])
    eye = sparse.identity(nobs, format="csr")
    A = sparse.hstack([sparse.csr_matrix(np.ones((nobs, 1))), sparse.csr_matrix(x.reshape(-1, 1)), eye, -eye], format="csr")
    res = linprog(c, A_eq=A, b_eq=y, bounds=[(None, None), (None, None)] + [(0, None)] * (2 * nobs), method="highs")
    assert res.status == 0, res.message
    return float(res.x[0]), float(res.x[1])


def main():
    ds = pd.read_pickle(DATA_IN)
    tr = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    assert set(ds.uid.unique()) <= tr, "train 사용자 외 행이 있음"
    ds = ds.sort_values(["uid", "t"]).reset_index(drop=True)
    uid = ds.uid.to_numpy()
    y = ds.y.to_numpy()
    x = ds.t5_ref_minus_m.to_numpy()
    pred = {"B0 (ŷ=0)": np.zeros(len(ds)), "B1 (ŷ=m−T5(기준))": -x}  # m − T5(ref) = −(T5(ref) − m)
    oof = np.zeros(len(ds))
    coefs = []
    for f, (trm, vam) in enumerate(ridge_lib.group_kfold_indices(uid, N_FOLDS, SEED)):
        assert set(uid[trm]).isdisjoint(set(uid[vam]))
        a, b = lad_fit(x[trm], y[trm])
        coefs.append((f + 1, int(trm.sum()), int(vam.sum()), a, b))
        oof[vam] = a + b * x[vam]
    pred["B2 (a+b×(T5(기준)−m), LAD)"] = oof
    names = list(pred)

    users, inv = np.unique(uid, return_inverse=True)
    U = len(users)
    cnt = np.bincount(inv, minlength=U).astype(float)
    dy = direction(y)

    def per_user_sum(v):
        return np.bincount(inv, weights=np.asarray(v, float), minlength=U)
    S = {}
    for nm in names:
        err = np.abs(y - pred[nm])
        agree = (direction(pred[nm]) == dy).astype(float)
        S[nm] = dict(mae=per_user_sum(err), agree=per_user_sum(agree))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, U, size=(N_BOOT, U))

    def stats(sums):
        """(사용자 평균, 풀링) 점추정과 부트스트랩 배열."""
        um = sums / cnt
        pt = (um.mean(), sums.sum() / cnt.sum())
        bs = (um[idx].mean(1), sums[idx].sum(1) / cnt[idx].sum(1))
        return pt, bs

    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return f"[{lo:.3f}, {hi:.3f}]"

    def cell(pt, bs, k, fmt="{:.3f}"):
        return f"{fmt.format(pt[k])} {ci(bs[k])}"
    res = {nm: {m: stats(S[nm][m]) for m in ("mae", "agree")} for nm in names}

    out = ["# 5km 기준치 B0~B2 — train 사용자 내부 결과 (test 미사용)", "",
           "생성: `39_baselines.py` (입력 `data/5km_dataset_train.pkl`, 구축 `38_build_5km_dataset.py`). 정의는 `docs/predeclaration_5km_final.md`(커밋 3fad402).",
           "**test 사용자의 정답 y와 T5는 계산하지 않았다.** 아래 결과는 train 사용자 평가 대상(개발 단계 평가)이며 최종 평가가 아니다. 시각은 start_time 원값(기준 미확인).", "",
           f"- 평가 단위: 기준 시점 {len(ds):,}개, 사용자 {U}명(train 사용자 중 평가 대상이 있는 사용자). 단위는 초(T5, y, MAE).",
           f"- 분할: train 사용자 GroupKFold({N_FOLDS}, seed {SEED}, `ridge_lib.group_kfold_indices`), 폴드별로 사용자가 겹치지 않음을 assert로 확인. B0·B1은 학습이 없고 B2의 a, b만 학습 폴드에서 추정해 검증 폴드에서 평가(out-of-fold).",
           f"- 신뢰구간: 사용자 단위 cluster bootstrap, 사용자를 복원추출({U}명씩) {N_BOOT}회, seed {SEED}, 뽑힌 사용자의 모든 기준 시점 유지, 모든 방법에 같은 추출(쌍대), 백분위 95% CI. "
           "사용자 평균 MAE = 사용자별 MAE의 동일 가중 평균(주 지표), 풀링 MAE = 기준 시점 전체 평균(보조).",
           f"- 방향 밴드 ±{BAND:g}초: y ≤ −10 개선, −10 < y < 10 변화 없음, y ≥ 10 악화(ŷ도 같은 규칙).",
           f"- 환경: numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__} (B2 LAD는 `scipy.optimize.linprog`, HiGHS).", ""]

    out += ["## 1. 정답 y 요약 (train 평가 대상, 분포 요약까지만)", ""]
    cls = pd.Series(dy).map({-1: "개선", 0: "변화 없음", 1: "악화"}).value_counts()
    out += [md_table([{"개수": f"{len(y):,}", "사용자": U, "중앙값(초)": f"{np.median(y):.1f}", "25% 분위(초)": f"{np.percentile(y, 25):.1f}",
                       "75% 분위(초)": f"{np.percentile(y, 75):.1f}"}]), "",
            md_table([{"방향": k, "개수": f"{int(cls.get(k, 0)):,}", "비율": f"{100 * cls.get(k, 0) / len(y):.1f}%"} for k in ["개선", "변화 없음", "악화"]]), "",
            "y = T5(정답 세션) − T5(기준 기록), 음수 = 개선. 한 사용자의 여러 기준 시점은 28일 입력 창이 겹칠 수 있어 서로 독립이 아니다.", ""]

    out += ["## 2. 기준치별 지표 (95% CI)", "",
            md_table([{"기준치": nm,
                       "사용자 평균 MAE(주)": cell(*res[nm]["mae"], 0), "풀링 MAE": cell(*res[nm]["mae"], 1),
                       "방향 일치율 풀링": cell(*res[nm]["agree"], 1, "{:.3f}"), "방향 일치율 사용자 평균": cell(*res[nm]["agree"], 0, "{:.3f}")} for nm in names]), "",
            "방향 일치율은 y와 ŷ를 같은 3범주 규칙으로 분류했을 때 같은 범주인 비율이다. B0은 항상 \"변화 없음\"이므로 일치율 = 변화 없음 범주의 비율이다. "
            "CI 구간이 겹친다고 같다는 뜻이 아니며(비교는 아래 쌍대 차이), CI가 0을 포함한 차이는 동등성의 증명이 아니다.", ""]

    out += ["## 3. 기준치 간 쌍대 차이 (같은 기준 시점·같은 bootstrap 추출)", ""]
    pairs = [(names[1], names[0]), (names[2], names[0]), (names[2], names[1])]
    rows = []
    for a, b in pairs:
        row = {"차이(앞 − 뒤)": f"{a.split(' ')[0]} − {b.split(' ')[0]}"}
        d_um = res[a]["mae"][0][0] - res[b]["mae"][0][0]
        bs_um = res[a]["mae"][1][0] - res[b]["mae"][1][0]
        d_pl = res[a]["mae"][0][1] - res[b]["mae"][0][1]
        bs_pl = res[a]["mae"][1][1] - res[b]["mae"][1][1]
        row["사용자 평균 MAE 차이(초)"] = f"{d_um:+.3f} {ci(bs_um)}"
        row["풀링 MAE 차이(초)"] = f"{d_pl:+.3f} {ci(bs_pl)}"
        row["CI 상한 < 0 (앞이 더 낮은 사용자 평균 MAE)"] = "예" if np.percentile(bs_um, 97.5) < 0 else "아니오"
        rows.append(row)
    out += [md_table(rows), "",
            "차이 = 앞 기준치의 MAE − 뒤 기준치의 MAE. 음수면 앞 기준치의 오차가 더 작다. 통계적 통과 기준(CI 상한 < 0)과 실용 기준(점추정 개선 ≥ 10초)은 모델 평가 단계에서 \"모델 vs 최고 기준치\"에 적용하며, "
            "여기서는 기준치끼리의 차이를 기술적으로만 보고한다.", ""]

    out += ["## 4. B2 계수 (학습 폴드에서 LAD로 추정)", "",
            md_table([{"폴드": f, "학습 기준 시점": ntr, "검증 기준 시점": nva, "a(초)": f"{a:.2f}", "b": f"{b:.3f}"} for f, ntr, nva, a, b in coefs]), "",
            "LAD 해는 유일하지 않을 수 있으며 HiGHS가 반환한 해를 썼다. a, b는 train 학습 폴드 자료에서만 추정했고 검증 폴드·test의 y를 쓰지 않았다. "
            "b를 효과 크기나 훈련 효과로 해석하지 않는다(T5(기준 기록) − m은 기준 기록의 상대 위치일 뿐이다).", ""]
    DOC_PATH.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC_PATH)


if __name__ == "__main__":
    main()
