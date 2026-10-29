# -*- coding: utf-8 -*-
"""
5km 과제 train 사용자 내부 모델 비교 (Ridge / GBM / MLP × 입력 세트 3개). **train 내부 참고용 — test 미사용.**
정의: docs/predeclaration_5km_final.md(3fad402) + docs/predeclaration_5km_addendum.md(9fb0d8f).

- 대상: data/5km_dataset_train.pkl(train 평가 대상 1,720개·278명). 같은 GroupKFold(5, seed 42, ridge_lib.group_kfold_indices).
- 하이퍼파라미터 선택과 성능 추정에 같은 폴드를 쓴다(addendum 5항): 설정마다 5개 폴드의 폴드 밖(out-of-fold) 예측을 만들고,
  사용자 평균 MAE가 가장 낮은 설정을 선택해 그 설정의 폴드 밖 예측을 보고한다 -> 낙관적으로 치우칠 수 있다.
- 모델/손실은 기존 구현 그대로: Ridge(ridge_lib.RidgePipeline, 제곱오차+L2), GBM(gbt_lib, absolute_error), MLP(27번의 torch L1Loss 학습 함수).
  MLP는 27번 방식(검증으로 에폭 수 결정 -> 전체 학습 폴드로 고정 에폭 재학습)을 따르되 검증은 학습 폴드 안 사용자 20%(내부 검증)에서만 한다.
  MLP 시드는 선택 시드 42 하나(27번의 5개 최종 시드 평균은 쓰지 않음).
- 입력: 기본 11개 / 기본 + 평균 페이스 / 기본 + 성별 one-hot(3). 전처리(표준화)는 학습 폴드에서만 적합. 타깃은 y(초) 그대로.
- 지표/CI: 39번과 같은 사용자 단위 cluster bootstrap(2000회, seed 42, 같은 추출), 사용자 평균 MAE(주)·풀링 MAE·방향 일치율(밴드 ±10초).
- 기준 B2는 39번과 같은 폴드·LAD로 다시 만들고 39번 값(209.263)과 일치하는지 assert.

폴드 밖 예측값은 data/model_cv_oof_5km.pkl(git 제외)에만 저장한다. test 사용자는 쓰지 않는다. sklearn 의존 없음.
출력: docs/model_cv_results_5km.md
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import gbt_lib as gl
import ridge_lib

DATA_IN = Path("data") / "5km_dataset_train.pkl"
OOF_OUT = Path("data") / "model_cv_oof_5km.pkl"
DOC_PATH = Path("docs") / "model_cv_results_5km.md"
N_FOLDS, SEED, N_BOOT, BAND = 5, 42, 2000, 10.0


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m27 = _load("27_mlp_torch.py", "m27")
m39 = _load("39_baselines.py", "m39")
m25 = m27.m25
md_table, direction, lad_fit = m39.md_table, m39.direction, m39.lad_fit

BASE = ["n_runs", "total_km", "wk1_km", "wk2_km", "wk3_km", "wk4_km", "longest_km", "days_since_last", "mean_hr", "t5_ref", "t5_ref_minus_m"]
SETS = {"기본(11)": BASE, "기본+평균 페이스(12)": BASE + ["mean_pace"],
        "기본+성별(14)": BASE + ["g_male", "g_female", "g_unknown"]}
RIDGE_GRID = m25.RIDGE_ALPHA_GRID
GBM_GRID = m25.HGB_PARAM_GRID
MLP_GRID = m27.MLP_PARAM_GRID


def fit_predict_ridge(params, Xtr, ytr, Xva, uid_tr):
    return ridge_lib.RidgePipeline(params["alpha"]).fit(Xtr, ytr).predict(Xva), None


def fit_predict_gbm(params, Xtr, ytr, Xva, uid_tr):
    return m25.build_gbt(params, "absolute_error").fit(Xtr, ytr).predict(Xva), None


def fit_predict_mlp(params, Xtr, ytr, Xva, uid_tr):
    inner_tr, inner_va = next(ridge_lib.group_kfold_indices(uid_tr, N_FOLDS, SEED))  # 학습 폴드 안 사용자 20%를 내부 검증으로
    _, _, _, n_ep = m27.train_mlp_early_stopping(Xtr[inner_tr], ytr[inner_tr], Xtr[inner_va], ytr[inner_va],
                                                   params["n_hidden_layers"], params["width"], params["weight_decay"], m27.SELECTION_SEED)
    model, scaler = m27.train_mlp_fixed_epochs(Xtr, ytr, params["n_hidden_layers"], params["width"], params["weight_decay"], m27.SELECTION_SEED, n_ep)
    return m27.predict_mlp(model, scaler, Xva), n_ep


MODELS = {"Ridge": (RIDGE_GRID and [{"alpha": a} for a in RIDGE_GRID], fit_predict_ridge),
          "GBM": (GBM_GRID, fit_predict_gbm),
          "MLP": (MLP_GRID, fit_predict_mlp)}


def user_mean_mae(err, inv, U, cnt):
    return (np.bincount(inv, weights=err, minlength=U) / cnt).mean()


def main():
    torch.set_num_threads(4)
    ds = pd.read_pickle(DATA_IN).sort_values(["uid", "t"]).reset_index(drop=True)
    tr = {int(x) for x in Path(config.TRAIN_USERS_PATH).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    assert set(ds.uid.unique()) <= tr
    for g in ["male", "female", "unknown"]:
        ds["g_" + g] = (ds.gender == g).astype(float)
    assert (ds[["g_male", "g_female", "g_unknown"]].sum(axis=1) == 1).all()
    uid, y = ds.uid.to_numpy(), ds.y.to_numpy()
    users, inv = np.unique(uid, return_inverse=True)
    U = len(users)
    cnt = np.bincount(inv, minlength=U).astype(float)
    folds = list(ridge_lib.group_kfold_indices(uid, N_FOLDS, SEED))
    dy = direction(y)

    # ---- 기준 B2 (39번과 같은 폴드·LAD) ----
    x = ds.t5_ref_minus_m.to_numpy()
    b2 = np.zeros(len(ds))
    for trm, vam in folds:
        a, b = lad_fit(x[trm], y[trm])
        b2[vam] = a + b * x[vam]
    assert abs(user_mean_mae(np.abs(y - b2), inv, U, cnt) - 209.263) < 5e-4, "39번 B2 값과 불일치"
    preds = {"B2": b2}

    # ---- 모델 × 입력 세트 ----
    selected, results_cfg = {}, []
    t0 = time.time()
    for mname, (grid, fp) in MODELS.items():
        for sname, cols in SETS.items():
            X = ds[cols].to_numpy(float)
            best = None
            for params in grid:
                oof = np.zeros(len(ds))
                eps = []
                for trm, vam in folds:
                    assert set(uid[trm]).isdisjoint(set(uid[vam]))
                    p, ep = fp(params, X[trm], y[trm], X[vam], uid[trm])
                    oof[vam] = p
                    if ep is not None:
                        eps.append(int(ep))
                score = user_mean_mae(np.abs(y - oof), inv, U, cnt)
                results_cfg.append((mname, sname, json.dumps(params), score))
                if best is None or score < best[0]:
                    best = (score, params, oof, eps)
            selected[(mname, sname)] = best
            preds[(mname, sname)] = best[2]
            print(f"{mname} / {sname}: 선택 {best[1]} 사용자 평균 MAE {best[0]:.3f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 부트스트랩 (39번과 같은 추출) ----
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, U, size=(N_BOOT, U))

    def stats(p):
        err = np.abs(y - p)
        agree = (direction(p) == dy).astype(float)
        out = {}
        for key, v in (("mae", err), ("agree", agree)):
            s = np.bincount(inv, weights=v, minlength=U)
            um = s / cnt
            out[key] = ((um.mean(), s.sum() / cnt.sum()), (um[idx].mean(1), s[idx].sum(1) / cnt[idx].sum(1)))
        return out
    R = {k: stats(p) for k, p in preds.items()}

    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return f"[{lo:.3f}, {hi:.3f}]"

    def cell(r, key, k, fmt="{:.3f}"):
        return f"{fmt.format(r[key][0][k])} {ci(r[key][1][k])}"

    def diff(a, b, key="mae", k=0):
        return R[a][key][0][k] - R[b][key][0][k], R[a][key][1][k] - R[b][key][1][k]

    out = ["# 5km 과제 train 내부 모델 비교 (Ridge·GBM·MLP) — 참고용, test 미사용", "",
           "생성: `40_models_cv_5km.py`. 정의: `docs/predeclaration_5km_final.md`(3fad402), `docs/predeclaration_5km_addendum.md`(9fb0d8f). "
           "**이 결과는 train 사용자 내부 참고용이며 공식 판정이 아니다.** 공식 판정은 test 1회 평가로 한다. test 사용자의 정답은 계산하지 않았다. 시각은 start_time 원값(기준 미확인), 상관관계까지만 주장한다.", "",
           f"- 대상: train 평가 대상 {len(ds):,}개·{U}명, 같은 GroupKFold({N_FOLDS}, seed {SEED}), 사용자는 폴드 사이에 겹치지 않음(assert). 단위는 초.",
           "- **선택 편향**: 하이퍼파라미터 선택과 성능 추정에 같은 폴드를 쓴다. 설정마다 5개 폴드의 폴드 밖 예측을 만들고 사용자 평균 MAE가 가장 낮은 설정을 골라 그 설정의 폴드 밖 예측을 보고하므로, 아래 점수는 낙관적으로 치우칠 수 있다.",
           "- 모델·손실(기존 구현 그대로): Ridge = 제곱오차+L2(alpha 격자 10개), GBM = `gbt_lib` absolute_error(learning_rate{0.03,0.1}×max_iter{100,300}×max_depth{2,3}, min_samples_leaf 20, n_bins 32), "
           "MLP = torch `L1Loss`(Adam lr 1e-3, 배치 2048, 최대 200에폭, patience 15; hidden_layers{1,2}×width{32,64}×weight_decay{0,1e-4}). MLP는 학습 폴드 안 사용자 20% 내부 검증으로 에폭 수를 정하고 학습 폴드 전체로 재학습(시드 42 하나).",
           "- 전처리(표준화)는 학습 폴드에서만 적합. 타깃은 y(초) 그대로. 선택 기준 = 폴드 밖 사용자 평균 MAE.",
           f"- CI: 사용자 단위 cluster bootstrap({U}명씩 복원추출 {N_BOOT}회, seed {SEED}), 뽑힌 사용자의 모든 기준 시점 유지, 모든 방법에 같은 추출(쌍대), 백분위 95% CI. 방향 밴드 ±{BAND:g}초.",
           f"- 환경: numpy {np.__version__}, pandas {pd.__version__}, torch {torch.__version__}.", ""]

    out += ["## 1. 선택된 하이퍼파라미터", ""]
    rows = []
    for (mname, sname), (score, params, _o, eps) in selected.items():
        rows.append({"모델": mname, "입력 세트": sname, "선택된 설정": json.dumps(params, ensure_ascii=False),
                     "폴드별 에폭(MLP)": ",".join(map(str, eps)) if eps else "-", "폴드 밖 사용자 평균 MAE": f"{score:.3f}"})
    out += [md_table(rows), "", "격자 전체 점수는 하단 부록에 있다.", ""]

    out += ["## 2. 지표 (폴드 밖 예측, 95% CI)", ""]
    rows = []
    for k in ["B2"] + list(selected):
        r = R[k]
        nm = "B2 (기준치)" if k == "B2" else f"{k[0]} / {k[1]}"
        rows.append({"방법": nm, "사용자 평균 MAE(주)": cell(r, "mae", 0), "풀링 MAE": cell(r, "mae", 1),
                     "방향 일치율 풀링": cell(r, "agree", 1), "방향 일치율 사용자 평균": cell(r, "agree", 0)})
    out += [md_table(rows), ""]

    out += ["## 3. 쌍대 차이 (같은 기준 시점·같은 추출, 차이 = 앞 − 뒤, 음수 = 앞이 더 낮은 오차)", "",
            "### 3-1. 각 모델(기본) − B2", ""]
    rows = []
    for m in MODELS:
        d0, bs = diff((m, "기본(11)"), "B2")
        dp, bp = diff((m, "기본(11)"), "B2", k=1)
        rows.append({"비교": f"{m}(기본) − B2", "사용자 평균 MAE 차이": f"{d0:+.3f} {ci(bs)}", "풀링 MAE 차이": f"{dp:+.3f} {ci(bp)}"})
    out += [md_table(rows), "", "### 3-2. 추가 비교 − 기본 (같은 모델)", ""]
    rows = []
    for m in MODELS:
        for s2 in ["기본+평균 페이스(12)", "기본+성별(14)"]:
            d0, bs = diff((m, s2), (m, "기본(11)"))
            dp, bp = diff((m, s2), (m, "기본(11)"), k=1)
            rows.append({"비교": f"{m}: {s2} − 기본", "사용자 평균 MAE 차이": f"{d0:+.3f} {ci(bs)}", "풀링 MAE 차이": f"{dp:+.3f} {ci(bp)}"})
    out += [md_table(rows), "",
            "추가 입력 세트마다 하이퍼파라미터를 다시 선택했다(선택 편향 아래 같은 방식). CI가 0을 포함한 차이는 동등성의 증명이 아니다.", ""]

    out += ["## 4. 통과 기준 표 (train 내부 참고용 — 공식 판정 아님)", "",
            "기준: 모델(기본) − B2의 사용자 평균 MAE 차이. 통계적 통과 = 95% CI 상한 < 0, 실용적 통과 = 점추정 개선 ≥ 10초(차이 ≤ −10). 둘을 따로 본다.", ""]
    rows = []
    for m in MODELS:
        d0, bs = diff((m, "기본(11)"), "B2")
        hi = np.percentile(bs, 97.5)
        rows.append({"모델(기본)": m, "차이(모델 − B2)": f"{d0:+.3f} {ci(bs)}", "통계적 통과(CI 상한 < 0)": "예" if hi < 0 else "아니오",
                     "실용적 통과(개선 ≥ 10초)": "예" if d0 <= -10 else "아니오"})
    out += [md_table(rows), "",
            "이 표는 선택 편향이 있는 train 내부 점수이며(위 설명), 공식 판정은 test 1회 평가로 한다.", "",
            "## 5. 특성 중요도", "",
            "이 문서에서는 특성 중요도를 계산·보고하지 않았다. 이후 보고하더라도 T5(기준 기록)·T5(기준 기록)−m 관련 항목은 훈련 효과로 해석하지 않는다.", "",
            "## 6. 구현상 관찰과 한계 (사전 선언한 설정을 바꾸지 않음)", "",
            "- **MLP가 최대 에폭(200)에서 멈췄다**: 모든 입력 세트·모든 폴드에서 선택된 에폭 수가 상한 200이었다(1절 표). 학습 폴드가 약 1,400행이라 배치 2048에서는 에폭당 1회 갱신뿐이며, "
            "학습 설정(배치 2048, lr 1e-3, 최대 200에폭, patience 15)은 약 8만 행의 이전 과제에 맞춰 고정된 값이다. 사전 선언(\"기존 격자\")에 따라 설정을 바꾸지 않았으므로, MLP의 낮은 순위는 모델 구조의 한계가 아니라 이 설정에서 충분히 학습되지 않았기 때문일 수 있다(미확인). 설정을 바꾼 재실행은 탐색적 변경으로 별도 표시해야 한다.",
            "- 입력에 T5(기준 기록)와 T5(기준 기록)−m이 모두 있어 B2의 선형 구조는 Ridge 입력 안에 포함된다. 그럼에도 Ridge가 B2를 사용자 평균 MAE에서 앞서지 못한 이유는 확인하지 않았다.",
            "- 풀링 MAE와 사용자 평균 MAE의 결론이 갈리는 경우(예: Ridge·GBM(기본) − B2)가 있다. 집계 단위가 다르며 원인은 확인하지 않았다.",
            "- 학습 행 수가 약 1,400개, 사용자 약 220명(폴드당)이라 모든 CI가 넓다. 한 사용자의 여러 기준 시점은 28일 입력 창이 겹쳐 독립이 아니다.", "",
            "## 부록: 격자 전체 점수 (폴드 밖 사용자 평균 MAE)", ""]
    out += [md_table([{"모델": a, "입력 세트": b, "설정": c, "사용자 평균 MAE": f"{d:.3f}"} for a, b, c, d in results_cfg]), ""]
    DOC_PATH.write_text("\n".join(out), encoding="utf-8", newline="\n")
    pd.DataFrame({"uid": uid, "t": ds.t.to_numpy(), "y": y, **{(k if k == "B2" else f"{k[0]}|{k[1]}"): v for k, v in preds.items()}}).to_pickle(OOF_OUT)
    print("saved", DOC_PATH, OOF_OUT)


if __name__ == "__main__":
    main()
