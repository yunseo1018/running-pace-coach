# -*- coding: utf-8 -*-
"""
탐색적 MLP(E-MLP)의 train 사용자 내부 결과 (test 미사용). 설정은 docs/evaluation_freeze_5km.md(f1857fb, 태그 freeze-5km-v1) 1절 그대로:
기존 MLP 구조 격자 유지, batch_size 64, 에폭 상한 1,000, 조기 종료 = 학습 폴드 안 사용자 20%(seed 42) 내부 검증 풀링 MAE가 50에폭 개선 없으면 중단,
선택된 에폭 수로 학습 폴드 전체 재학습, seed 42. 입력 세트 3가지. 40번과 같은 GroupKFold(5, seed 42)·같은 지표·같은 bootstrap 추출.

**train 결과를 본 뒤 설정을 바꾼 탐색적 모델이며 주 결론에 쓰지 않는다.** 선택 편향(같은 폴드로 선택·평가)이 있어 낙관적일 수 있다.
출력: docs/model_cv_results_5km.md의 <!-- EMLP_START -->~<!-- EMLP_END --> 블록만 갱신(40번이 문서를 다시 쓰면 41번을 이어서 실행). 폴드 밖 예측값은 data/emlp_oof_5km.pkl.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import ridge_lib

import importlib.util


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m40 = _load("40_models_cv_5km.py", "m40")
m27, lad_fit, direction, md_table = m40.m27, m40.lad_fit, m40.direction, m40.md_table
DOC_PATH = Path("docs") / "model_cv_results_5km.md"
OOF_40 = Path("data") / "model_cv_oof_5km.pkl"
OOF_OUT = Path("data") / "emlp_oof_5km.pkl"
START, END = "<!-- EMLP_START -->", "<!-- EMLP_END -->"
BATCH, MAX_EP, PATIENCE, CAP = 64, 1000, 50, 1000
N_FOLDS, SEED, N_BOOT = 5, 42, 2000


def emlp_fit_predict(params, Xtr, ytr, Xva, uid_tr):
    inner_tr, inner_va = next(ridge_lib.group_kfold_indices(uid_tr, N_FOLDS, SEED))
    _, _, _, n_ep = m27.train_mlp_early_stopping(Xtr[inner_tr], ytr[inner_tr], Xtr[inner_va], ytr[inner_va],
                                                   params["n_hidden_layers"], params["width"], params["weight_decay"], m27.SELECTION_SEED,
                                                   max_epochs=MAX_EP, patience=PATIENCE, batch_size=BATCH)
    model, scaler = m27.train_mlp_fixed_epochs(Xtr, ytr, params["n_hidden_layers"], params["width"], params["weight_decay"], m27.SELECTION_SEED, n_ep,
                                                 batch_size=BATCH)
    return m27.predict_mlp(model, scaler, Xva), n_ep


def main():
    torch.set_num_threads(4)
    ds = pd.read_pickle(m40.DATA_IN).sort_values(["uid", "t"]).reset_index(drop=True)
    for g in ["male", "female", "unknown"]:
        ds["g_" + g] = (ds.gender == g).astype(float)
    uid, y = ds.uid.to_numpy(), ds.y.to_numpy()
    users, inv = np.unique(uid, return_inverse=True)
    U = len(users)
    cnt = np.bincount(inv, minlength=U).astype(float)
    folds = list(ridge_lib.group_kfold_indices(uid, N_FOLDS, SEED))
    dy = direction(y)
    x = ds.t5_ref_minus_m.to_numpy()
    b2 = np.zeros(len(ds))
    for trm, vam in folds:
        a, b = lad_fit(x[trm], y[trm])
        b2[vam] = a + b * x[vam]
    assert abs(m40.user_mean_mae(np.abs(y - b2), inv, U, cnt) - 209.263) < 5e-4
    old = pd.read_pickle(OOF_40)
    assert np.array_equal(old.uid.to_numpy(), uid) and np.allclose(old["B2"].to_numpy(), b2)
    preds = {"B2": b2, "MLP(기존)|기본(11)": old["MLP|기본(11)"].to_numpy()}

    selected, cfg_rows = {}, []
    t0 = time.time()
    for sname, cols in m40.SETS.items():
        X = ds[cols].to_numpy(float)
        best = None
        for params in m40.MLP_GRID:
            oof = np.zeros(len(ds))
            eps = []
            for trm, vam in folds:
                assert set(uid[trm]).isdisjoint(set(uid[vam]))
                p, ep = emlp_fit_predict(params, X[trm], y[trm], X[vam], uid[trm])
                oof[vam] = p
                eps.append(int(ep))
            score = m40.user_mean_mae(np.abs(y - oof), inv, U, cnt)
            cfg_rows.append((sname, json.dumps(params), score, eps))
            if best is None or score < best[0]:
                best = (score, params, oof, eps)
            print(f"E-MLP {sname} {params}: {score:.3f} epochs {eps} ({time.time() - t0:.0f}s)", flush=True)
        selected[sname] = best
        preds[("E-MLP", sname)] = best[2]

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, U, size=(N_BOOT, U))

    def stats(p):
        err = np.abs(y - p)
        agree = (direction(p) == dy).astype(float)
        o = {}
        for key, v in (("mae", err), ("agree", agree)):
            s = np.bincount(inv, weights=v, minlength=U)
            um = s / cnt
            o[key] = ((um.mean(), s.sum() / cnt.sum()), (um[idx].mean(1), s[idx].sum(1) / cnt[idx].sum(1)))
        return o
    R = {k: stats(p) for k, p in preds.items()}

    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return f"[{lo:.3f}, {hi:.3f}]"

    def cell(r, key, k):
        return f"{r[key][0][k]:.3f} {ci(r[key][1][k])}"

    def diff(a, b, k=0):
        return R[a]["mae"][0][k] - R[b]["mae"][0][k], R[a]["mae"][1][k] - R[b]["mae"][1][k]

    names = [("E-MLP", s) for s in m40.SETS]
    out = [START, "## 7. 탐색적: E-MLP (train 결과를 본 뒤 설정을 바꾼 MLP, 주 결론에 쓰지 않음)", "",
           "**탐색적 모델이다.** 위 MLP의 에폭 상한 도달을 본 뒤 `docs/evaluation_freeze_5km.md`(f1857fb, 태그 `freeze-5km-v1`) 1절에서 학습 설정만 바꿨다: "
           f"배치 {BATCH}, 에폭 상한 {MAX_EP:,}, 조기 종료 patience {PATIENCE}(학습 폴드 안 사용자 20% 내부 검증의 풀링 MAE 기준). 구조 격자·손실(L1)·lr·시드(42)·폴드·지표·bootstrap 추출은 40번과 같다. "
           "선택 편향(같은 폴드로 선택·평가)이 있어 낙관적일 수 있고, test 미사용이다. 설정은 train 결과를 본 뒤에 정했으므로 이 절의 결과로 설정을 다시 바꾸지 않는다. "
           "비교를 여러 개 하므로 우연히 CI가 0을 벗어나는 경우가 생길 수 있다.", ""]
    rows = []
    for s in m40.SETS:
        sc, params, _o, eps = selected[s]
        rows.append({"입력 세트": s, "선택된 구조": json.dumps(params, ensure_ascii=False), "폴드별 선택 에폭": ",".join(map(str, eps)),
                     f"상한 {CAP:,}에 닿은 폴드": f"{sum(e >= CAP for e in eps)}/{len(eps)}", "폴드 밖 사용자 평균 MAE": f"{sc:.3f}"})
    allep = [e for s in m40.SETS for e in selected[s][3]]
    out += ["### 7-1. 선택된 구조와 에폭", "", md_table(rows), "",
            f"선택된 구조의 폴드별 선택 에폭(최적 검증 에폭) 15개: 최소 {min(allep)}, 중앙값 {int(np.median(allep))}, 최대 {max(allep)}. "
            f"\"상한에 닿음\"은 최적 에폭이 {CAP:,}인 경우다(조기 종료 없이 끝까지 간 경우 포함).", "",
            "### 7-2. 지표 (폴드 밖 예측, 95% CI)", "",
            md_table([{"방법": nm, "사용자 평균 MAE(주)": cell(R[k], "mae", 0), "풀링 MAE": cell(R[k], "mae", 1),
                       "방향 일치율 풀링": cell(R[k], "agree", 1), "방향 일치율 사용자 평균": cell(R[k], "agree", 0)}
                      for nm, k in [("B2 (기준치)", "B2"), ("MLP(기존 설정) / 기본(11) — 참고", "MLP(기존)|기본(11)")] + [(f"E-MLP / {s}", ("E-MLP", s)) for s in m40.SETS]]), "",
            "### 7-3. 쌍대 차이 (같은 기준 시점·같은 추출, 차이 = 앞 − 뒤)", ""]
    rows = []
    for nm, a, b in [("E-MLP(기본) − B2", ("E-MLP", "기본(11)"), "B2"),
                     ("E-MLP(기본) − MLP(기존 설정, 기본)", ("E-MLP", "기본(11)"), "MLP(기존)|기본(11)"),
                     ("E-MLP: 기본+평균 페이스(12) − 기본", ("E-MLP", "기본+평균 페이스(12)"), ("E-MLP", "기본(11)")),
                     ("E-MLP: 기본+성별(14) − 기본", ("E-MLP", "기본+성별(14)"), ("E-MLP", "기본(11)"))]:
        d0, bs = diff(a, b)
        dp, bp = diff(a, b, 1)
        rows.append({"비교": nm, "사용자 평균 MAE 차이": f"{d0:+.3f} {ci(bs)}", "풀링 MAE 차이": f"{dp:+.3f} {ci(bp)}"})
    d0, bs = diff(("E-MLP", "기본(11)"), "B2")
    out += [md_table(rows), "",
            f"참고용 통과 기준(탐색적, 공식 판정 아님): E-MLP(기본) − B2의 사용자 평균 MAE 차이 {d0:+.3f}, CI 상한 {np.percentile(bs, 97.5):.3f} → 통계적 통과 "
            f"{'예' if np.percentile(bs, 97.5) < 0 else '아니오'}, 실용적 통과(개선 ≥ 10초) {'예' if d0 <= -10 else '아니오'}.", "",
            "### 7-4. 격자 전체 점수 (폴드 밖 사용자 평균 MAE)", "",
            md_table([{"입력 세트": a, "설정": b, "사용자 평균 MAE": f"{c:.3f}", "폴드별 선택 에폭": ",".join(map(str, e))} for a, b, c, e in cfg_rows]), "", END]
    block = "\n".join(out)
    doc = DOC_PATH.read_text(encoding="utf-8")
    if START in doc and END in doc:
        doc = doc[:doc.index(START)] + block + doc[doc.index(END) + len(END):]
    else:
        doc = doc.rstrip("\n") + "\n\n" + block + "\n"
    DOC_PATH.write_text(doc, encoding="utf-8", newline="\n")
    pd.DataFrame({"uid": uid, "t": ds.t.to_numpy(), **{f"E-MLP|{s}": preds[("E-MLP", s)] for s in m40.SETS}}).to_pickle(OOF_OUT)
    print("saved", DOC_PATH, OOF_OUT)


if __name__ == "__main__":
    main()
