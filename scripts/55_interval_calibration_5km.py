# -*- coding: utf-8 -*-
"""
예측 범위 보정(탐색 결과). 사양: docs/interval_calibration_spec_5km.md (커밋 76b5922, 결과 계산 전 고정).
- 보정 자료는 train만: 사용자 단위 GroupKFold(5, seed 42, ridge_lib)로 폴드마다 학습 폴드에서 B2(a, b)를 LAD로 적합하고 폴드 밖 잔차를 모아 10%·90% 분위수(q_oof)를 구한다.
- 범위 규칙: yhat + [q10_oof, q90_oof] (yhat = 시연의 B2, train 전체 LAD 적합). 기존 범위: yhat + [q10_ins, q90_ins] (train 전체 적합의 표본 내 잔차 분위수).
- test(cold-start 352개·71명, personalized 442개·250명)는 이미 열람한 자료이므로 결과는 "탐색 결과"다. "검증됐다"는 표현은 쓰지 않는다. 결과로 규칙을 바꾸지 않는다.
- ID·행 값은 싣지 않는다(집계만). 사용법(저장소 루트): python scripts/55_interval_calibration_5km.py  (run_all에는 넣지 않음)
출력: docs/interval_calibration_5km.md
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import ridge_lib  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m39 = _load("39_baselines.py", "m39")
m43 = _load("43_within_person_5km.py", "m43")
m42 = m43.m42
md_table = m42.md_table
lad_fit = m39.lad_fit
SPEC_COMMIT = "76b5922"
DOC = Path("docs") / "interval_calibration_5km.md"
PARAMS = Path("demo") / "b2_params.json"
N_FOLDS, SEED, N_BOOT = 5, 42, 2000
NOMINAL = 0.80


def oof_quantiles(tr):
    """train(1,720개)에서 폴드 밖 잔차의 10%·90% 분위수. 반환: (q10, q90, 폴드별 (a, b) 목록, 폴드 밖 잔차 배열)."""
    tr = tr.sort_values(["uid", "t"]).reset_index(drop=True)
    uid, x, y = tr.uid.to_numpy(), tr.t5_ref_minus_m.to_numpy(float), tr.y.to_numpy(float)
    res = np.full(len(tr), np.nan)
    coefs = []
    for trm, vam in ridge_lib.group_kfold_indices(uid, N_FOLDS, SEED):
        assert set(uid[trm]).isdisjoint(set(uid[vam]))
        a, b = lad_fit(x[trm], y[trm])
        res[vam] = y[vam] - (a + b * x[vam])
        coefs.append((a, b))
    assert not np.isnan(res).any()
    q10, q90 = (float(v) for v in np.percentile(res, [10, 90]))
    return q10, q90, coefs, res


def user_stats(uid, inside):
    users, inv = np.unique(uid, return_inverse=True)
    n = np.bincount(inv).astype(float)
    k = np.bincount(inv, weights=inside.astype(float))
    return users, n, k


def cond_table(name, ev, a, b, ranges, idx):
    x, y = ev.t5_ref_minus_m.to_numpy(float), ev.y.to_numpy(float)
    yhat = a + b * x
    uid = ev.uid.to_numpy()
    rows, texts = [], []
    for rname, (q10, q90) in ranges.items():
        lo, hi = yhat + q10, yhat + q90
        inside = (y >= lo) & (y <= hi)
        users, n, k = user_stats(uid, inside)
        pc = k / n
        boot = (k[idx].sum(1)) / (n[idx].sum(1))
        cl, ch = np.percentile(boot, [2.5, 97.5])
        q1, med, q3 = np.percentile(pc, [25, 50, 75])
        rows.append({"평가 조건": name, "범위": rname, "포함률(풀링)": f"{100 * inside.mean():.1f}%", "95% CI(사용자 bootstrap)": f"[{100 * cl:.1f}%, {100 * ch:.1f}%]",
                     "범위 아래 / 위로 벗어남": f"{100 * (y < lo).mean():.1f}% / {100 * (y > hi).mean():.1f}%", "평균 폭(초)": f"{q90 - q10:.1f}",
                     "사용자별 포함률 중앙값 [IQR]": f"{100 * med:.1f}% [{100 * q1:.1f}%, {100 * q3:.1f}%]", "포함률 50% 미만 사용자": f"{100 * (pc < 0.5).mean():.1f}% ({int((pc < 0.5).sum())}/{len(pc)}명)"})
        if cl <= NOMINAL <= ch:
            read = "명목 80%에서 벗어났다고 확정할 수 없음"
        elif ch < NOMINAL:
            read = "명목 80%보다 낮음"
        else:
            read = "명목 80%보다 높음"
        texts.append((name, rname, inside.mean(), cl, ch, read))
    return rows, texts, (len(y), len(users), float(np.median(n)), np.percentile(n, 25), np.percentile(n, 75), int((n == 1).sum()))


def main():
    tr = pd.read_pickle(Path("data") / "5km_dataset_train.pkl")
    assert len(tr) == 1720 and tr.uid.nunique() == 278
    q10_oof, q90_oof, coefs, res = oof_quantiles(tr)
    # 시연의 B2(train 전체 적합)와 표본 내 잔차 분위수 재계산(고정본 assert)
    x, y = tr.t5_ref_minus_m.to_numpy(float), tr.y.to_numpy(float)
    a, b = lad_fit(x, y)
    ins = y - (a + b * x)
    q10_ins, q90_ins = (float(v) for v in np.percentile(ins, [10, 90]))
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    exp10 = p.get("resid_q10_insample", p["resid_q10"])
    exp90 = p.get("resid_q90_insample", p["resid_q90"])
    assert abs(a - p["a"]) < 1e-9 and abs(b - p["b"]) < 1e-9 and abs(q10_ins - exp10) < 1e-9 and abs(q90_ins - exp90) < 1e-9, "시연 파라미터와 다름"

    d, trs, te, grp, cache = m43.load_base()
    ds = m43.main_dataset(d, grp, cache)
    m43.verify_against_final(ds, trs, te)
    cold = ds[ds.uid.isin(te)].reset_index(drop=True)
    _ptr, pers, _u = m42.personalized_split(ds)
    assert (len(cold), cold.uid.nunique(), len(pers), pers.uid.nunique()) == (352, 71, 442, 250)

    ranges = {"기존(표본 내 잔차)": (q10_ins, q90_ins), "보정(폴드 밖 잔차)": (q10_oof, q90_oof)}
    rows, texts, info = [], [], {}
    for name, ev in (("cold-start test", cold), ("personalized", pers)):
        U = ev.uid.nunique()
        idx = np.random.default_rng(SEED).integers(0, U, size=(N_BOOT, U))
        r, t, i = cond_table(name, ev, a, b, ranges, idx)
        rows += r
        texts += t
        info[name] = i
    # train 폴드 밖 sanity(구성상 약 80%)
    in_oof = float(((res >= q10_oof) & (res <= q90_oof)).mean())
    in_ins = float(((ins >= q10_ins) & (ins <= q90_ins)).mean())
    in_oof_on_ins = float(((res >= q10_ins) & (res <= q90_ins)).mean())
    fa = [c[0] for c in coefs]
    fb = [c[1] for c in coefs]
    cov_cold = next(t for t in texts if t[0] == "cold-start test" and t[1].startswith("보정"))[2]

    out = ["# 예측 범위 보정 결과 (탐색 결과, 5km 과제)", "",
           f"생성: `scripts/55_interval_calibration_5km.py`. 사양: `docs/interval_calibration_spec_5km.md`(커밋 {SPEC_COMMIT}, 결과 계산 전 고정). 범위 규칙은 **train 안에서만** 정했고(사용자 단위 GroupKFold(5, seed 42), 폴드 밖 잔차), "
           "test는 이미 열람한 자료이므로 아래 test 결과는 **탐색 결과**이며 \"검증됐다\"고 쓰지 않는다. 주 결론(49c02f8)·모델·정의·분할·ŷ 계산식은 바꾸지 않았다. 모든 평가는 \"다음 7일 안에 적격 5km 기록이 관측된 경우\"에 조건부다. ID·행 값은 싣지 않았다.", "",
           "## 1. 보정 규칙", "",
           md_table([{"항목": "보정 자료", "값": "train 평가 대상 1,720개·278명"},
                     {"항목": "폴드별 LAD 적합 a_f 범위(초)", "값": f"{min(fa):.2f} ~ {max(fa):.2f}"},
                     {"항목": "폴드별 LAD 적합 b_f 범위", "값": f"{min(fb):.4f} ~ {max(fb):.4f}"},
                     {"항목": "폴드 밖 잔차 10%·90% 분위수 (q10_oof, q90_oof)", "값": f"{q10_oof:.1f}, {q90_oof:.1f}초 (폭 {q90_oof - q10_oof:.1f}초)"},
                     {"항목": "기존: 표본 내 잔차 10%·90% 분위수 (q10_ins, q90_ins)", "값": f"{q10_ins:.1f}, {q90_ins:.1f}초 (폭 {q90_ins - q10_ins:.1f}초)"},
                     {"항목": "train 안 범위 안 비율: 보정 범위 / 기존 범위(폴드 밖 잔차 기준) / 기존 범위(표본 내 잔차 기준)", "값": f"{100 * in_oof:.1f}% / {100 * in_oof_on_ins:.1f}% / {100 * in_ins:.1f}%"}]), "",
           "시연의 ŷ = a + b·x는 train 전체 LAD 적합(a = %.3f초, b = %.4f)으로 그대로다. 폴드별 a_f·b_f는 폴드 밖 잔차에만 썼다." % (a, b), "",
           "## 2. test 결과 (탐색 결과; 기존 범위와 보정 범위 나란히)", "",
           md_table(rows), "",
           f"- 포함률(풀링) = 정답 y가 범위 안에 들어간 평가 대상의 비율. 95% CI는 사용자 단위 bootstrap(사용자 복원추출 {N_BOOT:,}회, seed {SEED}, 백분위 CI, 모든 범위·조건에 같은 추출)이다. 평균 폭은 q90 − q10으로 모든 예측에 같다.",
           "- ŷ는 두 조건 모두 시연의 B2(cold-start train 적합)로 계산했다. personalized 평가 행에는 학습(train) 사용자의 뒤쪽 기준 시점이 섞여 있어 학습 자료와 완전히 독립이 아니다.", "",
           "평가 대상 수 분포(사용자당 중앙값 [25%, 75%], 1개뿐인 사용자 수):", "",
           md_table([{"평가 조건": nm, "평가 대상": f"{i[0]:,}", "사용자": f"{i[1]:,}", "사용자당 평가 대상 중앙값 [25%, 75%]": f"{i[2]:.0f} [{i[3]:.0f}, {i[4]:.0f}]", "평가 대상이 1개뿐인 사용자": f"{i[5]:,}"} for nm, i in info.items()]), "",
           "## 3. 해석 (사양 3절의 규칙)", ""]
    for nm, rn, pooled, cl, ch, read in texts:
        out.append(f"- {nm}, {rn}: 포함률 {100 * pooled:.1f}% [{100 * cl:.1f}%, {100 * ch:.1f}%] → **{read}**.")
    out += ["",
            f"- 보정 범위는 기존 범위보다 폭이 {(q90_oof - q10_oof) - (q90_ins - q10_ins):.1f}초 넓다. 포함률은 " + ", ".join(
                f"{nm}에서 {100 * [t for t in texts if t[0] == nm and t[1].startswith('보정')][0][2]:.1f}%(기존 {100 * [t for t in texts if t[0] == nm and t[1].startswith('기존')][0][2]:.1f}%)" for nm in info)
            + "이다(점추정). 폭과 포함률의 이 교환은 구성상의 것이며 범위가 개별 예측의 정확성을 뜻하지 않는다. 개인에게 80%를 보장한다고 쓰지 않는다.",
            "- 사용자별 포함률은 평가 대상이 1개뿐인 사용자에서 0% 또는 100%가 되므로 분포의 해석에 한계가 있다(특히 personalized). test 결과는 이미 열람한 자료의 탐색 결과다.",
            f"- **시연은 이 결과와 상관없이 보정 범위를 쓴다**(사양 1·4절): \"명목 80% 범위\", \"테스트 포함률 {100 * cov_cold:.1f}%(탐색 결과), 개인에게 80%를 보장하지 않음\"(보정 범위의 cold-start test 포함률).",
            "- 이 결과로 주 결론, 모델, 정의, 분할, ŷ의 계산식을 바꾸지 않는다.", ""]
    DOC.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC)
    for r in rows:
        print(r["평가 조건"], r["범위"], r["포함률(풀링)"], r["95% CI(사용자 bootstrap)"], r["평균 폭(초)"], r["사용자별 포함률 중앙값 [IQR]"], r["포함률 50% 미만 사용자"])
    print("q_oof", q10_oof, q90_oof, "q_ins", q10_ins, q90_ins, "coef a", min(fa), max(fa), "b", min(fb), max(fb))


if __name__ == "__main__":
    main()
