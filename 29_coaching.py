# -*- coding: utf-8 -*-
"""
10월 2주차 10~11단계: 코칭 1단계(고정 템플릿, LLM/RAG 없음) + 코칭 평가 분리.

**이 데이터에는 사용자가 입력한 실제 목표(거리/페이스/데드라인)가 없다. 모든 목표는 합성 시나리오이며
실제 사용자 목표 기반 코칭 성능이 아니다.**

사전 고정(결과를 보고 바꾸지 않음):
- 현재 실력 = 윈도우의 median-5(과거만), 목표 페이스 = 현재 실력 x (1-delta), delta in {3%,5%,10%},
  데드라인 8주(민감도 4주). 목표 거리는 문구 표시용, 판정에는 쓰지 않음.
- 주차 = ceil((타깃 시작 - 마지막 입력 세션 종료)/7일) (최소 1). 주차 > 데드라인 주수인 윈도우는 제외.
- 격차 = 예측 페이스 - 해당 주차 체크포인트(초/km, 양수=뒤처짐).
- 카테고리(격차/현재페이스): <=-3% 앞서감, (-3%,+3%] 계획대로, (+3%,+10%] 조금 뒤처짐, >+10% 크게 뒤처짐.
- 예측기: median-5, 전체 이력 중앙값, 0.5*median5+0.5*전체이력중앙값, 직접구현 GBM, MLP(둘 다 CV/검증 선택 모델).
코칭 모듈은 coaching_lib.py(모델 import 없음)로 예측기와 분리되어 있다.

산출물: docs/coaching_results.md
"""
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import config
import coaching_lib as cl

_spec = importlib.util.spec_from_file_location("m28", "28_model_comparison.py")
m28 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m28)
m21, m25, m27, m24 = m28.m21, m28.m25, m28.m27, m28.m24

OUTPUT_MD_PATH = Path("docs") / "coaching_results.md"
CACHE_PATH = Path(__import__("os").environ.get("TEMP", ".")) / "coaching_pred_cache.pkl"

DELTAS = [0.03, 0.05, 0.10]
DEADLINES = [8, 4]
PREDICTORS = ["median5", "hist_median", "blend", "gbm", "mlp"]
PRED_NAMES = {"median5": "median-5", "hist_median": "전체이력중앙값", "blend": "0.5*m5+0.5*전체이력중앙값",
              "gbm": "직접구현 GBM", "mlp": "MLP"}
N_BOOT = 2000
BOOT_SEED = 42
DIST_COLS = [0, 3, 6, 9, 12]  # X의 past[i].actual_distance_km 위치(PAST_FEATURE_COLUMNS 순서 기준)


def predict_all_models(train_samples, test_samples, split):
    """28단계와 동일한 하이퍼파라미터(재탐색 없음)로 GBM/MLP를 refit하고 5개 예측기의 예측을 반환."""
    X_tr, y_tr, _ = m25.samples_to_xy(train_samples)
    X_te, y_te, _ = m25.samples_to_xy(test_samples)
    med_tr, med_te = m25.median5_from_X(X_tr), m25.median5_from_X(X_te)
    hist_te = X_te[:, 18]  # user_history_median_pace (24단계에서 이력 감사 통과)
    if split == "cold-start":
        gbm = m25.build_gbt(m28.GBM_COLD_PARAMS, "absolute_error").fit(X_tr, y_tr)
        gbm_pred = gbm.predict(X_te)
        mp, ep = m28.MLP_COLD_PARAMS, m28.MLP_COLD_EPOCHS
    else:
        gbm = m25.build_gbt(m28.GBM_PERS_PARAMS, "absolute_error")
        gbm.fit(X_tr, y_tr - med_tr)
        gbm_pred = med_te + gbm.predict(X_te)
        mp, ep = m28.MLP_PERS_PARAMS, m28.MLP_PERS_EPOCHS
    form = m28.MLP_COLD_FORM if split == "cold-start" else m28.MLP_PERS_FORM  # 27번이 저장한 선택 결과(JSON)
    resid = form == "median5_residual"
    model, scaler = m27.train_mlp_fixed_epochs(X_tr, (y_tr - med_tr) if resid else y_tr, mp["n_hidden_layers"],
                                               mp["width"], mp["weight_decay"], m28.MLP_SEED, ep)
    mlp_pred = (med_te if resid else 0.0) + m27.predict_mlp(model, scaler, X_te)
    preds = {"median5": med_te, "hist_median": hist_te, "blend": 0.5 * med_te + 0.5 * hist_te,
             "gbm": gbm_pred, "mlp": mlp_pred}
    # 28단계 보고 수치와 일치하는지 확인(재현성)
    # GBM은 고정값, MLP는 27번 선택 결과를 따르므로 docs/model_comparison.md(28번 산출)의 같은 행과 대조한다.
    import re
    doc = (Path("docs") / "model_comparison.md").read_text(encoding="utf-8")
    for k, row_name in [("gbm", "7. 직접구현 GBM(주결과)"), ("mlp", "8. MLP(주결과)")]:
        mae = float(np.mean(np.abs(y_te - preds[k])))
        m = re.search(r"\| " + re.escape(row_name) + r" \| " + split + r" \| ([0-9.]+) \|", doc)
        assert m and abs(mae - float(m.group(1))) < 0.001, (split, k, mae, m and m.group(1))
    return preds


def build_split_data(split, train_samples, test_samples):
    preds = predict_all_models(train_samples, test_samples, split)
    y = np.array([s["y"] for s in test_samples], dtype=float)
    uid = np.array([s["user_id"] for s in test_samples])
    last_end = np.array([s["last_input_session_time"] for s in test_samples], dtype=float)
    tgt_start = np.array([s["target_start_time"] for s in test_samples], dtype=float)
    assert (last_end < tgt_start).all(), "last_input_session_time < target_start_time 위반"
    weeks = np.array([cl.weeks_until_target(a, b) for a, b in zip(last_end, tgt_start)])
    ext = np.array([any(m21.is_extreme_affected({**s, "X": s["X"][:15]})) for s in test_samples])
    dist = np.array([np.mean([s["X"][c] for c in DIST_COLS]) for s in test_samples])
    return {"y": y, "uid": uid, "weeks": weeks, "ext": ext, "preds": preds, "cur": preds["median5"],
            "dist": dist, "sid": np.array([s["target_session_id"] for s in test_samples]),
            "gap_days": (tgt_start - last_end) / 86400.0}


def category_idx(pred, cur, delta, weeks, D):
    """coaching_lib와 동일한 규칙의 벡터화 버전(0=앞서감,1=계획대로,2=조금 뒤처짐,3=크게 뒤처짐)."""
    cp = cur * (1 - delta * weeks / D)
    ratio = (pred - cp) / cur
    return np.where(ratio <= cl.AHEAD_MAX, 0, np.where(ratio <= cl.ON_TRACK_MAX, 1,
                                                        np.where(ratio <= cl.SLIGHT_MAX, 2, 3)))


def user_codes(uid):
    codes, uniq = pd.factorize(uid)
    return codes, len(uniq)


def boot_rate_ci(indicator, codes, n_users, rng_seed=BOOT_SEED):
    """사용자 단위 cluster bootstrap: indicator 평균(불일치율)의 95% CI와 분포."""
    rng = np.random.default_rng(rng_seed)
    s = np.bincount(codes, weights=indicator, minlength=n_users)
    n = np.bincount(codes, minlength=n_users).astype(float)
    W = np.stack([np.bincount(rng.integers(0, n_users, n_users), minlength=n_users) for _ in range(N_BOOT)])
    return (W @ s) / (W @ n)


def boot_pair_diff(ind_a, ind_b, codes, n_users):
    """같은 사용자 리샘플로 (a의 불일치율 - b의 불일치율). 양수면 b가 불일치 적음."""
    rng = np.random.default_rng(BOOT_SEED)
    sa = np.bincount(codes, weights=ind_a, minlength=n_users)
    sb = np.bincount(codes, weights=ind_b, minlength=n_users)
    n = np.bincount(codes, minlength=n_users).astype(float)
    W = np.stack([np.bincount(rng.integers(0, n_users, n_users), minlength=n_users) for _ in range(N_BOOT)])
    return (W @ sa) / (W @ n) - (W @ sb) / (W @ n)


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    md = []
    md.append("# 코칭 1단계 + 코칭 평가 분리 (10월 2주차 10~11단계)\n\n")
    md.append("> **중요: 이 데이터에는 사용자가 입력한 실제 목표(목표 거리/페이스/데드라인)가 없다. 아래 모든 목표는 "
              "'합성 시나리오'이며, 실제 사용자 목표 기반 코칭 성능이 아니다.** 이 문서는 \"예측 오차가 코칭 판정을 "
              "얼마나 흔드는가\"만 본다. 시나리오(목표 비율 delta, 데드라인, 카테고리 임계값)는 모두 결과를 보기 전에 고정한 합성 설정이다.\n\n")
    main_df = config.load_main_sessions()
    md.append(f"입력 메인 데이터: `{config.MAIN_DATA_PATH}` (세션 {len(main_df)}건, 사용자 {main_df['user_id'].nunique()}명, "
              f"assert 통과). 평가 윈도우는 7~9단계와 동일(24단계 확장 피처 윈도우), 모델은 9단계와 같은 하이퍼파라미터로 refit"
              f"(재탐색 없음, 9단계 MAE와 일치하는지 assert로 확인).\n\n")

    md.append("## 0. 설계 (사전 고정 — 결과를 보고 바꾸지 않음)\n")
    md.append("- **합성 목표**: 현재 실력 = 윈도우의 median-5(과거 5세션만 사용), 목표 페이스 = 현재 실력 × (1−delta), "
              "delta ∈ {3%, 5%, 10%}, 데드라인 8주(민감도: 4주). 목표 거리는 문구 표시용(윈도우 과거 5세션 평균 거리)이며 판정에 쓰지 않는다.\n")
    md.append("- **주간 체크포인트**: 현재 실력에서 목표 페이스까지 주 단위 선형 보간. 타깃이 몇 주차인지는 "
              "ceil((타깃 시작 − 마지막 입력 세션 종료) / 7일)(최소 1)로 계산하고, **주차 > 데드라인 주수인 윈도우는 제외**하며 제외 비율을 보고한다.\n")
    md.append("- **격차** = 예측 페이스 − 해당 주차 체크포인트 (초/km, 양수 = 뒤처짐).\n")
    md.append("- **카테고리(격차 ÷ 현재 페이스)**: 앞서감(≤ −3%), 계획대로(−3% 초과 ~ +3%), 조금 뒤처짐(+3% 초과 ~ +10%), "
              "크게 뒤처짐(+10% 초과). **임계값은 설계값이며 튜닝하지 않았다.**\n")
    md.append("- **고정 템플릿**: 카테고리별 3개 문구(coaching_lib.py), LLM/RAG 사용 없음. 코칭 모듈(coaching_lib.py)은 모델을 import하지 않는 별도 모듈이다.\n")
    md.append("- **예측기**: median-5 / 전체 이력 중앙값 / 0.5×median-5+0.5×전체 이력 중앙값 / 직접구현 GBM / MLP. "
              "후자 둘은 CV·검증 선택 모델(cold-start: GBM=absolute_error 직접, personalized: median5+잔차, MLP는 둘 다 median5+잔차). "
              "전체 이력 중앙값·블렌드는 9단계에서 ablation을 본 뒤 추가한 사후 baseline이다.\n\n")

    train = pd.read_pickle(m24.EXT_TRAIN_PATH)
    test = pd.read_pickle(m24.EXT_TEST_PATH)
    p_train = pd.read_pickle(m24.EXT_P_TRAIN_PATH)
    p_test = pd.read_pickle(m24.EXT_P_TEST_PATH)

    if CACHE_PATH.exists():
        data = pickle.loads(CACHE_PATH.read_bytes())
        print("예측 캐시 사용:", CACHE_PATH)
    else:
        data = {"cold-start": build_split_data("cold-start", train, test),
                "personalized": build_split_data("personalized", p_train, p_test)}
        CACHE_PATH.write_bytes(pickle.dumps(data))

    # ---- 상수 규칙용: 학습/이력 구간에서 '참조 카테고리 최빈값'을 정하기 위한 정보 (평가 데이터 사용 안 함) ----
    # cold-start 허용 학습 자료 = train 사용자 윈도우(train), personalized 허용 학습 자료 = history 구간 윈도우(p_train).
    train_ref_info = {}
    for split, smp in [("cold-start", train), ("personalized", p_train)]:
        X_s, y_s, _ = m25.samples_to_xy(smp)
        le = np.array([s["last_input_session_time"] for s in smp], dtype=float)
        ts = np.array([s["target_start_time"] for s in smp], dtype=float)
        assert (le < ts).all()
        train_ref_info[split] = {"cur": m25.median5_from_X(X_s), "y": y_s,
                                 "weeks": np.array([cl.weeks_until_target(a, b) for a, b in zip(le, ts)])}

    # ---- 구현 일치 검증: 벡터화 규칙 == coaching_lib 규칙 ----
    rng = np.random.default_rng(0)
    for split, d in data.items():
        idx = rng.choice(len(d["y"]), 300, replace=False)
        for D in DEADLINES:
            for delta in DELTAS:
                ok = d["weeks"][idx] <= D
                vec = category_idx(d["preds"]["gbm"][idx], d["cur"][idx], delta, d["weeks"][idx], D)
                for j, i in enumerate(idx):
                    if ok[j]:
                        c, _ = cl.categorize(d["preds"]["gbm"][i], d["cur"][i], delta, d["weeks"][i], D)
                        assert cl.CATEGORIES.index(c) == vec[j]
    print("벡터화 규칙 == coaching_lib 규칙 검증 통과")

    summary_idx = len(md)  # 요약은 결과 계산 후 여기에 삽입
    # ---- 1. 샘플 출력 10건 ----
    md.append("## 1. 샘플 출력 10건 (합성 시나리오: cold-start, delta=5%, 데드라인 8주, 예측기=직접구현 GBM)\n")
    d = data["cold-start"]
    D, delta = 8, 0.05
    ok = np.where(d["weeks"] <= D)[0]
    cat_pred = category_idx(d["preds"]["gbm"], d["cur"], delta, d["weeks"], D)
    cat_true = category_idx(d["y"], d["cur"], delta, d["weeks"], D)
    rng = np.random.default_rng(42)
    picked = []
    pools = {c: list(rng.permutation(ok[cat_pred[ok] == c])) for c in range(4)}
    while len(picked) < 10 and any(pools.values()):
        for c in range(4):
            if pools[c] and len(picked) < 10:
                picked.append(pools[c].pop())
    rows = []
    for i in picked:
        cname, g = cl.categorize(d["preds"][("gbm")][i], d["cur"][i], delta, d["weeks"][i], D)
        msg = cl.make_message(cname, d["preds"]["gbm"][i], d["cur"][i], delta, int(d["weeks"][i]), D,
                              round(float(d["dist"][i]) * 2) / 2 or 0.5, int(d["sid"][i]))
        rows.append({"user": d["uid"][i], "주차": int(d["weeks"][i]), "현재(med5)": cl.fmt_pace(d["cur"][i]),
                     "목표": cl.fmt_pace(cl.goal_pace(d["cur"][i], delta)),
                     "체크포인트": cl.fmt_pace(cl.checkpoint(d["cur"][i], delta, d["weeks"][i], D)),
                     "예측": cl.fmt_pace(d["preds"]["gbm"][i]), "실제": cl.fmt_pace(d["y"][i]),
                     "격차(초/km)": round(g, 1), "판정(예측)": cname, "판정(실제)": cl.CATEGORIES[cat_true[i]],
                     "문구": msg})
    md.append("카테고리별로 번갈아 무작위(seed=42) 추출했다(대표성 있는 표본 아님, 형식 확인용).\n\n")
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n")

    # ---- 2. 제외 비율 ----
    md.append("## 2. 데드라인 초과 윈도우 제외 비율\n")
    rows = []
    for split, d in data.items():
        wk = d["weeks"]
        for D in DEADLINES:
            rows.append({"평가조건": split, "데드라인(주)": D, "전체 윈도우": len(wk), "제외(주차>데드라인)": int((wk > D).sum()),
                         "제외 비율": pct((wk > D).mean()), "남은 윈도우": int((wk <= D).sum()),
                         "주차 중앙값(전체)": float(np.median(wk)), "간격(일) 중앙값": float(np.median(d["gap_days"]))})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n주의: 제외는 '다음 러닝까지 간격이 데드라인보다 긴 윈도우'를 버리는 것이라, 러닝 간격이 긴 사용자/윈도우가 남은 표본에서 과소대표될 수 있다(영향은 미확인).\n\n")

    # ---- 3. 불일치율 ----
    md.append("## 3. 코칭 판정 불일치율 (실제 페이스로 판정한 카테고리 vs 예측 페이스로 판정한 카테고리) — 합성 시나리오\n")
    md.append("불일치율 = 같은 윈도우에서 두 카테고리가 다른 비율. 같은 체크포인트·같은 현재 페이스를 쓰므로 "
              "불일치는 오직 (예측−실제) 페이스 차이가 카테고리 경계를 넘나드는 정도로 결정된다. "
              "**상수 규칙** = 항상 같은 카테고리 하나만 말하는 규칙. **그 카테고리는 평가 데이터가 아니라 허용된 학습 자료에서 정한다** "
              "(코드: `train_ref_info`): cold-start는 train 사용자 윈도우, personalized는 history 구간 윈도우에서, 같은 delta·데드라인 설정으로 "
              "실제 페이스 기준 참조 카테고리의 최빈값을 구해 평가 구간에 그대로 적용한다. (이전 버전은 평가 세트에서 최빈값을 골랐으나 "
              "그것은 평가 데이터를 쓰는 것이라 이 버전에서 학습/이력 구간 기준으로 바꿔 다시 계산했다.) CI는 사용자 단위 bootstrap 95%.\n\n")
    main_rows, diff_rows, ref_rows, mism = [], [], [], {}
    fair_rows, const_cat_rows = [], []
    for split, d in data.items():
        codes_all, nu_all = user_codes(d["uid"])
        for D in DEADLINES:
            keep = d["weeks"] <= D
            for delta in DELTAS:
                ref = category_idx(d["y"], d["cur"], delta, d["weeks"], D)[keep]
                codes, nu = user_codes(d["uid"][keep])
                ref_share = np.bincount(ref, minlength=4) / len(ref)
                ref_rows.append({"평가조건": split, "데드라인": D, "delta": f"{delta:.0%}", "n": int(keep.sum()),
                                 **{f"참조:{cl.CATEGORIES[c]}": pct(ref_share[c]) for c in range(4)}})
                ti = train_ref_info[split]
                tkeep = ti["weeks"] <= D
                tref = category_idx(ti["y"], ti["cur"], delta, ti["weeks"], D)[tkeep]
                tshare = np.bincount(tref, minlength=4) / len(tref)
                const_cat = int(np.argmax(tshare))
                const_ind = (ref != const_cat).astype(float)
                const_rate = const_ind.mean()
                const_cat_rows.append({"평가조건": split, "데드라인": D, "delta": f"{delta:.0%}",
                                       "상수 규칙 카테고리(학습/이력 구간 최빈)": cl.CATEGORIES[const_cat],
                                       "학습/이력 구간 비율": pct(tshare[const_cat]), "학습/이력 윈도우 수": int(tkeep.sum()),
                                       "평가 구간에서 그 카테고리 비율": pct(ref_share[const_cat])})
                inds = {}
                for p in PREDICTORS:
                    cat = category_idx(d["preds"][p], d["cur"], delta, d["weeks"], D)[keep]
                    inds[p] = (cat != ref).astype(float)
                    bs = boot_rate_ci(inds[p], codes, nu)
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    main_rows.append({"평가조건": split, "데드라인": D, "delta": f"{delta:.0%}", "예측기": PRED_NAMES[p],
                                      "불일치율": pct(inds[p].mean()), "95%CI": f"[{pct(lo)},{pct(hi)}]",
                                      "#users": nu, "#윈도우": int(keep.sum())})
                    mism[(split, D, delta, p)] = inds[p].mean()
                bs_c = boot_rate_ci(const_ind, codes, nu)
                clo, chi = np.percentile(bs_c, [2.5, 97.5])
                main_rows.append({"평가조건": split, "데드라인": D, "delta": f"{delta:.0%}", "예측기": "상수 규칙(학습/이력 구간 최빈)",
                                  "불일치율": pct(const_rate), "95%CI": f"[{pct(clo)},{pct(chi)}]", "#users": nu, "#윈도우": int(keep.sum())})
                for p in ["gbm", "mlp"]:
                    for ref_name, ref_ind in [("상수 규칙", const_ind), ("전체이력중앙값", inds["hist_median"])]:
                        diff = boot_pair_diff(ref_ind, inds[p], codes, nu)  # 양수: 모델이 불일치 적음
                        lo, hi = np.percentile(diff, [2.5, 97.5])
                        fair_rows.append({"평가조건": split, "데드라인": D, "delta": f"{delta:.0%}",
                                          "비교": f"{ref_name} − {PRED_NAMES[p]}", "불일치율 차이(%p)": 100 * diff.mean(),
                                          "하한(%p)": 100 * lo, "상한(%p)": 100 * hi, "0포함": bool(lo <= 0 <= hi)})
                for p in ["gbm", "mlp", "blend"]:
                    diff = boot_pair_diff(inds["median5"], inds[p], codes, nu)  # 양수: p가 불일치 적음
                    lo, hi = np.percentile(diff, [2.5, 97.5])
                    diff_rows.append({"평가조건": split, "데드라인": D, "delta": f"{delta:.0%}", "비교": f"median-5 − {PRED_NAMES[p]}",
                                      "불일치율 차이(%p)": 100 * diff.mean(), "하한(%p)": 100 * lo, "상한(%p)": 100 * hi,
                                      "0포함": bool(lo <= 0 <= hi)})
    md.append("### 3-1. 참조(실제 페이스) 카테고리 분포\n")
    md.append(m21.df_to_markdown(pd.DataFrame(ref_rows)))
    md.append("\n\n### 3-2. 예측기별 불일치율\n")
    md.append(m21.df_to_markdown(pd.DataFrame(main_rows)))
    md.append("\n\n### 3-3. 공정한 비교 (주 비교): GBM/MLP vs 상수 규칙, GBM/MLP vs 전체 이력 중앙값 (사용자 단위 bootstrap 95% CI)\n")
    md.append("차이 = (비교 기준의 불일치율) − (모델의 불일치율), %p. 양수 = 모델이 판정 불일치가 더 적음.\n\n")
    md.append(m21.df_to_markdown(pd.DataFrame(fair_rows)))
    md.append("\n\n**대표 시나리오(delta 5%, 데드라인 8주)**:\n")
    for r in [r for r in fair_rows if r["delta"] == "5%" and r["데드라인"] == 8]:
        md.append(f"- {r['평가조건']}, {r['비교']}: {r['불일치율 차이(%p)']:+.2f}%p (95% CI [{r['하한(%p)']:+.2f}, {r['상한(%p)']:+.2f}], "
                  f"{'0 포함 → 통계적으로 구분 안 됨' if r['0포함'] else '0 불포함'})\n")
    md.append("\n**통계적 유의성과 실용성은 별개다.** %p 차이가 작으면 사용자가 체감하는 코칭 문구 변화는 거의 없을 수 있다. "
              "위 수치는 합성 목표 기준이므로 실제 사용자 목표에서의 효과는 **미확인**이다.\n\n")
    md.append("### 3-4. 상수 규칙이 말하는 카테고리 (학습/이력 구간에서 결정)\n")
    md.append(m21.df_to_markdown(pd.DataFrame(const_cat_rows)))
    md.append("\n\n")

    md.append("\n### 3-5. median-5 예측기의 불일치율은 구조적으로 높다 (median-5 대비 비교를 주 결과로 쓰지 않는 이유)\n")
    rows = []
    for split, d in data.items():
        for D in DEADLINES:
            keep = d["weeks"] <= D
            for delta in DELTAS:
                cat_m5 = category_idx(d["preds"]["median5"], d["cur"], delta, d["weeks"], D)[keep]
                cat_gbm = category_idx(d["preds"]["gbm"], d["cur"], delta, d["weeks"], D)[keep]
                rows.append({"평가조건": split, "데드라인": D, "delta": f"{delta:.0%}",
                             "median-5가 '계획대로'라고 말한 비율": pct((cat_m5 == 1).mean()),
                             "median-5가 '조금 뒤처짐'": pct((cat_m5 == 2).mean()),
                             "GBM이 '계획대로'라고 말한 비율": pct((cat_gbm == 1).mean())})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n현재 실력을 median-5로 정의했기 때문에 median-5 예측기는 '예측 = 현재 실력'이고, 격차는 체크포인트 이동폭(= 현재×delta×주차/데드라인, 최대 delta)뿐이라 "
              "거의 항상 '계획대로'(또는 delta=10%·후반 주차에서 '조금 뒤처짐')만 말한다. 따라서 **3-6의 'median-5 대비 불일치율 감소'는 현재 실력을 median-5로 정의한 이 시나리오의 구조 효과가 크게 "
              "반영된 값이며, 모델의 코칭 판정 능력을 보여주는 수치가 아니다(구조 효과의 크기 분리는 미확인).** 그래서 이 문서의 요약·헤드라인에는 쓰지 않고, "
              "공정한 눈금으로 3-3의 **상수 규칙(학습/이력 구간에서 정한 최빈 카테고리)**, **전체이력중앙값** 대비 비교를 쓴다.\n\n")
    md.append("### 3-6. (참고용, 구조 효과 포함 — 헤드라인 아님) median-5 대비 불일치율 차이\n")
    md.append(m21.df_to_markdown(pd.DataFrame(diff_rows)))
    md.append("\n\n")

    # ---- 4. 오차 큰 사용자 / 극단 페이스 ----
    md.append("## 4. 예측 오차가 큰 사용자·극단 페이스 윈도우에서 불일치가 커지는가\n")
    md.append(f"- **오차 큰 사용자** = 사용자별 median-5 MAE 상위 10% 사용자(평가조건별로 정함; 예측기와 무관하게 같은 사용자 집합을 쓰려고 median-5 기준으로 고정). "
              f"**극단 페이스 윈도우** = 타깃 또는 입력 5세션 중 페이스 > {config.EXTREME_PACE_THRESHOLD_SEC_PER_KM}초/km인 윈도우(9단계까지와 동일 정의). "
              f"포함/제외 모두 보고한다.\n\n")
    rows = []
    for split, d in data.items():
        err = pd.DataFrame({"uid": d["uid"], "e": np.abs(d["y"] - d["preds"]["median5"])}).groupby("uid")["e"].mean()
        top = set(err[err >= err.quantile(0.9)].index)
        is_top = np.array([u in top for u in d["uid"]])
        for D in DEADLINES:
            for delta in DELTAS:
                keep = d["weeks"] <= D
                ref = category_idx(d["y"], d["cur"], delta, d["weeks"], D)
                for gname, gmask in [("오차 상위10% 사용자", is_top), ("나머지 사용자", ~is_top),
                                     ("극단 페이스 윈도우(포함)", d["ext"]), ("극단 제외", ~d["ext"])]:
                    m = keep & gmask
                    row = {"평가조건": split, "데드라인": D, "delta": f"{delta:.0%}", "그룹": gname, "#윈도우": int(m.sum()),
                           "#users": int(len(set(d["uid"][m])))}
                    for p in PREDICTORS:
                        cat = category_idx(d["preds"][p], d["cur"], delta, d["weeks"], D)
                        row[PRED_NAMES[p]] = pct((cat[m] != ref[m]).mean()) if m.sum() else "-"
                    rows.append(row)
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n불일치율이 그룹 간 크게 다르다면 그것은 오차 큰 사용자/극단 윈도우에 불일치가 몰려 있다는 관찰이다. 원인(예: 극단 페이스가 GPS 이상치인지 실제 휴식 포함인지)은 이 분석에서 검증하지 않았다(**추정/미확인**).\n\n")

    # ---- 5. 판정이 예측 오차에 지배되는가 ----
    md.append("## 5. 체크포인트 변화폭 vs 예측 오차: 판정이 예측 오차에 지배되는가\n")
    rows = []
    for split, d in data.items():
        cur_med = float(np.median(d["cur"]))
        mae_m5 = float(np.mean(np.abs(d["y"] - d["preds"]["median5"])))
        mae_gbm = float(np.mean(np.abs(d["y"] - d["preds"]["gbm"])))
        rel_err = np.abs(d["y"] - d["preds"]["gbm"]) / d["cur"]
        dev = (d["y"] - d["cur"]) / d["cur"]
        for D in DEADLINES:
            for delta in DELTAS:
                weekly = cur_med * delta / D
                rows.append({"평가조건": split, "데드라인": D, "delta": f"{delta:.0%}", "현재페이스 중앙값(초/km)": cur_med,
                             "주당 체크포인트 변화(초/km)": weekly, "전체 변화폭(초/km)": cur_med * delta,
                             "'계획대로' 폭 ±3%(초/km)": 0.03 * cur_med,
                             "median-5 MAE": mae_m5, "GBM MAE": mae_gbm,
                             "주당 변화/GBM MAE": weekly / mae_gbm,
                             "|GBM오차|>3%현재페이스 비율": pct((rel_err > 0.03).mean()),
                             "|실제−현재|>3% 비율": pct((np.abs(dev) > 0.03).mean())})
    md.append(m21.df_to_markdown(pd.DataFrame(rows)))
    md.append("\n\n**읽는 법**: 주당 체크포인트 변화는 예측 MAE의 몇 %인지(‘주당 변화/GBM MAE’)로 비교한다. 이 값이 작을수록 한 주 사이의 목표 이동은 "
              "예측 오차에 묻힌다. 또한 '계획대로' 폭(±3%)보다 큰 예측 오차를 가진 윈도우 비율이 높으면 카테고리 경계가 예측 오차 수준에 비해 좁다는 뜻이다. "
              "또 '|실제−현재|>3% 비율'은 실제 페이스 자체가 현재 실력에서 ±3% 넘게 벗어나는 윈도우 비율로, 이 비율이 크면 "
              "(delta·체크포인트 이동이 아니라) 개별 러닝의 변동이 판정을 가른다. 참조 카테고리 분포(3-1)가 delta를 3%→10%로 바꿔도 크게 안 변하는 점도 같은 방향이다.\n")
    md.append("\n이 숫자들은 '판정이 예측 오차(및 실제 러닝 간 변동)에 지배된다'는 정황이지, 원인을 분리 검증한 것은 아니다. "
              "예: 변동이 사용자 수준 차이인지 같은 사용자 내 변동인지, 휴식 포함 경과 시간 페이스 정의의 영향인지는 **추정/미확인**이다.\n\n")

    md.append("## 6. 한계\n")
    md.append("- 목표는 전부 합성이다: 실제 사용자의 목표 설정 방식, 목표 달성 여부, 코칭 문구에 대한 반응은 관찰하지 않았다(미확인).\n")
    md.append("- 현재 실력을 median-5로 정의했으므로 median-5 예측기는 '예측 = 현재 실력'이라 격차가 체크포인트 이동폭(최대 delta)에 의해서만 결정된다 — "
              "median-5 예측기의 불일치율에는 이 정의가 구조적으로 영향을 준다(다른 예측기와 대칭이 아님).\n")
    md.append("- 데드라인 초과 윈도우 제외로 표본이 달라지고(2절), 4주/8주 결과는 서로 다른 표본 위에서 계산된다.\n")
    md.append("- 불일치율 CI는 사용자 단위 bootstrap이며 모델 재학습 변동성은 반영하지 않는다(GBM은 결정적, MLP는 시드 1개 사용).\n")
    md.append("- 코칭 문구의 품질(자연스러움, 근거 일치)은 평가하지 않았다. RAG·LLM 평가는 이 단계 범위 밖이다.\n")
    md.append("\n### 향후 과제 (가설, 이번에 검증하지 않음)\n")
    md.append("- 러닝 한 번의 페이스는 변동이 커서(5절) 단일 세션 페이스를 예측해 카테고리를 판정하는 방식보다, 여러 세션을 집계한 지표(예: 최근 몇 주 평균/중앙값 추세)를 "
              "기반으로 한 코칭이 더 적합할 수 있다. 이는 가설이며 이번 단계에서는 어떤 분석도 하지 않았다.\n")

    # ---- 요약 (결과 계산 후 문서 앞부분에 삽입) ----
    f5 = {(r["평가조건"], r["비교"]): r for r in fair_rows if r["delta"] == "5%" and r["데드라인"] == 8}
    m5 = {(r["평가조건"], r["예측기"]): r for r in main_rows if r["delta"] == "5%" and r["데드라인"] == 8}
    sm = ["## 요약 (delta 5%, 데드라인 8주 기준; 모든 수치는 합성 시나리오)\n"]
    sm.append("- 판정 불일치율(실제 페이스로 판정한 카테고리와 다른 비율)은 median-5 예측기를 제외하면 62~67% 범위다(8주·delta 5% 기준). "
              "예측 오차(MAE 약 55~60초/km)에 비해 주당 체크포인트 이동(약 2초/km)이 매우 작아 판정은 예측 오차와 러닝 간 변동에 지배된다(5절).\n")
    for split in ["cold-start", "personalized"]:
        c = m5[(split, "상수 규칙(학습/이력 구간 최빈)")]["불일치율"]
        h = m5[(split, "전체이력중앙값")]["불일치율"]
        g = m5[(split, "직접구현 GBM")]["불일치율"]
        ml = m5[(split, "MLP")]["불일치율"]
        sm.append(f"- {split}: 상수 규칙 {c}, 전체이력중앙값 {h}, GBM {g}, MLP {ml}.\n")
        for key in ["상수 규칙 − 직접구현 GBM", "상수 규칙 − MLP", "전체이력중앙값 − 직접구현 GBM", "전체이력중앙값 − MLP"]:
            r = f5[(split, key)]
            sm.append(f"  - {key}: {r['불일치율 차이(%p)']:+.2f}%p (95% CI [{r['하한(%p)']:+.2f}, {r['상한(%p)']:+.2f}]"
                      f"{', 0 포함' if r['0포함'] else ', 0 불포함'})\n")
    sm.append("- median-5 예측기 대비 비교는 현재 실력을 median-5로 정의한 시나리오의 구조 효과(3-5절)가 커서 요약에 쓰지 않는다.\n\n")
    md[summary_idx:summary_idx] = sm

    OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
    print(f"저장 완료: {OUTPUT_MD_PATH.resolve()}")


if __name__ == "__main__":
    main()
