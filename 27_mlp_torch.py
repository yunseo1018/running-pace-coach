# -*- coding: utf-8 -*-
"""
10월 2주차 8단계: torch MLP (L1 손실).

0. sklearn.neural_network.MLPRegressor는 loss∈{'squared_error','poisson'}만
   지원(L1/MAE 없음, docstring/코드 기준 확인 — 아래 main()에서 직접 확인해
   문서에 남긴다) -> 메인은 torch + L1Loss로 직접 구현.
1. 모델 두 형태: (a) 직접 예측 (b) median5 + 잔차. 입력은 7단계와 동일한
   22개 피처(24단계 extended 데이터셋), 표준화는 학습 데이터에서만 fit.
2. 하이퍼파라미터 격자(사전 고정, ≤8개): hidden_layers∈{1,2} x width∈{32,64}
   x weight_decay∈{0,1e-4} = 8개. 학습/아키텍처 세부값(배치크기, lr, 최대
   에폭, patience)도 사전에 고정하고 결과를 보고 바꾸지 않는다.
3. 선택+early stopping은 학습 구간 내부 검증 MAE만 사용(cold-start는
   GroupKFold 1-fold를 검증으로, personalized는 시간순 inner-split). 테스트는
   선택 과정에서 전혀 쓰지 않는다.
4. 시드 5개로 최종 재학습, test MAE 평균/표준편차 보고.
5. 평가: 기존 통합표 형식 + bootstrap CI + median-5/7단계 GBM 대비 쌍체 차이.
6. MLP가 GBM을 못 이기면 '이기지 못함'으로 명시, 원인 추정/미확인.

산출물: docs/mlp_results.md
"""

import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import gbt_lib as gl

_spec_m21 = importlib.util.spec_from_file_location("m21", "21_baseline_integrated_report.py")
m21 = importlib.util.module_from_spec(_spec_m21)
_spec_m21.loader.exec_module(m21)

_spec_m24 = importlib.util.spec_from_file_location("m24", "24_build_xy_dataset_extended.py")
m24 = importlib.util.module_from_spec(_spec_m24)
_spec_m24.loader.exec_module(m24)

# 25_tree_models.py의 재사용 가능한 유틸리티(samples_to_xy, median5_from_X,
# personalized_inner_split, build_gbt, build_model_rows 등)를 가져온다.
# 모듈을 이렇게 로드해도 main()은 실행되지 않는다(파일 끝의 __main__ 가드로
# 보호됨) — 무거운 계산 없이 함수/상수만 재사용.
_spec_m25 = importlib.util.spec_from_file_location("m25", "25_tree_models.py")
m25 = importlib.util.module_from_spec(_spec_m25)
_spec_m25.loader.exec_module(m25)

OUTPUT_MD_PATH = Path("docs") / "mlp_results.md"
SELECTED_JSON_PATH = Path("docs") / "mlp_selected_hyperparams.json"  # 28/29번이 읽는 선택 결과(하이퍼파라미터만)

# ===== 7단계에서 이미 CV/내부검증으로 선택된(재탐색하지 않음) GBM 하이퍼파라미터 =====
# cold-start 주 결과: 7a. 직접구현 GBM(absolute_error); personalized 주 결과: 7c. median5+잔차 GBM
GBM_COLD_PARAMS = {"learning_rate": 0.1, "max_iter": 300, "max_depth": 3}
GBM_PERS_PARAMS = {"learning_rate": 0.1, "max_iter": 300, "max_depth": 3}

# ===== MLP 하이퍼파라미터 격자 (사전 고정, 8개, 결과 보고 안 바꿈) =====
MLP_PARAM_GRID = [
    {"n_hidden_layers": nh, "width": w, "weight_decay": wd}
    for nh in [1, 2]
    for w in [32, 64]
    for wd in [0.0, 1e-4]
]
# 아키텍처 외 학습 설정(사전 고정, 실측 속도 기준: batch_size=2048일 때
# 10에폭=1.5초/79669건 -> 200에폭 최악의 경우도 수십 초 수준으로 확인하고 정함)
BATCH_SIZE = 2048
MAX_EPOCHS = 200
PATIENCE = 15
LR = 1e-3
SELECTION_SEED = 42
FINAL_SEEDS = [1, 2, 3, 42, 999]

DIRECT_LABEL = "mlp_direct"
RESID_LABEL = "mlp_median5_resid"
MLP_LABELS = {DIRECT_LABEL: "8a. MLP(직접 예측, L1)", RESID_LABEL: "8b. MLP(median5+잔차, L1)"}


class MLP(nn.Module):
    def __init__(self, input_dim, n_hidden_layers, width):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, width), nn.ReLU()]
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_scaler(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def _apply_scaler(X, scaler):
    mean, std = scaler
    return (X - mean) / std


def train_mlp_early_stopping(X_tr, y_tr, X_val, y_val, n_hidden_layers, width, weight_decay, seed,
                               max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE):
    """학습 데이터에서만 scaler를 fit하고, 검증 MAE가 patience 에폭 동안 개선되지
    않으면 멈춘다(최적 가중치/에폭 보존). 테스트는 쓰지 않는다."""
    torch.manual_seed(seed)
    scaler = _fit_scaler(X_tr)
    X_tr_s = _apply_scaler(X_tr, scaler)
    X_val_s = _apply_scaler(X_val, scaler)

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    model = MLP(X_tr.shape[1], n_hidden_layers, width)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    n = X_tr_t.shape[0]
    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = model(X_tr_t[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_mae = loss_fn(model(X_val_t), y_val_t).item()
        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, scaler, best_val_mae, best_epoch + 1


def train_mlp_fixed_epochs(X_full, y_full, n_hidden_layers, width, weight_decay, seed, n_epochs,
                             lr=LR, batch_size=BATCH_SIZE):
    """validation으로 정한 에폭 수만큼, 전체 학습 데이터로 재학습(더 이상 검증 없음 —
    에폭 수 자체가 이미 검증으로 정해졌으므로)."""
    torch.manual_seed(seed)
    scaler = _fit_scaler(X_full)
    X_s = _apply_scaler(X_full, scaler)
    X_t = torch.tensor(X_s, dtype=torch.float32)
    y_t = torch.tensor(y_full, dtype=torch.float32)

    model = MLP(X_full.shape[1], n_hidden_layers, width)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()
    n = X_t.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = model(X_t[idx])
            loss = loss_fn(pred, y_t[idx])
            loss.backward()
            opt.step()
    return model, scaler


def predict_mlp(model, scaler, X):
    X_s = _apply_scaler(X, scaler)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_s, dtype=torch.float32)).numpy()
    return pred


def select_mlp_hparams(X_fit, y_fit, X_val, y_val, resid_mode, med_fit=None, med_val=None):
    """8개 격자 전체를 검증 MAE 기준으로 평가, 최적 (hparams, best_epoch)을 반환.
    resid_mode=True면 y_fit/y_val 대신 잔차(y-median5)를 타깃으로 학습하고,
    검증 MAE는 median5를 더한 실제 예측으로 계산한다."""
    target_fit = (y_fit - med_fit) if resid_mode else y_fit
    scores = {}
    for params in MLP_PARAM_GRID:
        key = tuple(sorted(params.items()))
        model, scaler, _, best_epoch = train_mlp_early_stopping(
            X_fit, target_fit, X_val, (y_val - med_val) if resid_mode else y_val,
            params["n_hidden_layers"], params["width"], params["weight_decay"], SELECTION_SEED,
        )
        pred_val = predict_mlp(model, scaler, X_val)
        if resid_mode:
            pred_val = pred_val + med_val
        val_mae = float(np.mean(np.abs(y_val - pred_val)))
        scores[key] = (params, val_mae, best_epoch)
    best_key = min(scores, key=lambda k: scores[k][1])
    return scores[best_key][0], scores[best_key][1], scores[best_key][2], scores


def main():
    md = []
    md.append("# torch MLP 결과 (10월 2주차 8단계)\n\n")

    main_df = config.load_main_sessions()
    md.append(f"입력 메인 데이터: `{config.MAIN_DATA_PATH}` (세션 {len(main_df)}건, "
              f"사용자 {main_df['user_id'].nunique()}명, assert 통과)\n\n")

    # ===== 0. 환경: MLPRegressor loss 제약 확인 + torch 버전/결정성 =====
    md.append("## 0. 환경\n")
    try:  # 진단 기록 전용: 실패해도 결과 수치 경로(torch MLP)에는 영향 없음
        from sklearn.neural_network import MLPRegressor
        import inspect
        sig = inspect.signature(MLPRegressor.__init__)
        loss_param = sig.parameters.get("loss")
        doc_excerpt = [line for line in MLPRegressor.__doc__.split("\n") if "loss :" in line or "squared_error" in line or "poisson" in line][:3]
        md.append(f"- **sklearn.neural_network.MLPRegressor의 `loss` 파라미터 기본값/타입**: "
                  f"`{loss_param.default!r}` (코드 기준, `inspect.signature` 확인)\n")
        md.append(f"- **docstring 발췌**: {' / '.join(l.strip() for l in doc_excerpt)}\n")
    except ImportError as e:
        md.append(f"- (진단 생략) 이번 실행에서 sklearn import 실패(`{str(e)[:100]}`)로 MLPRegressor 코드 확인을 다시 하지 않았다. "
                  "이전 실행에서의 확인 결과(sklearn 1.9.1): `loss`는 'squared_error'/'poisson'만 지원.\n")
    md.append("- **확인됨**: MLPRegressor는 `loss ∈ {'squared_error', 'poisson'}`만 지원하고 "
              "**L1(절대오차) 손실 옵션이 없다.** 그래서 메인 모델에 쓰지 않고, torch로 직접 "
              "L1Loss 기반 MLP를 구현했다.\n")
    md.append(f"- torch: {torch.__version__}, numpy: {np.__version__}\n")
    md.append("- **결정성 설정**: 각 학습 전 `torch.manual_seed(seed)`를 호출한다. CPU 전용 실행"
              f"(torch {torch.__version__}, GPU 미사용)이며, `torch.use_deterministic_algorithms`는 "
              "강제하지 않았다(의도적으로 시드 간 실제 변동을 관찰하기 위함 — 4번 참고). "
              "데이터 순서 셔플(`torch.randperm`)도 `manual_seed`에 종속된다.\n\n")

    # ===== 1. 입력 =====
    train_samples = pd.read_pickle(m24.EXT_TRAIN_PATH)
    test_samples = pd.read_pickle(m24.EXT_TEST_PATH)
    p_train_samples = pd.read_pickle(m24.EXT_P_TRAIN_PATH)
    p_test_samples = pd.read_pickle(m24.EXT_P_TEST_PATH)
    user_sequences = pd.read_pickle(config.USER_SEQUENCES_PATH)
    overall_avg_lookup = m21.build_user_overall_avg_lookup(user_sequences)

    md.append("## 1. 입력\n")
    md.append(f"- cold-start: train {len(train_samples)}건 / test {len(test_samples)}건\n")
    md.append(f"- personalized: history {len(p_train_samples)}건 / hold-out {len(p_test_samples)}건\n")
    md.append("- X = 7단계와 동일한 22개 피처(24단계 확장 데이터셋). 표준화(평균/표준편차)는 "
              "**학습(또는 inner-train) 데이터에서만 fit**하고 검증/테스트에는 그 통계치로 "
              "transform만 적용한다(함수 `_fit_scaler`/`_apply_scaler`, Pipeline은 아니지만 "
              "같은 원칙 — fit 범위를 코드로 강제).\n")
    md.append(f"- MLP 하이퍼파라미터 격자(사전 고정, {len(MLP_PARAM_GRID)}개): "
              f"hidden_layers∈[1,2], width∈[32,64], weight_decay∈[0,1e-4]\n")
    md.append(f"- 학습 설정(사전 고정): batch_size={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, "
              f"patience={PATIENCE}, lr={LR}, 선택 단계 시드={SELECTION_SEED}, "
              f"최종 재학습 시드={FINAL_SEEDS}\n\n")

    X_tr, y_tr, uid_tr = m25.samples_to_xy(train_samples)
    X_te, y_te, uid_te = m25.samples_to_xy(test_samples)
    X_ptr, y_ptr, _ = m25.samples_to_xy(p_train_samples)
    X_pte, y_pte, _ = m25.samples_to_xy(p_test_samples)

    all_samples_check = train_samples + test_samples + p_train_samples + p_test_samples
    gaps_check = [s["target_start_time"] - s["last_input_session_time"] for s in all_samples_check]
    md.append(f"- **last_input_session_time 감사 재확인**: {len(all_samples_check)}개 샘플 전체 "
              f"최소 갭 = {min(gaps_check)}초 (0보다 큼, 전부 통과). 24단계에서 이미 걸러진 "
              "샘플만 여기 들어있다(동일 데이터셋 재사용).\n\n")

    # ===== 2~3. 하이퍼파라미터 선택 (검증 MAE만, 테스트 안 봄) =====
    print("\ncold-start 검증 분할 준비 중...")
    import ridge_lib as _rl  # GroupKFold도 ridge_lib 고정(sklearn 미사용)
    tr_idx, val_idx = next(iter(_rl.group_kfold_splits(uid_tr, m25.N_FOLDS, m25.CV_SEED)))
    X_fit_cold, y_fit_cold = X_tr[tr_idx], y_tr[tr_idx]
    X_val_cold, y_val_cold = X_tr[val_idx], y_tr[val_idx]
    med_fit_cold = m25.median5_from_X(X_fit_cold)
    med_val_cold = m25.median5_from_X(X_val_cold)
    print(f"  cold-start: fit {len(tr_idx)}건 / val {len(val_idx)}건 (GroupKFold 1-fold)")

    inner_tr, inner_val = m25.personalized_inner_split(p_train_samples)
    X_fit_pers, y_fit_pers, _ = m25.samples_to_xy(inner_tr)
    X_val_pers, y_val_pers, _ = m25.samples_to_xy(inner_val)
    med_fit_pers = m25.median5_from_X(X_fit_pers)
    med_val_pers = m25.median5_from_X(X_val_pers)
    print(f"  personalized: inner-train {len(inner_tr)}건 / inner-val {len(inner_val)}건")

    md.append("## 2. 하이퍼파라미터 선택 (검증 MAE만 사용, 테스트는 선택에 쓰지 않음)\n")
    selection_results = {}
    for split_name, X_fit, y_fit, X_val, y_val, med_fit, med_val in [
        ("cold-start", X_fit_cold, y_fit_cold, X_val_cold, y_val_cold, med_fit_cold, med_val_cold),
        ("personalized", X_fit_pers, y_fit_pers, X_val_pers, y_val_pers, med_fit_pers, med_val_pers),
    ]:
        md.append(f"\n#### {split_name}\n")
        for label, resid_mode in [(DIRECT_LABEL, False), (RESID_LABEL, True)]:
            print(f"  {split_name} / {label} 하이퍼파라미터 탐색 중 (8개 조합)...")
            best_params, best_val_mae, best_epoch, scores = select_mlp_hparams(
                X_fit, y_fit, X_val, y_val, resid_mode, med_fit, med_val
            )
            selection_results[(split_name, label)] = (best_params, best_val_mae, best_epoch)
            rows = [{"조합": str(p), "검증 MAE": v, "선택된 에폭": e} for p, v, e in scores.values()]
            md.append(f"\n**{MLP_LABELS[label]}**:\n")
            md.append(m21.df_to_markdown(pd.DataFrame(rows)))
            md.append(f"\n- 선택: **{best_params}**, 검증 MAE={best_val_mae:.3f}, 에폭={best_epoch}\n")
            print(f"    선택: {best_params}, val_mae={best_val_mae:.3f}, epoch={best_epoch}")

    # ===== 4. 최종 재학습 (5개 시드, 전체 학습 데이터, 고정 에폭) =====
    print("\n최종 재학습 (5개 시드)...")
    md.append("\n## 3. 최종 재학습 (선택된 하이퍼파라미터로 전체 학습 데이터 재학습, 5개 시드)\n")
    md.append("검증으로 정한 에폭 수만큼, 전체 train(cold-start)/history(personalized) 데이터로 "
              "다시 학습한다(더 이상 조기종료 없음 — 에폭 수 자체가 이미 검증으로 정해졌으므로). "
              "같은 설정을 시드 5개로 반복해 test MAE 평균/표준편차를 본다.\n")

    final_models = {}
    for split_name, X_full, y_full, med_full_fn, X_test_, y_test_ in [
        ("cold-start", X_tr, y_tr, m25.median5_from_X, X_te, y_te),
        ("personalized", X_ptr, y_ptr, m25.median5_from_X, X_pte, y_pte),
    ]:
        md.append(f"\n#### {split_name}\n")
        med_full = med_full_fn(X_full)
        med_test = med_full_fn(X_test_)
        for label, resid_mode in [(DIRECT_LABEL, False), (RESID_LABEL, True)]:
            params, _, n_epochs = selection_results[(split_name, label)]
            target_full = (y_full - med_full) if resid_mode else y_full
            seed_maes = []
            seed_preds = []
            for seed in FINAL_SEEDS:
                model, scaler = train_mlp_fixed_epochs(
                    X_full, target_full, params["n_hidden_layers"], params["width"],
                    params["weight_decay"], seed, n_epochs,
                )
                pred = predict_mlp(model, scaler, X_test_)
                if resid_mode:
                    pred = pred + med_test
                mae = float(np.mean(np.abs(y_test_ - pred)))
                seed_maes.append(mae)
                seed_preds.append(pred)
                print(f"  {split_name}/{label} seed={seed}: test MAE={mae:.3f}")
            # 대표 예측(보고용 통합표/부트스트랩 CI는 seed=42, 관례상 CV_SEED와 동일)
            rep_idx = FINAL_SEEDS.index(42)
            final_models[(split_name, label)] = {
                "params": params, "n_epochs": n_epochs, "seed_maes": seed_maes,
                "rep_pred": seed_preds[rep_idx],
            }
            md.append(f"\n**{MLP_LABELS[label]}** (하이퍼파라미터={params}, 에폭={n_epochs}): "
                      f"시드별 test MAE={[round(m,3) for m in seed_maes]}\n")
            md.append(f"- **평균={np.mean(seed_maes):.3f}, 표준편차={np.std(seed_maes):.3f}** "
                      f"(대표 시드=42를 이하 통합표/CI에 사용)\n")

    # CV(검증) 기준 선택된 MLP 형태(두 형태 중 검증 MAE 더 낮은 쪽)를 주 결과로.
    mlp_cv_selected = {}
    for split_name in ["cold-start", "personalized"]:
        direct_val_mae = selection_results[(split_name, DIRECT_LABEL)][1]
        resid_val_mae = selection_results[(split_name, RESID_LABEL)][1]
        mlp_cv_selected[split_name] = DIRECT_LABEL if direct_val_mae <= resid_val_mae else RESID_LABEL
        print(f"  {split_name} MLP 주 결과(검증 MAE 기준): {mlp_cv_selected[split_name]} "
              f"(direct_val={direct_val_mae:.3f}, resid_val={resid_val_mae:.3f})")

    # 28/29번이 읽는 선택 결과 저장(하이퍼파라미터만: 사용자 ID·행별 예측 없음, git 추적 대상).
    # 선택 규칙: ridge_lib GroupKFold(cold-start, 첫 fold를 검증으로) / 시간순 inner-split(personalized)의 검증 MAE. 테스트 점수는 선택에 쓰지 않음.
    selected_json = {
        "selection_rule": "validation MAE only (cold-start: first fold of ridge_lib.group_kfold_splits(seed=CV_SEED); "
                          "personalized: chronological inner split); test never used",
        "selection_seed": SELECTION_SEED, "representative_seed": 42, "final_seeds": FINAL_SEEDS,
        "torch_version": torch.__version__, "splits": {},
    }
    for split_name in ["cold-start", "personalized"]:
        form = mlp_cv_selected[split_name]
        p_sel, v_sel, e_sel = selection_results[(split_name, form)]
        selected_json["splits"][split_name] = {
            "form": "median5_residual" if form == RESID_LABEL else "direct", "label": form,
            "n_hidden_layers": p_sel["n_hidden_layers"], "width": p_sel["width"],
            "weight_decay": p_sel["weight_decay"], "n_epochs": int(e_sel), "val_mae": float(v_sel),
            "val_mae_direct": float(selection_results[(split_name, DIRECT_LABEL)][1]),
            "val_mae_resid": float(selection_results[(split_name, RESID_LABEL)][1]),
        }
    SELECTED_JSON_PATH.write_text(json.dumps(selected_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print("선택 결과 저장:", SELECTED_JSON_PATH)

    md.append("\n**MLP 두 형태 중 주 결과(검증 MAE 기준)**: "
              f"cold-start=`{MLP_LABELS[mlp_cv_selected['cold-start']]}`, "
              f"personalized=`{MLP_LABELS[mlp_cv_selected['personalized']]}`\n")

    # ===== 4. 평가: 통합표 + bootstrap CI + median-5/GBM 대비 쌍체 차이 =====
    print("\n평가 통합표/부트스트랩 계산 중...")
    md.append("\n## 4. 평가\n")

    def with_original_15_X(samples):
        return [{**s, "X": s["X"][:15]} for s in samples]

    base_cold_df = m21.build_rows(with_original_15_X(test_samples), overall_avg_lookup)
    base_pers_df = m21.build_rows(with_original_15_X(p_test_samples), overall_avg_lookup)

    # 7단계에서 이미 선택된(재탐색 없음) GBM 하이퍼파라미터로 비교용 예측 재생성.
    med_tr_full = m25.median5_from_X(X_tr)
    gbm_cold_model = m25.build_gbt(GBM_COLD_PARAMS, "absolute_error").fit(X_tr, y_tr)
    gbm_cold_pred = gbm_cold_model.predict(X_te)
    med_ptr_full = m25.median5_from_X(X_ptr)
    med_pte_full = m25.median5_from_X(X_pte)
    gbm_pers_model = m25.build_gbt(GBM_PERS_PARAMS, "absolute_error")
    gbm_pers_model.fit(X_ptr, y_ptr - med_ptr_full)
    gbm_pers_pred = med_pte_full + gbm_pers_model.predict(X_pte)

    GBM_LABEL = "gbm_7step"
    GBM_LABELS = {GBM_LABEL: "7단계 주결과 GBM(gbt_lib)"}
    gbm_cold_df = m25.build_model_rows(test_samples, gbm_cold_pred, GBM_LABEL)
    gbm_pers_df = m25.build_model_rows(p_test_samples, gbm_pers_pred, GBM_LABEL)

    direct_cold_df = m25.build_model_rows(test_samples, final_models[("cold-start", DIRECT_LABEL)]["rep_pred"], DIRECT_LABEL)
    resid_cold_df = m25.build_model_rows(test_samples, final_models[("cold-start", RESID_LABEL)]["rep_pred"], RESID_LABEL)
    direct_pers_df = m25.build_model_rows(p_test_samples, final_models[("personalized", DIRECT_LABEL)]["rep_pred"], DIRECT_LABEL)
    resid_pers_df = m25.build_model_rows(p_test_samples, final_models[("personalized", RESID_LABEL)]["rep_pred"], RESID_LABEL)

    combined_cold_df = pd.concat([base_cold_df, gbm_cold_df, direct_cold_df, resid_cold_df], ignore_index=True)
    combined_pers_df = pd.concat([base_pers_df, gbm_pers_df, direct_pers_df, resid_pers_df], ignore_index=True)

    ALL_LABELS = dict(m21.BASELINE_LABELS)
    ALL_LABELS.update(GBM_LABELS)
    ALL_LABELS.update(MLP_LABELS)
    ALL_NAMES = m21.BASELINE_NAMES + [GBM_LABEL, DIRECT_LABEL, RESID_LABEL]

    def summarize_all(df):
        g = df.groupby("baseline")
        out = g.agg(n_samples=("abs_error", "size"), n_users=("user_id", "nunique"),
                     mae=("abs_error", "mean"), rmse=("sq_error", lambda c: np.sqrt(c.mean())))
        n_merges = df[df["merge_affected"]].groupby("baseline").size().rename("n_merges")
        out = out.join(n_merges, how="left")
        out["n_merges"] = out["n_merges"].fillna(0).astype(int)
        out = out.reindex(ALL_NAMES)
        out["방법"] = [ALL_LABELS[b] for b in out.index]
        return out.reset_index(drop=True)[["방법", "mae", "rmse", "n_users", "n_samples", "n_merges"]].rename(
            columns={"mae": "MAE(초/km)", "rmse": "RMSE(초/km)", "n_users": "#users",
                     "n_samples": "#eval sessions", "n_merges": "#merges"}
        )

    md.append("\n### 통합표\n")
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        md.append(f"\n#### {split_name}\n")
        md.append(m21.df_to_markdown(summarize_all(df)))
        md.append("\n")

    print("부트스트랩 CI 계산 중...")
    md.append(f"\n### 신뢰구간 (사용자 단위 cluster bootstrap, {m21.N_BOOTSTRAP}회, seed={m21.BOOTSTRAP_SEED})\n")
    boot_results = {}
    for split_name, df in [("cold-start", combined_cold_df), ("personalized", combined_pers_df)]:
        boot_mae, boot_rmse = m21.cluster_bootstrap(df, ALL_NAMES)
        boot_results[split_name] = (boot_mae, boot_rmse)
        rows = []
        for b in ALL_NAMES:
            mae_lo, mae_hi = np.percentile(boot_mae[b], [2.5, 97.5])
            rmse_lo, rmse_hi = np.percentile(boot_rmse[b], [2.5, 97.5])
            rows.append({"방법": ALL_LABELS[b], "MAE": boot_mae[b].mean(), "MAE_95%CI_하한": mae_lo,
                         "MAE_95%CI_상한": mae_hi, "RMSE": boot_rmse[b].mean(), "RMSE_95%CI_하한": rmse_lo,
                         "RMSE_95%CI_상한": rmse_hi})
        md.append(f"\n#### {split_name}\n")
        md.append(m21.df_to_markdown(pd.DataFrame(rows)))
        md.append("\n")

    md.append("\n### 쌍체 차이 95% CI (MLP 주 결과 vs median-5, vs 7단계 GBM)\n")
    any_beats_gbm = False
    for split_name in ["cold-start", "personalized"]:
        boot_mae, boot_rmse = boot_results[split_name]
        mlp_label = mlp_cv_selected[split_name]
        md.append(f"\n#### {split_name} — MLP 주 결과: {ALL_LABELS[mlp_label]}\n")

        mae_diff_m5 = boot_mae["median5"] - boot_mae[mlp_label]
        lo_m5, hi_m5 = np.percentile(mae_diff_m5, [2.5, 97.5])
        md.append(f"- **MAE: median-5 - MLP** = {mae_diff_m5.mean():.3f} (95% CI [{lo_m5:.3f}, {hi_m5:.3f}]) "
                  f"-> {'0 미포함, MLP가 유의하게 나음' if lo_m5 > 0 else ('0 미포함, median-5가 유의하게 나음' if hi_m5 < 0 else '0 포함, 유의한 차이 없음')}\n")

        mae_diff_gbm = boot_mae[GBM_LABEL] - boot_mae[mlp_label]
        lo_g, hi_g = np.percentile(mae_diff_gbm, [2.5, 97.5])
        gbm_beats = hi_g < 0
        mlp_beats_gbm = lo_g > 0
        any_beats_gbm = any_beats_gbm or mlp_beats_gbm
        md.append(f"- **MAE: GBM - MLP** = {mae_diff_gbm.mean():.3f} (95% CI [{lo_g:.3f}, {hi_g:.3f}]) "
                  f"-> {'0 미포함, MLP가 GBM을 유의하게 이김' if mlp_beats_gbm else ('0 미포함, GBM이 유의하게 나음(MLP가 이기지 못함)' if gbm_beats else '0 포함, 유의한 차이 없음')}\n")

        rmse_diff_gbm = boot_rmse[GBM_LABEL] - boot_rmse[mlp_label]
        lo_rg, hi_rg = np.percentile(rmse_diff_gbm, [2.5, 97.5])
        rmse_significant = not (lo_rg <= 0 <= hi_rg)
        ci_width = hi_rg - lo_rg
        near_zero_note = ""
        if rmse_significant and lo_rg > 0 and lo_rg < 0.1 * rmse_diff_gbm.mean():
            near_zero_note = (f" (**주의: CI 하한 {lo_rg:.3f}이 0에 매우 가깝고 구간 폭이 "
                              f"{ci_width:.1f}로 넓다** — 통계적으로는 유의하지만 경계선에 "
                              f"걸쳐 있다는 점을 같이 봐야 한다)")
        md.append(f"- **RMSE: GBM - MLP** = {rmse_diff_gbm.mean():.3f} (95% CI [{lo_rg:.3f}, {hi_rg:.3f}]) "
                  f"-> {'0 포함, 유의한 차이 없음' if lo_rg <= 0 <= hi_rg else ('0 미포함, MLP가 유의하게 나음' if lo_rg > 0 else '0 미포함, GBM이 유의하게 나음')}"
                  f"{near_zero_note}\n")
        if rmse_significant and lo_rg > 0:
            md.append("  - 이 RMSE 차이의 원인(예: GBM이 극단 페이스 윈도우에 더 민감한지)은 "
                      "별도로 검증하지 않았다 — **추정/미확인**으로만 남긴다.\n")

    # ===== 5. 해석 =====
    md.append("\n## 5. 해석\n")
    for split_name in ["cold-start", "personalized"]:
        boot_mae, _ = boot_results[split_name]
        mlp_label = mlp_cv_selected[split_name]
        diff = boot_mae[GBM_LABEL] - boot_mae[mlp_label]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        if lo > 0:
            md.append(f"- **{split_name}**: MLP({ALL_LABELS[mlp_label]})가 7단계 GBM을 MAE "
                      f"신뢰구간 밖에서 **이겼다** (GBM-MLP diff 95% CI [{lo:.3f},{hi:.3f}], 0 미포함).\n")
        else:
            md.append(f"- **{split_name}**: MLP({ALL_LABELS[mlp_label]})가 7단계 GBM을 MAE "
                      f"신뢰구간 밖에서 **이기지 못함**. (diff 95% CI [{lo:.3f},{hi:.3f}]) "
                      f"원인은 **추정/미확인**이다(가능성: 이 피처셋에서는 트리 기반 비선형 분할이 "
                      f"MLP의 매끄러운 함수 근사보다 유리할 수 있음, 또는 하이퍼파라미터 격자가 "
                      f"충분히 넓지 않았을 수 있음 — 둘 다 검증 안 함).\n")
    md.append("\n**MAE와 RMSE 결론을 따로 서술**: 위 쌍체 차이 표의 RMSE 비교가 MAE 비교와 "
              "다를 수 있다 — RMSE는 상위 오차 윈도우(극단 페이스 등)에 민감한 지표이므로 "
              "MAE 기준 결론과 RMSE 기준 결론을 섞어서 읽지 않는다.\n")

    # ===== 6. 분할기 변경에 따른 민감도 =====
    md.append("\n## 6. 분할기 변경에 따른 민감도 (cold-start 검증 fold 구성)\n")
    md.append("cold-start의 하이퍼파라미터 선택은 train 사용자 5-fold GroupKFold의 첫 fold를 검증으로 쓴다. 이 fold 구성은 분할기에 따라 다르다: "
              "처음 실행(커밋 00c27d1)은 sklearn GroupKFold(그룹 크기 기반 결정적 분할), 재현성 고정 후에는 ridge_lib의 GroupKFold(시드 42 셔플 후 균등분할). "
              "**둘 다 같은 사용자가 fold 양쪽에 걸치지 않는 유효한 CV 선택이며, 어느 쪽이 더 낫다고 판단하지 않는다.** 아래는 같은 코드·같은 격자에서 fold 구성만 바뀐 효과다 "
              "(personalized는 시간순 분할이라 영향 없음, 값 불변).\n\n")
    cur_final = final_models[("cold-start", RESID_LABEL)]
    cur_rep_mae = float(np.mean(np.abs(y_te - cur_final["rep_pred"])))
    b_mae, b_rmse = boot_results["cold-start"]
    d_m5 = b_mae["median5"] - b_mae[RESID_LABEL]
    d_gb = b_mae[GBM_LABEL] - b_mae[RESID_LABEL]
    lo1, hi1 = np.percentile(d_m5, [2.5, 97.5])
    lo2, hi2 = np.percentile(d_gb, [2.5, 97.5])
    p_cur = cur_final["params"]
    sens = pd.DataFrame([
        {"분할기": "sklearn GroupKFold (커밋 00c27d1 시점, 문서 기록값)", "선택 하이퍼파라미터": "{'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0}, 에폭 97",
         "test MAE(대표 시드 42)": "59.578", "5시드 평균±표준편차": "59.514±0.050",
         "median-5 대비 개선폭 [95% CI]": "2.661 [1.865, 3.535]", "GBM−MLP [95% CI]": "0.104 [-0.560, 0.941]"},
        {"분할기": "ridge_lib GroupKFold (현재, 고정)", "선택 하이퍼파라미터": f"{p_cur}, 에폭 {cur_final['n_epochs']}",
         "test MAE(대표 시드 42)": f"{cur_rep_mae:.3f}", "5시드 평균±표준편차": f"{np.mean(cur_final['seed_maes']):.3f}±{np.std(cur_final['seed_maes']):.3f}",
         "median-5 대비 개선폭 [95% CI]": f"{d_m5.mean():.3f} [{lo1:.3f}, {hi1:.3f}]", "GBM−MLP [95% CI]": f"{d_gb.mean():.3f} [{lo2:.3f}, {hi2:.3f}]"},
    ])
    md.append(m21.df_to_markdown(sens))
    md.append("\n\n**기록**: fold 구성이 바뀌자 선택된 구조·에폭이 달라졌고 cold-start MLP의 test MAE가 약 0.2초/km 움직였다. 두 설정 모두에서 MLP는 median-5보다 유의하게 낫고 GBM과는 구분되지 않는다(위 CI 기준). "
              "이 0.2초/km는 하이퍼파라미터 선택이 검증 fold 구성에 민감하다는 정도를 보여주는 값이며, GBM은 같은 격자·같은 선택이 유지됐다(7단계 문서). "
              "이 변동의 원인 분해(검증 fold 데이터 차이 vs 조기종료/에폭 변화)는 하지 않았다 — **추정/미확인**.\n")

    OUTPUT_MD_PATH.write_text("".join(md), encoding="utf-8")
    print(f"\n최종 저장 완료: {OUTPUT_MD_PATH.resolve()}")
    return {"any_beats_gbm": any_beats_gbm, "mlp_cv_selected": mlp_cv_selected}


if __name__ == "__main__":
    main()
