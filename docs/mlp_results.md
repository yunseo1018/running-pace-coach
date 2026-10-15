# torch MLP 결과 (10월 2주차 8단계)

입력 메인 데이터: `data\run_sessions_deduped_merged.pkl` (세션 105709건, 사용자 854명, assert 통과)

## 0. 환경
- **sklearn.neural_network.MLPRegressor의 `loss` 파라미터 기본값/타입**: `'squared_error'` (코드 기준, `inspect.signature` 확인)
- **docstring 발췌**: loss : {'squared_error', 'poisson'}, default='squared_error' / "squared error" and "poisson" losses actually implement / "half squares error" and "half poisson deviance" to simplify the
- **확인됨**: MLPRegressor는 `loss ∈ {'squared_error', 'poisson'}`만 지원하고 **L1(절대오차) 손실 옵션이 없다.** 그래서 메인 모델에 쓰지 않고, torch로 직접 L1Loss 기반 MLP를 구현했다.
- torch: 2.14.0+cpu, numpy: 2.5.3
- **결정성 설정**: 각 학습 전 `torch.manual_seed(seed)`를 호출한다. CPU 전용 실행(torch 2.14.0+cpu, GPU 미사용)이며, `torch.use_deterministic_algorithms`는 강제하지 않았다(의도적으로 시드 간 실제 변동을 관찰하기 위함 — 4번 참고). 데이터 순서 셔플(`torch.randperm`)도 `manual_seed`에 종속된다.

## 1. 입력
- cold-start: train 79669건 / test 21977건
- personalized: history 81018건 / hold-out 20628건
- X = 7단계와 동일한 22개 피처(24단계 확장 데이터셋). 표준화(평균/표준편차)는 **학습(또는 inner-train) 데이터에서만 fit**하고 검증/테스트에는 그 통계치로 transform만 적용한다(함수 `_fit_scaler`/`_apply_scaler`, Pipeline은 아니지만 같은 원칙 — fit 범위를 코드로 강제).
- MLP 하이퍼파라미터 격자(사전 고정, 8개): hidden_layers∈[1,2], width∈[32,64], weight_decay∈[0,1e-4]
- 학습 설정(사전 고정): batch_size=2048, max_epochs=200, patience=15, lr=0.001, 선택 단계 시드=42, 최종 재학습 시드=[1, 2, 3, 42, 999]

- **last_input_session_time 감사 재확인**: 203292개 샘플 전체 최소 갭 = 1초 (0보다 큼, 전부 통과). 24단계에서 이미 걸러진 샘플만 여기 들어있다(동일 데이터셋 재사용).

## 2. 하이퍼파라미터 선택 (검증 MAE만 사용, 테스트는 선택에 쓰지 않음)

#### cold-start

**8a. MLP(직접 예측, L1)**:
| 조합 | 검증 MAE | 선택된 에폭 |
|---|---|---|
| {'n_hidden_layers': 1, 'width': 32, 'weight_decay': 0.0} | 57.054 | 199 |
| {'n_hidden_layers': 1, 'width': 32, 'weight_decay': 0.0001} | 57.339 | 200 |
| {'n_hidden_layers': 1, 'width': 64, 'weight_decay': 0.0} | 56.446 | 199 |
| {'n_hidden_layers': 1, 'width': 64, 'weight_decay': 0.0001} | 56.470 | 199 |
| {'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0} | 56.360 | 112 |
| {'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0001} | 56.349 | 112 |
| {'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0} | 56.180 | 131 |
| {'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0001} | 56.237 | 86 |
- 선택: **{'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0}**, 검증 MAE=56.180, 에폭=131

**8b. MLP(median5+잔차, L1)**:
| 조합 | 검증 MAE | 선택된 에폭 |
|---|---|---|
| {'n_hidden_layers': 1, 'width': 32, 'weight_decay': 0.0} | 56.151 | 196 |
| {'n_hidden_layers': 1, 'width': 32, 'weight_decay': 0.0001} | 56.151 | 200 |
| {'n_hidden_layers': 1, 'width': 64, 'weight_decay': 0.0} | 56.120 | 199 |
| {'n_hidden_layers': 1, 'width': 64, 'weight_decay': 0.0001} | 56.118 | 197 |
| {'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0} | 56.092 | 112 |
| {'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0001} | 56.048 | 112 |
| {'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0} | 56.107 | 54 |
| {'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0001} | 56.102 | 54 |
- 선택: **{'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0001}**, 검증 MAE=56.048, 에폭=112

#### personalized

**8a. MLP(직접 예측, L1)**:
| 조합 | 검증 MAE | 선택된 에폭 |
|---|---|---|
| {'n_hidden_layers': 1, 'width': 32, 'weight_decay': 0.0} | 53.000 | 200 |
| {'n_hidden_layers': 1, 'width': 32, 'weight_decay': 0.0001} | 53.250 | 200 |
| {'n_hidden_layers': 1, 'width': 64, 'weight_decay': 0.0} | 52.163 | 200 |
| {'n_hidden_layers': 1, 'width': 64, 'weight_decay': 0.0001} | 52.166 | 200 |
| {'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0} | 52.146 | 101 |
| {'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0001} | 52.167 | 101 |
| {'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0} | 52.056 | 79 |
| {'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0001} | 52.097 | 61 |
- 선택: **{'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0}**, 검증 MAE=52.056, 에폭=79

**8b. MLP(median5+잔차, L1)**:
| 조합 | 검증 MAE | 선택된 에폭 |
|---|---|---|
| {'n_hidden_layers': 1, 'width': 32, 'weight_decay': 0.0} | 51.882 | 193 |
| {'n_hidden_layers': 1, 'width': 32, 'weight_decay': 0.0001} | 51.882 | 193 |
| {'n_hidden_layers': 1, 'width': 64, 'weight_decay': 0.0} | 51.824 | 142 |
| {'n_hidden_layers': 1, 'width': 64, 'weight_decay': 0.0001} | 51.822 | 142 |
| {'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0} | 51.927 | 101 |
| {'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0001} | 51.928 | 101 |
| {'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0} | 51.882 | 63 |
| {'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0001} | 51.867 | 63 |
- 선택: **{'n_hidden_layers': 1, 'width': 64, 'weight_decay': 0.0001}**, 검증 MAE=51.822, 에폭=142

## 3. 최종 재학습 (선택된 하이퍼파라미터로 전체 학습 데이터 재학습, 5개 시드)
검증으로 정한 에폭 수만큼, 전체 train(cold-start)/history(personalized) 데이터로 다시 학습한다(더 이상 조기종료 없음 — 에폭 수 자체가 이미 검증으로 정해졌으므로). 같은 설정을 시드 5개로 반복해 test MAE 평균/표준편차를 본다.

#### cold-start

**8a. MLP(직접 예측, L1)** (하이퍼파라미터={'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0}, 에폭=131): 시드별 test MAE=[59.577, 59.6, 59.669, 59.582, 59.612]
- **평균=59.608, 표준편차=0.033** (대표 시드=42를 이하 통합표/CI에 사용)

**8b. MLP(median5+잔차, L1)** (하이퍼파라미터={'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0001}, 에폭=112): 시드별 test MAE=[59.484, 59.223, 59.444, 59.374, 59.371]
- **평균=59.379, 표준편차=0.089** (대표 시드=42를 이하 통합표/CI에 사용)

#### personalized

**8a. MLP(직접 예측, L1)** (하이퍼파라미터={'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0}, 에폭=79): 시드별 test MAE=[53.851, 53.852, 53.77, 53.675, 53.715]
- **평균=53.773, 표준편차=0.071** (대표 시드=42를 이하 통합표/CI에 사용)

**8b. MLP(median5+잔차, L1)** (하이퍼파라미터={'n_hidden_layers': 1, 'width': 64, 'weight_decay': 0.0001}, 에폭=142): 시드별 test MAE=[53.457, 53.434, 53.505, 53.485, 53.507]
- **평균=53.478, 표준편차=0.028** (대표 시드=42를 이하 통합표/CI에 사용)

**MLP 두 형태 중 주 결과(검증 MAE 기준)**: cold-start=`8b. MLP(median5+잔차, L1)`, personalized=`8b. MLP(median5+잔차, L1)`

## 4. 평가

### 통합표

#### cold-start
| 방법 | MAE(초/km) | RMSE(초/km) | #users | #eval sessions | #merges |
|---|---|---|---|---|---|
| 1. 최근 5회 평균 | 70.804 | 270.771 | 149 | 21977 | 4187 |
| 2. last-session | 83.012 | 326.977 | 149 | 21977 | 4187 |
| 3. median-5 | 62.219 | 273.952 | 149 | 21977 | 4187 |
| 4. 사용자 전체 평균 | 68.771 | 261.865 | 149 | 21977 | 4187 |
| 5. 단순 선형 추세 | 94.244 | 337.516 | 149 | 21977 | 4187 |
| 7단계 주결과 GBM(gbt_lib) | 59.664 | 284.865 | 149 | 21977 | 4187 |
| 8a. MLP(직접 예측, L1) | 59.582 | 270.858 | 149 | 21977 | 4187 |
| 8b. MLP(median5+잔차, L1) | 59.374 | 270.872 | 149 | 21977 | 4187 |

#### personalized
| 방법 | MAE(초/km) | RMSE(초/km) | #users | #eval sessions | #merges |
|---|---|---|---|---|---|
| 1. 최근 5회 평균 | 63.352 | 253.237 | 742 | 20628 | 3254 |
| 2. last-session | 74.933 | 303.985 | 742 | 20628 | 3254 |
| 3. median-5 | 56.449 | 254.192 | 742 | 20628 | 3254 |
| 4. 사용자 전체 평균 | 63.435 | 250.532 | 742 | 20628 | 3254 |
| 5. 단순 선형 추세 | 84.243 | 306.244 | 742 | 20628 | 3254 |
| 7단계 주결과 GBM(gbt_lib) | 53.600 | 253.092 | 742 | 20628 | 3254 |
| 8a. MLP(직접 예측, L1) | 53.675 | 245.306 | 742 | 20628 | 3254 |
| 8b. MLP(median5+잔차, L1) | 53.485 | 245.790 | 742 | 20628 | 3254 |

### 신뢰구간 (사용자 단위 cluster bootstrap, 2000회, seed=42)

#### cold-start
| 방법 | MAE | MAE_95%CI_하한 | MAE_95%CI_상한 | RMSE | RMSE_95%CI_하한 | RMSE_95%CI_상한 |
|---|---|---|---|---|---|---|
| 1. 최근 5회 평균 | 71.250 | 57.252 | 89.328 | 267.440 | 161.381 | 374.568 |
| 2. last-session | 83.453 | 68.970 | 102.154 | 323.383 | 203.071 | 450.678 |
| 3. median-5 | 62.542 | 50.726 | 78.003 | 269.909 | 157.143 | 384.139 |
| 4. 사용자 전체 평균 | 69.230 | 54.860 | 87.293 | 258.877 | 157.533 | 363.271 |
| 5. 단순 선형 추세 | 94.822 | 77.362 | 117.675 | 333.886 | 209.692 | 464.977 |
| 7단계 주결과 GBM(gbt_lib) | 59.984 | 48.193 | 75.706 | 280.085 | 157.942 | 404.053 |
| 8a. MLP(직접 예측, L1) | 59.892 | 48.562 | 75.064 | 266.918 | 156.472 | 379.840 |
| 8b. MLP(median5+잔차, L1) | 59.681 | 48.333 | 74.779 | 266.933 | 156.274 | 379.761 |

#### personalized
| 방법 | MAE | MAE_95%CI_하한 | MAE_95%CI_상한 | RMSE | RMSE_95%CI_하한 | RMSE_95%CI_상한 |
|---|---|---|---|---|---|---|
| 1. 최근 5회 평균 | 63.230 | 56.390 | 70.765 | 250.965 | 188.448 | 314.728 |
| 2. last-session | 74.794 | 67.145 | 83.179 | 301.045 | 223.607 | 385.019 |
| 3. median-5 | 56.354 | 50.524 | 62.938 | 251.827 | 186.403 | 316.355 |
| 4. 사용자 전체 평균 | 63.350 | 56.561 | 70.856 | 248.200 | 184.466 | 312.848 |
| 5. 단순 선형 추세 | 84.082 | 75.164 | 94.533 | 303.361 | 226.996 | 385.312 |
| 7단계 주결과 GBM(gbt_lib) | 53.511 | 47.814 | 59.967 | 250.707 | 185.336 | 315.405 |
| 8a. MLP(직접 예측, L1) | 53.591 | 48.073 | 59.795 | 242.843 | 181.757 | 305.633 |
| 8b. MLP(median5+잔차, L1) | 53.399 | 47.935 | 59.564 | 243.315 | 181.619 | 306.374 |

### 쌍체 차이 95% CI (MLP 주 결과 vs median-5, vs 7단계 GBM)

#### cold-start — MLP 주 결과: 8b. MLP(median5+잔차, L1)
- **MAE: median-5 - MLP** = 2.861 (95% CI [2.178, 3.597]) -> 0 미포함, MLP가 유의하게 나음 (윈도우 풀링 MAE 기준, 22피처 기준; 사용자별 MAE 평균(macro)에서는 일부 조건에서 CI가 0을 포함하고, cold-start GBM22처럼 점추정이 오히려 높은 경우가 있음 — `docs/same_scope_comparison.md` 7-3절)
- **MAE: GBM - MLP** = 0.304 (95% CI [-0.361, 1.203]) -> 0 포함, 유의한 차이 없음
- **RMSE: GBM - MLP** = 13.152 (95% CI [0.869, 26.030]) -> 0 미포함, MLP가 유의하게 나음 (**주의: CI 하한 0.869이 0에 매우 가깝고 구간 폭이 25.2로 넓다** — 통계적으로는 유의하지만 경계선에 걸쳐 있다는 점을 같이 봐야 한다)
  - 이 RMSE 차이의 원인(예: GBM이 극단 페이스 윈도우에 더 민감한지)은 별도로 검증하지 않았다 — **추정/미확인**으로만 남긴다.

#### personalized — MLP 주 결과: 8b. MLP(median5+잔차, L1)
- **MAE: median-5 - MLP** = 2.955 (95% CI [2.174, 3.849]) -> 0 미포함, MLP가 유의하게 나음 (윈도우 풀링 MAE 기준, 22피처 기준; 사용자별 MAE 평균(macro)에서는 일부 조건에서 CI가 0을 포함하고, cold-start GBM22처럼 점추정이 오히려 높은 경우가 있음 — `docs/same_scope_comparison.md` 7-3절)
- **MAE: GBM - MLP** = 0.112 (95% CI [-0.433, 0.806]) -> 0 포함, 유의한 차이 없음
- **RMSE: GBM - MLP** = 7.392 (95% CI [-3.803, 20.858]) -> 0 포함, 유의한 차이 없음

## 5. 해석
- **cold-start**: MLP(8b. MLP(median5+잔차, L1))가 7단계 GBM을 MAE 신뢰구간 밖에서 **이기지 못함**. (diff 95% CI [-0.361,1.203]) 원인은 **추정/미확인**이다(가능성: 이 피처셋에서는 트리 기반 비선형 분할이 MLP의 매끄러운 함수 근사보다 유리할 수 있음, 또는 하이퍼파라미터 격자가 충분히 넓지 않았을 수 있음 — 둘 다 검증 안 함).
- **personalized**: MLP(8b. MLP(median5+잔차, L1))가 7단계 GBM을 MAE 신뢰구간 밖에서 **이기지 못함**. (diff 95% CI [-0.433,0.806]) 원인은 **추정/미확인**이다(가능성: 이 피처셋에서는 트리 기반 비선형 분할이 MLP의 매끄러운 함수 근사보다 유리할 수 있음, 또는 하이퍼파라미터 격자가 충분히 넓지 않았을 수 있음 — 둘 다 검증 안 함).

**MAE와 RMSE 결론을 따로 서술**: 위 쌍체 차이 표의 RMSE 비교가 MAE 비교와 다를 수 있다 — RMSE는 상위 오차 윈도우(극단 페이스 등)에 민감한 지표이므로 MAE 기준 결론과 RMSE 기준 결론을 섞어서 읽지 않는다.

## 6. 분할기 변경에 따른 민감도 (cold-start 검증 fold 구성)
cold-start의 하이퍼파라미터 선택은 train 사용자 5-fold GroupKFold의 첫 fold를 검증으로 쓴다. 이 fold 구성은 분할기에 따라 다르다: 처음 실행(커밋 00c27d1)은 sklearn GroupKFold(그룹 크기 기반 결정적 분할), 재현성 고정 후에는 ridge_lib의 GroupKFold(시드 42 셔플 후 균등분할). **둘 다 같은 사용자가 fold 양쪽에 걸치지 않는 유효한 CV 선택이며, 어느 쪽이 더 낫다고 판단하지 않는다.** 아래는 같은 코드·같은 격자에서 fold 구성만 바뀐 효과다 (personalized는 시간순 분할이라 영향 없음, 값 불변).

| 분할기 | 선택 하이퍼파라미터 | test MAE(대표 시드 42) | 5시드 평균±표준편차 | median-5 대비 개선폭 [95% CI] | GBM−MLP [95% CI] |
|---|---|---|---|---|---|
| sklearn GroupKFold (커밋 00c27d1 시점, 문서 기록값) | {'n_hidden_layers': 2, 'width': 64, 'weight_decay': 0.0}, 에폭 97 | 59.578 | 59.514±0.050 | 2.661 [1.865, 3.535] | 0.104 [-0.560, 0.941] |
| ridge_lib GroupKFold (현재, 고정) | {'n_hidden_layers': 2, 'width': 32, 'weight_decay': 0.0001}, 에폭 112 | 59.374 | 59.379±0.089 | 2.861 [2.178, 3.597] | 0.304 [-0.361, 1.203] |

**기록**: fold 구성이 바뀌자 선택된 구조·에폭이 달라졌고 cold-start MLP의 test MAE가 약 0.2초/km 움직였다. 두 설정 모두에서 MLP는 median-5보다 유의하게 낫고 GBM과는 구분되지 않는다(위 CI 기준) (윈도우 풀링 MAE 기준, 22피처 기준; 사용자별 MAE 평균(macro)에서는 일부 조건에서 CI가 0을 포함하고, cold-start GBM22처럼 점추정이 오히려 높은 경우가 있음 — `docs/same_scope_comparison.md` 7-3절). 이 0.2초/km는 하이퍼파라미터 선택이 검증 fold 구성에 민감하다는 정도를 보여주는 값이며, GBM은 같은 격자·같은 선택이 유지됐다(7단계 문서). 이 변동의 원인 분해(검증 fold 데이터 차이 vs 조기종료/에폭 변화)는 하지 않았다 — **추정/미확인**.
