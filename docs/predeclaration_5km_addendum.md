# 사전 선언 보충 (3fad402 이후, 모델 결과 이전)

작성: 10월 5주차(학기 9주차). 이 문서는 `docs/predeclaration_5km_final.md`(커밋 3fad402)를 바꾸지 않고 보충한다. **모델 학습·평가 결과를 보기 전에 작성한다.**
이 시점까지 본 5km 과제 결과는 개수(`docs/dataset_counts_5km.md`)와 train 내부 기준치 B0~B2의 결과(`docs/baseline_results_train.md`, 커밋 337e6d9)뿐이다. test 사용자의 정답 y와 T5는 계산하지 않았다.
시각은 "start_time 원값(기준 미확인)"이며 상관관계까지만 주장한다.

1. **구현 세부(337e6d9에서 적용, 확정)**
   - 평균 심박·평균 페이스 = 직전 28일 창 안 세션 값의 단순 평균.
   - 주별 거리 4개 = 종료 시각 기준으로 창의 오래된 구간부터 7일씩.
   - 마지막 달리기 후 일수 = 창 안 가장 늦은 종료 시각부터 t까지(일, 소수).
   - m = 종료 시각 < t인 비병합·비극단 5km 후보 전체의 T5 중앙값(기준 기록 포함). 1개뿐이면 m = T5(기준 기록).
2. **성별 부호화** = one-hot 3범주(male, female, unknown).
3. **비교 상대 = B2로 고정.** 근거: train GroupKFold(5, seed 42)에서 사용자 평균 MAE 점추정이 가장 낮다(209.3초; B1 211.1초, B0 243.4초; `docs/baseline_results_train.md`). B2와 B1의 차이는 CI가 0을 포함해 구분되지 않았으므로(B2−B1 −1.8초 [−6.2, +2.4]), 이는 "점추정 최저"에 따른 사전 고정이며 B2가 B1보다 낫다는 주장이 아니다. 통과 기준은 **모델 − B2**로 판정한다(통계적 통과 = 95% CI 상한 < 0, 실용적 통과 = 점추정 개선 ≥ 10초, 둘을 따로 보고).
4. **모델 손실 함수 = 기존 구현 그대로.** 코드에서 확인한 내용:
   - Ridge(`ridge_lib.py`, `RidgePipeline`): 제곱오차 + L2 벌점(alpha). 표준화(`ManualStandardScaler`)를 학습 폴드에서만 적합, 절편은 벌점 없음, 닫힌 해(closed-form). alpha 격자 [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5](`25_tree_models.py` `RIDGE_ALPHA_GRID`).
   - GBM(`gbt_lib.py`, `SimpleGradientBoostingRegressor`): `loss="absolute_error"`(절대오차) — 초기값 = y의 중앙값, 분할 탐색의 기울기 = 잔차 부호, 리프 값 = 잔차 중앙값. `min_samples_leaf=20`, `n_bins=32` 고정, 격자 learning_rate∈{0.03, 0.1} × max_iter∈{100, 300} × max_depth∈{2, 3}(8개; `25_tree_models.py` `HGB_PARAM_GRID`). `squared_error` 변형과 median-5+잔차 변형은 이 과제에서 쓰지 않는다.
   - MLP(`27_mlp_torch.py`): `torch.nn.L1Loss`(절대오차), Adam(lr 1e-3), 배치 2048, 최대 200에폭, 검증 MAE patience 15, 입력 표준화는 학습 자료에서만 적합. 격자 hidden_layers∈{1, 2} × width∈{32, 64} × weight_decay∈{0, 1e-4}(8개; `MLP_PARAM_GRID`).
   - 타깃은 모든 모델에서 y 자체(기준 기록 대비 변화, 초)이며 median-5+잔차 같은 변환은 쓰지 않는다.
5. **선택 편향 명시**: 하이퍼파라미터 선택과 성능 추정에 같은 GroupKFold(5, seed 42)를 쓰므로 train 내부 점수는 낙관적으로 치우칠 수 있다(선택된 설정의 폴드 밖 예측 점수를 그대로 보고하며, 별도 바깥 검증 층이 없다). **공식 판정은 test 1회 평가로 한다.** train 내부 결과는 참고용이다.
6. **추가 비교(+평균 페이스, +성별)는 Ridge·GBM·MLP 세 모델 모두에 같은 방식으로 적용한다**(같은 격자, 같은 선택 기준, 같은 폴드).
7. **personalized 평가 자료**(평가 기준 시점 442개, 그중 test 사용자 78개 포함; `docs/dataset_counts_5km.md` 3절)는 cold-start test와 함께 **마지막에 1회만** 평가한다.
8. **test·분할 밖 사용자의 T5**는 극단 판정(T5 > 5,000초) 개수 산출을 위해 임시 계산한 뒤 폐기했고, 값은 저장·출력·열람하지 않았다(`docs/results_viewed_log.md`에도 기록).
9. **환경 기록**: `39_baselines.py` 첫 실행에서 scipy DLL 로드 오류(Windows App Control) 1회가 있었고, 재실행 후 결과가 같았다(`docs/baseline_results_train.md`가 재실행 값).
