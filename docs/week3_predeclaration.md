# 10월 3주차 2·3번 사전 선언 (실행 전 작성, 결과 확인 후 변경하지 않음)

이 문서는 `31_same_scope_comparison.py`를 **실행하기 전에** 작성했다. 아래 후보·지표·비교쌍·선택 규칙·중단 조건은 결과를 본 뒤 바꾸지 않는다. 새 모델 종류는 추가하지 않는다(기존 Ridge/GBM/MLP의 입력 피처 변형만). 모델 선택은 기존과 같은 CV(`ridge_lib` GroupKFold, 시드 42)와 내부 검증으로만 하고 **테스트 점수로 고르지 않는다**.

## 1. 평가 조건과 cold-start 정의
- cold-start: 사용자 단위 80/20 분할(03단계 배정 재사용). **학습에 없던 사용자이지만 타깃 이전 기록이 5회 이상 있는 조건**이다(전체 이력 중앙값 피처는 5회보다 더 먼 과거까지 사용). 기록이 전혀 없는 신규 사용자는 평가에 없다.
- personalized: 사용자별 시간순 history/hold-out(06단계와 동일 80/20).
- 평가 윈도우는 7~9단계와 같은 24단계 확장 피처 윈도우(cold-start test 149명/21,977, personalized hold-out 742명/20,628).

## 2. 비교 그룹 (그룹 안에서만 순위·유의성 보고, **그룹을 가로지르는 순위는 매기지 않는다**)
**그룹 (a) 최근 5회만 쓰는 방법**: median-5, Ridge15, GBM15, MLP15.
**그룹 (b) 타깃 이전 전체 이력을 쓰는 방법**: 전체 이력 중앙값, (d) 0.5×median-5+0.5×전체 이력 중앙값(탐색적 baseline), Ridge22, GBM22, MLP22.
- (d)는 0.5/0.5 튜닝 없는 고정 상수이나 9단계 결과를 본 뒤 추가한 탐색적 baseline이다.
- (b)의 median-5는 비교 참고용으로 표에 함께 두되 그룹 (a)의 구성원이며 (b)의 순위에는 넣지 않는다. 그룹 (b)의 기준(reference)은 전체 이력 중앙값이다.

## 3. 후보 정의와 선택 규칙 (신규 학습이 필요한 것: Ridge15, GBM15, MLP15)
- **입력**: 15피처 = 과거 5세션 × (실제 거리, 평균 페이스, 평균 심박) (`X[:, :15]`). 22피처 = 15 + 7개 요약 피처(전체 이력 중앙값·누적 세션 수 포함).
- **Ridge15**: `ridge_lib.RidgePipeline`, alpha는 기존 격자 `RIDGE_ALPHA_GRID`(10개)에서 cold-start는 train 사용자 GroupKFold(5-fold, 시드 42) 평균 CV MAE, personalized는 시간순 inner-split 검증 MAE가 최소인 값. (기존 종합표 6a는 22피처용 alpha=1e4를 15피처에 썼다 — 참고용 "레거시 6a"로 별도 표시하고 값은 바꾸지 않는다.)
- **GBM15**: `gbt_lib` 직접 구현, 기존 격자 `HGB_PARAM_GRID` 8개(learning_rate∈{0.03,0.1}, max_iter∈{100,300}, max_depth∈{2,3}), min_samples_leaf=20, n_bins=32 고정. 두 형태(absolute_error 직접 예측 / median5+잔차)별로 격자를 CV(cold-start GroupKFold 5-fold 평균 MAE, personalized 시간순 inner-split MAE)로 선택하고, 두 형태 중 CV MAE가 더 낮은 쪽을 주결과로 쓴다(7단계와 동일 규칙).
- **MLP15**: 기존 27번 코드(`select_mlp_hparams`, `train_mlp_fixed_epochs`) 재사용, 격자 8개(은닉층 {1,2}×폭 {32,64}×weight decay {0,1e-4}), 두 형태(직접/median5+잔차)별로 검증 MAE(cold-start는 ridge_lib GroupKFold 첫 fold, personalized는 시간순 inner-split)로 선택, 두 형태 중 검증 MAE가 낮은 쪽이 주결과. 선택 시드 42, 최종 재학습은 전체 학습 자료에서 선택된 에폭 수로 고정, 대표 시드 42의 예측으로 평가표를 만들고 5개 시드(1,2,3,42,999) 평균±표준편차를 참고로 병기.
- **그룹 (b)의 기존 방법(Ridge22 alpha=1e4, GBM22, MLP22, 전체 이력 중앙값, (d))**: 재탐색하지 않고 기존 선택을 그대로 쓴다(GBM22 하이퍼파라미터는 `28_model_comparison.py` 상수, MLP22는 `docs/mlp_selected_hyperparams.json`).
- **중단 조건**: 그룹 (b) 방법과 레거시 6a의 MAE/RMSE가 `docs/model_comparison.md`의 값과 소수 셋째 자리에서 다르면 즉시 중단하고 보고한다.

## 4. 지표와 통계 (모든 방법 공통)
- **풀링 MAE(윈도우 평균)**, **풀링 RMSE**, **macro MAE(사용자별 MAE의 평균, 사용자 1명 = 1표)**. macro RMSE는 계산하지 않는다.
- **신뢰구간**: 사용자 단위 cluster bootstrap — 평가 사용자를 복원추출(사용자 수만큼, 2000회, seed 42), 뽑힌 사용자의 **모든 윈도우**를 함께 사용, 모든 방법에 같은 추출을 적용한 쌍체 비교, 95% percentile CI. macro MAE는 뽑힌 사용자(중복 포함)의 사용자별 MAE를 중복 횟수 가중으로 평균한다.
- **비교쌍(결과 확인 후 변경 없음)**: 그룹 안의 **모든 쌍**(a: 6쌍, b: 10쌍)에 대해 풀링 MAE, macro MAE, 풀링 RMSE 차이(앞−뒤, 양수면 뒤가 더 낮음)와 CI. 다중 비교 보정은 하지 않는다(탐색적 쌍 비교 다수 → 개별 CI를 확정 근거로 쓰지 않음).
- **개선폭 표기(분모 구분)**: 그룹 기준 방법(a: median-5, b: 전체 이력 중앙값) 대비 (1) 절대 개선(초/km), (2) 'reference MAE 대비 %', (3) '평가 윈도우 타깃 페이스 중앙값 대비 %'를 각각 구분해 쓴다.
- **유의하지 않은 비교**에는 "CI가 0을 포함 — 동등성(차이가 없음)의 입증이 아님"이라고 쓴다.
- **사용 데이터 수**: 각 표에 평가 사용자 수·윈도우 수·재표집 방법을 본문에 명시한다.

## 5. 무결성 검사 (코드에 assert로 둔다)
- (i) 모든 방법의 예측 길이가 평가 윈도우 수와 같고, 윈도우 순서(사용자 ID, 타깃 세션 ID 열)가 방법 간 동일하다. bootstrap 입력의 사용자별 윈도우 수가 방법 간 동일하다.
- (iv) median-10/20이 쓰는 `session_order` 조밀 인덱스 전제(`session_order == 0..n-1`, 사용자별 시간순)를 assert로 검증한다.

## 6. 산출물
- `docs/history_scope_table.md`: 예측기별 사용 이력 범위 표(파일·줄 번호 포함).
- `docs/same_scope_comparison.md`: 그룹별 결과표, 쌍별 차이 CI, 전/후 비교.
- 스크립트: `31_same_scope_comparison.py`.

## 7. 사후 추가(탐색) — 결과를 본 뒤 추가한 분석 (2번 결과표 확인 후 사용자 요청으로 추가)
위 1~6절은 실행 전에 고정한 사전 선언이다. 아래는 `same_scope_comparison.md`의 결과를 본 **뒤에** 추가한 탐색 분석이며, 사전 선언된 비교쌍이 아니다. 기술통계만 쓰고 인과 해석은 하지 않는다.
- (가) 평가 사용자별 윈도우 수 분포(분위수: 최소, 10/25/50/75/90분위, 최대, 평균).
- (나) 윈도우 수 기준 하/중/상 3분위 사용자 그룹(사용자 수가 같도록 윈도우 수 정렬 후 3등분, 동률은 안정 정렬 순서)별 median-5 대비 GBM15·MLP15의 MAE 차이(풀링·macro, 그룹 내 사용자 단위 bootstrap 2000회 seed 42).
- (다) 풀링 MAE와 macro MAE 결론이 갈린다는 점을 문서 문구에 병기하기 위해, median-5 대비 GBM22·MLP22의 풀링/macro MAE 차이(전체 평가 사용자 기준, 같은 bootstrap)를 확인용으로 추가한다. 이 쌍은 사전 선언(그룹 내 쌍)에 없는 그룹 간 쌍이므로 순위가 아니라 문구 근거 확인용이다.
