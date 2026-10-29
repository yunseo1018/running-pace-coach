# 결과 열람 기록 (results viewed log)

목적: 어떤 자료에서 어떤 **성능 결과**를 이미 봤는지 남겨, 이후 사전 선언·모델 선택·최종 확인이 열람한 자료에 오염되지 않았는지 따질 근거로 쓴다.
작성: 10월 5주차(학기 9주차). 작성 근거는 저장소 문서(`docs/`)와 커밋 이력이며, 아래 표에 없는 열람은 **확인 불가**다(대화 중 임시 출력 등). 새 열람이 생기면 이 표에 추가한다.

## 1. 새 타깃(5km 기록 변화)

| 항목 | 상태 |
|---|---|
| 5km 완주 시간, 기록 변화량(미래 − 직전), 5km 페이스의 분포·평균·모델 오차 | **어떤 자료에서도 계산·열람하지 않았다.** 근거: `docs/week4_data_audit.md`·`docs/week5_target_check.md`는 개수와 가능 여부만 싣고(스크립트가 5km 도달 시각 자체를 계산·저장하지 않음), `docs/week4_predeclaration_5km.md`는 결과 열람 없는 초안이다. |
| 새 타깃에 대한 baseline·모델 점수 | 없음(미실행). |
| 개수 점검(열람함) | 5km 세션 수, 기준 시점 수, 사용자당 개수 분포, 공백 구간별 쌍 수, 기준 시점 이동별 개수, 점 간 시간 간격(타임스탬프 간격이며 기록 시간이 아님). 사전 선언 항목(D1·D8 등)의 선택지 근거로 쓰인 것은 개수뿐이다. |

## 2. 이전 과제(세션 단위 다음 러닝 평균 페이스 예측)에서 이미 본 결과

이 과제는 타깃이 다르지만 **같은 사용자 집단**에서 나왔고, 아래 결과를 이미 열람했다. 새 과제의 "최종 확인용 자료"를 정할 때 이 사실을 감안한다.

| 자료 | 열람한 결과 | 근거 문서 |
|---|---|---|
| cold-start test 사용자(분할 파일 기준 150명, 6세션 이상 149명; 평가 윈도우 21,977개) | median-5, 이력 기반 baseline, Ridge, GBM(직접 구현), MLP의 MAE·RMSE, 사용자 단위 bootstrap CI, 사용자별 MAE 분포, history 길이별·거리 구간별(사후)·페이스 구간별 세분화, 코칭 판정 불일치율, 극단 세션 민감도 | `docs/baseline_results.md`, `docs/ridge_results.md`, `docs/tree_results.md`, `docs/mlp_results.md`, `docs/model_comparison.md`, `docs/same_scope_comparison.md`, `docs/coaching_results.md`, `docs/coaching_abstain_results.md`, `docs/extreme_session_window_map.md` |
| cold-start train 사용자(598명; 학습 윈도우 79,669개) | 모델 선택용 사용자 단위 GroupKFold CV 점수와 선택된 하이퍼파라미터(`docs/mlp_selected_hyperparams.json`) | `docs/ridge_results.md`, `docs/tree_results.md`, `docs/mlp_results.md`, `docs/model_comparison.md` |
| personalized 평가(742명; 사용자별 시간 80/20, hold-out 윈도우 20,628개) | 같은 모델들의 hold-out MAE·RMSE, bootstrap CI, 코칭 불일치율 | 위와 같음 |
| 분할 재정의 전후 | 분할기 변경(`ridge_lib` GroupKFold 고정) 전후 MLP cold-start 수치 | `docs/mlp_results.md` 6절 |
| 다중 세션 집계(28일 입력 → 다음 7일) | **가용성 개수만**(기준 시점 수·사용자 수). 모델 학습·baseline 오차는 계산하지 않았다. | `docs/multisession_coaching_design.md`, `34_multisession_availability.py` |

## 3. 분할 사용자 집합과 새 과제의 관계 (결정 대기)

- 새 과제의 cold-start 분할은 기존 분할 파일(`data/train_user_ids.txt` 598명, `data/test_user_ids.txt` 150명)을 재사용할 계획이다. 이 test 사용자들의 **이전 과제 결과는 이미 열람**했으므로, 새 과제의 테스트가 "열람 전 자료"는 아니다. 단 새 타깃의 값은 열람하지 않았다.
- 새 과제용 **최종 확인용 자료**(결과 열람 없이 마지막에 한 번만 평가하는 자료)를 따로 둘지, 둔다면 방식(예: 새로 분리한 사용자 집합, 마지막 시간 구간 등)은 **결정 대기**이며 정해지지 않았다. 5km 평가 대상 사용자 수가 작아(`docs/week4_data_audit.md`: 전체 470명, test 91명) 추가 분리하면 표본이 더 줄어드는 비용이 있다.
- 11/15 테스트 1회 계획(`CLAUDE.md` 일정)은 위 결정이 정해진 뒤 확정한다.

## 4. 기록 규칙
- 새 결과를 열람하면 날짜·자료·열람한 지표를 이 표에 추가한다. 열람한 뒤에는 그 자료로 모델·기준치·판정 기준을 바꾸지 않는다(바꾸면 "탐색적 변경"으로 표기).
- 테스트 점수로 모델을 선택하지 않는다.
