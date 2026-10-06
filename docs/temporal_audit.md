# 과거 정보 감사(Temporal Audit) 기록

10월 2주차 2번 작업 산출물. "예측 행(윈도우)마다 입력에 쓰인 마지막 세션의
종료 시각과 타깃 세션 시작 시각을 저장하고, 모든 행에서
`last_input_session_time < 타깃 시작 시각`임을 assert로 검증한다"
(CLAUDE.md v3 leakage 규칙 3번)를 실제로 구현하고 발견한 내용을 기록한다.

## 1. 감사 로직

`04_build_xy_dataset.py`(cold-start)와 `06_personalized_split.py`
(personalized) 둘 다, 윈도우를 만들 때마다:

- `last_input_session_time` = 과거 5개 세션 중 가장 늦은 세션의
  `start_time + duration_sec` (그 세션이 끝난 시각)
- `target_start_time` = 타깃(6번째) 세션의 `start_time`

을 계산해 샘플에 같이 저장하고, `last_input_session_time < target_start_time`
을 확인한다. 이 두 값은 데이터셋 파일(`train_dataset.pkl` 등)에 그대로
저장되므로, 이후 다른 스크립트(07단계 등)에서도 재검증할 수 있다.

## 2. 발견: 일부 세션의 시간 겹침 (실제 데이터 문제)

처음 이 감사를 켰을 때 **2.45%의 윈도우(전체 110,425개 중 2,700개,
748명 중 227명 영향)**에서 검증이 실패했다. 최대 겹침은 17,696초
(약 4.9시간)에 달했다.

**원인: 추정(미확인)**. 1단계에서 이미 발견했던 "`actual_distance_km`은
짧은데 `duration_sec`만 비정상적으로 긴" 세션들(워치를 켜놓은 채 거의
움직이지 않은 세션 등, 극단적인 `avg_pace_sec_per_km`를 만드는 바로 그
세션들)이 `duration_sec`을 과도하게 늘려 다음 실제 세션이 이미 시작된
뒤까지도 "안 끝난 것"처럼 계산되는 것 아닐까 추정하고 있으나, 실제로
확인하지는 않았다. 세션 쪼개짐/겹쳐 기록된 기기 등 다른 원인이 섞여
있을 가능성도 있다 — 추가 조사는 하지 않았으므로 원인은 미확인 상태로
남겨둔다.

**처리 방식 (사용자 확인 후 결정)**: 이 2,700개 윈도우는 조용히 넘기지
않고 **train/test/personalized 데이터셋 생성 단계에서 전부 제외**한다.
`build_windows_for_user()`가 감사에 실패한 윈도우를 건너뛰고(`continue`),
제외 건수를 반환해 `main()`이 투명하게 출력한다.

| 데이터셋 | 제외 전 윈도우 | 제외 건수 | 제외 후 |
|---|---|---|---|
| cold-start train | - | 1,955 | 84,125 |
| cold-start test | - | 745 | 23,600 |
| personalized (train+test 합계) | 110,425 | 2,700 | 107,725 |

제외 후 남은 모든 샘플에서 `last_input_session_time < target_start_time`이
성립함을 재확인했다 (04/06단계 실행 출력 참고).

## 3. baseline 4(사용자 전체 평균)의 추가 감사

baseline 4는 과거 5개보다 더 이전 세션까지 참조하므로, 과거 5개만 보는
감사(위 2번)와는 별도로 검증이 필요하다. `07_baseline_comparison.py`의
`build_user_overall_avg_lookup()`에서 각 (사용자, 타깃) 쌍에 대해:

- `cum_mean_before` (타깃 이전 모든 세션의 페이스 평균) 뿐 아니라
- `cum_max_time_before` (그 평균을 계산하는 데 참조한 세션 중 가장 늦은
  `start_time`)

을 같이 계산해, `evaluate_split()`에서 `cum_max_time_before <
target_start_time`을 매 샘플마다 assert로 검증한다 (전부 통과).

baseline 5(단순 선형 추세)는 과거 5개 세션만 쓰므로 위 2번 감사로 이미
커버되며, `07_baseline_comparison.py`에서도 `sample["last_input_session_time"]
< sample["target_start_time"]`을 다시 한번 명시적으로 assert해 재확인한다.

## 4. 허용 학습 자료 (조건별 명시)

- **Cold-start** (`03_train_test_split.py`가 고정한 `train_user_ids`/
  `test_user_ids`): 평가(test)에는 `test_user_ids`의 기록만 쓴다. baseline
  1~5는 fit이 없는 규칙 기반이라 "학습 자료"가 현재는 실질적으로 쓰이지
  않지만, 이후 Ridge 등 실제로 fit이 필요한 모델이 붙으면
  **`train_dataset.pkl`(= train 사용자의 기록)만 fit에 사용하고,
  `test_dataset.pkl`(= test 사용자의 기록)은 평가에만 사용**해야 한다
  (처음 보는 사용자 일반화를 보는 것이 cold-start의 목적이므로, test
  사용자의 기록이 fit에 섞이면 그 목적 자체가 깨진다).
- **Personalized** (`06_personalized_split.py`가 만든 history/hold-out):
  각 사용자의 `personalized_train_dataset.pkl`(=history, 시간순 앞부분
  80%)만 그 사용자의 "학습 자료"로 허용되고, `personalized_test_dataset.pkl`
  (=hold-out)의 정보가 같은 사용자의 history 쪽 계산에 섞이면 안 된다.
  baseline 4(사용자 전체 평균)는 "타깃 이전의 모든 세션"을 참조하는데,
  history/hold-out이 시간순으로 잘려 있어서(앞부분=history, 뒷부분=
  hold-out) 어떤 hold-out 타깃을 보더라도 그 이전 세션은 전부 history
  구간에 속한다 — 즉 위 3번 감사가 이 경계도 자동으로 지켜주고 있음을
  확인했다.

## 5. 감사 반영 후 baseline 5종 MAE/RMSE (최종)

(2,700개 윈도우 제외 후, 과거 정보 감사 전부 통과 상태에서 재평가)

| baseline | cold-start MAE | cold-start RMSE | personalized MAE | personalized RMSE |
|---|---|---|---|---|
| 1. 최근 5회 평균 | 74.211 | 276.458 | 67.556 | 263.019 |
| 2. last-session | 86.267 | 335.568 | 79.922 | 316.706 |
| 3. median-5 | 65.085 | 278.601 | 59.893 | 260.660 |
| 4. 사용자 전체 평균 | 71.024 | 266.970 | 66.148 | 257.022 |
| 5. 단순 선형 추세 | 99.265 | 346.507 | 89.804 | 329.009 |

감사 전(2,700개 포함) 수치와 비교하면, MAE 기준 순위(median-5가 cold-start/
personalized 둘 다 최저)는 바뀌지 않았다. RMSE 기준으로는 cold-start는
순위가 그대로지만, **personalized RMSE에서 2~3위 순서가 바뀌었다** —
감사 전에는 1번(최근 5회 평균, 267.639)이 3번(median-5, 269.838)보다
낮았는데, 감사 후에는 3번(260.660)이 1번(263.019)보다 낮아졌다. 1위(사용자
전체 평균)와 나머지 순위는 바뀌지 않았다. 전체적으로 절대값 차이는 1~2
단위 수준으로 작지만, 이 한 곳에서는 순위가 실제로 바뀌었다는 점을
정확히 기록해둔다. 이 2.45%의 윈도우(원인 미확인)가 baseline 전체 결론을
뒤집을 정도는 아니었지만, 예측 시점에 없어야 할 정보가 섞인 채로
보고하는 것 자체가 문제이므로 제외하는 것이 맞다고 판단했다.
