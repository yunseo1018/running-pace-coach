# 합성 예시 사례표 (시연 `demo/coach_demo.py`)

생성: `python demo/make_demo_cases.py`(저장소 루트). `demo/examples/`의 직접 만든 합성 예시 ①~⑰을 기준 시점 2025-03-03으로 **실제로 실행**해 프로그램 반응 칸의 문구를 그대로 옮겼다. 다시 생성한 결과가 이 파일과 같은지는 테스트(`test_demo_cases_doc_matches_regeneration`)와 `python demo/make_demo_cases.py --check`로 확인한다. 실제 데이터와 무관하다. B2 계산식·파라미터·명목 80% 범위는 이 정리에서 바꾸지 않았다. 모든 예측은 "다음 7일 안에 5km 기록이 관측된 경우"에 조건부다.

분류: 정상 예측 / 조건별 안내 / 예측 불가(자격 미달) / 입력 오류 / 목표 비교 / 뛴 다음의 피드백. 한 예시가 여러 분류에 걸쳐도 대표 분류 하나를 적었다(실제 반응 칸에 안내가 함께 나온다).

## 1. 예시 ①~⑰

| 번호 | 예시 파일 | 분류 | 입력 조건 | 프로그램 반응(실제 출력의 문구) | 해당 테스트 |
|---|---|---|---|---|---|
| ① | `ex1_steady.csv` | 정상 예측 | 8주, 5km 4회(평소 약 25분), 마지막 5km도 평소 수준, stop_sec 열 없음 | 예측 가능: 다음 7일 안 5km 예상 기록: 24:41 / 명목 80% 범위 21:19 ~ 29:19<br>안내: 과거 유효 5km가 2~4개뿐 (교차검증 포함률 73.1%)<br>안내: 정지 시간 정보가 없어 이 조건은 확인하지 않았습니다<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고] | `test_example_outputs_match_live_runs_and_cover_each_stratum`<br>`test_two_term_split_and_coverage_text` |
| ② | `ex2_slower_last.csv` | 정상 예측 | 마지막 5km만 평소보다 약 150초 느림 | 예측 가능: 다음 7일 안 5km 예상 기록: 25:22 / 명목 80% 범위 22:00 ~ 30:00<br>안내: 과거 유효 5km가 2~4개뿐 (교차검증 포함률 73.1%)<br>안내: 직전 기록이 평소와 2분 넘게 차이 (교차검증 포함률 67.8%)<br>안내: 정지 시간 정보가 없어 이 조건은 확인하지 않았습니다<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고] | `test_example_outputs_match_live_runs_and_cover_each_stratum`<br>`test_two_term_split_and_coverage_text` |
| ③ | `ex3_no_5km.csv` | 예측 불가(자격 미달) | 달리기는 충분하나 5km 기록 없음 | 상태 표시: [입력 부족]<br>**예측 불가**: 직전 28일 안에 유효한 5km 기록이 없습니다(1회 이상 필요, 1회 부족). | `test_shortfall_shows_what_and_how_many_are_missing` |
| ④ | `ex4_few_runs.csv` | 예측 불가(자격 미달) | 달리기 2회, 5km 기록이 5.0~5.5km 범위 밖(5.8km) | 상태 표시: [입력 부족]<br>**예측 불가**: 직전 28일 안에 끝난 달리기가 2회입니다(4회 이상 필요, 2회 부족). 직전 28일 안에 유효한 5km 기록이 없습니다(1회 이상 필요, 1회 부족). 5km 기록 1개가 있으나 5.0~5.5km 범위 밖 1개라 제외했습니다. | `test_shortfall_shows_what_and_how_many_are_missing`<br>`test_5km_outside_range_or_extreme_latest_falls_back_to_earlier_valid_record_and_says_so` |
| ⑤ | `ex1_steady.csv`<br>`--goal 20:00` | 목표 비교 | ①과 같은 기록, 목표 20:00 | 예측 가능: 다음 7일 안 5km 예상 기록: 24:41 / 명목 80% 범위 21:19 ~ 29:19<br>안내: 과거 유효 5km가 2~4개뿐 (교차검증 포함률 73.1%)<br>안내: 정지 시간 정보가 없어 이 조건은 확인하지 않았습니다<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고]<br>목표 비교(목표 20:00): 명목 범위보다 빠름<br>예측값과 목표의 차이(예측 − 목표): +281초 (목표가 예측보다 281초 빠름) | `test_goal_three_outcomes_and_boundaries` |
| ⑥ | `ex1_steady.csv`<br>`--goal 24:00` | 목표 비교 | ①과 같은 기록, 목표 24:00 | 예측 가능: 다음 7일 안 5km 예상 기록: 24:41 / 명목 80% 범위 21:19 ~ 29:19<br>안내: 과거 유효 5km가 2~4개뿐 (교차검증 포함률 73.1%)<br>안내: 정지 시간 정보가 없어 이 조건은 확인하지 않았습니다<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고]<br>목표 비교(목표 24:00): 명목 범위 안<br>예측값과 목표의 차이(예측 − 목표): +41초 (목표가 예측보다 41초 빠름) | `test_goal_three_outcomes_and_boundaries` |
| ⑦ | `ex1_steady.csv`<br>`--goal 30:00` | 목표 비교 | ①과 같은 기록, 목표 30:00 | 예측 가능: 다음 7일 안 5km 예상 기록: 24:41 / 명목 80% 범위 21:19 ~ 29:19<br>안내: 과거 유효 5km가 2~4개뿐 (교차검증 포함률 73.1%)<br>안내: 정지 시간 정보가 없어 이 조건은 확인하지 않았습니다<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고]<br>목표 비교(목표 30:00): 명목 범위보다 느림<br>예측값과 목표의 차이(예측 − 목표): −319초 (목표가 예측보다 319초 느림) | `test_goal_three_outcomes_and_boundaries` |
| ⑧ | `ex5_past1.csv` | 조건별 안내 | 과거 유효 5km가 직전 기록 1개뿐(stop_sec 열 있음, 정지 40초) | 예측 가능: 다음 7일 안 5km 예상 기록: 24:45 / 명목 80% 범위 21:23 ~ 29:23<br>안내: 과거 유효 5km가 직전 기록 1개뿐 (교차검증 포함률 70.6%)<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고] | `test_stratum_a_boundaries`<br>`test_example_outputs_match_live_runs_and_cover_each_stratum` |
| ⑨ | `ex6_past3.csv` | 조건별 안내 | 과거 유효 5km 3개 | 예측 가능: 다음 7일 안 5km 예상 기록: 24:45 / 명목 80% 범위 21:23 ~ 29:23<br>안내: 과거 유효 5km가 2~4개뿐 (교차검증 포함률 73.1%)<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고] | `test_stratum_a_boundaries`<br>`test_example_outputs_match_live_runs_and_cover_each_stratum` |
| ⑩ | `ex7_far_from_usual.csv` | 조건별 안내 | 직전 5km가 평소(1500초)보다 180초 느림, 과거 5km 5개 | 예측 가능: 다음 7일 안 5km 예상 기록: 25:22 / 명목 80% 범위 22:00 ~ 30:00<br>안내: 직전 기록이 평소와 2분 넘게 차이 (교차검증 포함률 67.8%)<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고] | `test_stratum_b_boundary_120_seconds`<br>`test_example_outputs_match_live_runs_and_cover_each_stratum` |
| ⑪ | `ex8_long_stop.csv` | 조건별 안내 | 직전 기록 세션의 정지 420초, 과거 5km 5개 | 예측 가능: 다음 7일 안 5km 예상 기록: 24:45 / 명목 80% 범위 21:23 ~ 29:23<br>안내: 직전 기록 세션에 5분 넘는 정지가 있음 (교차검증 포함률 39.1%)<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고] | `test_stratum_e_boundary_and_missing_stop_info`<br>`test_example_outputs_match_live_runs_and_cover_each_stratum` |
| ⑫ | `ex10_multi.csv` | 조건별 안내 | 과거 5km 3개 + 직전이 평소보다 300초 느림 + 정지 400초 | 예측 가능: 다음 7일 안 5km 예상 기록: 25:47 / 명목 80% 범위 22:25 ~ 30:25<br>안내: 과거 유효 5km가 2~4개뿐 (교차검증 포함률 73.1%)<br>안내: 직전 기록이 평소와 2분 넘게 차이 (교차검증 포함률 67.8%)<br>안내: 직전 기록 세션에 5분 넘는 정지가 있음 (교차검증 포함률 39.1%)<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고] | `test_multiple_strata_show_all_lines_in_order_and_width_line_once`<br>`test_example_outputs_match_live_runs_and_cover_each_stratum` |
| ⑬ | `ex9_no_note.csv` | 정상 예측 | 과거 5km 5개, 평소와 차이 0초, 정지 40초(안내 조건 해당 없음) | 예측 가능: 다음 7일 안 5km 예상 기록: 24:45 / 명목 80% 범위 21:23 ~ 29:23<br>조건별 범위 안내 없음 | `test_run_prints_notices_before_prediction_and_no_notice_when_clean`<br>`test_example_outputs_match_live_runs_and_cover_each_stratum` |
| ⑭ | `ex11_trend_smaller_a.csv` | 정상 예측 | 직전 5km가 평소보다 약 7초 빠름(방향 문장 (다)) | 예측 가능: 다음 7일 안 5km 예상 기록: 24:38 / 명목 80% 범위 21:16 ~ 29:16<br>안내: 과거 유효 5km가 2~4개뿐 (교차검증 포함률 73.1%)<br>안내: 정지 시간 정보가 없어 이 조건은 확인하지 않았습니다<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고] | `test_trend_sentence_five_cases_match_final_direction`<br>`test_example_outputs_match_live_runs_and_cover_each_stratum` |
| ⑮ | `ex12_trend_larger_a.csv` | 정상 예측 | 직전 5km가 평소보다 약 40초 빠름(방향 문장 (라)) | 예측 가능: 다음 7일 안 5km 예상 기록: 24:31 / 명목 80% 범위 21:09 ~ 29:09<br>안내: 과거 유효 5km가 2~4개뿐 (교차검증 포함률 73.1%)<br>안내: 정지 시간 정보가 없어 이 조건은 확인하지 않았습니다<br>상태 표시: [조건별 검증 부족: 개발 결과 기준 경고] | `test_trend_sentence_five_cases_match_final_direction`<br>`test_example_outputs_match_live_runs_and_cover_each_stratum` |
| ⑯ | `ex13_feedback_inside.csv`<br>`--feedback` | 뛴 다음의 피드백 | ⑬과 같은 과거 기록 + 기준 시점 2일 뒤 5km 24:30 | 예측 가능: 다음 7일 안 5km 예상 기록: 24:45 / 명목 80% 범위 21:23 ~ 29:23<br>알림: 기준 시점 이후에 끝난 기록 1개는 사용하지 않았습니다(예측 시점에 관측 가능한 기록만 사용).<br>조건별 범위 안내 없음<br>피드백: 실제 기록: 2025-03-05 5km 시간 24:30<br>피드백: 차이(실제 − 예측): −15초 (음수 = 예측보다 빠름)<br>피드백: 명목 범위 대비 위치: 범위 안<br>피드백: 학습 자료에서 예측보다 15초 이상 빨랐던 경우: 46.0% (train 폴드 밖 B2 잔차 기준) | `test_feedback_inside_range_slower_than_prediction`<br>`test_feedback_example_outputs_match_live_runs` |
| ⑰ | `ex14_feedback_faster.csv`<br>`--feedback` | 뛴 다음의 피드백 | ⑬과 같은 과거 기록 + t5 없는 5km 1개 + 기준 시점 3일 뒤 5km 20:30 | 예측 가능: 다음 7일 안 5km 예상 기록: 24:45 / 명목 80% 범위 21:23 ~ 29:23<br>알림: 기준 시점 이후에 끝난 기록 2개는 사용하지 않았습니다(예측 시점에 관측 가능한 기록만 사용).<br>조건별 범위 안내 없음<br>피드백: 실제 기록: 2025-03-06 5km 시간 20:30<br>피드백: 이보다 먼저 뛴 5.0~5.5km 기록 1개는 5km 시간(t5_sec)이 없어 건너뛰었습니다<br>피드백: 차이(실제 − 예측): −255초 (음수 = 예측보다 빠름)<br>피드백: 명목 범위 대비 위치: 범위보다 빠름<br>피드백: 학습 자료에서 예측보다 255초 이상 빨랐던 경우: 7.3% (train 폴드 밖 B2 잔차 기준) | `test_feedback_faster_and_slower_than_range`<br>`test_feedback_skips_runs_without_t5_and_reports_count`<br>`test_feedback_example_outputs_match_live_runs` |

## 2. 입력 오류 (예시 파일 없음: 합성 입력을 임시로 만들어 실행)

입력·옵션 문제는 멈추지 않고 "오류: 이유 / 해결: 방법" 두 줄과 종료 코드 2로 알린다(추적 메시지 없음).

| 번호 | 입력 조건 | 프로그램 반응(실제 출력 전문) | 해당 테스트 |
|---|---|---|---|
| E1 | CSV 파일이 없음 | 오류: 입력 파일을 찾을 수 없습니다: none.csv<br>해결: 파일 경로를 확인하세요(작업 폴더 기준 상대 경로나 전체 경로). 예: python demo/coach_demo.py demo/examples/ex1_steady.csv --asof 2025-03-03T00:00:00 | `test_input_file_missing_empty_and_directory` |
| E2 | 필수 열(duration_sec, t5_sec)이 없음 | 오류: 필요한 열이 없습니다: duration_sec, t5_sec<br>해결: 첫 줄(헤더)에 열 이름 start_time, end_time, distance_km, duration_sec, t5_sec가 모두 있어야 합니다(쉼표로 구분, 엑셀의 세미콜론 저장 주의). | `test_missing_columns_named` |
| E3 | start_time이 엑셀식 날짜 형식(3/3/2025 7:00) | 오류: 2번째 줄의 start_time 또는 end_time 형식을 읽을 수 없습니다.<br>해결: ISO 일시(예: 2025-03-03T07:00:00) 또는 에포크 초(숫자)로 고치세요. 엑셀이 바꾼 날짜 형식(예: 3/3/2025 7:00)은 텍스트로 다시 저장해야 합니다. | `test_time_number_and_order_format_errors_with_line_numbers` |
| E4 | --goal이 mm:ss 형식이 아님(24) | 오류: 목표 형식이 올바르지 않습니다(mm:ss 필요).<br>해결: 분:초로 입력하세요(초는 00~59). 예: --goal 24:30 | `test_goal_format_and_range_errors` |
| E5 | --asof가 현재보다 미래(2999-01-01) | 오류: 기준 시점이 현재 시각보다 미래입니다.<br>해결: 오늘 또는 그 이전 시각으로 지정하세요(미래 기록은 관측할 수 없어 예측 시점으로 쓸 수 없습니다). | `test_future_dated_records_excluded_with_notice_and_future_asof_rejected` |

## 3. 이 프로그램이 하지 않는 것

- **처방**: 훈련·휴식을 권하거나 계획을 제시하지 않는다.
- **확률**: 목표 달성 확률이나 달성 가능 여부를 말하지 않는다(목표 비교와 피드백의 범위 안·밖은 확률이 아니다).
- **원인 설명**: 예측이나 범위 안·밖의 원인을 설명하지 않는다(상관이며 인과가 아니다).
- **실력 변화 판정**: 실력이 늘었다·줄었다고 판정하지 않는다. 범위 안·밖은 실력 변화의 증거가 아니며, 한 번의 기록에는 일상적인 흔들림이 포함된다.

뛴 다음의 피드백 블록(`--feedback`)에는 아래 금지 표현이 없음을 테스트(`test_feedback_has_no_forbidden_expressions`)가 확인한다(사양 [`demo_feedback_spec.md`](demo_feedback_spec.md) 7절).

- 훈련·휴식 권고: "훈련", "휴식", "쉬", "권장", "추천", "하세요", "해야", "늘리", "줄이"
- 실력 변화 판정: "늘었", "줄었", "향상", "좋아졌", "나빠졌", "나아졌", "퇴보", "저하"
- 목표 달성 확률: "확률", "달성"
- 원인 설명: "때문", "원인", "덕분", "탓", "결과로"
