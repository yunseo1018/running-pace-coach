# 재현 점검: 원본 데이터부터 새 폴더에서 최종 숫자까지 (5km 과제)

**점검만 했다.** 코드·문서·숫자는 고치지 않았고, 원본 저장소의 `data/`와 `final_eval_done.flag`는 건드리지 않았다. 원본 사용자 행·ID·좌표는 싣지 않았다(집계·파일 단위 비교만). 점검 대상 코드는 커밋 `fa09830`(저장소 상태 그대로의 clone 사본)이다. 이 문서는 점검 뒤에 작성했다.

## 1. 조건

| 항목 | 내용 |
|---|---|
| 사본 | 저장소를 새 폴더에 clone(HEAD `fa09830`). 사본의 `data/`는 비어 있었고 `run_all`과 각 단계 스크립트가 만들게 했다. 원본의 `data/`와 `data/final_eval_done.flag`는 사본에서 보이지 않았다(연결하지 않음). |
| 원본 데이터 | `endomondoHR.json` **파일 하나만** 사본 루트에 연결(복사 아님). 심볼릭 링크는 관리자 권한이 필요해 실패했고, 파일은 정션 대상이 아니라서 같은 드라이브의 **NTFS 하드 링크**(같은 파일 데이터를 가리킴, 6,568,384,411바이트)로 연결했다. `.gitignore`(`*.json`)에 걸려 커밋 대상이 아니며 사본은 저장소 밖 임시 폴더다. |
| 원본 sha256 | `1c021e3f8cb6428aefcb14ceff9d58ab2b6b2163bcbedfca71cb9f55616836e5` — 기대값 `1c021e3f…36e5`와 일치(`docs/week5_target_check.md` E절 값과 동일) |
| 사본에 복사해 넣은 입력 3개(예외) | `data/train_user_ids.txt`, `data/test_user_ids.txt`, `data/run_sessions_deduped_merged_with_moving.pkl`. **저장소 코드만으로는 만들 수 없어서**(아래 4절 ①②) 원본 `data/`에서 이 3개 파일만 복사했다. 해시는 `docs/environment.md`의 값과 일치함을 확인했다(분할 ID 두 개는 같은 방식의 정렬 문자열 해시). 이 3개 외 `data/`의 모든 파일은 사본에서 새로 생성했다. |
| 42번 | 코드·잠금 장치 수정 없음. 사본에 flag가 없던 상태에서 정상 실행되어 사본에 flag가 생겼다(정상). 막히지 않았다. |
| 환경 | Windows 11, Python 3.14.7(원본과 같은 환경). 사본의 `logs/`에 단계별 로그. |

## 2. 실행 순서·시간·결과 (사본에서)

README의 절차를 그대로 따를 수 없어 아래 순서로 실행했다(4절 ③). 고정 입력 3개 복사 후 시작.

| 단계 | 명령 | 소요 | 결과 |
|---|---|---|---|
| 01 원본 파싱 | 01_parse_run_sessions.py | 546초 | 성공 |
| 08 병합 | 08_session_merge.py | 494초 | 성공 |
| 16 중복 제거 | 16_duplicate_dedup.py | 478초 | 성공 |
| 17 중복 제거→병합 | 17_dedup_merge_rebuild.py | 95초 | 성공 |
| 02·04·06·10·24 | 02, 04, 06, 10, 24번 | 1·39·30·34·95초 | 성공 |
| 21·23 (baseline·Ridge 문서) | 21, 23번 | 7·37초 | 성공 |
| run_all 기본 | `python run_all.py` (27→28→29→31→32) | 6,646초(약 110.8분) | **27(15.8분)·28(2.7분)·29(2.3분) 성공, 31 실패로 중단(32는 실행되지 않음)** — 31은 실행 중 약 90분 뒤 실패(전체 시간에서 27~29 합을 뺀 추정값, 로그에 단계별 시간 없음) |
| 32 단독 | `python run_all.py --only 32` | 388초(6.4분) | 성공 |
| run_all 나머지 | `--only 25,30,33,34,35,36,37,38,39,40,41` | 3,653초 | 성공(25: 2,356초, 41: 961초, 40: 130초, 33: 80초, 37: 69초, 38: 33초 등) |
| 42 최종 평가 | `python 42_final_eval_5km.py` (1회) | 71초 | 성공(`final_results_5km.md` 생성) |
| 43·44·45 | `python run_all.py --only 43,44,45` | 79초 | 성공(33초·40초·2초) |
| 데모 점검 | `demo/make_b2_params.py`, `demo/test_coach_demo.py` | — | 성공, 단위 테스트 11 tests passed |

**전체 실행 시간**: 단계 실행 합계 12,695초(약 3시간 31분) = 준비 단계 1,858초 + `run_all` 기본 6,646초 + 32 단독 388초 + 나머지 3,653초 + 42 71초 + 43~45 79초. 대기·비교 시간은 제외. README의 기본 명령 예상 소요(약 27분)와 달리 `run_all` 기본은 실패한 31번 때문에 약 111분 걸렸다. 이전 환경(원본)에서 31번이 14.5분이었다는 README 기록과도 차이가 있으나 원인은 확인하지 않았다(사본 실행 당시 다른 작업 병행 여부 등 미확인).

## 3. 실패한 단계와 에러 메시지

| 단계 | 에러 | 원인(코드·문서 확인) |
|---|---|---|
| 31 `31_same_scope_comparison.py` | `AssertionError: ('hist_median', 'cold-start')` (`parse_doc_rows`, 31번 189행; 로그 `run_all_31.log`) | 31번이 `docs/model_comparison.md` 종합표에서 행 이름 **"P1. 사용자 전체 이력 중앙값(사후추가)"**(31번 60행 `DOC_ROW`)를 정규식으로 찾는데, 현재 문서의 행 이름은 **"P1. 사용자 전체 이력 중앙값(탐색적)"**(`docs/model_comparison.md` 54행)이다. 커밋 `e2fadea`(3주차 8번 정리, '사후추가'→'탐색적' 표현 정정)가 28번의 라벨과 문서는 바꿨지만 31번의 `DOC_ROW` 두 줄(60·61행)은 바꾸지 않아서 생긴 불일치다. HEAD `fa09830`의 코드만으로는 `run_all` 기본 명령이 31번에서 항상 실패할 것으로 보인다(원본 저장소에서 같은 명령을 다시 실행하지는 않았다). 코드는 수정하지 않았다. |

그 밖의 단계는 실패하지 않았다. 31번이 실패하면서 `docs/same_scope_comparison.md`는 사본에서 재생성되지 않았다(체크아웃된 문서 그대로이므로 이 문서의 재현 여부는 이번 점검에서 확인하지 못했다).

## 4. 다른 폴더에서 깨지는 점·재현성 문제

**하드코딩 경로**: 저장소 파이썬 파일에 절대 경로(드라이브·사용자 폴더)는 없다. 모든 경로가 저장소 루트 기준 상대 경로(`data/…`, `docs/…`, `Path("endomondoHR.json")`)이며, 임시 폴더 깊은 경로(한글 없음, 긴 경로)의 사본에서도 위 31번 외의 경로 문제는 없었다. 단 **스크립트는 저장소 루트를 작업 디렉터리로 실행해야** 한다. `env_tools.py`의 `C:\Users\<name>` 패턴은 로그 마스킹용 정규식이다.

그 밖에 이번 점검에서 확인한 재현성 문제(코드는 고치지 않음):

| # | 문제 | 근거 |
|---|---|---|
| ① | **분할 ID를 코드로 다시 만들 수 없음.** `17`(중복 제거→병합)이 `data/train_user_ids.txt`·`test_user_ids.txt`를 읽는데, 이 파일은 `03_train_test_split.py`가 `02`(user_sequences)에서 만들고 `02`는 `17`의 산출물을 입력으로 쓰므로 순환 의존이다. 사본의 파이프라인 결과로 `03`을 다시 돌려 보면(임시 폴더, 사본 데이터에는 덮어쓰지 않음) train 593명·test 149명이 나오고 고정 목록(train 598명·test 150명)과 사용자 집합이 다르다(겹침 train 504명·test 60명). 고정 분할 ID 파일이 필수 입력이다(해시는 `docs/environment.md`). | `03`/`02`/`17` 입력·출력 경로, 임시 실행 |
| ② | **`run_sessions_deduped_merged_with_moving.pkl`을 만드는 스크립트가 저장소에 없음.** 24·33번이 읽는다. 원본 `session_moving_duration.pkl`로 `merged_session_ids` 합산을 시도하면 42개 세션(0.04%)의 `moving_duration_sec`이 원본 파일과 다르다(원인 미확인; 별도 단계가 있었을 가능성). 이 파일은 복사해 넣었다. | `docs/target_pace_definition.md` 19행, 해당 파일 읽는 스크립트 grep |
| ③ | README의 "02~04, 06, 16~18, 21, 23, 24 … 번호순으로 실행"은 따를 수 없다(02가 17의 산출물을 요구). 이 점검은 01→08→16→17→02→04→06→10→24→21→23 순서로 실행했다(10은 23이 필요로 하는 posthoc 데이터셋). | 실행 성공 순서 |
| ④ | `run_all.py --all`은 43번을 42번보다 먼저 실행하지만 43번은 42번이 만든 `data/final_preds_5km.pkl`(49c02f8의 평가 행)을 확인한다. 42번은 `run_all`에 없으므로 `--all` 한 번으로는 끝까지 갈 수 없다고 코드상 판단된다(`--all` 자체는 실행하지 않았고, 42를 사이에 둔 별도 실행으로 통과). | `run_all.py` ALL_ORDER, `43`의 `verify_against_final` |
| ⑤ | `docs/environment.md`는 `--save-predictions`를 쓰지 않은 사본에서 5절(예측값 파일)이 "현재 저장된 예측값 파일이 없다"로 달라진다. 선택 기능이므로 예상된 차이다. | 문서 diff 7줄 |

## 5. 문서·결과 파일 비교 (사본 생성본 vs 원본 저장소의 커밋된 문서, 줄바꿈 정규화 후)

사본 `docs/`의 파일 37개(사본 이후 커밋으로 생긴 `posthoc_spec_hr_5km.md`, `hr_posthoc_results_5km.md`, `results_viewed_log.md` 수정분은 비교 대상에서 제외) 중 **이번 실행이 다시 쓴 파일 21개: 동일 20개, 다름 1개.** 나머지 16개는 이번 실행이 다시 쓰지 않은 파일(체크아웃본 그대로, 재현 여부 확인 안 됨)이다. 다른 1개는 아래 `environment.md`뿐이다.

| 파일 | 이번 실행이 재생성 | 원본 커밋본과 동일 | 다르면 어떤 차이 |
|---|---|---|---|
| `docs/baseline_results.md` | 예 | 동일 |  |
| `docs/baseline_results_train.md` | 예 | 동일 |  |
| `docs/coaching_abstain_results.md` | 예 | 동일 |  |
| `docs/coaching_definition.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/coaching_results.md` | 예 | 동일 |  |
| `docs/dataset_counts_5km.md` | 예 | 동일 |  |
| `docs/demo_interval_check.md` | 예 | 동일 |  |
| `docs/demo_performance.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/duplicate_audit.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/environment.md` | 예 | 다름 | 7줄 차이: 예측값 파일(`--save-predictions`) 절 — 원본은 파일 2개의 sha256 표, 사본은 "저장된 예측값 파일이 없다" 문구(수치 차이 아님) |
| `docs/evaluation_freeze_5km.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/extreme_session_window_map.md` | 예 | 동일 |  |
| `docs/feature_availability.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/final_results_5km.md` | 예 | 동일 |  |
| `docs/history_scope_table.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/mlp_results.md` | 예 | 동일 |  |
| `docs/mlp_selected_hyperparams.json` | 예 | 동일 |  |
| `docs/model_comparison.md` | 예 | 동일 |  |
| `docs/model_cv_results_5km.md` | 예 | 동일 |  |
| `docs/multisession_coaching_design.md` | 예 | 동일 |  |
| `docs/posthoc_spec_5km.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/predeclaration_5km_addendum.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/predeclaration_5km_final.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/ridge_discrepancy.md` | 예 | 동일 |  |
| `docs/ridge_results.md` | 예 | 동일 |  |
| `docs/same_scope_comparison.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/sensitivity_results_5km.md` | 예 | 동일 |  |
| `docs/session_merge_audit.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/sklearn_import_issue.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/target_pace_definition.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/temporal_audit.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/tree_results.md` | 예 | 동일 |  |
| `docs/week3_predeclaration.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/week4_data_audit.md` | 예 | 동일 |  |
| `docs/week4_predeclaration_5km.md` | 아니오(재현 확인 안 됨) | 동일 |  |
| `docs/week5_target_check.md` | 예 | 동일 |  |
| `docs/within_person_results_5km.md` | 예 | 동일 |  |

**데이터 산출물**(사본에서 생성한 파일 vs 원본 `data/`)

| 파일 | 결과 |
|---|---|
| `data/5km_dataset_train.pkl` | 바이트 동일 |
| `data/emlp_oof_5km.pkl` | 바이트 동일 |
| `data/extended_personalized_test_dataset.pkl` | 바이트 동일 |
| `data/extended_personalized_train_dataset.pkl` | 바이트 동일 |
| `data/extended_test_dataset.pkl` | 바이트 동일 |
| `data/extended_train_dataset.pkl` | 바이트 동일 |
| `data/extreme_session_window_map.csv` | 바이트 동일 |
| `data/final_eval_done.flag` | 바이트 동일 |
| `data/final_preds_5km.pkl` | 바이트 동일 |
| `data/model_cv_oof_5km.pkl` | 바이트 동일 |
| `data/personalized_test_dataset.pkl` | 바이트 동일 |
| `data/personalized_train_dataset.pkl` | 바이트 동일 |
| `data/posthoc_test_dataset.pkl` | 바이트 동일 |
| `data/posthoc_train_dataset.pkl` | 바이트 동일 |
| `data/run_sessions.pkl` | 객체 동일(바이트 다름) |
| `data/run_sessions_deduped.pkl` | 객체 동일(바이트 다름) |
| `data/run_sessions_deduped_merged.pkl` | 바이트 동일 |
| `data/run_sessions_deduped_merged_with_moving.pkl` | 복사해 넣은 입력(재생성 불가) |
| `data/run_sessions_merged.pkl` | 객체 동일(바이트 다름) |
| `data/session_gps_endpoints.pkl` | 바이트 동일 |
| `data/session_point_counts.pkl` | 바이트 동일 |
| `data/t5_main_all.pkl` | 바이트 동일 |
| `data/test_dataset.pkl` | 바이트 동일 |
| `data/test_user_ids.txt` | 복사해 넣은 입력(재생성 불가) |
| `data/train_dataset.pkl` | 바이트 동일 |
| `data/train_user_ids.txt` | 복사해 넣은 입력(재생성 불가) |
| `data/user_sequences.pkl` | 바이트 동일 |

"객체 동일(바이트 다름)"은 `pd.read_pickle`로 읽은 DataFrame이 `equals`로 같고 pickle 직렬화 바이트만 다른 경우다. `final_preds_5km.pkl`의 평가 행(사용자·시점·정답)과 15개 방법의 예측값은 원본과 정확히 일치한다(최대 절대 차이 0.0초, cold 352개·pers 442개).

**그림 fig1~4**: 그림을 만드는 입력 수치가 같다 — 입력은 `docs/final_results_5km.md`·`within_person_results_5km.md`·`sensitivity_results_5km.md`(위 표에서 동일)와 `data/final_preds_5km.pkl`(위에서 동일)이다. 참고로 PNG 파일 바이트도 4개 모두 원본과 같다.

## 6. 핵심 숫자 대조 (사본 생성본)

| 항목 | 기대값 | 사본 생성값 | 결과 |
|---|---|---|---|
| 평가 대상 / 사용자(전체) | 2,072 / 349 | 2,072 / 349명 | 일치 |
| 평가 대상 / 사용자(train) | (참고) 1,720 / 278 | 1,720 / 278명 | 일치 |
| 평가 대상 / 사용자(test) | 352 / 71 | 352 / 71명 | 일치 |
| B0 사용자 평균 MAE(cold-start) | 266.3 | 266.258 [197.922, 340.025] | 일치 |
| B2 사용자 평균 MAE(cold-start) | 235.1 | 235.117 [171.885, 311.648] | 일치 |
| 모델 − B2, Ridge(cold-start) | +9.4 | +9.442 [-1.071, 19.907] | 일치 |
| 모델 − B2, GBM(cold-start) | +6.2 | +6.192 [-5.070, 17.647] | 일치 |
| 모델 − B2, MLP(cold-start) | +7.2 | +7.214 [-8.185, 23.401] | 일치 |
| 모델 − B2, Ridge(personalized) | (참고, 원본 문서와 동일해야 함) | +1.855 [-5.618, 9.618] | 원본과 일치 |
| 모델 − B2, GBM(personalized) | (참고, 원본 문서와 동일해야 함) | +8.464 [-2.118, 19.043] | 원본과 일치 |
| 모델 − B2, MLP(personalized) | (참고, 원본 문서와 동일해야 함) | +23.240 [8.812, 38.457] | 원본과 일치 |
| γ (같은 사람 안 분석) | −0.845 | -0.845 [-0.934, -0.757] | 일치 |
| 80% 범위 실제 포함률(cold-start test / personalized) | 73.9% | 73.9% / 77.8% | 일치(73.9%) |
| 민감도 S1~S5에서 B2를 넘은 모델 수 | 모두 0 | 모두 0 | 일치 |

모델 − B2의 점추정과 신뢰구간은 원본 `docs/final_results_5km.md`와 글자 그대로 같다(위 5절에서 동일 판정). 통계적·실용적 통과 여부도 같다(모두 "아니오").

## 7. 요약과 한계

- 원본 `endomondoHR.json`(sha256 일치)에서 출발해 사본에서 새로 만든 세션 표 6종·기준 시점·5km 데이터셋·예측값은 원본과 같고, 최종 평가(42번) 이후 문서의 수치는 글자 그대로 같았다. 수치가 달라진 문서는 없다(다른 문서는 `environment.md`의 예측값 파일 절뿐).
- 단 **고정 입력 3개를 원본에서 복사했고**(분할 ID 2개, with_moving 1개) 이를 코드로 재생성할 수 없다는 점, 그리고 `run_all` 기본 명령이 31번에서 실패하는 점은 재현성의 결함이다. 31번이 쓰는 `docs/same_scope_comparison.md`의 재현은 확인하지 못했다.
- 확인하지 않은 것: `run_all.py --all` 한 번 실행, `--repeat`, 다른 PC·다른 Python/패키지 버전, 사본 이후 커밋(`9d2efda`·`9b0d6ab`의 심박 사후 탐색)의 재현, 31번 실패 원인 외의 시간 차이 원인.
