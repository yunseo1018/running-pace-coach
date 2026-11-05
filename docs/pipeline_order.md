# 실행 순서와 단계별 의존 관계 (입력 파일 / 출력 파일)

이 표는 스크립트의 입력·출력 경로를 코드에서 읽어 만든 것이다. **계산은 바꾸지 않았고 순서와 의존 관계만 문서화했다.** 모든 명령은 저장소 루트에서 실행한다. `data/`는 커밋하지 않는 파생 캐시이고, 원본 `endomondoHR.json`은 저장소에 없다(sha256 `1c021e3f…36e5`, `docs/week5_target_check.md` E절).

## 1. 먼저 알아야 할 점

- **번호순 실행은 불가능하다.** 02는 17의 산출물(`run_sessions_deduped_merged.pkl`)을 입력으로 쓰고, 17은 03이 만드는 분할 ID 파일을 읽는다. 이 순환은 **03이 정제 전 세션 표 `run_sessions.pkl`(01 산출물)에서 6세션 이상 사용자로 분할 ID를 만들도록 고쳐서** 끊었다. 03은 01 직후, 17보다 먼저 실행하며 생성한 목록이 고정본과 sha256까지 같음을 assert로 확인한다(`docs/split_provenance.md`).
- `run_sessions_deduped_merged_with_moving.pkl`은 `18c_attach_moving_duration.py`가 만든다(18b가 만든 조각별 이동 시간을 합산해 `duration_sec`로 자름, 잘리는 세션 42개; 생성 내용이 고정본과 같음을 assert로 확인, `docs/split_provenance.md` 4절). 24·33번이 읽는다.
- 데이터 준비 단계(01, 03, 08, 16, 17, 02, 04, 06, 10, 18b, 18c, 24, 21, 23)는 `python run_all.py --prep`(원본 `endomondoHR.json`이 있을 때, `data/` 파생 파일을 만들고 덮어씀)으로 아래 2절 순서대로 실행할 수 있다. `--prep --all`은 준비 뒤 3절 전체를 실행한다. 준비 단계의 고정 입력은 이제 원본 `endomondoHR.json` 하나뿐이다(분할 ID는 03, `_with_moving`은 18c가 코드로 만든다). 기본·`--all` 실행은 준비 단계를 하지 않는다.
- 42번(test 1회 최종 평가)은 `data/final_eval_done.flag`가 있으면 스스로 중단한다. `run_all.py`는 flag가 있으면 42번을 건너뛰고(결과 고정), 없으면(새 폴더) 한 번 실행한 뒤 43~45를 실행한다.

## 2. 데이터 준비 (직접 실행, 이 순서)

| 순서 | 스크립트 | 입력 파일 | 출력 파일 | 선행 |
|---|---|---|---|---|
| 0 | (입력) | 원본 `endomondoHR.json`(sha256 `1c021e3f…36e5`) | — | — |
| 1 | `01_parse_run_sessions.py` | `endomondoHR.json` | `data/run_sessions.pkl` | 0 |
| 2 | `03_train_test_split.py` | `data/run_sessions.pkl`(6세션 이상 사용자 748명, 정제 전) | `data/train_user_ids.txt`(598명), `data/test_user_ids.txt`(150명). **고정본과 sha256이 같음을 assert**(다르면 쓰지 않고 중단, 이미 있으면 바이트만 비교) | 1 |
| 3 | `08_session_merge.py` | `endomondoHR.json`, `data/run_sessions.pkl` | `data/session_gps_endpoints.pkl`, `data/run_sessions_merged.pkl` | 1 |
| 4 | `16_duplicate_dedup.py` | `endomondoHR.json`, `data/run_sessions.pkl`, `data/session_gps_endpoints.pkl` | `data/session_point_counts.pkl`, `data/run_sessions_deduped.pkl`, `docs/merge_audit/dedup_*.csv`(git 제외) | 2 |
| 5 | `17_dedup_merge_rebuild.py` | `data/run_sessions.pkl`, `data/run_sessions_merged.pkl`, `data/run_sessions_deduped.pkl`, `data/session_gps_endpoints.pkl`, **분할 ID 두 파일(03 산출물)** | `data/run_sessions_deduped_merged.pkl`(메인 세션 표, 105,709건·854명) | 2, 4 |
| 6 | `02_build_user_sequences.py` | `data/run_sessions_deduped_merged.pkl` | `data/user_sequences.pkl` | 5 |
| 7 | `04_build_xy_dataset.py` | `data/user_sequences.pkl`, 분할 ID 두 파일 | `data/train_dataset.pkl`, `data/test_dataset.pkl` | 2, 6 |
| 8 | `06_personalized_split.py` | `data/user_sequences.pkl` | `data/personalized_train_dataset.pkl`, `data/personalized_test_dataset.pkl` | 6 |
| 9 | `10_posthoc_distance_condition.py` | `data/user_sequences.pkl` | `data/posthoc_train_dataset.pkl`, `data/posthoc_test_dataset.pkl` | 6 |
| 10 | `18b_moving_duration_extract.py` | `endomondoHR.json` | `data/session_moving_duration.pkl`(원본 세션 id별 이동 시간) | 0 |
| 11 | `18c_attach_moving_duration.py` | `data/run_sessions_deduped_merged.pkl`, `data/session_moving_duration.pkl` | `data/run_sessions_deduped_merged_with_moving.pkl`. 조각 합 → `duration_sec`로 자름(잘리는 세션 42개) → 세 열 파생. **내용이 고정본과 같음을 assert**(출력이 이미 있으면 DataFrame이 정확히 같은지 확인하고 덮어쓰지 않음) | 5, 10 |
| 12 | `24_build_xy_dataset_extended.py` | `data/run_sessions_deduped_merged.pkl`, `data/run_sessions_deduped_merged_with_moving.pkl` | `data/extended_train_dataset.pkl`, `data/extended_test_dataset.pkl`, `data/extended_personalized_train_dataset.pkl`, `data/extended_personalized_test_dataset.pkl` | 5, 11 |
| 13 | `21_baseline_integrated_report.py`(선택) | 세션 표, `data/user_sequences.pkl`, 04·06 데이터셋, `data/session_point_counts.pkl` | `docs/baseline_results.md` | 4, 6, 7, 8 |
| 14 | `23_ridge_regression.py`(선택) | 04·06·10 데이터셋, `data/user_sequences.pkl` | `docs/ridge_results.md` | 7, 8, 9 |

03은 02의 산출물이 아니라 01의 산출물을 읽도록 고쳤다(이전에는 02의 산출물 `user_sequences.pkl`을 읽어 17과 순환했고, 현재 메인 데이터로 다시 돌리면 593/149가 나왔다). 지금의 03은 고정 목록(598/150)을 sha256까지 같게 재생성한다(`docs/split_provenance.md`).

이전 과제의 부가 단계(결과 문서와 감사용): `05`, `07`, `09`, `11`, `13`, `14`, `15`, `18a`, `19`, `20`, `22`. 위 표의 본 파이프라인 산출물에는 필요하지 않다.

## 3. 결과 재생성 (`python run_all.py`가 실행)

고정 순서: **25 → 27 → 28 → 29 → 30 → 31 → 32 → 33 → 34 → 35 → 36 → 37 → 38 → 39 → 40 → 41 → 42 → 43 → 44 → 45** (`--all`). 기본 실행은 27 → 28 → 29 → 31 → 32이고, 42의 선행(25, 30, 33~41)까지 한 번에 돌려 42(flag가 없을 때만)에서 끝나는 이름 붙은 옵션이 `--through42`(= `--all`에서 43~45를 뺀 순서)다. 실행하지 않고 순서만 확인하려면 `--list`를 덧붙인다(예: `python run_all.py --prep --through42 --list`). 순서를 바꾸면 안 되는 이유는 오른쪽 열의 선행 단계다.

| 단계 | 스크립트 | 입력 파일 | 출력 파일 | 선행 |
|---|---|---|---|---|
| 25 | `25_tree_models.py` | 04·06 데이터셋, 24 데이터셋, `data/user_sequences.pkl` | `docs/tree_results.md` | 준비 6~12 |
| 27 | `27_mlp_torch.py` | 24 데이터셋, `data/user_sequences.pkl` | `docs/mlp_results.md`, `docs/mlp_selected_hyperparams.json` | 준비 10 |
| 28 | `28_model_comparison.py` | 24 데이터셋, `docs/mlp_selected_hyperparams.json` | `docs/model_comparison.md` | 27 |
| 29 | `29_coaching.py` | 24 데이터셋, `docs/mlp_selected_hyperparams.json`, `docs/model_comparison.md` | `docs/coaching_results.md` | 28 |
| 30 | `30_ridge_discrepancy.py` | 04·06 데이터셋, 24 데이터셋 | `docs/ridge_discrepancy.md` | 준비 7, 8, 12 |
| 31 | `31_same_scope_comparison.py` | 24 데이터셋, `docs/mlp_selected_hyperparams.json`, **`docs/model_comparison.md`의 행 이름(예: "P1. …(탐색적)")** | `docs/same_scope_comparison.md` | 28 |
| 32 | `32_coaching_abstain.py` | 24 데이터셋, `docs/mlp_selected_hyperparams.json`(28 경유) | `docs/coaching_abstain_results.md` | 28 |
| 33 | `33_extreme_session_map.py` | 24 데이터셋, `data/user_sequences.pkl`, `data/run_sessions_deduped_merged_with_moving.pkl` | `docs/extreme_session_window_map.md`, `data/extreme_session_window_map.csv`(git 제외) | 준비 12, 28 |
| 34 | `34_multisession_availability.py` | `data/user_sequences.pkl` | `docs/multisession_coaching_design.md` | 준비 6 |
| 35 | `35_data_audit_5km.py` | 메인 세션 표(원본 JSON은 파일 정보 표기용) | `docs/week4_data_audit.md` | 준비 5 |
| 36 | `36_timezone_check.py` | 메인 세션 표, `data/session_gps_endpoints.pkl` | `docs/week4_data_audit.md`에 덧씀 | 35 |
| 37 | `37_target_feasibility.py` | 원본 JSON, `data/run_sessions.pkl`, `data/run_sessions_deduped.pkl`, 메인 세션 표 | `docs/week5_target_check.md` | 준비 5 |
| 38 | `38_build_5km_dataset.py` | 원본 JSON, 메인 세션 표, 분할 ID 두 파일 | `data/5km_dataset_train.pkl`, `docs/dataset_counts_5km.md` | 준비 5 |
| 39 | `39_baselines.py` | `data/5km_dataset_train.pkl` | `docs/baseline_results_train.md` | 38 |
| 40 | `40_models_cv_5km.py` | `data/5km_dataset_train.pkl` | `docs/model_cv_results_5km.md`, `data/model_cv_oof_5km.pkl` | 39 |
| 41 | `41_emlp_cv_5km.py` | `data/5km_dataset_train.pkl`, `data/model_cv_oof_5km.pkl` | `docs/model_cv_results_5km.md`의 E-MLP 절, `data/emlp_oof_5km.pkl` | 40(40이 문서를 다시 쓰므로 반드시 40 뒤) |
| 42 | `42_final_eval_5km.py` | 원본 JSON, 메인 세션 표, 분할 ID, `data/5km_dataset_train.pkl`, `docs/model_cv_results_5km.md`(선택된 하이퍼파라미터와 E-MLP 설정) | `docs/final_results_5km.md`, `data/final_preds_5km.pkl`, `data/final_eval_done.flag` | 38, 40, 41. **1회만**: flag가 있으면 스스로 중단하고 `run_all`은 건너뜀 |
| 43 | `43_within_person_5km.py` | **`data/final_preds_5km.pkl`**(49c02f8의 평가 행과 일치 확인), 원본 JSON(T5 캐시가 없을 때) | `docs/within_person_results_5km.md`, `data/t5_main_all.pkl`(캐시) | **42** |
| 44 | `44_sensitivity_5km.py` | `data/final_preds_5km.pkl`, `data/t5_main_all.pkl`, 원본 JSON | `docs/sensitivity_results_5km.md` | 42, 43 |
| 45 | `45_figures_5km.py` | `docs/final_results_5km.md`, `docs/within_person_results_5km.md`, `docs/sensitivity_results_5km.md`, `data/final_preds_5km.pkl` | `docs/figures/fig1~4_*.png` | 42, 43, 44 |

`run_all.py`는 마지막에 `env_tools.py --environment`로 `docs/environment.md`를 다시 만든다.

## 4. 5km 시연·사후 탐색 (`run_all`에 없음)

| 스크립트 | 입력 | 출력 | 선행 |
|---|---|---|---|
| `demo/make_b2_params.py` | `data/5km_dataset_train.pkl`, `data/final_preds_5km.pkl`, 원본 JSON 캐시(`data/t5_main_all.pkl`) | `demo/b2_params.json`, `docs/demo_interval_check.md` | 42, 43 |
| `demo/make_examples.py`, `demo/coach_demo.py`, `demo/test_coach_demo.py` | `demo/b2_params.json`(예시 CSV는 합성) | `demo/examples/*.csv`, 화면 출력 | `demo/make_b2_params.py` |
| `demo/bench_demo.py` | `data/5km_dataset_train.pkl`, `docs/model_cv_results_5km.md`, `demo/b2_params.json` | `docs/demo_performance.md` | 40~42 |
| `scripts/46_fill_readme.py` | `docs/final_results_5km.md`, `docs/demo_interval_check.md`, `docs/demo_performance.md`, `demo/examples/*.csv` | `README.md`(맨 위 5km 절) | 위 결과 문서 |
| `scripts/47_hr_posthoc_5km.py` | `data/final_preds_5km.pkl`, `data/t5_main_all.pkl`, 메인 세션 표 | `docs/hr_posthoc_results_5km.md` | 42, 43 |

## 5. 새 폴더에서 처음부터 재현하는 순서 (요약)

1. 저장소를 clone하고 `endomondoHR.json`을 루트에 둔다(sha256 확인). 다른 입력 파일은 필요 없다(분할 ID는 03, `_with_moving`은 18c가 만든다).
2. 2절의 1~12번(필요하면 13·14번)을 순서대로 실행한다(`python run_all.py --prep`로 한 번에 가능).
3. `python run_all.py --through42`(기본 5단계 + 42의 선행 + 42; flag가 없으면 42번을 한 번 실행)를 실행하고, 이어서 `python run_all.py --only 43,44,45`를 실행한다. `python run_all.py --all`은 이 둘을 합친 순서다. (`python run_all.py` 기본 실행은 27 → 28 → 29 → 31 → 32만 한다.)
4. (선택) `demo/make_b2_params.py`, `demo/test_coach_demo.py`.

측정한 시간(재현 점검, `docs/repro_check_5km.md`, Windows 11 PC 한 대): 데이터 준비 1,858초(약 31분), `run_all` 기본(31번 수정 전 실패 시점까지) 약 111분, 전체 단계 합계 약 3시간 31분. 31번을 수정한 뒤 원본 저장소에서 `python run_all.py --only 31,32`는 13.8분(31번 11.3분, 32번 2.5분)이었고 결과 문서가 커밋본과 같았다. 사본에서 31번이 약 90분 뒤 실패한 시간(111분에 포함)과 11.3분의 차이 원인은 확인하지 않았다(실행 환경의 다른 부하 등 미확인). `run_all.py`의 42번 건너뛰기·43~45 입력 확인 중단은 코드에서 확인했고, flag가 있는 원본 저장소에서 `--only 42`가 건너뛰는 것을 실행해 확인했다(43~45가 예측값 파일 없이 중단하는 경로는 실행하지 않았다).
