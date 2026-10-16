# Ridge 15피처 수치 차이 분해 (6단계 문서 vs 9단계 종합표)

한 번에 한 요인만 바꿔 어디서 숫자가 움직이는지 본다. 새 튜닝 없음.

## 0. 윈도우 구성 차이 (04/06단계 vs 24단계)
| 평가조건 | 구간 | 04/06 윈도우 수 | 24단계 윈도우 수 | 04/06에만 있음 | 24단계에만 있음 |
|---|---|---|---|---|---|
| cold-start | train/history | 79671 | 79669 | 2 | 0 |
| cold-start | test/holdout | 21978 | 21977 | 1 | 0 |
| personalized | train/history | 81020 | 81018 | 3 | 1 |
| personalized | test/holdout | 20629 | 20628 | 1 | 0 |

## 1. 단계별 분해 (Ridge, 15피처 = 6단계와 같은 피처)
alpha_6단계=0.0001, alpha_9단계종합표=10000.0. 학습/평가 윈도우 표기: old=04/06단계, new=24단계. 모든 계산은 `ridge_lib`(numpy)로 이 스크립트가 직접 수행한다(sklearn 의존 없음).

| 평가조건 | 단계 | n_eval | MAE | RMSE |
|---|---|---|---|---|
| cold-start | S0 6단계 재현: ridge_lib, train old/test old, alpha 6단계 | 21978 | 69.928 | 262.069 |
| cold-start | S1 +평가 윈도우만 new: ridge_lib, train old/test new | 21977 | 69.930 | 262.075 |
| cold-start | S2 +학습 윈도우만 new: ridge_lib, train new/test old | 21978 | 69.928 | 262.069 |
| cold-start | S3 윈도우 둘 다 new: ridge_lib, train new/test new | 21977 | 69.929 | 262.075 |
| cold-start | S4 +alpha 1e4: ridge_lib, new/new (= 현재 28번 종합표 6a) | 21977 | 70.274 | 262.419 |
| cold-start | S7 old 윈도우에서 alpha만 1e4: ridge_lib, old/old | 21978 | 70.273 | 262.414 |
| cold-start | S5 (참고) 과거 sklearn Pipeline으로 처음 산출한 종합표 6a 기록값(소수 3자리) | 21977 | 70.274 | 262.419 |
| personalized | S0 6단계 재현: ridge_lib, train old/test old, alpha 6단계 | 20629 | 62.054 | 239.762 |
| personalized | S1 +평가 윈도우만 new: ridge_lib, train old/test new | 20628 | 62.056 | 239.767 |
| personalized | S2 +학습 윈도우만 new: ridge_lib, train new/test old | 20629 | 62.054 | 239.762 |
| personalized | S3 윈도우 둘 다 new: ridge_lib, train new/test new | 20628 | 62.056 | 239.767 |
| personalized | S4 +alpha 1e4: ridge_lib, new/new (= 현재 28번 종합표 6a) | 20628 | 62.270 | 239.443 |
| personalized | S7 old 윈도우에서 alpha만 1e4: ridge_lib, old/old | 20629 | 62.268 | 239.437 |
| personalized | S5 (참고) 과거 sklearn Pipeline으로 처음 산출한 종합표 6a 기록값(소수 3자리) | 20628 | 62.270 | 239.443 |

**S6 행(sklearn으로 구현만 전환)을 제외한 이유**: 이 문서는 sklearn을 호출하지 않는다(sklearn import가 정책으로 간헐 차단되고 파이프라인이 sklearn에 의존하지 않으므로, 새로 계산하는 모든 단계를 ridge_lib로만 수행). 이전 버전에는 S6이 있었으나 sklearn import가 차단된 실행에서는 미실행이었고, S5는 과거 sklearn Pipeline으로 처음 산출해 기록한 값을 참고용으로만 남겼다.

## 2. 6단계 문서 수치 재현 여부 (S0)
- cold-start: 재계산 MAE=69.928493/RMSE=262.069142 vs 6단계 문서 69.928493/262.069142 -> MAE 차이 -0.000000, RMSE 차이 +0.000000
- personalized: 재계산 MAE=62.053868/RMSE=239.761553 vs 6단계 문서 62.053868/239.761552 -> MAE 차이 -0.000000, RMSE 차이 +0.000001

## 3. 요인별 이동폭 (MAE 기준)
| 평가조건 | 평가 윈도우 변경 | 학습 윈도우 변경 | 둘 다 | alpha 변경(ridge_lib, new) | 구현 전환 참고(S5 기록값 − S4) | 총 변화(S0→S4) |
|---|---|---|---|---|---|---|
| cold-start | +0.0011 | -0.0002 | +0.0010 | +0.3443 | +0.0003 | +0.3452 |
| personalized | +0.0017 | -0.0000 | +0.0017 | +0.2142 | +0.0002 | +0.2160 |

**'총 변화' 기준 변경**: 이전 버전은 '총 변화'를 S5−S0(과거 sklearn 기록값 기준)으로 정의했으나, sklearn 호출 없이 이 스크립트가 직접 계산한 값만으로 정의하도록 S4−S0(ridge_lib 재현값 기준)으로 바꿨다. 변경 전후 값: cold-start 0.3455→0.3452, personalized 0.2161→0.2160 (차이는 S5가 소수 3자리로 반올림된 기록값이기 때문이다).

**읽는 법**: 어느 열의 이동폭이 총 변화의 대부분을 차지하는지가 원인이다. 요인 간 상호작용은 분리하지 않았다(순차 변경이라 순서에 의존). '구현 전환 참고'는 과거 sklearn Pipeline으로 처음 산출한 값(S5)과 ridge_lib로 재현한 값(S4)의 차이이며 소수 3자리 반올림 수준이다.

## 4. 사용 구현과 import 폴백 확인
- 이 스크립트의 모든 계산은 `ridge_lib`(numpy, ManualStandardScaler)로 수행한다. sklearn 의존이 없고 sklearn import 성공 여부가 문서 내용에 영향을 주지 않는다.
- 과거 이력: 9단계 종합표(28번)의 Ridge 6a/6b와 7단계(25번)의 Ridge는 처음에는 sklearn Pipeline(StandardScaler, Ridge)으로 산출했고, 재현성 정리(커밋 a0bcfe8) 이후 `ridge_lib.RidgePipeline`으로 고정되어 같은 값을 재현한다. 6단계 문서(23번)의 Ridge는 처음부터 ridge_lib였다.
- try/except import 폴백 확인(코드 검색 결과): 23/25/27/28/29/31/32 스크립트에는 Ridge·GroupKFold·scaler를 'sklearn 실패 시 ridge_lib로 대체'하는 폴백이 **없다**. sklearn import 진단은 `env_tools.py`가 `logs/sklearn_diagnostics.log`(git 제외)에만 기록하며 GBM 구현 선택에도 쓰이지 않는다(항상 gbt_lib).
- sklearn 로드 오류 기록: `docs/sklearn_import_issue.md`.
