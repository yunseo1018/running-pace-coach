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
alpha_6단계=0.0001, alpha_9단계종합표=10000.0. 학습/평가 윈도우 표기: old=04/06단계, new=24단계.

| 평가조건 | 단계 | n_eval | MAE | RMSE |
|---|---|---|---|---|
| cold-start | S0 6단계 재현: ridge_lib, train old/test old, alpha 6단계 | 21978 | 69.928 | 262.069 |
| cold-start | S1 +평가 윈도우만 new: ridge_lib, train old/test new | 21977 | 69.930 | 262.075 |
| cold-start | S2 +학습 윈도우만 new: ridge_lib, train new/test old | 21978 | 69.928 | 262.069 |
| cold-start | S3 윈도우 둘 다 new: ridge_lib, train new/test new | 21977 | 69.929 | 262.075 |
| cold-start | S4 +alpha 1e4: ridge_lib, new/new | 21977 | 70.274 | 262.419 |
| cold-start | S5 +구현 전환(sklearn Pipeline): new/new, alpha 1e4 (= 9단계 종합표 6a) [미실행: sklearn import 차단 -> 28단계 기록값 사용, 소수 3자리] | 21977 | 70.274 | 262.419 |
| cold-start | S6 구현만 전환(alpha 6단계): sklearn, new/new [미실행: sklearn import 차단] | 21977 | nan | nan |
| cold-start | S7 old 윈도우에서 alpha만 1e4: ridge_lib, old/old | 21978 | 70.273 | 262.414 |
| personalized | S0 6단계 재현: ridge_lib, train old/test old, alpha 6단계 | 20629 | 62.054 | 239.762 |
| personalized | S1 +평가 윈도우만 new: ridge_lib, train old/test new | 20628 | 62.056 | 239.767 |
| personalized | S2 +학습 윈도우만 new: ridge_lib, train new/test old | 20629 | 62.054 | 239.762 |
| personalized | S3 윈도우 둘 다 new: ridge_lib, train new/test new | 20628 | 62.056 | 239.767 |
| personalized | S4 +alpha 1e4: ridge_lib, new/new | 20628 | 62.270 | 239.443 |
| personalized | S5 +구현 전환(sklearn Pipeline): new/new, alpha 1e4 (= 9단계 종합표 6a) [미실행: sklearn import 차단 -> 28단계 기록값 사용, 소수 3자리] | 20628 | 62.270 | 239.443 |
| personalized | S6 구현만 전환(alpha 6단계): sklearn, new/new [미실행: sklearn import 차단] | 20628 | nan | nan |
| personalized | S7 old 윈도우에서 alpha만 1e4: ridge_lib, old/old | 20629 | 62.268 | 239.437 |

## 2. 6단계 문서 수치 재현 여부 (S0)
- cold-start: 재계산 MAE=69.928493/RMSE=262.069142 vs 6단계 문서 69.928493/262.069142 -> MAE 차이 -0.000000, RMSE 차이 +0.000000
- personalized: 재계산 MAE=62.053868/RMSE=239.761553 vs 6단계 문서 62.053868/239.761552 -> MAE 차이 -0.000000, RMSE 차이 +0.000001
## 3. 요인별 이동폭 (MAE 기준)
| 평가조건 | 평가 윈도우 변경 | 학습 윈도우 변경 | 둘 다 | alpha 변경(ridge_lib, new) | 구현 전환(alpha 1e4) | 구현 전환(alpha 6단계) | 총 변화(S0→S5) |
|---|---|---|---|---|---|---|---|
| cold-start | +0.0011 | -0.0002 | +0.0010 | +0.3443 | +0.0003(S5=28단계 기록값) | 미실행(sklearn import 차단) | +0.3455 |
| personalized | +0.0017 | -0.0000 | +0.0017 | +0.2142 | +0.0002(S5=28단계 기록값) | 미실행(sklearn import 차단) | +0.2161 |

**읽는 법**: 어느 열의 이동폭이 총 변화의 대부분을 차지하는지가 원인이다. 요인 간 상호작용은 분리하지 않았다(순차 변경이라 순서에 의존).

## 4. 사용 구현과 import 폴백 확인
- 이 분해에서 `ridge_lib`(numpy, ManualStandardScaler) 단계와 `sklearn 1.9.1(기록값; 이번 실행에서는 import 차단)` Pipeline(StandardScaler+Ridge) 단계를 명시적으로 구분해 썼다.
- **9단계 종합표(28_model_comparison.py)의 Ridge 6a/6b는 sklearn Pipeline(StandardScaler, Ridge)** 이다. 7단계(25_tree_models.py)의 Ridge도 sklearn.
- **6단계 문서(23_ridge_regression.py)의 Ridge는 ridge_lib(numpy 직접 구현)** 이다. 6단계 이후 sklearn이 import 가능해지자 7단계부터 sklearn으로 전환했다.
- try/except import 폴백 확인(코드 검색 결과): 23/25/27/28/29 스크립트에는 Ridge·GroupKFold·scaler를 'sklearn 실패 시 ridge_lib로 대체'하는 폴백이 **없다**. 25_tree_models.py의 `try: import sklearn / except ImportError`는 실패 시 sys.exit(1)로 중단하는 용도이고, HistGradientBoosting import 시도는 진단 기록용일 뿐 GBM 구현 선택에 쓰이지 않는다(항상 gbt_lib). 폴백 없음.
- **이번 실행 환경 메모**: 이 실행에서 sklearn import가 Windows 애플리케이션 제어 정책으로 차단되어(`DLL load failed`, 실행마다 차단 모듈이 달라지는 간헐 현상과 같은 계열, 원인 미확인) sklearn 단계(S5, S6)는 새로 계산하지 못했다. 정책 우회는 시도하지 않았다. S5는 28단계가 이미 산출·기록한 6a 값(70.274/62.270)을 출처를 밝혀 썼고, S6은 미실행으로 남겼다. 구현 전환 효과는 S4(ridge_lib)와 S5(기록된 sklearn 값)가 소수 3자리까지 같다는 사실까지만 확인됐다. 참고: 이 진단 스크립트(30)의 `fit_pred_sk_guarded`는 결과를 대체하는 폴백이 아니라 차단 시 해당 단계를 미실행으로 표기하는 보고용 가드다.
