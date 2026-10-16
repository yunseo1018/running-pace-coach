# 실행 환경 기록 (자동 생성: `python env_tools.py --environment`)

타임스탬프를 넣지 않아 같은 환경·같은 데이터에서는 바이트 단위로 같다. 사용자·세션 ID와 개인 경로는 쓰지 않는다.

## 1. 소프트웨어
- Python 3.14.7
- OS: Windows-11-10.0.26200-SP0
- numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, torch 2.14.0+cpu (CPU 스레드 24)
- scikit-learn 배포 버전 1.9.1: **이 파이프라인은 sklearn에 의존하지 않는다**(진단 전용, import 실패해도 결과 불변). 실패 기록은 `docs/sklearn_import_issue.md`
- 직접 구현: `ridge_lib.py`(Ridge/GroupKFold/표준화), `gbt_lib.py`(GBM: 손실 absolute_error/squared_error, 격자·고정값은 `docs/tree_results.md` 2절), MLP는 torch(L1 손실)

## 2. 실행 명령
```
pip install -r requirements.txt
python run_all.py                      # 27 → 28 → 29 → 31 → 32 (기본)
python run_all.py --all                # 25, 27, 28, 29, 30, 31, 32, 33 전체
python run_all.py --only 28,29         # 일부만 (고정 순서 유지)
python run_all.py --check              # 각 단계 후 결과 문서를 git HEAD 문서와 대조
python run_all.py --repeat 2           # 같은 명령을 두 번 실행하고 docs/ 바이트 동일성 확인
python run_all.py --save-predictions   # data/predictions/에 가명화 예측값 저장(git 제외)
python env_tools.py --environment      # 이 문서 재생성
python env_tools.py --sklearn-issue    # docs/sklearn_import_issue.md 재생성
```

## 3. 시드·분할
- 사용자 분할(03단계): RANDOM_SEED=42, TRAIN_RATIO=0.8 (사용자 단위 80/20)
- personalized: 사용자별 시간순 history/hold-out 비율 0.8; 모델 선택용 내부 분할 비율 0.8
- GroupKFold(ridge_lib): 5-fold, 시드 42
- 사용자 단위 cluster bootstrap(21·28·31): 2000회, 시드 42
- 코칭 bootstrap(29·32): 2000회, 시드 42; 32 무작위 보류 통제 1000회
- MLP(27): 선택 시드 42, 최종 시드 [1, 2, 3, 42, 999], 대표 시드 42 (28·29·31·32는 `MLP_SEED`=42)
- GBM(`gbt_lib`): 행·피처 서브샘플링이 없어 난수를 쓰지 않는다(결정적; `random_state`는 결과에 영향 없음)

## 4. 데이터·분할·하이퍼파라미터 해시 (sha256)
| 파일 | sha256 |
|---|---|
| `data/run_sessions_deduped_merged.pkl` | 2826cb2548fd5dc33875c8ac9f076361eb2691a44c87ec6a72a2d618c8e1b81d |
| `data/run_sessions_deduped_merged_with_moving.pkl` | a847a424d22f336ed03c6813e233225235d4732b3077287b9433727fc01640b3 |
| `data/user_sequences.pkl` | 0ceb942619a7e46b69de604bc6be5dac490287157f2a705be6dc61730e1a7492 |
| `data/extended_train_dataset.pkl` | 98e59da8debf6f974ee502b4a81d84dc053463bd2d47efd51139f904d75fe8bd |
| `data/extended_test_dataset.pkl` | 8137c26c55735d9403de3715fcc582a30cdf2c2d2964794d2a2b1c50d1d85294 |
| `data/extended_personalized_train_dataset.pkl` | aa5102d18483bc8d5daaee45ac6d224afed6f28e20b29eef5adb0810ed4a32a0 |
| `data/extended_personalized_test_dataset.pkl` | e32562c93f1d1fdfeb4dde00bd4f43957842030b9d0b7de525b835c5d69dc818 |
| 분할 ID 목록 `data/train_user_ids.txt` (정렬된 ID 598개를 개행으로 이은 문자열의 해시; ID 자체는 기록하지 않음) | 9787f2803ec632ac45bc886e459095f2a0e6289853f12714aaef987fade50e8c |
| 분할 ID 목록 `data/test_user_ids.txt` (정렬된 ID 150개를 개행으로 이은 문자열의 해시; ID 자체는 기록하지 않음) | f6a5aece07b03f6e5e593926bde302bc2e3b3bc0dec45c7f40be1cd9548a66c6 |
| `docs/mlp_selected_hyperparams.json` (27번이 저장, 28·29·31·32가 읽음) | 07073ff078ea11e2d8ac97caae0a5bdeca6ed32da67759915c66b3153c3141d6 |

## 5. 평가 행별 예측값 파일 (선택 기능, git 제외)
`python run_all.py --save-predictions`로 만든 파일. 사용자·세션 ID는 고정 salt 해시로 대체했다(가명화일 뿐 익명화가 아니므로 파일 자체는 커밋 금지; `data/`는 `.gitignore` 대상).

| 파일 | sha256 |
|---|---|
| `data/predictions/cold_start_predictions.csv` | 38c28c5cfccf6e57c785346e2184eda5a4a826f5b878020a4edf3e607ae7002b |
| `data/predictions/personalized_predictions.csv` | 40db5b864c560b539660957b2befb1a7fcf9d71286bb15a1f3aa261873155dde |
