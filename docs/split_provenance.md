# 분할 ID와 `_with_moving.pkl`의 출처 (읽기 전용 조사, 문서화만)

작성: 재현 점검(`docs/repro_check_5km.md`)에서 드러난 두 가지 재현 갭을 조사한 기록이다. **조사와 문서화만 했고 코드·데이터·숫자는 바꾸지 않았다.** 사용자 ID 목록은 이 문서에 싣지 않았고 파일 이름·sha256·사용자 수·커밋만 적었다(`data/`는 커밋하지 않는다).

## 1. 분할 ID 파일 (`data/train_user_ids.txt`, `data/test_user_ids.txt`)

| 항목 | train | test |
|---|---|---|
| 사용자 수 | 598 | 150 |
| 파일 바이트 sha256 | `4e9c1f40680f3f24da0f7a5ffbb3658486620ea16db5f269edd364ee69292f9a` | `57d4c9ad592b48e720c8cf08bd6cd27003ad34ccf0af53f063e92370b5e7ff79` |
| `docs/environment.md`의 해시(문자열 정렬한 ID를 개행으로 이은 문자열의 sha256) | `9787f2803ec632ac45bc886e459095f2a0e6289853f12714aaef987fade50e8c` | `f6a5aece07b03f6e5e593926bde302bc2e3b3bc0dec45c7f40be1cd9548a66c6` |
| 파일 수정 시각(로컬 파일 시스템) | 2026-09-26 23:25 | 2026-09-26 23:25 |
| 만든 코드·커밋 | `03_train_test_split.py` — 커밋 `9a8ee8b`(2026-09-26, "사용자 단위 train/test 80/20 분할") | 같음 |
| 파일 자체의 커밋 | 없음(`data/`는 `.gitignore` 대상). 파일을 만든 코드(03)만 커밋돼 있다 | 없음 |
| 이후 변경 | 없음. 파일은 한 번 만들어진 뒤 그대로 쓰였다(`config.intersect_train_test_with_eligible`의 docstring: "기존 03단계 train/test 배정(748명 기준)을 새 eligible 집합과 교집합해, 재분할 없이 기존 배정을 유지"; 04번 docstring도 같은 설명). 전체 748명 = 598 + 150 | |

`03_train_test_split.py`의 규칙: `data/user_sequences.pkl`의 사용자 목록을 `np.random.default_rng(42).permutation`으로 섞어 앞 80%(`int(n × 0.8)`)를 train, 나머지를 test로 쓴다.

## 2. 03 ↔ 17이 순환하는 방식 (코드 기준)

| 단계 | 입력 | 출력 | 분할 ID 사용 |
|---|---|---|---|
| 01 | `endomondoHR.json` | `data/run_sessions.pkl` | 없음 |
| 02 | `config.load_main_sessions()` = `data/run_sessions_deduped_merged.pkl`(**17의 산출물**) | `data/user_sequences.pkl` | 없음 |
| 03 | `data/user_sequences.pkl`(02의 산출물) | `data/train_user_ids.txt`, `data/test_user_ids.txt` | **만든다** |
| 17 | `run_sessions.pkl`, `run_sessions_merged.pkl`, `run_sessions_deduped.pkl`, GPS 끝점 캐시, **분할 ID 두 파일** | `data/run_sessions_deduped_merged.pkl` | **시작할 때 읽는다**(248·249행) |

- 02가 17의 산출물을 입력으로 쓰고 03이 02의 산출물을 쓰므로, 17 → 02 → 03 → (ID 파일) → 17이 순환이다.
- 17은 ID 파일을 읽지만 메인 산출물(`run_sessions_deduped_merged.pkl`)을 만든 **뒤에** 비교 표를 계산하는 데만 쓴다(251~260행). 산출물 자체는 ID 파일에 의존하지 않지만 파일이 없으면 17이 시작하자마자 실패한다.
- 이 순환은 10월 7일 메인 데이터를 "중복 제거 → 병합"으로 재배선한 커밋(`17fd70c`)에서 생겼다. 03·02는 9월 26일에 원본 `run_sessions.pkl` 기준으로 만들어졌고, 그 뒤 02의 입력만 메인 데이터로 바뀌었다.

## 3. 03을 다시 돌리면 593/149가 되는 원인 (단계별 사용자 수, 집계만)

고정 목록(748명) 대비, 각 단계 세션 표에서 세션 6개 이상인 사용자(= 02가 남기는 자격 사용자):

| 세션 표 | 세션 수 | 사용자 수(전체) | 6세션 이상 사용자 | 고정 748명과 교집합 | 고정 목록에만 있음 |
|---|---|---|---|---|---|
| `run_sessions.pkl` (01 산출물) | 114,425 | 854 | **748** | 748 | 0 |
| `run_sessions_merged.pkl` (병합만) | 108,392 | 854 | 745 | 745 | 3 |
| `run_sessions_deduped.pkl` (중복 제거만) | 111,738 | 854 | 745 | 745 | 3 |
| `run_sessions_deduped_merged.pkl` (현재 메인) | 105,709 | 854 | 742 | 742 | 6 |
| `user_sequences.pkl` (02 산출물, 현재) | — | 742 | 742 | 742 | 6 |

- **원인**: 고정 목록의 748명은 `run_sessions.pkl`(정제 전)에서 세션 6개 이상인 사용자 748명이다. 중복 제거·병합이 세션을 합치고 줄여 6명이 6세션 미만이 되면서 현재 02의 자격 사용자는 742명이다. 03을 현재 데이터로 다시 돌리면 742명을 `permutation`하므로(길이와 순서가 달라짐) 전혀 다른 배정(train 593명, test 149명)이 나온다. 재생성한 목록은 고정 목록과 train 504명, test 60명만 겹친다.
- **재구성 확인**: 03과 같은 규칙(시드 42, `permutation`, 80%)을 `run_sessions.pkl`의 6세션 이상 사용자 748명(user_id 정렬 순서, 또는 `(user_id, start_time)` 정렬 뒤 `unique` 순서)에 적용하면 train 598명과 test 150명의 **목록이 파일의 순서까지 그대로** 재현됐다. 즉 고정 목록은 "01 산출물 기준 03"의 결과이며 우연이 아니다(이 확인은 이번 조사에서만 하고 저장하지 않았다).

### 정정 결과 (P1-3, 구현됨): 코드로 재생성됨, 해시 일치

- `03_train_test_split.py`의 입력을 `data/run_sessions.pkl`(01번 산출물, 정제 전)에서 **세션 6개 이상인 사용자**로 바꿨다(위 제안 1번). 사용자 순서는 `(user_id, start_time)` 정렬 뒤 `unique` 순서이고 규칙(시드 42, `permutation`, 80%)은 그대로다. 이 입력은 01번만 필요로 하므로 `run_all.py --prep`의 순서에서 17보다 먼저(01 바로 뒤) 실행한다.
- **고정본과 sha256까지 같음을 assert로 확인한다**(다르면 아무것도 쓰지 않고 중단). 확인 결과(원본 저장소 `python 03_train_test_split.py`, `python run_all.py --only 03`): 생성한 목록의 sha256이 고정본과 같다 — train `4e9c1f40680f3f24da0f7a5ffbb3658486620ea16db5f269edd364ee69292f9a`, test `57d4c9ad592b48e720c8cf08bd6cd27003ad34ccf0af53f063e92370b5e7ff79`(위 1절 표의 값). 사용자 수 598/150도 같다.
- ID 파일이 없는 빈 `data/`에서도 실행해 확인했다: 03이 고정본과 같은 바이트로 파일을 만들었다(줄바꿈 CRLF로 고정해 쓰므로 OS와 무관하게 바이트가 같다). 파일이 이미 있으면 바이트가 같은지만 확인하고 덮어쓰지 않는다.
- 이제 분할 ID는 고정 입력이 아니라 코드로 재생성된다. ID 목록 자체는 커밋하지 않는다(`data/`는 `.gitignore`).
- 위 3절의 "03을 현재 데이터(메인)로 다시 돌리면 593/149"는 정정 전 코드의 동작이다. 정정 뒤 03은 메인 데이터를 읽지 않는다.

### 고칠 방법 제안 (정정 전 기록, 위 정정 결과로 1번 구현됨)

1. `03_train_test_split.py`의 입력을 `data/run_sessions.pkl`의 6세션 이상 사용자(`config.RAW_REFERENCE_DATA_PATH`)로 바꾸고 파이프라인에서 17 앞에 둔다. 위 재구성으로 같은 목록이 나오는 것을 먼저 확인한 뒤 적용한다(출력 해시가 `docs/environment.md`와 같아야 함).
2. 또는 고정 ID 파일을 "고정 입력"으로 인정하고 `config.py`가 시작할 때 sha256(위 `environment.md` 해시)을 검사하게 한다. 파일은 데이터 라이선스상 공개 저장소에 올리지 않으므로 별도 전달 경로를 문서에 적는다.
3. 17이 ID 파일을 비교 표에만 쓰므로 파일이 없을 때 비교 표만 건너뛰는 옵션도 가능하다(분할 배정은 03이 먼저 만든다).

## 4. `data/run_sessions_deduped_merged_with_moving.pkl`의 출처

| 항목 | 내용 |
|---|---|
| 파일 sha256 | `a847a424d22f336ed03c6813e233225235d4732b3077287b9433727fc01640b3`(`docs/environment.md`와 같음) |
| 파일 수정 시각 | 2026-10-07 11:20(같은 시각에 `run_sessions_deduped_merged.pkl`과 `run_sessions_deduped.pkl`이 만들어짐) |
| 파일을 처음 언급한 커밋 | `d1efa9c`(2026-10-07 11:22:28, "타깃 페이스 정의 확정…"). `docs/target_pace_definition.md`가 "`merged_session_ids`로 각 조각의 moving_duration을 합산해 붙인 `run_sessions_deduped_merged_with_moving.pkl`을 썼다"고 설명 |
| 그 커밋이 추가한 코드 | `18b_moving_duration_extract.py`는 조각 단위 `session_moving_duration.pkl`(원본 세션 id별 이동 시간)만 쓴다. `18a`, `19`, `20`도 이 파일을 만들지 않는다 |
| 찾아본 곳 | `git log --all -S"with_moving"`, `-S"moving_duration_sec"`, `-S"moving_pace"`: 읽기·설명하는 커밋만 나오고 **이 파일을 쓰는(`to_pickle`) 코드는 어떤 커밋에도 없음**. 삭제된 파일 목록(감사 CSV 등)에도 없음. 스태시 없음, 브랜치는 `main`뿐. 저장소 밖 파일은 찾지 못했다(추적되지 않은 `ddsaf`는 `git log` 출력 붙여넣기로 코드가 아님) |
| 결론 | **생성 코드 미확인, 해시로 고정된 입력.** 아마도 10월 7일 임시(인라인) 실행으로 만들어졌을 것으로 추정되나 확인하지 못했다 |
| 이 파일을 읽는 단계 | 24(확장 피처), 33(극단 세션 대응), 그 밖에 29·환경 기록이 경로·해시를 참조 |

### 재구성 방법 (이번 조사에서 확인, 구현하지 않음)

`data/run_sessions_deduped_merged.pkl`에 아래 규칙을 적용하면 `_with_moving.pkl`의 네 열(`moving_duration_sec`, `moving_pace`, `gap_sec`, `gap_ratio`)이 **전부 정확히 같은 값**(105,709행, 최대 절대 차이 0.0, 열 순서 동일)으로 재현된다.

- `moving_duration_sec` = min(`merged_session_ids`에 속한 조각들의 `session_moving_duration.pkl`(18b 산출물) `moving_duration_sec` 합, `duration_sec`)
- `moving_pace` = `moving_duration_sec` / `actual_distance_km`
- `gap_sec` = `duration_sec` − `moving_duration_sec`, `gap_ratio` = `gap_sec` / `duration_sec`

**조각 합을 `duration_sec`로 자르는 단계가 없는 단순 합은 42개 세션(0.04%)에서 달라진다**(`repro_check_5km.md`에서 처음 관찰한 차이). 아래는 그 42개의 집계다.

| 항목 | 값 |
|---|---|
| 차이가 나는 세션 | 42 / 105,709 (0.04%), 모두 병합되지 않은 단일 조각 세션 |
| 부호 | 42개 모두 단순 합이 저장값보다 큼(저장값은 `duration_sec`와 같음 = 상한으로 잘린 값) |
| 차이 크기 | 최소 1.0초, 중앙값 2.5초, 90% 분위 약 1,572초, 최대 7,168초 |
| 구간별 개수 | 1초 이하 20개, 1~10초 4개, 10~100초 10개, 100초 초과 8개 |
| 원인 | 18b의 조각 이동 시간 합이 세션 표의 `duration_sec`(경과 시간)보다 큰 세션이 42개 있어, 저장 파일은 이를 `duration_sec`로 잘랐다(`gap_sec` ≥ 0 보장). 단순 합이면 `gap_sec`가 음수가 된다 |

### 고칠 방법 제안 (구현하지 않음)

위 재구성 규칙을 `17` 뒤에 두는 단계(예: `18c_attach_moving_duration.py`)로 저장소에 추가하고, 출력 해시가 `a847a424…`와 같은지 확인한다. 18b가 `session_moving_duration_30s.pkl`·`_120s.pkl` 변형(10월 7일 10:57·11:06 생성)도 만들었지만 커밋된 18b 코드는 기본 한 파일만 쓴다. 변형은 어떤 단계도 읽지 않아 이번 재현에는 영향이 없다(단순 합이 달라지는 것으로 확인, 위 42개와 무관).
