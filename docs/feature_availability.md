# 피처 가용성(Feature Availability) 표

10월 2주차 1번 작업 산출물. "예측 시점(타깃 세션 시작 전)에 앱이 이미 가지고
있는 값인가"를 확정하기 위해, raw 파일의 전체 필드 목록을 실제로 확인하고
각 피처가 러닝 전/후 중 언제 확보되는지 표로 정리한다.

## 1. 원본(`endomondoHR.json`)에 실제로 존재하는 전체 필드 목록

전체 6.12GB 파일을 `grep -oE "'[a-zA-Z_]+':"`로 **전수 스캔**해서 확인한
결과, 다음 11개 키 외에는 존재하지 않는다 (추측 아님, 전체 파일 스캔 결과):

| 필드명 | 타입 | 값 예시 |
|---|---|---|
| `speed` | list[float], 세션당 보통 500개 | `[6.8652, 16.4736, 19.1988, ...]` |
| `altitude` | list[float] | `[41.6, 40.6, 40.6, ...]` |
| `gender` | str | `'male'` |
| `heart_rate` | list[int] | `[100, 111, 120, ...]` |
| `id` | int | `396826535` (세션/워크아웃 id) |
| `url` | str | `'https://www.endomondo.com/users/<USER_ID>/workouts/<WORKOUT_ID>'` |
| `userId` | int | `10921915` |
| `timestamp` | list[int], unix epoch seconds | `[1408898746, 1408898754, 1408898765, ...]` |
| `longitude` | list[float] | `[24.64977040886879, ...]` |
| `latitude` | list[float] | `[60.173348765820265, ...]` |
| `sport` | str | `'bike'`, `'run'` 등 49종 |

**계획/목표 거리 필드 확인 결과: 없음 (확정).** 위 11개 키가 전부이며,
`plan`/`goal`/`target`/`distance`/`duration`이라는 문자열 자체가 (대소문자
무관) 전체 파일에 **단 한 번도 등장하지 않는다** — `grep -ci`로 전체 파일
검색 결과 전부 0건. 즉 사용자가 러닝 전에 입력한 계획/목표 거리에 해당할
만한 필드가 원본에 전혀 없다는 것을 전수 조사로 확정했다 (추측이 아님).

## 2. 거리 컬럼 분리: `planned_distance_km` / `actual_distance_km`

| 컬럼명 | 출처 원본 필드 | 계산 방법 | 확보 시점 | 상태 |
|---|---|---|---|---|
| `planned_distance_km` | (없음) | - | 러닝 전 | **사용 불가** — 원본에 해당 필드가 없음을 위 전수 조사로 확정. 이 컬럼은 어떤 모델 입력에도 쓰지 않는다. |
| `actual_distance_km` | `latitude`, `longitude` | 세션 내 연속 GPS 좌표 간 haversine 거리를 누적 합산 (GPS 점프 구간은 제외, `01_parse_run_sessions.py` 참고) | **러닝 후** (러닝이 끝나야 전체 좌표를 다 알 수 있음) | 사용 가능 — 단, **과거(이미 끝난) 세션에 대해서만** 입력으로 사용 가능. 타깃(예측 대상) 세션에는 입력 금지. |

**이름 변경**: 기존 10월 1주차(v2) 파이프라인에서 `target.distance_km`이라고
불렀던 값은 실제로는 GPS(haversine)로 계산한 **완주 거리**이며, 이는
러닝이 끝나야 확정되는 결과값이다. "계획값"이라는 이름이 틀렸으므로,
이번 주부터 `actual_distance_km`으로 이름을 바로잡는다. 이름을 바꿔도
예측 시점에 존재하지 않는 정보라는 사실은 변하지 않으므로, 메인
예측 태스크의 입력에서는 제외하고 "사후 거리 조건 분석"으로만 남긴다
(`10_posthoc_distance_condition.py` 참고).

## 3. 그 외 피처들의 확보 시점

| 컬럼명 | 출처 원본 필드 | 계산 방법 | 확보 시점 |
|---|---|---|---|
| `avg_pace_sec_per_km` | `timestamp`, `latitude`, `longitude` (파생) | `duration_sec / actual_distance_km` | 러닝 후 |
| `avg_heart_rate` | `heart_rate` | 생리학적 범위(30~230bpm) 밖 값 제외 후 평균 | 러닝 후 |
| `duration_sec` | `timestamp` | 마지막 타임스탬프 - 첫 타임스탬프 | 러닝 후 |
| `gender` | `gender` | 그대로 사용 | 가입 시 (세션과 무관하게 항상 사용 가능) |

과거(이미 끝난) 세션의 `actual_distance_km`/`avg_pace_sec_per_km`/
`avg_heart_rate`/`duration_sec`는 "과거 기록"이므로 입력으로 사용 가능하다.
금지되는 것은 **타깃(예측 대상) 세션 자신의** 위 값들이다.

## 4. 날씨

`weather`, `temperature`, `humidity` 등 날씨 관련 필드도 전체 파일 스캔
결과 존재하지 않음을 이미 확인했다 (1주차에 확인, 이번 주 재확인 결과 동일).
입력/타깃 정의에서 계속 제외한다.

## 5. 재현 방법 (실제로 실행한 명령과 결과 전문)

누구나 저장소 루트(`endomondoHR.json`이 있는 폴더)에서 아래 명령을 그대로
실행해 위 결론을 재현할 수 있다. (Git Bash 기준. 전체 6.12GB 파일을 한 번에
메모리에 올리지 않는 스트리밍 방식 — `grep`은 라인 단위로 처리한다.)

### 5-1. 전체 필드(키) 목록 전수 추출

```bash
grep -oE "'[a-zA-Z_]+':" endomondoHR.json | sort -u
```

실행 결과 (11개 전부):
```
'altitude':
'gender':
'heart_rate':
'id':
'latitude':
'longitude':
'speed':
'sport':
'timestamp':
'url':
'userId':
```

### 5-2. 계획/목표 거리 관련 키워드 존재 여부 (대소문자 무관, 전체 파일)

```bash
for kw in plan goal target distance duration; do
  cnt=$(grep -ci "$kw" endomondoHR.json)
  echo "$kw: $cnt"
done
```

실행 결과:
```
plan: 0
goal: 0
target: 0
distance: 0
duration: 0
```

### 5-3. 날씨 관련 키워드 존재 여부 (재확인)

```bash
for kw in weather temperature humidity; do
  cnt=$(grep -ci "$kw" endomondoHR.json)
  echo "$kw: $cnt"
done
```

실행 결과:
```
weather: 0
temperature: 0
humidity: 0
```

### 5-4. 샘플 레코드 1건의 필드별 값/타입 확인

```bash
python -c "
import ast
with open('endomondoHR.json', 'r', encoding='utf-8') as f:
    line = f.readline()
rec = ast.literal_eval(line)
for k, v in rec.items():
    if isinstance(v, list):
        print(f'{k}: list(len={len(v)}), sample={v[:3]}')
    else:
        print(f'{k}: {v!r}')
"
```

실행 결과:
```
speed: list(len=500), sample=[6.8652, 16.4736, 19.1988]
altitude: list(len=500), sample=[41.6, 40.6, 40.6]
gender: 'male'
heart_rate: list(len=500), sample=[100, 111, 120]
id: 396826535
url: 'https://www.endomondo.com/users/<USER_ID>/workouts/<WORKOUT_ID>'
userId: 10921915
timestamp: list(len=500), sample=[1408898746, 1408898754, 1408898765]
longitude: list(len=500), sample=[24.64977040886879, 24.65014273300767, 24.650910682976246]
latitude: list(len=500), sample=[60.173348765820265, 60.173239801079035, 60.17298021353781]
sport: 'bike'
```

**참고**: `endomondoHR.json`은 유효한 JSON이 아니라 한 줄에 파이썬 dict
리터럴(싱글쿼트)이 하나씩 있는 형식이라, `json.load()`가 아니라
`ast.literal_eval()`로 한 줄씩 읽어야 한다 (`01_parse_run_sessions.py` 참고).
위 5-1, 5-2, 5-3의 `grep` 명령은 파싱 없이 텍스트 패턴만 찾는 것이라
이 형식 문제와 무관하게 전체 파일에 대해 정확히 동작한다.

## 6. 알려진 한계

- `distance_km` → `actual_distance_km` 컬럼명 리네임(`11_migrate_rename_distance_column.py`)
  후 01단계(`01_parse_run_sessions.py`)의 동작 검증은 원본 6.12GB 파일 전체가
  아니라 **앞부분 1만 줄 샘플**로만 스모크 테스트했다. 6GB 전체를 01단계로
  다시 돌려서 끝까지 확인한 것은 아니며(캐시된 pkl은 마이그레이션 스크립트로
  컬럼명만 바꾼 것), 1만 줄 밖의 구간에서 예기치 못한 스키마 차이가 있다면
  아직 발견되지 않았을 수 있다.
