# 가져오기 판정 기준 정렬 사양 (`demo/import_apple_health.py`)

목적: 애플 건강 내보내기에서 만드는 시연용 CSV의 종료 시각·경과 시간·거리를 연구 데이터(FitRec)가 계산된 방식과 맞춘다. B2 식·파라미터·명목 80% 범위·시연 계산식(`coach_demo.py`)은 바꾸지 않는다. 근거는 `docs/source_trace.md`(Workout 종료−시작이 `duration` 속성·GPX 점 시각과 60초 넘게 다른 기록이 87개 중 20개, Workout 거리와 GPX 재계산 거리의 차이 −5.9%~+2.9%; 원인은 미확인).

## 1. 종료 시각과 경과 시간

- **경로(GPX)가 있고 점이 2개 이상 읽힌 기록**: `start_time` = GPX 첫 점 시각, `end_time` = GPX 마지막 점 시각, `duration_sec` = 마지막 점 − 첫 점(초, 반올림). 연구 데이터가 점 시각으로 경과 시간을 계산한 것(`start_time` = 첫 timestamp, `compute_t5`의 `ts[0]`)과 같은 방식이다. 시각은 기존 규칙대로 현지 벽시계 값(시간대 표기 없음)으로 적는다: GPX의 UTC 시각에 그 Workout 시작 시각의 UTC 오프셋(예: +0900)을 더한다.
  - `start_time`도 GPX 첫 점으로 바꾸는 이유: 종료만 바꾸면 `end_time − start_time`이 `duration_sec`과 어긋난다(Workout 시작과 첫 점의 차이는 초 단위).
- **경로가 없거나 읽을 수 없는 기록**: `start_time` = Workout 시작(벽시계), `duration_sec` = Workout `duration` 속성(단위 min/sec/hr를 초로 환산), `end_time` = start_time + duration_sec. `duration` 속성이 없으면 기존처럼 종료−시작을 쓰고 그 개수를 요약에 표시한다.
- **요약 표시**: 기존 방식(Workout 종료−시작)과 새 경과 시간이 60초 넘게 다른 기록 수.

## 2. 거리

- **경로가 있는 기록**: `distance_km` = GPX 점에서 연구와 같은 함수(`feas.cum_with_skip` = `01_parse_run_sessions.total_distance_km`의 haversine·스킵 규칙)로 다시 계산한 거리(소수 3자리). 건강 앱 거리는 `distance_km_health` 열에 따로 남긴다. `distance_src` = `gpx`.
- **경로가 없거나 읽을 수 없는 기록**: `distance_km` = 건강 앱 거리, `distance_src` = `health`, `distance_km_health` = 같은 값. 건강 앱 거리도 없으면 기존처럼 제외한다.
- 새 열: `source` 바로 뒤에 `distance_src`, `distance_km_health`를 둔다(`coach_demo.py`는 추가 열을 무시한다).
- **5.0~5.5km 판정은 `distance_km` 기준**이다. 건강 앱 거리 기준 판정과 달라진 기록 수(후보가 됨 / 후보에서 빠짐)를 요약에 표시한다. `t5_sec`·`stop_sec` 계산은 그대로다(연구 `compute_t5`).

## 3. 하지 않는 것

연구 문서의 결과·정의·분할, `b2_params.json`, `stratum_notes.json`, `coach_demo.py`의 예측·범위·문구를 바꾸지 않는다. 좌표는 저장·출력하지 않는다. 개인 CSV는 저장소 밖에만 둔다.

## 4. 테스트(합성 XML·GPX)

종료 시각 오류 사례(Workout 종료가 GPX 끝보다 훨씬 뒤: end_time·duration_sec은 점 시각), 경로 없는 기록의 duration 속성 환산, 거리 출처 2종(gpx/health)과 건강 앱 거리 열, 5km 판정 변경 개수(후보가 됨/빠짐), 60초 초과 차이 개수, 기존 테스트 전부(기대값이 이 사양으로 바뀐 곳만 갱신).
