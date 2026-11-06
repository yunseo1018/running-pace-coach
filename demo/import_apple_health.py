# -*- coding: utf-8 -*-
"""
애플 건강 내보내기 ZIP -> 시연용 CSV (demo/coach_demo.py 입력).
사용: python demo/import_apple_health.py --zip 내보내기.zip --out <저장소 밖 폴더>/runs.mine.csv

- ZIP 안 apple_health_export/의 메인 XML(내보내기.xml 또는 export.xml)을 iterparse로 스트리밍해 읽는다(약 1GB도 메모리에 올리지 않음).
- 달리기 = Workout(workoutActivityType=HKWorkoutActivityTypeRunning). 거리 = 자식 WorkoutStatistics(DistanceWalkingRunning)의 sum(km로 환산).
- 경로 = 자식 WorkoutRoute/FileReference의 path -> workout-routes/*.gpx. 경로가 있는 기록(주로 애플워치)에만 t5_sec·stop_sec를 채운다.
- t5_sec: 경로가 있고 거리가 5.0~5.5km일 때만. 연구 파이프라인(38_build_5km_dataset.py의 compute_t5 + 37의 cum_with_skip·01의 haversine/속도 임계)을 그대로 불러 쓴다
  (누적 haversine 5.000km 도달 시각의 인접 두 점 선형 보간, 세션 첫 점부터의 경과 시간). 이를 위해 numpy/pandas가 필요하다(시연 본체 coach_demo.py는 표준 라이브러리만 쓴다).
- stop_sec: 같은 GPX에서 시간차 > 0이고 속도 1km/h 미만인 점 사이 구간의 시간 합(세션 전체; docs/posthoc_spec_b2_failure_5km.md 5절 3).
- 시간대: start_time·end_time은 +0900 등 표기를 바꾸지 않고 현지 벽시계 시각 그대로(시간대 표기 없이) 적는다. coach_demo는 표기 없는 시각을 UTC로 두고 일관되게 쓰므로 --asof도 같은 벽시계(한국 시각)로 지정하면 된다.
- 좌표는 어디에도 저장하거나 출력하지 않는다. 출력 CSV(개인 건강 데이터)는 저장소 밖에 두며 --out이 저장소 안이면 경고하고 멈춘다.
"""
import argparse
import csv
import io
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNNING = "HKWorkoutActivityTypeRunning"
DIST_TYPE = "HKQuantityTypeIdentifierDistanceWalkingRunning"
HR_TYPE = "HKQuantityTypeIdentifierHeartRate"
KM_LO, KM_HI = 5.0, 5.5
STOP_KMH = 1.0
COLUMNS = ["start_time", "end_time", "distance_km", "duration_sec", "avg_hr", "t5_sec", "stop_sec", "source"]
_RESEARCH = None


def load_research():
    """연구 파이프라인의 T5 함수(38_build_5km_dataset.compute_t5)와 haversine·속도 임계(37·01)를 그대로 가져온다. 반환 (모듈 38, parse)."""
    global _RESEARCH
    if _RESEARCH is None:
        old = os.getcwd()
        os.chdir(REPO)                      # 38·37이 상대 경로로 서로(와 01)를 읽는다
        sys.path.insert(0, str(REPO))
        saved = (sys.stdout, sys.stderr)    # config.py가 import 때 stdout/stderr를 감싸 바꾸므로 끝난 뒤 원래대로 되돌린다
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        try:
            spec = spec_from_file_location("build38", REPO / "38_build_5km_dataset.py")
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            _RESEARCH = (mod, mod.feas.import_parser())
        finally:
            sys.stdout, sys.stderr = saved
            os.chdir(old)
            sys.path.pop(0)
    return _RESEARCH


def inside_repo(path):
    p = Path(path).resolve()
    try:
        p.relative_to(REPO)
        return True
    except ValueError:
        return False


def parse_apple_time(text):
    """'2025-10-12 07:00:00 +0900' -> (벽시계 naive datetime, 에포크 초). 벽시계는 시간대 표기를 떼기만 하고 변환하지 않는다."""
    dt = datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S %z")
    return dt.replace(tzinfo=None), dt.timestamp()


def find_members(zf):
    """(메인 XML 멤버 이름, {gpx 파일 이름: 멤버 이름})."""
    main = None
    for n in zf.namelist():
        base = n.rsplit("/", 1)[-1]
        if base in ("내보내기.xml", "export.xml"):
            main = n
    if main is None:
        raise SystemExit("오류: ZIP 안에서 메인 XML(내보내기.xml 또는 export.xml)을 찾지 못했습니다.")
    gpx = {n.rsplit("/", 1)[-1]: n for n in zf.namelist() if n.lower().endswith(".gpx")}
    return main, gpx


def _km(value, unit):
    v = float(value)
    return v / 1000.0 if unit == "m" else (v * 1.609344 if unit == "mi" else v)


def _workout(el):
    """Workout 요소 -> dict. 자식 요소는 이 시점에 모두 읽혀 있다."""
    d = {"start": el.get("startDate"), "end": el.get("endDate"), "src": el.get("sourceName") or "", "km": None, "hr": None, "route": None}
    for ch in el:
        if ch.tag == "WorkoutStatistics":
            if ch.get("type") == DIST_TYPE and ch.get("sum") is not None:
                d["km"] = _km(ch.get("sum"), ch.get("unit") or "km")
            elif ch.get("type") == HR_TYPE and ch.get("average") is not None:
                d["hr"] = float(ch.get("average"))
        elif ch.tag == "WorkoutRoute":
            for fr in ch:
                if fr.tag == "FileReference" and fr.get("path"):
                    d["route"] = fr.get("path").rsplit("/", 1)[-1]
    if d["km"] is None and el.get("totalDistance") is not None:
        d["km"] = _km(el.get("totalDistance"), el.get("totalDistanceUnit") or "km")
    return d


def iter_workouts(stream):
    """XML 스트림을 iterparse로 읽어 달리기 Workout만 차례로 내준다(처리한 최상위 요소는 즉시 비워 메모리를 쓰지 않음)."""
    root, depth = None, 0
    for ev, el in ET.iterparse(stream, events=("start", "end")):
        if ev == "start":
            if root is None:
                root = el
            depth += 1
            continue
        depth -= 1
        if depth == 1:                                   # 최상위 자식 하나가 끝남
            if el.tag == "Workout" and el.get("workoutActivityType") == RUNNING:
                yield _workout(el)
            root.clear()


def read_gpx(stream):
    """GPX -> (epoch 초 목록, 위도 목록, 경도 목록). 좌표는 메모리 안에서만 쓰고 저장·출력하지 않는다."""
    ts, lat, lon = [], [], []
    cur = None
    for ev, el in ET.iterparse(stream, events=("start", "end")):
        tag = el.tag.rsplit("}", 1)[-1]
        if ev == "start":
            if tag == "trkpt":
                cur = [float(el.get("lat")), float(el.get("lon")), None]
            continue
        if tag == "time" and cur is not None:
            cur[2] = datetime.fromisoformat((el.text or "").strip().replace("Z", "+00:00")).timestamp()
        elif tag == "trkpt":
            if cur and cur[2] is not None:
                lat.append(cur[0])
                lon.append(cur[1])
                ts.append(cur[2])
            cur = None
            el.clear()
    return ts, lat, lon


def stop_seconds(parse, ts, lat, lon):
    """시간차 > 0이고 속도 1km/h 미만인 점 사이 구간의 시간 합(초, 세션 전체)."""
    total = 0.0
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        if dt > 0 and parse.haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i]) / dt * 3600.0 < STOP_KMH:
            total += dt
    return total


def t5_from_points(ts, lat, lon):
    """연구 파이프라인의 compute_t5 그대로. 경로 누적 거리가 5.000km에 닿지 않으면 None."""
    mod, parse = load_research()
    cum, _ = mod.feas.cum_with_skip(parse, lat, lon, ts)
    if not (len(cum) > 1 and cum[-1] >= 5.0):
        return None
    return float(mod.compute_t5(parse, {"latitude": lat, "longitude": lon, "timestamp": ts}, ts[0]))


def build_rows(zip_path):
    """ZIP -> (CSV 행 목록, 이유별 개수). 좌표·개별 값은 반환하지 않는다."""
    _mod, parse = load_research()
    st = {"달리기 전체": 0, "CSV에 쓴 달리기": 0, "거리 정보 없어 제외": 0, "경로가 있는 기록": 0, "t5 계산됨": 0,
          "t5 비움: 거리가 5.0~5.5km 아님": 0, "t5 비움: 5km 후보인데 경로 없음": 0, "t5 비움: 경로 파일 없음·읽기 실패": 0, "t5 비움: 경로 거리 5km 미달": 0}
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        main, gpx = find_members(zf)
        with zf.open(main) as xml:
            for w in iter_workouts(xml):
                st["달리기 전체"] += 1
                if w["km"] is None or not w["start"] or not w["end"]:
                    st["거리 정보 없어 제외"] += 1
                    continue
                s_wall, s_ep = parse_apple_time(w["start"])
                e_wall, e_ep = parse_apple_time(w["end"])
                cand = KM_LO <= w["km"] <= KM_HI
                t5 = stop = None
                if w["route"]:
                    st["경로가 있는 기록"] += 1
                    member = gpx.get(w["route"])
                    pts = None
                    if member:
                        try:
                            with zf.open(member) as g:
                                pts = read_gpx(io.BytesIO(g.read()))
                        except (ET.ParseError, ValueError, KeyError):
                            pts = None
                    if not pts or len(pts[0]) < 2:
                        if cand:
                            st["t5 비움: 경로 파일 없음·읽기 실패"] += 1
                    else:
                        ts, lat, lon = pts
                        stop = stop_seconds(parse, ts, lat, lon)
                        if cand:
                            t5 = t5_from_points(ts, lat, lon)
                            if t5 is None:
                                st["t5 비움: 경로 거리 5km 미달"] += 1
                            else:
                                st["t5 계산됨"] += 1
                        del pts, ts, lat, lon
                elif cand:
                    st["t5 비움: 5km 후보인데 경로 없음"] += 1
                if not cand:
                    st["t5 비움: 거리가 5.0~5.5km 아님"] += 1
                src = "애플워치" if ("Watch" in w["src"] or "워치" in w["src"]) else "기타"
                rows.append([s_wall.strftime("%Y-%m-%dT%H:%M:%S"), e_wall.strftime("%Y-%m-%dT%H:%M:%S"), round(w["km"], 3), int(round(e_ep - s_ep)),
                             "" if w["hr"] is None else round(w["hr"], 1), "" if t5 is None else round(t5, 3), "" if stop is None else round(stop, 1), src])
                st["CSV에 쓴 달리기"] += 1
    rows.sort(key=lambda r: r[0])
    return rows, st


def main(argv=None):
    ap = argparse.ArgumentParser(description="애플 건강 내보내기 ZIP에서 시연용 CSV 만들기")
    ap.add_argument("--zip", required=True, help="애플 건강 내보내기 ZIP 경로")
    ap.add_argument("--out", required=True, help="출력 CSV 경로(저장소 밖)")
    a = ap.parse_args(argv)
    if inside_repo(a.out):
        print("경고: --out이 저장소 폴더 안입니다. 개인 건강 데이터가 커밋될 수 있어 중단합니다. 저장소 밖 경로(예: <저장소 밖 폴더>/my_runs/runs.mine.csv)를 지정하세요.")
        return 2
    if not Path(a.zip).is_file():
        print("오류: ZIP 파일을 찾을 수 없습니다: %s" % a.zip)
        return 2
    rows, st = build_rows(a.zip)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print("저장: %s (%d행)" % (a.out, len(rows)))
    for k, v in st.items():
        print("  %s: %d" % (k, v))
    print("참고: 시각은 현지 벽시계 그대로(시간대 표기 없음)이므로 --asof도 같은 벽시계로 지정하세요. 좌표는 저장·출력하지 않았습니다.")
    return 0


if __name__ == "__main__":
    for _st in (sys.stdout, sys.stderr):
        if hasattr(_st, "reconfigure"):
            _st.reconfigure(encoding="utf-8")
    sys.exit(main())
