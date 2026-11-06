# -*- coding: utf-8 -*-
"""import_apple_health 단위 테스트(합성 XML·합성 GPX만 사용). pytest 없이 `python demo/test_import_apple_health.py`로 실행된다. numpy/pandas 필요."""
import contextlib
import csv
import io
import math
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coach_demo as cd  # noqa: E402
import import_apple_health as ia  # noqa: E402

STEP_RAD = 0.1 / 6371.0088            # haversine 거리가 정확히 0.1km가 되는 위도 증가분(rad)
LAT0, LON0 = 37.0, 127.0


def gpx(n_points, stop_points=0, step_sec=30):
    """합성 GPX: 제자리 stop_points개 뒤 북쪽으로 100m씩 이동(30초 간격). 5km는 50번째 이동 구간 끝."""
    pts, t = [], 0
    lat = LAT0
    for i in range(n_points):
        if i >= stop_points:
            lat += math.degrees(STEP_RAD)
        pts.append((lat, t))
        t += step_sec
    body = ""
    for la, tt in pts:
        body += '<trkpt lat="%.9f" lon="%.9f"><time>2025-10-12T%02d:%02d:%02dZ</time></trkpt>' % (la, LON0, tt // 3600, (tt % 3600) // 60, tt % 60)
    return '<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>%s</trkseg></trk></gpx>' % body


def workout(start, end, km, src, route=None, hr=None, kind="HKWorkoutActivityTypeRunning"):
    ch = '<WorkoutStatistics type="HKQuantityTypeIdentifierDistanceWalkingRunning" sum="%s" unit="km"/>' % km
    if hr:
        ch += '<WorkoutStatistics type="HKQuantityTypeIdentifierHeartRate" average="%s" unit="count/min"/>' % hr
    if route:
        ch += '<WorkoutRoute sourceName="x"><FileReference path="/workout-routes/%s"/></WorkoutRoute>' % route
    return '<Workout workoutActivityType="%s" sourceName="%s" startDate="%s" endDate="%s">%s</Workout>' % (kind, src, start, end, ch)


def make_zip(tmp, workouts, gpxs, xml_name="내보내기.xml"):
    xml = ('<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE HealthData [<!ELEMENT HealthData (Record*,Workout*)>]><HealthData locale="ko_KR">'
           '<Record type="HKQuantityTypeIdentifierStepCount" value="1"/>' + "".join(workouts) + "</HealthData>")
    p = Path(tmp) / "export.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("apple_health_export/" + xml_name, xml)
        zf.writestr("apple_health_export/내보내기_cda.xml", "<ClinicalDocument/>")
        for name, text in gpxs.items():
            zf.writestr("apple_health_export/workout-routes/" + name, text)
    return str(p)


def sample_zip(tmp):
    w = [workout("2025-10-12 07:00:00 +0900", "2025-10-12 07:26:30 +0900", 5.2, "Apple Watch", "route_a.gpx", hr=151.4),
         workout("2025-10-13 07:00:00 +0900", "2025-10-13 07:31:00 +0900", 5.2, "Nike Run Club"),                    # 5km 후보인데 경로 없음
         workout("2025-10-14 07:00:00 +0900", "2025-10-14 07:20:00 +0900", 3.0, "Apple Watch", "route_c.gpx"),         # 5km 아님(경로 있음)
         workout("2025-10-15 07:00:00 +0900", "2025-10-15 07:50:00 +0900", 5.3, "Apple Watch", "route_d.gpx"),         # 경로 거리 5km 미달
         workout("2025-10-16 07:00:00 +0900", "2025-10-16 07:30:00 +0900", 4.0, "Apple Watch", kind="HKWorkoutActivityTypeWalking"),   # 달리기 아님
         workout("2025-10-17 07:00:00 +0900", "2025-10-17 07:50:00 +0900", 5.1, "Apple Watch", "route_e.gpx")]        # 제자리 270초 + 5km
    g = {"route_a.gpx": gpx(60), "route_c.gpx": gpx(20), "route_d.gpx": gpx(30), "route_e.gpx": gpx(70, stop_points=10)}
    return make_zip(tmp, w, g)


def _run(tmp, zip_path):
    out = str(Path(tmp) / "runs.csv")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = ia.main(["--zip", zip_path, "--out", out])
    return code, buf.getvalue(), out


def _read(out):
    with open(out, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_t5_matches_research_function_and_known_value():
    with tempfile.TemporaryDirectory() as tmp:
        code, _text, out = _run(tmp, sample_zip(tmp))
        assert code == 0
        rows = {r["start_time"]: r for r in _read(out)}
        a = rows["2025-10-12T07:00:00"]
        assert abs(float(a["t5_sec"]) - 1500.0) < 1e-3 and float(a["stop_sec"]) == 0.0      # 100m/30s 균일 이동 -> 5km는 정확히 1500초
        # 연구 파이프라인 함수(38.compute_t5)에 같은 점을 직접 넣은 값과 같다
        ts, lat, lon = ia.read_gpx(io.BytesIO(gpx(60).encode("utf-8")))
        mod, parse = ia.load_research()
        direct = mod.compute_t5(parse, {"latitude": lat, "longitude": lon, "timestamp": ts}, ts[0])
        assert abs(float(a["t5_sec"]) - direct) < 1e-3
        assert a["avg_hr"] == "151.4" and a["source"] == "애플워치" and a["distance_km"] == "5.2"


def test_stop_seconds_and_t5_include_stationary_time():
    with tempfile.TemporaryDirectory() as tmp:
        _code, _text, out = _run(tmp, sample_zip(tmp))
        e = {r["start_time"]: r for r in _read(out)}["2025-10-17T07:00:00"]
        assert abs(float(e["stop_sec"]) - 270.0) < 1e-6          # 제자리 점 10개 = 구간 9개 x 30초
        assert abs(float(e["t5_sec"]) - 1770.0) < 1e-3           # 경과 시간 기준: 정지 270초 + 이동 1500초


def test_no_route_and_non_candidates_are_blank_and_reasons_counted():
    with tempfile.TemporaryDirectory() as tmp:
        _code, text, out = _run(tmp, sample_zip(tmp))
        rows = {r["start_time"]: r for r in _read(out)}
        n = rows["2025-10-13T07:00:00"]
        assert n["t5_sec"] == "" and n["stop_sec"] == "" and n["source"] == "기타"
        c = rows["2025-10-14T07:00:00"]
        assert c["t5_sec"] == "" and c["stop_sec"] != ""        # 5km 아님: t5 없음, 경로가 있으니 정지 시간은 있음
        d = rows["2025-10-15T07:00:00"]
        assert d["t5_sec"] == ""                                 # 경로가 5km에 못 미침
        assert "2025-10-16T07:00:00" not in rows                 # 걷기는 제외
        assert "달리기 전체: 5" in text and "CSV에 쓴 달리기: 5" in text and "t5 계산됨: 2" in text
        assert "5km 후보인데 경로 없음: 1" in text and "경로 거리 5km 미달: 1" in text and "거리가 5.0~5.5km 아님: 1" in text


def test_times_are_wall_clock_without_conversion_and_duration_is_elapsed():
    with tempfile.TemporaryDirectory() as tmp:
        _code, _text, out = _run(tmp, sample_zip(tmp))
        a = {r["start_time"]: r for r in _read(out)}["2025-10-12T07:00:00"]
        assert a["end_time"] == "2025-10-12T07:26:30" and a["duration_sec"] == "1590"      # +0900 표기는 떼기만 하고 변환하지 않음


def test_out_inside_repo_warns_and_stops():
    with tempfile.TemporaryDirectory() as tmp:
        z = sample_zip(tmp)
        target = ia.REPO / "demo" / "should_not_exist.mine.csv"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = ia.main(["--zip", z, "--out", str(target)])
        assert code == 2 and "저장소 폴더 안" in buf.getvalue() and not target.exists()


def test_no_coordinates_in_output_or_csv():
    with tempfile.TemporaryDirectory() as tmp:
        _code, text, out = _run(tmp, sample_zip(tmp))
        blob = text + Path(out).read_text(encoding="utf-8")
        assert "127.0" not in blob and "37.0" not in blob and "lat" not in blob.lower()


def test_generated_csv_works_with_demo_and_avg_hr_is_optional():
    with tempfile.TemporaryDirectory() as tmp:
        _code, _text, out = _run(tmp, sample_zip(tmp))
        asof = "2025-10-20T00:00:00"
        res, text = cd.run(out, asof, cd.load_params())
        assert res["eligible"] is False or "다음 7일" in text           # 입력이 충분하지 않아도 멈추지 않는다
        # avg_hr 열을 지우거나 비워도 예측은 같다
        with open(out, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        p_no = Path(tmp) / "no_hr.csv"
        with open(p_no, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[c for c in rows[0] if c != "avg_hr"])
            w.writeheader()
            for r in rows:
                w.writerow({k: v for k, v in r.items() if k != "avg_hr"})
        p_blank = Path(tmp) / "blank_hr.csv"
        with open(p_blank, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(dict(r, avg_hr=""))
        t_full = cd.run(out, asof, cd.load_params())[1]
        assert cd.run(str(p_no), asof, cd.load_params())[1] == t_full and cd.run(str(p_blank), asof, cd.load_params())[1] == t_full


def test_main_xml_is_found_even_when_korean_file_name_is_garbled():
    with tempfile.TemporaryDirectory() as tmp:
        garbled = "내보내기.xml".encode("utf-8").decode("cp437") + ".xml"           # 실제 ZIP에서 본 형태: UTF-8 이름이 cp437로 읽힌 글자
        z = make_zip(tmp, [workout("2025-10-12 07:00:00 +0900", "2025-10-12 07:26:30 +0900", 5.2, "Apple Watch", "route_a.gpx")], {"route_a.gpx": gpx(60)}, xml_name=garbled)
        code, _text, out = _run(tmp, z)
        assert code == 0 and len(_read(out)) == 1


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print("ok", n)
    print(len(tests), "tests passed")
