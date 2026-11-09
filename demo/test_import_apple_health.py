# -*- coding: utf-8 -*-
"""import_apple_health 단위 테스트(합성 XML·합성 GPX만 사용). pytest 없이 `python demo/test_import_apple_health.py`로 실행된다. numpy/pandas 필요."""
import contextlib
import csv
import io
import math
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coach_demo as cd  # noqa: E402
import import_apple_health as ia  # noqa: E402

STEP_RAD = 0.1 / 6371.0088            # haversine 거리가 정확히 0.1km가 되는 위도 증가분(rad)
LAT0, LON0 = 37.0, 127.0


def gpx(n_points, stop_points=0, step_sec=30, t0="2025-10-11T22:00:00"):
    """합성 GPX: 제자리 stop_points개 뒤 북쪽으로 100m씩 이동(30초 간격). 5km는 50번째 이동 구간 끝. 시각은 UTC t0부터(기본 22:00Z = 07:00 +0900)."""
    base = datetime.fromisoformat(t0)
    pts, t = [], 0
    lat = LAT0
    for i in range(n_points):
        if i >= stop_points:
            lat += math.degrees(STEP_RAD)
        pts.append((lat, t))
        t += step_sec
    body = ""
    for la, tt in pts:
        body += '<trkpt lat="%.9f" lon="%.9f"><time>%sZ</time></trkpt>' % (la, LON0, (base + timedelta(seconds=tt)).strftime("%Y-%m-%dT%H:%M:%S"))
    return '<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>%s</trkseg></trk></gpx>' % body


def workout(start, end, km, src, route=None, hr=None, kind="HKWorkoutActivityTypeRunning", dur=None):
    ch = '<WorkoutStatistics type="HKQuantityTypeIdentifierDistanceWalkingRunning" sum="%s" unit="km"/>' % km
    if hr:
        ch += '<WorkoutStatistics type="HKQuantityTypeIdentifierHeartRate" average="%s" unit="count/min"/>' % hr
    if route:
        ch += '<WorkoutRoute sourceName="x"><FileReference path="/workout-routes/%s"/></WorkoutRoute>' % route
    da = ' duration="%s" durationUnit="min"' % dur if dur is not None else ""
    return '<Workout workoutActivityType="%s" sourceName="%s" startDate="%s" endDate="%s"%s>%s</Workout>' % (kind, src, start, end, da, ch)


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
    """합성 달리기 6개(걷기 1개 제외). 건강 앱 거리와 GPX 재계산 거리가 다르고, Workout 종료 시각이 GPX 끝과 다른 사례를 포함한다."""
    w = [workout("2025-10-12 07:00:00 +0900", "2025-10-12 07:31:00 +0900", 5.2, "Apple Watch", "route_a.gpx", hr=151.4, dur=25.5),     # 종료 시각 오류: GPX는 1530초, 종료−시작 1860초
         workout("2025-10-13 07:00:00 +0900", "2025-10-13 08:00:00 +0900", 5.2, "Nike Run Club", dur=31),                              # 5km 후보인데 경로 없음, duration 속성 31분(종료−시작은 60분)
         workout("2025-10-14 07:00:00 +0900", "2025-10-14 07:20:00 +0900", 3.0, "Apple Watch", "route_c.gpx", dur=9.5),               # 5km 아님(경로 있음)
         workout("2025-10-15 07:00:00 +0900", "2025-10-15 07:50:00 +0900", 5.3, "Apple Watch", "route_d.gpx", dur=14.5),              # 건강 앱 5.3km, GPX 2.9km -> 후보에서 빠짐
         workout("2025-10-16 07:00:00 +0900", "2025-10-16 07:30:00 +0900", 4.0, "Apple Watch", kind="HKWorkoutActivityTypeWalking"),   # 달리기 아님
         workout("2025-10-17 07:00:00 +0900", "2025-10-17 07:50:00 +0900", 5.1, "Apple Watch", "route_e.gpx", dur=30.5),              # 제자리 270초 + 5km, GPX 5.2km
         workout("2025-10-18 07:00:00 +0900", "2025-10-18 07:25:30 +0900", 4.8, "Apple Watch", "route_f.gpx", dur=25.5)]              # 건강 앱 4.8km, GPX 5.1km -> 후보가 됨
    g = {"route_a.gpx": gpx(52), "route_c.gpx": gpx(20, t0="2025-10-13T22:00:00"), "route_d.gpx": gpx(30, t0="2025-10-14T22:00:00"),
         "route_e.gpx": gpx(62, stop_points=10, t0="2025-10-16T22:00:00"), "route_f.gpx": gpx(52, t0="2025-10-17T22:00:00")}
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
        ts, lat, lon = ia.read_gpx(io.BytesIO(gpx(52).encode("utf-8")))
        mod, parse = ia.load_research()
        direct = mod.compute_t5(parse, {"latitude": lat, "longitude": lon, "timestamp": ts}, ts[0])
        assert abs(float(a["t5_sec"]) - direct) < 1e-3
        assert a["avg_hr"] == "151.4" and a["source"] == "애플워치"


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
        assert d["t5_sec"] == ""                                 # GPX 거리가 5km에 못 미침(건강 앱 5.3km였으나 후보에서 빠짐)
        assert "2025-10-16T07:00:00" not in rows                 # 걷기는 제외
        assert "달리기 전체: 6" in text and "CSV에 쓴 달리기: 6" in text and "t5 계산됨: 3" in text
        assert "5km 후보인데 경로 없음: 1" in text and "경로 거리 5km 미달: 0" in text and "거리가 5.0~5.5km 아님: 2" in text


def test_end_time_and_duration_use_gpx_points_when_workout_end_is_wrong():
    with tempfile.TemporaryDirectory() as tmp:
        _code, text, out = _run(tmp, sample_zip(tmp))
        rows = {r["start_time"]: r for r in _read(out)}
        a = rows["2025-10-12T07:00:00"]          # Workout 종료 07:31:00(1860초)이지만 GPX 첫~마지막 점은 1530초
        assert a["end_time"] == "2025-10-12T07:25:30" and a["duration_sec"] == "1530"
        e = rows["2025-10-17T07:00:00"]          # 종료 07:50:00(3000초) vs 점 시각 61x30 = 1830초
        assert e["end_time"] == "2025-10-17T07:30:30" and e["duration_sec"] == "1830"
        # 점 시각 UTC에 Workout 시작의 +0900 오프셋을 더해 벽시계로 적는다(시간대 표기 없음)
        assert all(len(r["start_time"]) == 19 and "+" not in r["start_time"] and "Z" not in r["end_time"] for r in rows.values())
        # 경로 없는 기록: duration 속성 31분으로 종료 시각을 정한다(종료−시작 60분 아님)
        n = rows["2025-10-13T07:00:00"]
        assert n["end_time"] == "2025-10-13T07:31:00" and n["duration_sec"] == "1860"
        # 요약: 기존 방식과 60초 넘게 다른 기록 수 = a, c, d, e, n (f는 1530초로 같음)
        assert "기존 방식(종료−시작)과 경과 시간이 60초 넘게 다른 기록: 5" in text
        assert "종료 시각·경과 시간을 GPX 점 시각으로 정한 기록: 5" in text and "종료 시각을 시작+duration 속성으로 정한 기록(경로 없음): 1" in text


def test_distance_source_gpx_vs_health_and_judgement_changes():
    with tempfile.TemporaryDirectory() as tmp:
        _code, text, out = _run(tmp, sample_zip(tmp))
        rows = {r["start_time"]: r for r in _read(out)}
        a = rows["2025-10-12T07:00:00"]
        assert a["distance_src"] == "gpx" and a["distance_km"] == "5.1" and a["distance_km_health"] == "5.2"      # GPX 재계산 5.1km, 건강 앱 5.2km는 따로 남김
        n = rows["2025-10-13T07:00:00"]
        assert n["distance_src"] == "health" and n["distance_km"] == "5.2" and n["distance_km_health"] == "5.2"   # 경로 없음: 건강 앱 거리
        d = rows["2025-10-15T07:00:00"]
        assert d["distance_src"] == "gpx" and float(d["distance_km"]) < 3.0 and d["distance_km_health"] == "5.3"
        f = rows["2025-10-18T07:00:00"]
        assert f["distance_km"] == "5.1" and f["distance_km_health"] == "4.8" and f["t5_sec"] != ""             # 건강 앱 4.8km였으나 GPX 거리로 5km 후보가 되어 t5 계산
        assert "거리 출처 gpx(GPX 점 재계산): 5" in text and "거리 출처 health(건강 앱): 1" in text
        assert "후보가 됨(건강 앱 거리 기준 아님 → GPX 거리 기준 5.0~5.5km): 1" in text and "후보에서 빠짐(건강 앱 거리 기준 5.0~5.5km → GPX 거리 기준 아님): 1" in text
        # 연구와 같은 함수로 다시 계산한 값과 같다(스킵 규칙 포함)
        ts, lat, lon = ia.read_gpx(io.BytesIO(gpx(52).encode("utf-8")))
        mod, parse = ia.load_research()
        cum, _ = mod.feas.cum_with_skip(parse, lat, lon, ts)
        assert abs(float(a["distance_km"]) - float(cum[-1])) < 1e-3


def test_missing_duration_attribute_falls_back_to_end_minus_start_and_is_counted():
    with tempfile.TemporaryDirectory() as tmp:
        z = make_zip(tmp, [workout("2025-10-13 07:00:00 +0900", "2025-10-13 07:31:00 +0900", 5.2, "Nike Run Club")], {})
        code, text, out = _run(tmp, z)
        r = _read(out)[0]
        assert code == 0 and r["end_time"] == "2025-10-13T07:31:00" and r["duration_sec"] == "1860" and r["distance_src"] == "health"
        assert "duration 속성이 없어 종료−시작을 쓴 기록: 1" in text


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
