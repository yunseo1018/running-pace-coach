# -*- coding: utf-8 -*-
"""
사례 D: 내 애플워치 달리기 1개의 GPS 거리(재계산)와 건강 앱 거리 차이를 집계로만 점검해 docs/source_trace.md의 "사례 D" 절에 쓴다.
- 새 분석이 아니라 기존 계산(scripts/59의 추적과 같은 cum_with_skip·haversine, 시연 가져오기 스크립트의 GPX 읽기)을 한 기록에 적용한다.
- 날짜·좌표·심박·사용자 ID는 출력·문서에 싣지 않는다(점 개수, 간격, 속도, 시작 기준 경과 시각의 구간, 거리 합계만).
- 대상: 경로 파일 이름의 날짜 접두가 --route-date(YYYY-MM-DD)인 달리기 Workout 중 애플워치 기록.
사용법(저장소 루트): python scripts/62_case_d_trace.py --zip <저장소 밖 폴더>/health.zip --route-date YYYY-MM-DD [--write]
"""
import argparse
import importlib.util
import io
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
import import_apple_health as ia  # noqa: E402

OUT = ROOT / "docs" / "source_trace.md"
BEGIN, END = "<!-- case-d:begin -->", "<!-- case-d:end -->"
DIST_REC = "HKQuantityTypeIdentifierDistanceWalkingRunning"
BUCKET = 300


def mmss(sec):
    sec = int(round(sec))
    return "%d:%02d" % (sec // 60, sec % 60)


def find_workout(zf, main, route_date):
    with zf.open(main) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag == "Workout" and el.get("workoutActivityType") == ia.RUNNING:
                w = ia._workout(el)
                if w["route"] and route_date in w["route"]:
                    w["events"] = [(c.get("type"), c.get("date")) for c in el if c.tag == "WorkoutEvent"]
                    w["duration_attr"] = (el.get("duration"), el.get("durationUnit"))
                    return w
            if el.tag in ("Workout", "Record"):
                el.clear()
    raise SystemExit("대상 Workout을 찾지 못했습니다")


def watch_distance_records(zf, main, lo, hi):
    """애플워치가 쓴 DistanceWalkingRunning 기록 중 [lo, hi]와 겹치는 것: (시작, 끝, km)."""
    out = []
    with zf.open(main) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag == "Record" and el.get("type") == DIST_REC and "Watch" in (el.get("sourceName") or ""):
                a, b = ia.parse_apple_time(el.get("startDate"))[1], ia.parse_apple_time(el.get("endDate"))[1]
                if b > lo and a < hi and el.get("unit") == "km":
                    out.append((a, b, float(el.get("value"))))
            el.clear()
    return out


def overlap_share(a, b, lo, hi):
    """기록 [a,b]의 값 중 구간 [lo,hi]에 시간 비례로 들어가는 비율."""
    if b <= a:
        return 1.0 if lo <= a < hi else 0.0
    return max(0.0, min(b, hi) - max(a, lo)) / (b - a)


def compute(zip_path, route_date):
    mod = importlib.util.spec_from_file_location("m38", ROOT / "38_build_5km_dataset.py")
    m38 = importlib.util.module_from_spec(mod)
    mod.loader.exec_module(m38)
    feas, parse = m38.feas, m38.feas.import_parser()
    with zipfile.ZipFile(zip_path) as zf:
        main, gpx = ia.find_members(zf)
        w = find_workout(zf, main, route_date)
        ts, lat, lon = ia.read_gpx(io.BytesIO(zf.read(gpx[w["route"]])))
        s_ep, e_ep = ia.parse_apple_time(w["start"])[1], ia.parse_apple_time(w["end"])[1]
        recs = watch_distance_records(zf, main, s_ep, e_ep)
    ts = [float(x) for x in ts]
    n = len(ts)
    cum, skipped = feas.cum_with_skip(parse, lat, lon, ts)
    seg = np.array([parse.haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i]) for i in range(1, n)])
    dts = np.diff(np.array(ts))
    spd = np.array([seg[i] / dts[i] * 3600.0 for i in range(n - 1) if dts[i] > 0])
    sk = [i for i in range(1, n) if skipped[i]]
    r = dict(n=n, dt_min=float(dts.min()), dt_med=float(np.median(dts)), dt_max=float(dts.max()), n_dt_le0=int((dts <= 0).sum()), n_gap60=int((dts > 60).sum()),
             spd_max=float(spd.max()), spd_med=float(np.median(spd)), n_skip=len(sk), skip_km=float(sum(seg[i - 1] for i in sk)), cum_km=float(cum[-1]), raw_km=float(seg.sum()),
             wk_km=w["km"], elapsed_wk=e_ep - s_ep, elapsed_pts=ts[-1] - ts[0], first_off=ts[0] - s_ep, last_off=ts[-1] - e_ep, dur_attr=w["duration_attr"], events=w["events"],
             rec_n=len(recs), rec_km=float(sum(v * overlap_share(a, b, s_ep, e_ep) for a, b, v in recs)),
             skips=[(ts[i - 1] - ts[0], ts[i] - ts[i - 1], seg[i - 1] * 1000, (seg[i - 1] / (ts[i] - ts[i - 1]) * 3600.0) if ts[i] > ts[i - 1] else None) for i in sk])
    # 5분 구간별: GPS 누적 증가(스킵 규칙 적용)와 애플워치 거리 기록(시간 비례) 합
    nb = int(np.ceil((ts[-1] - s_ep) / BUCKET))
    rows = []
    for k in range(nb):
        lo, hi = s_ep + k * BUCKET, s_ep + (k + 1) * BUCKET
        idx = [i for i in range(1, n) if lo <= ts[i] < hi]
        g = float(sum(cum[i] - cum[i - 1] for i in idx))
        h = float(sum(v * overlap_share(a, b, lo, hi) for a, b, v in recs))
        rows.append((k * BUCKET, min((k + 1) * BUCKET, ts[-1] - s_ep), g, h))
    r["buckets"] = rows
    # 일시정지 이벤트(시작 기준 경과 시각)
    r["events_off"] = [(t, ia.parse_apple_time(d)[1] - s_ep) for t, d in w["events"] if d]
    return r


def render(r):
    L = []
    P = L.append
    P(BEGIN)
    P("")
    P("## 6. 사례 D: 건강 앱 거리와 GPX 재계산 거리가 크게 다른 애플워치 달리기 1개 (집계만)")
    P("")
    P("가져오기 요약에서 GPX 재계산 거리가 5km 범위 밖으로 나와(건강 앱 거리 기준으로는 범위 안) t5가 비게 된 기록 1개를 같은 방식으로 풀어 본 것이다. 날짜·좌표·심박은 싣지 않고 시작 기준 경과 시각과 합계만 적는다(`scripts/62_case_d_trace.py`, 새 ZIP 기준). 위 3절의 사례 A·B·C와 개수(달리기 87개)는 이전 ZIP 기준이며 이 절은 새 ZIP 기준이다.")
    P("")
    P("| 항목 | 값 |")
    P("|---|---|")
    P("| 점 개수 | %d점 |" % r["n"])
    P("| 점 간격 | 중앙값 %g초 · 최소 %g초 · 최대 %g초 · 0 이하 %d개 · 60초 초과 %d개 |" % (r["dt_med"], r["dt_min"], r["dt_max"], r["n_dt_le0"], r["n_gap60"]))
    P("| 점 사이 속도 | 중앙값 %.1f km/h · 최대 %.1f km/h(임계 25/3.6 m/s = 25.0 km/h 초과 구간은 스킵) |" % (r["spd_med"], r["spd_max"]))
    P("| `cum_with_skip` 스킵 구간 | %d구간 · 해당 구간 직선거리 합 %.3f km |" % (r["n_skip"], r["skip_km"]))
    P("| GPX 재계산 거리 | %.3f km(스킵 규칙 적용) / %.3f km(스킵 없음) |" % (r["cum_km"], r["raw_km"]))
    P("| 건강 앱 거리 | Workout 거리 통계 %.3f km · 같은 시간대 애플워치 거리 기록 %d개의 시간 비례 합 %.3f km |" % (r["wk_km"], r["rec_n"], r["rec_km"]))
    P("| 차이(건강 앱 − GPX 재계산) | Workout 통계 기준 %+.3f km · 거리 기록 합 기준 %+.3f km |" % (r["wk_km"] - r["cum_km"], r["rec_km"] - r["cum_km"]))
    P("| 경과 시간 | Workout 종료−시작 %d초 · 점 시각(마지막−처음) %d초 · duration 속성 %s %s · GPX 첫 점 − 시작 %+d초, 마지막 점 − 종료 %+d초 |" % (r["elapsed_wk"], r["elapsed_pts"], "%.2f" % float(r["dur_attr"][0]), r["dur_attr"][1], r["first_off"], r["last_off"]))
    kinds = {}
    for t, o in r["events_off"]:
        kinds[t.replace("HKWorkoutEventType", "")] = kinds.get(t.replace("HKWorkoutEventType", ""), 0) + 1
    P("| Workout 이벤트 | %d개(종류별: %s; 일시정지·재개 이벤트 %s) |" % (len(r["events_off"]), ", ".join("%s %d" % kv for kv in sorted(kinds.items())), "있음" if any(("Pause" in t or "Resume" in t) for t in kinds) else "없음"))
    P("")
    P("스킵 구간(시작 기준 경과 시각 · 점 간격 · 직선거리 · 암시 속도):")
    P("")
    if r["skips"]:
        for off, dt, m, v in r["skips"][:12]:
            P("- %s · %g초 · %.1f m · %s" % (mmss(off), dt, m, ("%.1f km/h" % v) if v is not None else "시간차 0 이하"))
    else:
        P("- 없음")
    P("")
    P("5분 구간별 거리(시작 기준 경과 시각; GPS = `cum_with_skip` 누적 증가, 건강 앱 = 같은 시간대 애플워치 거리 기록을 시간 비례로 나눈 합):")
    P("")
    P("| 구간 | GPS(km) | 건강 앱 기록(km) | GPS − 건강 앱(km) | 누적 차이(km) |")
    P("|---|---|---|---|---|")
    c = 0.0
    for a, b, g, h in r["buckets"]:
        c += g - h
        P("| %s~%s | %.3f | %.3f | %+.3f | %+.3f |" % (mmss(a), mmss(b), g, h, g - h, c))
    P("")
    P("### 확인된 사실")
    P("- 위 표의 수치(점 수, 간격, 속도, 스킵, 거리 합계, 구간별 거리)는 이 기록에서 계산한 값이다.")
    P("- GPX 재계산 거리가 건강 앱 Workout 거리보다 %.3f km 크다(%+.1f%%)." % (r["cum_km"] - r["wk_km"], (r["cum_km"] - r["wk_km"]) / r["wk_km"] * 100))
    P("- 스킵 구간의 직선거리 합은 %.3f km라 차이 %.3f km의 대부분을 스킵 규칙으로 설명할 수 없다(스킵 없이 합쳐도 %.3f km)." % (r["skip_km"], r["cum_km"] - r["wk_km"], r["raw_km"]))
    big = max(r["buckets"], key=lambda b: b[2] - b[3])
    rise = [b for b in r["buckets"] if b[2] - b[3] > 0]
    P("- 5분 구간별 비교에서 GPS가 건강 앱 기록보다 큰 구간은 %s이고(합 %+.3f km), 가장 큰 구간은 %s~%s(%+.3f km)이다. 나머지 구간에서는 GPS가 같거나 작다(합 %+.3f km)." % (
        ", ".join("%s~%s" % (mmss(b[0]), mmss(b[1])) for b in rise), sum(b[2] - b[3] for b in rise), mmss(big[0]), mmss(big[1]), big[2] - big[3], sum(b[2] - b[3] for b in r["buckets"] if b not in rise)))
    P("")
    P("### 미확인")
    P("- 건강 앱 거리와 GPX 재계산 거리가 다른 이유는 이 자료로 알 수 없다. 원인은 추정하지 않는다.")
    P("- 시간 비례로 나눈 거리 기록은 근사이며, 건강 앱 Workout 거리 통계가 어떤 기록을 합친 값인지는 미확인이다.")
    P("- 이 한 기록의 관측이며 다른 기록이나 다른 기기로 일반화하지 않는다.")
    P("")
    P(END)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--route-date", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    text = render(compute(a.zip, a.route_date))
    if a.write:
        doc = OUT.read_text(encoding="utf-8")
        if BEGIN in doc:
            doc = doc[:doc.index(BEGIN)].rstrip("\n") + "\n\n" + text
        else:
            doc = doc.rstrip("\n") + "\n\n" + text
        OUT.write_text(doc, encoding="utf-8", newline="\n")
        print("wrote", OUT)
    else:
        print(text)


if __name__ == "__main__":
    main()
