# -*- coding: utf-8 -*-
"""
원자료 → 5km 시간(T5) 추적표: docs/source_trace.md. 새 분석이 아니라 기존 계산(01_parse_run_sessions의 haversine·스킵 규칙, 38번 compute_t5, 시연 가져오기 스크립트)을 단계별로 풀어 보인다.
- 연구 데이터(FitRec) 2개(사례 F1·F2): 후보(비병합 5.0~5.5km, T5 ≤ 5,000초)이면서 T5 캐시(train 사용자)에 있는 세션 중 F1 = session_id가 가장 작은 것, F2 = 경과−이동 시간 비율(gap_ratio)이 가장 큰 것(정지·공백 처리 대조용).
- 합성 GPX 2개(사례 S1·S2): demo/test_import_apple_health.py의 합성 GPX 생성기(균일 100m/30초; S2는 제자리 점 10개 먼저). 합성 Workout과 함께 가져오기 코드를 그대로 통과시킨다.
- 내 애플워치 기록 3개(사례 A·B·C): 건강 내보내기 ZIP에서 t5가 계산된 달리기 중 A = 가장 이른 것, B = 정지 시간(stop_sec) 최대, C = 가장 늦은 것(중복이면 다음 순위). 날짜·좌표·심박·사용자 ID는 문서에 싣지 않는다(집계·형식 패턴만).
사용법(저장소 루트): python scripts/59_source_trace.py --zip <저장소 밖 폴더>/health.zip   (원본 endomondoHR.json과 data/ 필요, 약 5분)
"""
import argparse
import ast
import importlib.util
import io
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
import config  # noqa: E402
import import_apple_health as ia  # noqa: E402

OUT = ROOT / "docs" / "source_trace.md"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m38 = _load("38_build_5km_dataset.py", "m38")
feas = m38.feas
parse = feas.import_parser()


def mmss(sec):
    t = int(round(sec))
    return "%d:%02d" % (t // 60, t % 60)


def pattern(text):
    """숫자를 모두 #로 가린 형식 패턴(날짜·시각 값은 싣지 않음)."""
    return re.sub(r"\d", "#", text)


def trace_points(ts, lat, lon):
    """점 배열 하나에 대한 단계별 값(좌표는 저장·반환하지 않고 집계만)."""
    ts = [float(x) for x in ts]
    n = len(ts)
    dts = np.diff(np.array(ts))
    cum, skipped = feas.cum_with_skip(parse, lat, lon, ts)
    seg_all = [parse.haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i]) for i in range(1, n)]
    stop = 0.0
    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        if dt > 0 and seg_all[i - 1] / dt * 3600.0 < 1.0:
            stop += dt
    big = dts[dts > 60]
    j = int(np.searchsorted(cum, 5.0, "left"))
    reached = bool(cum[-1] >= 5.0)
    r = dict(n=n, dt_med=float(np.median(dts)), dt_min=float(dts.min()), dt_max=float(dts.max()), n_dt_le0=int((dts <= 0).sum()),
             n_gap60=int(len(big)), sum_gap60=float(big.sum()), cum_km=float(cum[-1]), raw_sum_km=float(sum(seg_all)), n_skipped=int(skipped.sum()),
             elapsed=float(ts[-1] - ts[0]), stop=float(stop), reached=reached)
    if reached:
        frac = (5.0 - cum[j - 1]) / (cum[j] - cum[j - 1])
        r.update(j=j, dt_prev=(ts[j - 1] - ts[j - 2]) if j >= 2 else None, dt_cross=ts[j] - ts[j - 1], dt_next=(ts[j + 1] - ts[j]) if j + 1 < n else None,
                 seg_m=float((cum[j] - cum[j - 1]) * 1000), frac=float(frac), cross_skipped=bool(skipped[j]), t5=float((ts[j - 1] + frac * (ts[j] - ts[j - 1])) - ts[0]))
        t5_lib = float(m38.compute_t5(parse, {"latitude": lat, "longitude": lon, "timestamp": ts}, ts[0]))
        assert abs(t5_lib - r["t5"]) < 1e-6, (t5_lib, r["t5"])
    return r


# ------------------------------------------------------------------ FitRec
def pick_fitrec():
    d = pd.read_pickle(Path("data") / "run_sessions_deduped_merged_with_moving.pkl")
    cache = pd.read_pickle(Path("data") / "t5_main_all.pkl")
    c = d[(d.merged_count == 1) & d.actual_distance_km.between(5.0, 5.5)]
    c = c[[int(s) in cache["t5"] and not cache["ext"].get(int(s), False) for s in c.session_id]].sort_values("session_id")
    f1 = c.iloc[0]
    f2 = c.sort_values(["gap_ratio", "session_id"], ascending=[False, True]).iloc[0]
    assert int(f1.session_id) != int(f2.session_id)
    return [("F1", f1), ("F2", f2)], len(c)


def scan_fitrec(rows):
    want = {int(r.session_id): nm for nm, r in rows}
    found = {}
    with open("endomondoHR.json", "rb") as f:
        for line in f:
            m = feas.ID_RE.search(line)
            if not m or int(m.group(1)) not in want:
                continue
            rec = ast.literal_eval(line.decode("utf-8"))
            sid = int(m.group(1))
            if rec.get("sport") != "run" or rec["id"] != sid:
                continue
            found[want[sid]] = rec
            if len(found) == len(want):
                break
    assert len(found) == len(want)
    return found


def fitrec_case(nm, row, rec):
    ts, lat, lon = rec["timestamp"], rec["latitude"], rec["longitude"]
    assert len(ts) == len(lat) == len(lon) and all(isinstance(x, (int, np.integer)) for x in ts)
    t = trace_points(ts, lat, lon)
    assert abs(t["cum_km"] - float(row.actual_distance_km)) < 1e-9, "세션 표 거리와 다시 계산한 거리가 다름"
    t.update(kind="fit", keys=sorted(k for k in rec.keys() if k not in ("latitude", "longitude")), table_km=float(row.actual_distance_km),
             table_dur=float(row.duration_sec), table_gap=float(row.gap_sec), table_moving=float(row.moving_duration_sec), ts_digits=len(str(int(ts[0]))), start_eq=int(ts[0]) == int(row.start_time))
    return t


# ------------------------------------------------------------------ Apple / 합성
def wk_pattern(text):
    m = re.match(r"(.*?)(\s[+-]\d{4})?$", text or "")
    return pattern(m.group(1)) + (m.group(2) or "")


def workout_ext(el):
    """Workout -> 요약(좌표 없음): 시작·종료 문자열, 거리(km), duration 속성, 이벤트."""
    w = ia._workout(el)
    w["duration_attr"] = (el.get("duration"), el.get("durationUnit"))
    ev = []
    for ch in el:
        if ch.tag == "WorkoutEvent":
            ev.append((ch.get("type"), ch.get("date")))
    w["events"] = ev
    return w


def iter_workouts_ext(stream):
    root, depth = None, 0
    for evn, el in ET.iterparse(stream, events=("start", "end")):
        if evn == "start":
            if root is None:
                root = el
            depth += 1
            continue
        depth -= 1
        if depth == 1:
            if el.tag == "Workout" and el.get("workoutActivityType") == ia.RUNNING:
                yield workout_ext(el)
            root.clear()


def paused_seconds(events):
    """HKWorkoutEventTypePause → Resume 쌍의 시간 합(초)과 이벤트 개수."""
    t_pause, tot, n, n_res, n_paired = None, 0.0, 0, 0, 0
    for typ, date in events:
        if not date:
            continue
        if typ and typ.endswith("Pause"):
            t_pause = ia.parse_apple_time(date)[1]
            n += 1
        elif typ and typ.endswith("Resume"):
            n_res += 1
            if t_pause is not None:
                tot += ia.parse_apple_time(date)[1] - t_pause
                t_pause = None
                n_paired += 1
    return tot, n, len(events), n_res, n_paired


def apple_case(w, gpx_bytes, ts, lat, lon):
    t = trace_points(ts, lat, lon)
    s_ep = ia.parse_apple_time(w["start"])[1]
    e_ep = ia.parse_apple_time(w["end"])[1]
    p_tot, n_pause, n_ev, n_res, n_paired = paused_seconds(w["events"])
    first_txt = re.search(rb"<time>([^<]+)</time>", gpx_bytes).group(1).decode("utf-8")
    t.update(kind="apple", wk_km=w["km"], wk_elapsed=e_ep - s_ep, wk_start_pat=wk_pattern(w["start"]), gpx_time_pat=pattern(first_txt),
             gpx_first_minus_wk_start=float(ts[0] - s_ep), gpx_end_minus_wk_end=float(ts[-1] - e_ep), dur_attr=w["duration_attr"], n_events=n_ev, n_pause=n_pause, n_resume=n_res, n_paired=n_paired,
             paused_sec=p_tot, wk_tz=(re.search(r"([+-]\d{4})$", w["start"]).group(1) if re.search(r"([+-]\d{4})$", w["start"]) else "없음"))
    return t


def collect_apple(zip_path):
    cands = []
    stat = dict(n=0, no_dur=0, diff60=0)          # 모든 달리기: Workout 종료−시작과 duration 속성의 차이(개수만)
    with zipfile.ZipFile(zip_path) as zf:
        main, gpx = ia.find_members(zf)
        with zf.open(main) as xml:
            for w in iter_workouts_ext(xml):
                stat["n"] += 1
                v, u = w["duration_attr"]
                if v is None or not w["start"] or not w["end"]:
                    stat["no_dur"] += 1
                else:
                    el = ia.parse_apple_time(w["end"])[1] - ia.parse_apple_time(w["start"])[1]
                    if abs(el - float(v) * (60 if u == "min" else 1)) > 60:
                        stat["diff60"] += 1
                if w["km"] is None or not w["start"] or not w["end"] or not w["route"] or not (ia.KM_LO <= w["km"] <= ia.KM_HI):
                    continue
                member = gpx.get(w["route"])
                if not member:
                    continue
                data = zf.read(member)
                try:
                    ts, lat, lon = ia.read_gpx(io.BytesIO(data))
                except (ET.ParseError, ValueError):
                    continue
                if len(ts) < 2:
                    continue
                cum, _ = feas.cum_with_skip(parse, lat, lon, ts)
                if cum[-1] < 5.0:
                    continue
                stop = ia.stop_seconds(parse, ts, lat, lon)
                cands.append(dict(w=w, data=data, pts=(ts, lat, lon), stop=stop, start=ia.parse_apple_time(w["start"])[1]))
    cands.sort(key=lambda c: c["start"])
    assert len(cands) >= 3, len(cands)
    order = [("A", 0), ("B", max(range(len(cands)), key=lambda i: (cands[i]["stop"], -i))), ("C", len(cands) - 1)]
    chosen, used = [], set()
    for nm, i in order:
        step = 1 if nm != "C" else -1
        while i in used:
            i += step
        used.add(i)
        chosen.append((nm, cands[i]))
    return chosen, len(cands), stat


def synthetic_cases():
    import test_import_apple_health as T
    cases = []
    for nm, g, km, n_int in (("S1", T.gpx(60), 5.9, 59), ("S2", T.gpx(70, stop_points=10), 6.0, 69)):
        w = {"start": "2025-10-12 09:00:00 +0900", "end": ia_end(n_int * 30), "src": "Apple Watch", "km": km, "hr": None, "route": "r.gpx", "duration_attr": (None, None), "events": []}
        ts, lat, lon = ia.read_gpx(io.BytesIO(g.encode("utf-8")))
        cases.append((nm, apple_case(w, g.encode("utf-8"), ts, lat, lon)))
    return cases


def ia_end(sec):
    h, rem = divmod(9 * 3600 + sec, 3600)
    return "2025-10-12 %02d:%02d:%02d +0900" % (h, rem // 60, rem % 60)


# ------------------------------------------------------------------ 문서
def fmt_dt(x):
    return "-" if x is None else "%g" % x


def cell(c, key):
    t = c
    if key == "시각 표기":
        if t["kind"] == "fit":
            return "정수 에포크 초(%d자리), 시간대 표기 없음" % t["ts_digits"]
        return "GPX `<time>` `%s`(Z = UTC) / Workout `%s`(시간대 %s)" % (t["gpx_time_pat"], t["wk_start_pat"], t["wk_tz"])
    if key == "점 수 · 점 간격":
        return "%d점 · 간격 중앙값 %s초(최소 %s, 최대 %s, 0 이하 %d개)" % (t["n"], fmt_dt(t["dt_med"]), fmt_dt(t["dt_min"]), fmt_dt(t["dt_max"]), t["n_dt_le0"])
    if key == "거리 계산 방법 · 단위":
        return "점 사이 haversine(지구 반지름 6371.0088km), 단위 km; 시간차 ≤ 0 또는 암시 속도 > 25/3.6 m/s 구간은 0 누적(%d구간 스킵)" % t["n_skipped"]
    if key == "총 거리 원자료 표기":
        if t["kind"] == "fit":
            return "없음(원자료 키에 거리 필드 없음) · 세션 표 %.3f km" % t["table_km"]
        return "Workout 거리 통계 %.3f km" % t["wk_km"]
    if key == "총 거리 haversine 재계산":
        return "%.3f km(스킵 규칙 적용) / %.3f km(스킵 없음)" % (t["cum_km"], t["raw_sum_km"])
    if key == "거리 차이(원자료 − 재계산)":
        if t["kind"] == "fit":
            return "세션 표와 재계산 %.1e km(assert 통과)" % abs(t["table_km"] - t["cum_km"])
        d = t["wk_km"] - t["cum_km"]
        return "%+.3f km (%+.2f%%)" % (d, 100 * d / t["cum_km"])
    if key == "총 경과 시간":
        if t["kind"] == "fit":
            return "원자료 표기 없음 · 점 시각 %s초 · 세션 표 %s초" % (fmt_dt(t["elapsed"]), fmt_dt(t["table_dur"]))
        return "Workout 종료−시작 %s초 · 점 시각(마지막−처음) %s초 · duration 속성 %s" % (fmt_dt(t["wk_elapsed"]), fmt_dt(t["elapsed"]), "없음" if t["dur_attr"][0] is None else "%.2f %s(%d초)" % (float(t["dur_attr"][0]), t["dur_attr"][1], round(float(t["dur_attr"][0]) * (60 if t["dur_attr"][1] == "min" else 1))))
    if key == "경과 시간 차이":
        if t["kind"] == "fit":
            return "표 − 점 시각 %+g초" % (t["table_dur"] - t["elapsed"])
        return "Workout − 점 시각 %+g초 (GPX 첫 점 − 시작 %+g초, 마지막 점 − 종료 %+g초)" % (t["wk_elapsed"] - t["elapsed"], t["gpx_first_minus_wk_start"], t["gpx_end_minus_wk_end"])
    if key == "일시정지 · 정지 처리":
        base = "간격 60초 초과 %d개(합 %s초) · 속도 1km/h 미만 구간 합 stop_sec %s초" % (t["n_gap60"], fmt_dt(t["sum_gap60"]), fmt_dt(round(t["stop"], 1)))
        if t["kind"] == "fit":
            return base + " · 연구 표 경과−이동 %s초" % fmt_dt(t["table_gap"])
        return base + " · Workout 이벤트 %d개(일시정지 %d회, 재개 %d회, 짝지어진 %d쌍의 합 %s초)" % (t["n_events"], t["n_pause"], t["n_resume"], t["n_paired"], fmt_dt(round(t["paused_sec"], 1)))
    if key == "5.000km 보간 앞뒤 점 간격":
        return "앞 %s초 · 교차 구간 %s초(%.1f m) · 뒤 %s초; 보간 비율 %.3f; 교차 구간 스킵 %s" % (fmt_dt(t["dt_prev"]), fmt_dt(t["dt_cross"]), t["seg_m"], fmt_dt(t["dt_next"]), t["frac"], "예" if t["cross_skipped"] else "아니오")
    if key == "최종 t5":
        return "%s초 (%s)" % (fmt_dt(round(t["t5"], 3)), mmss(t["t5"]))
    raise KeyError(key)


KEYS = ["시각 표기", "점 수 · 점 간격", "거리 계산 방법 · 단위", "총 거리 원자료 표기", "총 거리 haversine 재계산", "거리 차이(원자료 − 재계산)", "총 경과 시간", "경과 시간 차이",
        "일시정지 · 정지 처리", "5.000km 보간 앞뒤 점 간격", "최종 t5"]


def table(cases):
    names = [n for n, _ in cases]
    out = ["| 단계 | " + " | ".join("사례 " + n for n in names) + " |", "|---|" + "---|" * len(names)]
    for k in KEYS:
        out.append("| %s | %s |" % (k, " | ".join(cell(c, k).replace("|", "\\|") for _, c in cases)))
    return "\n".join(out)


def build_doc(fit, syn, apple, n_fit_cand, n_apple_cand, stat):
    FIT_KEYS = fit[0][1]["keys"]
    L = ["# 원자료 → 5km 시간(T5) 추적표", "",
         "생성: `scripts/59_source_trace.py`(저장소 루트에서 `python scripts/59_source_trace.py --zip <저장소 밖 폴더>/health.zip`; 원본 `endomondoHR.json`과 `data/` 필요). 새 분석이 아니라 기존 계산(`01_parse_run_sessions.py`의 haversine·스킵 규칙, `38_build_5km_dataset.py:compute_t5`, `demo/import_apple_health.py`)을 단계별로 풀어 보인 것이다. "
         "결과 숫자·정의·모형·범위는 바꾸지 않았다. 연구 데이터는 \"academic use only, 재배포 금지\"라 세션 ID·사용자 ID·좌표·원본 값 목록은 싣지 않고 사례 이름(F1·F2)과 집계값만 적는다. "
         "내 애플워치 기록은 저장소 밖 자료라 사례 A·B·C로만 적었고 날짜·좌표·심박 값은 뺐다(시각은 숫자를 가린 형식 패턴만 적음). 시각은 start_time 원값(기준 미확인)이며 모든 T5는 \"정지 시간 포함(경과 시간 기준)\"이다.", "",
         "## 1. 사례 선정 규칙", "",
         f"- FitRec 원자료의 키(좌표 제외): {', '.join(FIT_KEYS)}. 거리·경과 시간 필드는 없다.",
         f"- 사례 F1·F2(FitRec): 후보(비병합 세션, 5.0~5.5km, T5 ≤ 5,000초)이면서 T5 캐시에 있는 세션 {n_fit_cand:,}개 중 F1 = session_id가 가장 작은 것, F2 = 경과−이동 시간 비율(`gap_ratio`)이 가장 큰 것(정지·공백 처리 대조용). 사람이 고른 것이 아니라 규칙으로 정했다.",
         "- 사례 S1·S2(합성 GPX): `demo/test_import_apple_health.py`의 합성 GPX 생성기. S1 = 균일 100m/30초로 60점(59구간), S2 = 제자리 점 10개 뒤 같은 속도로 70점. 합성 Workout(거리 5.9km·6.0km, 종료−시작 = 점 시각)과 함께 가져오기 코드를 그대로 통과시켰다.",
         f"- 사례 A·B·C(내 애플워치 기록): 건강 내보내기 ZIP의 달리기 중 5.0~5.5km·경로 있음·경로 거리 5km 이상인 {n_apple_cand}개에서 A = 가장 이른 것, B = `stop_sec` 최대, C = 가장 늦은 것(겹치면 다음 순위).", "",
         "## 2. 연구 데이터(FitRec)와 합성 GPX", "", table(fit + syn), "",
         "## 3. 내 애플워치 기록", "", table(apple), "",
         "각 사례에서 `trace_points`가 계산한 T5가 연구 파이프라인 함수 `compute_t5`의 값과 1e-6초 이내로 같음을 assert로 확인했다(모든 사례). FitRec 사례의 재계산 거리는 세션 표의 `actual_distance_km`와 1e-9km 이내로 같다.", ""]
    # 확인된 차이 / 미확인
    L += ["## 4. 확인된 차이 / 미확인", "", "원인은 추정하지 않는다. 숫자로 확인된 것만 \"확인된 차이\"에 적고, 이 자료로 알 수 없는 것은 \"미확인\"에 적는다.", "", "### 확인된 차이"]
    for nm, c in apple:
        L.append("- 사례 %s: Workout 거리 − 재계산 거리 = %+.3f km(%+.2f%%). Workout 종료−시작 − 점 시각 경과 = %+g초. GPX 첫 점 − Workout 시작 = %+g초, 마지막 점 − 종료 = %+g초. 간격 60초 초과 %d개(합 %s초), Workout 일시정지 %d회·재개 %d회(짝지어진 %d쌍의 합 %s초)." % (nm, c["wk_km"] - c["cum_km"], 100 * (c["wk_km"] - c["cum_km"]) / c["cum_km"], c["wk_elapsed"] - c["elapsed"], c["gpx_first_minus_wk_start"], c["gpx_end_minus_wk_end"], c["n_gap60"], fmt_dt(c["sum_gap60"]), c["n_pause"], c["n_resume"], c["n_paired"], fmt_dt(round(c["paused_sec"], 1))))
        if abs(c["wk_elapsed"] - c["elapsed"]) > 60:
            L.append("  - 사례 %s는 CSV의 `end_time`·`duration_sec`가 Workout의 종료−시작(%s초)에서 나오므로 점 시각 경과(%s초)·duration 속성과 크게 다르다. 가져오기 스크립트(`demo/import_apple_health.py`)는 이 값을 고치지 않았다. 원인은 미확인이다. T5와 `stop_sec`는 GPX 점 시각만 쓰므로 영향이 없지만, `end_time`은 \"기준 시점 전에 끝난 기록\" 판정에 쓰인다." % (nm, fmt_dt(c["wk_elapsed"]), fmt_dt(c["elapsed"])))
    for nm, c in syn:
        L.append("- 사례 %s(합성): Workout 거리 − 재계산 거리 = %+.3f km. 점 시각 경과와 Workout 경과 차이 %+g초. 제자리 구간 stop_sec %s초가 T5(경과 시간 기준)에 포함된다." % (nm, c["wk_km"] - c["cum_km"], c["wk_elapsed"] - c["elapsed"], fmt_dt(round(c["stop"], 1))))
    for nm, c in fit:
        L.append("- 사례 %s(FitRec): 재계산 거리 = 세션 표 거리(차이 %.1e km). 세션 표 경과 − 점 시각 경과 = %+g초. 연구 표의 경과−이동 시간 차이 %s초와 속도 1km/h 미만 구간 합 %s초는 서로 다른 정의라 같지 않을 수 있다(정의별로 따로 적음)." % (nm, abs(c["table_km"] - c["cum_km"]), c["table_dur"] - c["elapsed"], fmt_dt(c["table_gap"]), fmt_dt(round(c["stop"], 1))))
    L.append("- 내 애플워치 기록 전체(달리기 %d개): Workout 종료−시작이 `duration` 속성과 60초를 넘게 다른 기록이 %d개(duration 속성이 없는 기록 %d개). 가져오기 CSV의 `end_time`·`duration_sec`는 종료−시작에서 나온다. 원인은 미확인이고 값은 고치지 않았다." % (stat["n"], stat["diff60"], stat["no_dur"]))
    L += ["", "### 미확인",
          "- FitRec 시각(정수 에포크 초)의 시간대 기준(UTC 여부)은 판정할 수 없다(`docs/week4_data_audit.md`). T5는 같은 세션 안 시각 차이라 시간대와 무관하지만, `start_time`을 달력 날짜로 읽는 용도(월요일 격자)는 가정이다.",
          "- FitRec 원자료의 점 기록 규칙(샘플링 주기, 일시정지 중 점이 기록되는지)과 일시정지 이벤트는 원자료에 없어 알 수 없다. 위 표의 \"간격 60초 초과\"는 점 시각 간격만 보여 줄 뿐 일시정지인지는 구분하지 못한다.",
          "- 애플 Workout 거리 통계가 어떤 센서·알고리즘으로 계산되는지, GPX 경로 거리와의 차이의 원인은 이 자료로 알 수 없다. 위 차이는 이 사례들에서 관측된 값일 뿐이며 다른 기록으로 일반화하지 않는다.",
          "- 애플 GPX 점 간격 규칙과, 일시정지 구간이 GPX에서 점 간격으로 남는지(또는 점이 비는지)는 이 사례들에서 관측된 것만 적었다. 기기·운영체제 버전별 동작은 미확인이다.",
          "- 가져오기 CSV의 `start_time`·`end_time`은 Workout 시각의 시간대 표기(예: +0900)를 떼고 현지 벽시계 값으로 적는다(변환하지 않음). GPX `<time>`은 UTC(Z)라 같은 순간이 두 표기로 달라 보일 수 있으나, T5·`stop_sec`는 GPX 안 시각 차이만 쓰므로 영향이 없다. `--asof`는 CSV와 같은 벽시계 기준으로 지정해야 한다.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    a = ap.parse_args()
    rows, n_fit = pick_fitrec()
    recs = scan_fitrec(rows)
    fit = [(nm, fitrec_case(nm, r, recs[nm])) for nm, r in rows]
    syn = synthetic_cases()
    chosen, n_apple, stat = collect_apple(a.zip)
    apple = []
    for nm, c in chosen:
        ts, lat, lon = c["pts"]
        apple.append((nm, apple_case(c["w"], c["data"], ts, lat, lon)))
    OUT.write_text(build_doc(fit, syn, apple, n_fit, n_apple, stat), encoding="utf-8", newline="\n")
    print("saved", OUT.name, "| fit candidates", n_fit, "| apple candidates", n_apple)


if __name__ == "__main__":
    main()
