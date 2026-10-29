# -*- coding: utf-8 -*-
"""
정직한 기대치 코치 (시연 프로토타입, 명령줄)
다음 7일 안 5km 기록을 '평균 회귀' 규칙(B2)으로 예측하고 80% 범위와 함께 보여준다.
표준 라이브러리만 사용한다. 모델 학습 코드는 없다(a, b, 잔차 분위수는 demo/b2_params.json, 49c02f8 cold-start 최종 적합).

입력 CSV 열: start_time, end_time, distance_km, duration_sec, avg_hr, t5_sec
  - start_time/end_time: 에포크 초 또는 ISO 일시(시간대 표기가 없으면 UTC로 두고 일관되게만 사용; 시각 기준 미확인)
  - t5_sec: 5.0~5.5km 세션에서 5.000km 지점 도달 시간(초). 그 외 세션은 비움
사용: python demo/coach_demo.py 세션.csv --asof 2025-01-06T00:00:00
자격(주 분석 정의 그대로): 기준 시점 직전 28일 안에 끝난 달리기 4회 이상 + 그 창 안 유효 5km 기록 1회 이상(t5_sec <= 5000초, 5.0~5.5km).
기준 기록 = 창 안 가장 늦게 시작한 유효 5km. 평소 기록 m = 기준 시점 전에 끝난 유효 5km의 중앙값(1회뿐이면 m = 기준 기록).
예측: y_hat = a + b x (기준 기록 - m), 예상 기록 = 기준 기록 + y_hat.
기준 시점까지 끝나지 않은 세션은 사용하지 않는다.
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

DAY = 86400
WIN_DAYS = 28
MIN_RUNS = 4
EXTREME_SEC = 5000.0
KM_LO, KM_HI = 5.0, 5.5
PARAMS_PATH = Path(__file__).resolve().parent / "b2_params.json"
REQUIRED_COLS = ["start_time", "end_time", "distance_km", "duration_sec", "avg_hr", "t5_sec"]


def load_params(path=PARAMS_PATH):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("a", "b", "resid_q10", "resid_q90"):
        float(p[k])
    return p


def parse_time(v):
    """에포크 초(숫자) 또는 ISO 일시 -> 초(float). 시간대 표기가 없는 값은 UTC로 둔다."""
    s = str(v).strip()
    try:
        return float(s)
    except ValueError:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()


def read_sessions(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in (rd.fieldnames or [])]
        if missing:
            raise ValueError("CSV에 없는 열: " + ", ".join(missing))
        out = []
        for r in rd:
            t5 = (r["t5_sec"] or "").strip()
            out.append({"start": parse_time(r["start_time"]), "end": parse_time(r["end_time"]),
                        "km": float(r["distance_km"]), "t5": float(t5) if t5 else None})
    return out


def is_valid_5km(s):
    return s["t5"] is not None and KM_LO <= s["km"] <= KM_HI and 0 < s["t5"] <= EXTREME_SEC


def assess(sessions, asof, params):
    """자격 판정과 예측 계산. 반환 dict(eligible, shortfalls, n_runs, 값들)."""
    t0 = asof - WIN_DAYS * DAY
    win = [s for s in sessions if t0 <= s["end"] < asof]
    n_runs = len(win)
    valid_win = sorted((s for s in win if is_valid_5km(s)), key=lambda s: s["start"])
    short = []
    if n_runs < MIN_RUNS:
        short.append("직전 28일 안에 끝난 달리기가 %d회입니다(4회 이상 필요)." % n_runs)
    if not valid_win:
        short.append("직전 28일 안에 유효한 5km 기록이 없습니다(1회 이상 필요).")
    res = {"eligible": not short, "shortfalls": short, "n_runs": n_runs}
    if short:
        return res
    ref = valid_win[-1]["t5"]
    past = [s["t5"] for s in sessions if s["end"] < asof and is_valid_5km(s)]
    m = median(past)
    if len(past) == 1:
        m = ref
    y_hat = predict_change(ref, m, params)
    res.update(ref=ref, m=m, y_hat=y_hat, pred=ref + y_hat,
               lo=ref + y_hat + params["resid_q10"], hi=ref + y_hat + params["resid_q90"])
    return res


def predict_change(ref, m, params):
    """B2: y_hat = a + b * (기준 기록 - m)  (초, 음수 = 빨라짐)."""
    return params["a"] + params["b"] * (ref - m)


def mmss(sec):
    t = int(round(sec))
    sign = "-" if t < 0 else ""
    t = abs(t)
    return "%s%02d:%02d" % (sign, t // 60, t % 60)


def format_report(res):
    if not res["eligible"]:
        return "예측할 수 없습니다. 부족한 조건:\n" + "\n".join("- " + s for s in res["shortfalls"])
    ref_s, m_s = int(round(res["ref"])), int(round(res["m"]))
    n = abs(ref_s - m_s)
    if ref_s > m_s:
        trend = "직전 기록이 평소보다 %d초 느렸기 때문에 평소 쪽으로 돌아오는 경향을 반영했습니다." % n
    elif ref_s < m_s:
        trend = "직전 기록이 평소보다 %d초 빨랐기 때문에 평소 쪽으로 돌아오는 경향을 반영했습니다." % n
    else:
        trend = "직전 기록이 평소와 같아 평소 쪽으로 돌아오는 보정이 거의 없습니다."
    return ("다음 7일 안 5km 예상 기록: %s (80%% 범위 %s ~ %s)\n" % (mmss(res["pred"]), mmss(res["lo"]), mmss(res["hi"]))
            + "평소 기록(과거 중앙값): %s, 직전 기록: %s\n" % (mmss(res["m"]), mmss(res["ref"]))
            + trend + "\n"
            + "검증 결과 지난 4주의 거리·횟수 정보는 이 예측을 더 좋게 만들지 못해(49c02f8) 평균 회귀 규칙만 사용합니다.\n"
            + "이 예측은 원인 분석이 아니며, 주마다 기록이 크게 흔들릴 수 있습니다.")


# 템플릿에 고정된 숫자(다음 7일, 5km, 80%, 지난 4주, 28일, 4회)와 커밋 해시. 계산값이 아니므로 따로 허용한다.
TEMPLATE_INTS = {"4", "5", "7", "28", "80"}
HASH = "49c02f8"
_TOKEN = re.compile(r"49c02f8|-?\d+:\d{2}|\d+")


def check_numbers(text, res):
    """출력 문구 속 모든 숫자가 계산값(또는 템플릿 고정 숫자)과 일치하는지 검사한다. 불일치 시 ValueError."""
    tokens = _TOKEN.findall(text)
    if not res["eligible"]:
        allowed = {str(res["n_runs"]), "1"} | TEMPLATE_INTS | {HASH}   # 1 = "5km 기록 1회 이상 필요"
        for t in tokens:
            if t not in allowed:
                raise ValueError("계산값과 맞지 않는 숫자: " + t)
        return True
    ref_s, m_s = int(round(res["ref"])), int(round(res["m"]))
    needed = [mmss(res[k]) for k in ("pred", "lo", "hi", "m", "ref")]
    found = [t for t in tokens if ":" in t]
    if found != needed:
        raise ValueError("시간 표기가 계산값과 다름: %s != %s" % (found, needed))
    ints = {str(abs(ref_s - m_s))} | TEMPLATE_INTS
    for t in tokens:
        if t != HASH and ":" not in t and t not in ints:
            raise ValueError("계산값과 맞지 않는 숫자: " + t)
    if ref_s != m_s and str(abs(ref_s - m_s)) not in tokens:
        raise ValueError("직전-평소 차이(초)가 문구에 없음")
    return True


def run(csv_path, asof, params=None):
    params = params or load_params()
    res = assess(read_sessions(csv_path), parse_time(asof), params)
    text = format_report(res)
    check_numbers(text, res)
    return res, text


def main(argv=None):
    ap = argparse.ArgumentParser(description="다음 7일 5km 기록 예상(평균 회귀 규칙) 시연")
    ap.add_argument("csv", help="세션 CSV (열: " + ", ".join(REQUIRED_COLS) + ")")
    ap.add_argument("--asof", required=True, help="기준 시점(에포크 초 또는 ISO 일시)")
    a = ap.parse_args(argv)
    try:
        _res, text = run(a.csv, a.asof)
    except (ValueError, OSError) as e:
        print("오류:", e, file=sys.stderr)
        return 2
    print(text)
    return 0


if __name__ == "__main__":
    for _st in (sys.stdout, sys.stderr):
        if hasattr(_st, "reconfigure"):
            _st.reconfigure(encoding="utf-8")
    sys.exit(main())
