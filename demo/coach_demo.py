# -*- coding: utf-8 -*-
"""
정직한 기대치 코치 (시연 프로토타입, 명령줄)
다음 7일 안 5km 기록을 '평균 회귀' 규칙(B2)으로 예측하고 명목 80% 범위와 함께 보여준다.
표준 라이브러리만 사용한다. 모델 학습 코드는 없다(a, b, 잔차 분위수는 demo/b2_params.json, 49c02f8 cold-start 최종 적합 + 보정 범위).

입력 CSV 열: start_time, end_time, distance_km, duration_sec, t5_sec (선택: avg_hr, stop_sec, source 등 그 밖의 열은 무시)
  - avg_hr: 선택 열이며 예측에 쓰지 않는다(없거나 비어 있어도 예측 동일)
  - start_time/end_time: 에포크 초 또는 ISO 일시(시간대 표기가 없으면 UTC로 두고 일관되게만 사용; 시각 기준 미확인)
  - t5_sec: 5.0~5.5km 세션에서 5.000km 지점 도달 시간(초). 그 외 세션은 비움
  - stop_sec(선택 열): 세션 안 정지 시간(초; 시간차>0이고 속도 1km/h 미만인 점 사이 구간의 합). 없으면 정지 시간 조건은 확인하지 않는다
조건별 안내(demo/stratum_notes.json): train 폴드 밖 포함률의 95% CI 상한이 80% 미만인 층에 해당하면 안내 줄을 모두 표시한다(범위 폭은 바꾸지 않음; docs/demo_stratum_note_spec.md).
사용: python demo/coach_demo.py 세션.csv --asof 2025-01-06T00:00:00 [--goal 24:30]
자격(주 분석 정의 그대로): 기준 시점 직전 28일 안에 끝난 달리기 4회 이상 + 그 창 안 유효 5km 기록 1회 이상(t5_sec <= 5000초, 5.0~5.5km).
기준 기록 = 창 안 가장 늦게 시작한 유효 5km. 평소 기록 m = 기준 시점 전에 끝난 유효 5km의 중앙값(1회뿐이면 m = 기준 기록).
가장 최근 5km 기록이 5.0~5.5km 범위 밖이거나 극단(T5 > 5,000초)이면 사전 선언 규칙대로 그 이전의 유효 기록을 기준 기록으로 쓰고 그 사실을 표시한다.
예측: y_hat = a + b x (기준 기록 - m), 예상 기록 = 기준 기록 + y_hat.
기준 시점까지 끝나지 않은 세션은 사용하지 않는다. 입력 검사는 예측 시점에 관측 가능한 값만 쓴다(미래 값으로 거르는 규칙 없음).
입력·옵션 문제(파일, 형식, 날짜, --goal)는 프로그램이 멈추지 않고 '오류: 이유 / 해결: 방법'을 한국어로 출력한다(종료 코드 2).
기본 출력은 요약 화면([결과]/[상태]/[안내]/[근거] 또는 예측 불가일 때 [해결]/[주의]); --detail은 뒤에 기존 전체 출력(자세히 보기)을 그대로 붙인다(docs/demo_screen_spec.md).
--goal mm:ss: 예측 범위와 목표를 비교해 '명목 범위보다 빠름 / 명목 범위 안 / 명목 범위보다 느림' 중 하나를 보여준다(달성 확률·훈련 처방 아님).
--feedback: 예측 뒤에 기준 시점 이후 7일 안 첫 5km 기록과 예측의 비교(범위 안·밖, 차이, 학습 자료 기준 비율)를 붙인다(처방·원인·실력 판정 아님; docs/demo_feedback_spec.md).
"""
import argparse
import bisect
import csv
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

DAY = 86400
WIN_DAYS = 28
MIN_RUNS = 4
EXTREME_SEC = 5000.0
KM_LO, KM_HI = 5.0, 5.5
GOAL_MIN_SEC, GOAL_MAX_SEC = 600, 5000
PARAMS_PATH = Path(__file__).resolve().parent / "b2_params.json"
NOTES_PATH = Path(__file__).resolve().parent / "stratum_notes.json"
QUANT_PATH = Path(__file__).resolve().parent / "feedback_quantiles.json"
REQUIRED_COLS = ["start_time", "end_time", "distance_km", "duration_sec", "t5_sec"]      # avg_hr는 선택 열(예측에 쓰지 않음)
STOP_COL = "stop_sec"          # 선택 열(초): 세션 안 정지 시간. docs/demo_stratum_note_spec.md 5절
EXAMPLE_CMD = "python demo/coach_demo.py demo/examples/ex1_steady.csv --asof 2025-03-03T00:00:00"


class InputProblem(Exception):
    """입력·옵션 문제: 이유(reason)와 해결 방법(fix)을 한국어로 보여 주기 위한 예외."""

    def __init__(self, reason, fix):
        super().__init__(reason)
        self.reason = reason
        self.fix = fix


def problem_text(e):
    return "오류: %s\n해결: %s" % (e.reason, e.fix)


def load_params(path=PARAMS_PATH):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("a", "b", "resid_q10", "resid_q90", "test_coverage_cold"):
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


def _line_list(lines):
    shown = lines[:3]
    more = "" if len(lines) <= 3 else " 등 %d곳" % len(lines)
    return "·".join("%d" % n for n in shown) + "번째 줄" + more


def load_sessions(path, now=None):
    """CSV를 읽어 (세션 목록, 안내 목록)을 돌려준다. 입력 문제는 InputProblem. 미래 날짜 기록과 같은 시각의 중복 기록은 제외하고 안내한다."""
    now = time.time() if now is None else now
    p = Path(path)
    if not p.exists():
        raise InputProblem("입력 파일을 찾을 수 없습니다: %s" % path, "파일 경로를 확인하세요(작업 폴더 기준 상대 경로나 전체 경로). 예: " + EXAMPLE_CMD)
    if p.is_dir():
        raise InputProblem("입력 경로가 파일이 아니라 폴더입니다: %s" % path, "달리기 기록이 담긴 CSV 파일을 지정하세요.")
    try:
        if p.stat().st_size == 0:
            raise InputProblem("입력 파일이 비어 있습니다(0바이트).", "헤더 줄과 달리기 기록이 담긴 CSV를 지정하세요(열: %s)." % ", ".join(REQUIRED_COLS))
        with open(p, encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames or []
            rows = list(rd)
    except UnicodeDecodeError:
        raise InputProblem("파일을 UTF-8로 읽을 수 없습니다.", "엑셀 등에서 'CSV UTF-8(쉼표로 분리)'로 다시 저장하세요.")
    except (OSError, csv.Error) as e:
        raise InputProblem("파일을 읽는 중 문제가 생겼습니다(%s)." % type(e).__name__, "파일이 다른 프로그램에서 열려 있지 않은지, 읽기 권한이 있는지 확인하세요.")
    missing = [c for c in REQUIRED_COLS if c not in fields]
    if missing:
        raise InputProblem("필요한 열이 없습니다: " + ", ".join(missing), "첫 줄(헤더)에 열 이름 %s가 모두 있어야 합니다(쉼표로 구분, 엑셀의 세미콜론 저장 주의)." % ", ".join(REQUIRED_COLS))
    data = [(i, r) for i, r in enumerate(rows, start=2) if any((v or "").strip() for v in r.values() if v is not None)]
    if not data:
        raise InputProblem("데이터 행이 없습니다(헤더만 있거나 모두 빈 줄).", "달리기 한 번당 한 줄씩 기록을 추가하세요(열: %s)." % ", ".join(REQUIRED_COLS))

    sessions, bad_time, bad_num, bad_order, bad_stop = [], [], [], [], []
    for line, r in data:
        try:
            st, en = parse_time(r["start_time"]), parse_time(r["end_time"])
            if not (math.isfinite(st) and math.isfinite(en)):
                raise ValueError("not finite")
        except (ValueError, OverflowError, OSError, TypeError):
            bad_time.append(line)
            continue
        try:
            km = float(r["distance_km"])
            t5s = (r["t5_sec"] or "").strip()
            t5 = float(t5s) if t5s else None
            if not math.isfinite(km) or (t5 is not None and not math.isfinite(t5)):
                raise ValueError("not finite")
        except (ValueError, TypeError):
            bad_num.append(line)
            continue
        stop_s = (r.get(STOP_COL) or "").strip() if STOP_COL in fields else ""
        try:
            stop = float(stop_s) if stop_s else None
            if stop is not None and (not math.isfinite(stop) or stop < 0):
                raise ValueError("bad stop")
        except (ValueError, TypeError):
            bad_stop.append(line)
            continue
        if en < st:
            bad_order.append(line)
            continue
        sessions.append({"start": st, "end": en, "km": km, "t5": t5, "stop": stop})
    if bad_time:
        raise InputProblem("%s의 start_time 또는 end_time 형식을 읽을 수 없습니다." % _line_list(bad_time),
                           "ISO 일시(예: 2025-03-03T07:00:00) 또는 에포크 초(숫자)로 고치세요. 엑셀이 바꾼 날짜 형식(예: 3/3/2025 7:00)은 텍스트로 다시 저장해야 합니다.")
    if bad_num:
        raise InputProblem("%s의 distance_km 또는 t5_sec가 숫자가 아닙니다." % _line_list(bad_num), "km와 초를 숫자로 입력하세요(t5_sec는 5km 기록이 아니면 비워 둡니다).")
    if bad_stop:
        raise InputProblem("%s의 stop_sec가 0 이상의 숫자가 아닙니다." % _line_list(bad_stop), "정지 시간을 0 이상의 초 단위 숫자로 입력하거나(선택 열) 비워 두세요.")
    if bad_order:
        raise InputProblem("%s의 end_time이 start_time보다 이릅니다." % _line_list(bad_order), "시작·종료 시각이 바뀌지 않았는지 확인하세요.")

    notices = []
    kept = [s for s in sessions if s["start"] <= now and s["end"] <= now]
    n_future = len(sessions) - len(kept)
    if n_future:
        notices.append("현재 시각보다 미래 날짜의 기록 %d개는 아직 일어나지 않은 기록이라 사용하지 않았습니다. start_time·end_time의 연도와 날짜가 맞는지 확인하세요." % n_future)
    seen, uniq = set(), []
    for s in kept:
        key = (s["start"], s["end"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    n_dup = len(kept) - len(uniq)
    if n_dup:
        notices.append("시작·종료 시각이 같은 중복 기록 %d개는 먼저 나온 한 개만 남기고 제외했습니다. 중복 줄을 지우면 이 안내가 사라집니다." % n_dup)
    uniq, n_in, n_out = merge_overlapping(uniq)
    if n_in:
        notices.append("시간이 겹치는 기록 %d건을 %d건으로 처리했습니다(시작 시각 차이 %d초 이하이고 시간 구간이 겹치는 기록은 하나로 세며, 경로가 있는 기록을 남김)." % (n_in, n_out, OVERLAP_START_SEC))
    return uniq, notices


OVERLAP_START_SEC = 60     # 연구 파이프라인(16_duplicate_dedup.py NEAR_SIMULTANEOUS_START_SEC)과 같은 값


def merge_overlapping(sessions):
    """연구 파이프라인의 중복 제거 규칙(시작 시각 차이 <= 60초 + 시간 구간 겹침; 시작 좌표 50m 조건은 CSV에 좌표가 없어 적용 불가)을 같게 적용한다.
    겹치는 기록 묶음에서 하나만 남긴다: 경로가 있는 기록(stop_sec 있음) → 더 긴 기록 → 먼저 시작한 기록 순. 돌려주는 값: (세션 목록, 묶음에 든 기록 수, 묶음 수)."""
    order = sorted(range(len(sessions)), key=lambda i: (sessions[i]["start"], sessions[i]["end"]))
    parent = list(range(len(sessions)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_pos, i in enumerate(order):
        a = sessions[i]
        for j in order[a_pos + 1:]:
            b = sessions[j]
            if b["start"] - a["start"] > OVERLAP_START_SEC:
                break
            if max(a["start"], b["start"]) < min(a["end"], b["end"]):
                parent[find(j)] = find(i)
    groups = {}
    for i in order:
        groups.setdefault(find(i), []).append(i)
    keep, n_in, n_groups = set(), 0, 0
    for members in groups.values():
        if len(members) > 1:
            n_in += len(members)
            n_groups += 1
        best = min(members, key=lambda i: (sessions[i]["stop"] is None, -(sessions[i]["end"] - sessions[i]["start"]), sessions[i]["start"], i))
        keep.add(best)
    return [s for i, s in enumerate(sessions) if i in keep], n_in, n_groups


def read_sessions(path):
    """세션 목록만 필요할 때(성능 측정 등). 입력 문제는 InputProblem."""
    return load_sessions(path)[0]


def parse_asof(asof, now):
    try:
        v = parse_time(asof)
        if not math.isfinite(v):
            raise ValueError("not finite")
    except (ValueError, OverflowError, OSError, TypeError):
        raise InputProblem("기준 시점(--asof)을 읽을 수 없습니다.", "ISO 일시(예: 2025-03-03T00:00:00) 또는 에포크 초(숫자)로 지정하세요.")
    if v > now:
        raise InputProblem("기준 시점이 현재 시각보다 미래입니다.", "오늘 또는 그 이전 시각으로 지정하세요(미래 기록은 관측할 수 없어 예측 시점으로 쓸 수 없습니다).")
    return v


def parse_goal(text):
    s = str(text).strip()
    m = re.fullmatch(r"(\d{1,3}):([0-5]\d)", s)
    if not m:
        raise InputProblem("목표 형식이 올바르지 않습니다(mm:ss 필요).", "분:초로 입력하세요(초는 00~59). 예: --goal 24:30")
    sec = int(m.group(1)) * 60 + int(m.group(2))
    if sec < GOAL_MIN_SEC:
        raise InputProblem("목표가 10분 미만입니다.", "10:00 이상 83:20 이하(5,000초 이하)로 입력하세요.")
    if sec > GOAL_MAX_SEC:
        raise InputProblem("목표가 5,000초(83:20)를 넘습니다.", "83:20 이하로 입력하세요(5,000초를 넘는 기록은 극단 기록으로 분류됩니다).")
    return sec


def is_valid_5km(s):
    return s["t5"] is not None and KM_LO <= s["km"] <= KM_HI and 0 < s["t5"] <= EXTREME_SEC


def _invalid_reasons(cands):
    """유효하지 않은 5km 기록의 이유별 개수 설명(0개인 항목은 생략)."""
    n_out = sum(1 for s in cands if not (KM_LO <= s["km"] <= KM_HI))
    n_ext = sum(1 for s in cands if (KM_LO <= s["km"] <= KM_HI) and s["t5"] > EXTREME_SEC)
    n_bad = len(cands) - n_out - n_ext
    parts = []
    if n_out:
        parts.append("5.0~5.5km 범위 밖 %d개" % n_out)
    if n_ext:
        parts.append("극단(T5 > 5,000초) %d개" % n_ext)
    if n_bad:
        parts.append("T5 값 오류(0 이하) %d개" % n_bad)
    return ", ".join(parts), {str(x) for x in (n_out, n_ext, n_bad) if x}


def assess(sessions, asof, params):
    """자격 판정과 예측 계산. 반환 dict(eligible, shortfalls, n_runs, 값들). 기준 시점까지 끝난 기록만 쓴다."""
    t0 = asof - WIN_DAYS * DAY
    win = [s for s in sessions if t0 <= s["end"] < asof]
    n_after = sum(1 for s in sessions if s["end"] >= asof)
    n_runs = len(win)
    cands = sorted((s for s in win if s["t5"] is not None), key=lambda s: s["start"])
    valid_win = [s for s in cands if is_valid_5km(s)]
    notices = []
    if n_after:
        notices.append("기준 시점 이후에 끝난 기록 %d개는 사용하지 않았습니다(예측 시점에 관측 가능한 기록만 사용)." % n_after)
    short, ints = [], {str(n_runs)}
    if n_runs < MIN_RUNS:
        short.append("직전 28일 안에 끝난 달리기가 %d회입니다(4회 이상 필요, %d회 부족)." % (n_runs, MIN_RUNS - n_runs))
        ints.add(str(MIN_RUNS - n_runs))
    if not valid_win:
        msg = "직전 28일 안에 유효한 5km 기록이 없습니다(1회 이상 필요, 1회 부족)."
        if cands:
            why, why_ints = _invalid_reasons(cands)
            msg += " 5km 기록 %d개가 있으나 %s라 제외했습니다." % (len(cands), why)
            ints |= why_ints | {str(len(cands))}
        short.append(msg)
        ints.add("1")
    res = {"eligible": not short, "shortfalls": short, "n_runs": n_runs, "ints": ints, "notices": notices,
           "fix": "직전 28일 안의 달리기 기록(4회 이상)과 5.0~5.5km의 5km 기록 1회 이상(t5_sec 포함)을 CSV에 추가하거나, 기록이 충분한 기준 시점(--asof)을 지정하세요."}
    if short:
        return res
    ref_s = valid_win[-1]
    ref = ref_s["t5"]
    skipped = [s for s in cands if not is_valid_5km(s) and s["start"] > ref_s["start"]]
    if skipped:
        why, _ = _invalid_reasons(skipped)
        days = int(round((asof - ref_s["end"]) / DAY))
        notices.append("가장 최근 5km 기록 %d개(%s)는 사전 선언 규칙(5.0~5.5km, T5 ≤ 5,000초)에 맞지 않아 쓰지 않고, 그 이전의 유효 기록(%d일 전)을 기준 기록으로 사용했습니다. 해당 줄의 distance_km·t5_sec를 확인하세요."
                       % (len(skipped), why, days))
    past = [s["t5"] for s in sessions if s["end"] < asof and is_valid_5km(s)]
    m = median(past)
    if len(past) == 1:
        m = ref
    y_hat = predict_change(ref, m, params)
    res.update(n_past=len(past), x=ref - m, days_ref=(asof - ref_s["end"]) / DAY, ref_stop=ref_s.get("stop"),
               ref=ref, m=m, y_hat=y_hat, pred=ref + y_hat, a_term=params["a"], b_coef=params["b"], b_term=params["b"] * (ref - m),
               tot_int=int(round(y_hat)), cov=params["test_coverage_cold"],
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


def signed(sec):
    """부호 있는 정수 초 표기(반올림). 예: -15, +113."""
    t = int(round(sec))
    return ("+%d" % t) if t > 0 else ("−%d" % abs(t) if t < 0 else "0")


def signed1(sec):
    """부호 있는 소수 1자리 초 표기(반올림만, 실제 값 그대로). 예: -15.3."""
    t = round(sec, 1)
    return ("+%.1f" % t) if t > 0 else ("−%.1f" % abs(t) if t < 0 else "0.0")


def cov_text(res):
    return "%.1f%%" % (100.0 * res["cov"])


TREND_TAIL = "(기록이 평소 주위에서 흔들리는 통계적 현상이며, 원인 효과가 아님)"


def trend_text(res):
    """두 항(기본 변화량 a, 평소 대비 보정 b×x)의 방향과 최종 예측의 방향(직전 기록 대비)을 각각 말하는 문장.
    (가) x = 0 / (나) 직전이 평소보다 느림 / (다) 직전이 평소보다 빠르고 |보정| < |a| / (라) 같은 경우에 |보정| > |a| / (마) |예상 변화| < 0.5초."""
    ref_s, m_s = int(round(res["ref"])), int(round(res["m"]))
    n = abs(ref_s - m_s)
    a_t, b_t, tot = res["a_term"], res["b_term"], res["tot_int"]
    t_abs = abs(tot)
    b_txt, a_txt = "%s초" % signed1(b_t), "%s초" % signed1(a_t)
    if abs(res["y_hat"]) < 0.5 or tot == 0:                                           # (마)
        why = "직전 기록이 평소와 같아" if ref_s == m_s else "직전 기록이 평소보다 %d초 %s" % (n, "느렸고" if ref_s > m_s else "빨랐고")
        return "%s 보정 항(%s)과 기본 변화량(%s)이 거의 상쇄되어 최종 예측은 직전 기록과 거의 같습니다. %s" % (why, b_txt, a_txt, TREND_TAIL)
    final = "최종 예측은 직전 기록보다 %d초 %s" % (t_abs, "빠릅니다" if tot < 0 else "느립니다")
    if ref_s == m_s:                                                                  # (가)
        return "직전 기록이 평소와 같아 보정 항은 없고(%s), 기본 변화량(%s)만 반영해 %s. %s" % (b_txt, a_txt, final, TREND_TAIL)
    if ref_s > m_s and b_t < 0 and a_t < 0:                                           # (나)
        return "직전 기록이 평소보다 %d초 느렸던 만큼 보정 항(%s)이 평소 쪽(빠른 쪽)으로 움직였고, 기본 변화량(%s)도 같은 방향이라 %s. %s" % (n, b_txt, a_txt, final, TREND_TAIL)
    if ref_s < m_s and b_t > 0 and a_t < 0:
        if tot < 0:                                                                   # (다)
            return "보정 항은 직전 기록이 평소보다 %d초 빨랐던 만큼 평소 쪽으로 %s 움직였지만, 기본 변화량(%s)이 더 커서 %s. %s" % (n, b_txt, a_txt, final, TREND_TAIL)
        return "보정 항은 직전 기록이 평소보다 %d초 빨랐던 만큼 평소 쪽으로 %s 움직였고 기본 변화량(%s)보다 커서 %s(평소 쪽). %s" % (n, b_txt, a_txt, final, TREND_TAIL)   # (라)
    return "보정 항(%s)과 기본 변화량(%s)을 더해 %s. %s" % (b_txt, a_txt, final, TREND_TAIL)        # 위 경우에 없는 부호 조합(현재 파라미터에서는 나오지 않음)


def format_report(res):
    if not res["eligible"]:
        return "[입력 부족] 예측할 수 없습니다. 부족한 조건:\n" + "\n".join("- " + s for s in res["shortfalls"]) + "\n해결: " + res["fix"]
    trend = trend_text(res)
    return ("다음 7일 안 5km 예상 기록: %s / 명목 80%% 범위 %s ~ %s\n" % (mmss(res["pred"]), mmss(res["lo"]), mmss(res["hi"]))
            + "다음 7일 안 적격 5km 기록이 관측되는 경우에 대한 예측입니다(명목 80% 범위)\n"
            + "실측 포함률 %s(cold-start test, 탐색 결과), 개인에게 80%%를 보장하지 않음\n" % cov_text(res)
            + "평소 기록(과거 중앙값): %s, 직전 기록: %s\n" % (mmss(res["m"]), mmss(res["ref"]))
            + "예상 변화 약 %s초 = 기본 변화량 a(%s초) + 평소 대비 보정 b×(직전 − 평소)(%s초)\n"
            % (signed(res["tot_int"]), signed1(res["a_term"]), signed1(res["b_term"]))
            + "직전 기록이 평소와 같았던 경우, 학습 데이터에서 다음 7일 첫 5km 변화의 중앙값은 약 %s초였음(상관, 인과 아님).\n"
            % signed(res["a_term"])
            + trend + "\n"
            + "현재 자료로는 지난 4주의 거리·횟수 정보가 이 예측을 더 좋게 만든다는 뚜렷한 증거를 확인하지 못해 평균 회귀 규칙만 사용합니다. 근거: 최종 평가 결과(docs/final_results_5km.md)\n"
            + "이 예측은 원인 분석이 아니며, 주마다 기록이 크게 흔들릴 수 있습니다.")


def format_goal(res, goal_sec, params):
    """--goal 비교 블록. 범위 밖/안만 알려 주며 달성 확률이나 훈련 처방이 아니다. 반환 (문구, 점검용 정보)."""
    lo_s, hi_s, pred_s = int(round(res["lo"])), int(round(res["hi"])), int(round(res["pred"]))
    if goal_sec < lo_s:
        label = "명목 범위보다 빠름"
    elif goal_sec > hi_s:
        label = "명목 범위보다 느림"
    else:
        label = "명목 범위 안"
    diff = pred_s - goal_sec                      # 예측 − 목표(초)
    if diff > 0:
        dtxt = "목표가 예측보다 %d초 빠름" % diff
    elif diff < 0:
        dtxt = "목표가 예측보다 %d초 느림" % (-diff)
    else:
        dtxt = "목표와 예측이 같음"
    q10, q90 = params["resid_q10"], params["resid_q90"]
    width_min = int(round((q90 - q10) / 60.0))
    fast = abs(int(round(q10)))
    s1 = "범위 안·밖은 달성 확률이나 달성 가능 여부를 뜻하지 않으며, 훈련 처방을 제공하지 않습니다."
    s2 = ("현재 범위 폭은 약 %d분(%s초 ~ %s초)으로 넓어, 예측보다 약 %d분 %d초 빠른 목표까지 범위 안으로 표시됩니다."
          % (width_min, signed(q10), signed(q90), fast // 60, fast % 60))
    text = ("목표 비교(목표 %s): %s\n" % (mmss(goal_sec), label)
            + "예측값과 목표의 차이(예측 − 목표): %s초 (%s)\n" % (signed(diff), dtxt)
            + s1 + "\n" + s2)
    info = {"goal": mmss(goal_sec), "ints": {str(abs(diff)), str(width_min), str(abs(int(round(q10)))), str(abs(int(round(q90)))), str(fast // 60), str(fast % 60)}, "label": label, "s1": s1, "s2": s2}
    return text, info


def load_notes(path=NOTES_PATH):
    """demo/stratum_notes.json(docs/b2_failure_5km.md의 train 폴드 밖 행에서 make_stratum_notes.py가 만듦)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["strata"]


# 층별 판정 조건(예측 시점에 관측 가능한 값만)과 문구의 <조건>. 키 = (차원, 문서의 층 이름). docs/demo_stratum_note_spec.md 3·4절
STRATUM_RULES = {
    ("a", "1개(x = 0)"): (lambda r: r["n_past"] == 1, "과거 유효 5km가 직전 기록 1개뿐"),
    ("a", "2~4개"): (lambda r: 2 <= r["n_past"] <= 4, "과거 유효 5km가 2~4개뿐"),
    ("a", "5개 이상"): (lambda r: r["n_past"] >= 5, "과거 유효 5km가 5개 이상"),
    ("b", "0초"): (lambda r: r["x"] == 0, "직전 기록이 평소와 같음"),
    ("b", "0초 초과 ~ 30초"): (lambda r: 0 < abs(r["x"]) <= 30, "직전 기록과 평소의 차이가 30초 이하"),
    ("b", "30초 초과 ~ 120초"): (lambda r: 30 < abs(r["x"]) <= 120, "직전 기록과 평소의 차이가 30초 넘고 2분 이하"),
    ("b", "120초 초과"): (lambda r: abs(r["x"]) > 120, "직전 기록이 평소와 2분 넘게 차이"),
    ("c", "0~7일"): (lambda r: r["days_ref"] < 7, "직전 기록이 7일 이내"),
    ("c", "7~14일"): (lambda r: 7 <= r["days_ref"] < 14, "직전 기록이 7~14일 전"),
    ("c", "14~28일"): (lambda r: r["days_ref"] >= 14, "직전 기록이 14일 이상 전"),
    ("d", "4~7회"): (lambda r: r["n_runs"] <= 7, "직전 28일 달리기가 4~7회"),
    ("d", "8~15회"): (lambda r: 8 <= r["n_runs"] <= 15, "직전 28일 달리기가 8~15회"),
    ("d", "16회 이상"): (lambda r: r["n_runs"] >= 16, "직전 28일 달리기가 16회 이상"),
    ("e", "300초 초과"): (lambda r: r["ref_stop"] is not None and r["ref_stop"] > 300, "직전 기록 세션에 5분 넘는 정지가 있음"),
}
NOTE_FMT = "※ 이 조건(%s)에서는 학습 자료 교차검증에서 명목 80%% 범위의 실제 포함률이 %.1f%%였습니다. 범위를 더 넓게 보셔야 합니다."
WIDTH_LINE = "※ 위 안내는 범위 폭을 바꾸지 않습니다(명목 80% 범위는 모든 조건에서 같은 폭입니다)."
NO_STOP_LINE = "※ 정지 시간 정보가 없어 이 조건은 확인하지 않았습니다"


def stratum_lines(res, notes=None):
    """해당하는 층의 안내 줄 전부(+ 범위 폭 불변 줄, + 정지 시간 정보 없음 줄). 반환 (줄 목록, 해당 층 목록). 범위 자체는 바꾸지 않는다."""
    if not res["eligible"]:
        return [], []
    notes = load_notes() if notes is None else notes
    matched = [s for s in notes if STRATUM_RULES[(s["dim"], s["label"])][0](res)]
    lines = [NOTE_FMT % (STRATUM_RULES[(s["dim"], s["label"])][1], s["cov"]) for s in matched]
    if lines:
        lines.append(WIDTH_LINE)
    if res["ref_stop"] is None and any(s["dim"] == "e" for s in notes):
        lines.append(NO_STOP_LINE)
    return lines, matched


def check_stratum_lines(lines, matched, res):
    """안내 줄의 문구와 숫자가 JSON 값과 맞는지 검사한다. 불일치 시 ValueError."""
    if not res["eligible"]:
        if lines:
            raise ValueError("자격 미달인데 층 안내가 있음")
        return True
    expect = [NOTE_FMT % (STRATUM_RULES[(s["dim"], s["label"])][1], s["cov"]) for s in matched]
    if matched:
        expect.append(WIDTH_LINE)
    if res["ref_stop"] is None and NO_STOP_LINE in lines:
        expect.append(NO_STOP_LINE)
    if lines != expect:
        raise ValueError("층 안내 문구가 JSON 값과 다름")
    return True


# 템플릿에 고정된 숫자(다음 7일, 5km, 80%, 지난 4주, 28일, 4회, 5,000초, 5.0~5.5km)와 파일 이름 속 5. 계산값이 아니므로 따로 허용한다.
TEMPLATE_INTS = {"4", "5", "7", "28", "80", "000"}
TEMPLATE_DECS = {"5.0", "5.5"}
_TOKEN = re.compile(r"\d+:\d{2}|\d+\.\d+|\d+")


def _abs_text(txt):
    return txt.lstrip("+-−")


def check_numbers(text, res):
    """출력 문구 속 모든 숫자가 계산값(또는 템플릿 고정 숫자)과 일치하는지 검사한다. 불일치 시 ValueError."""
    tokens = _TOKEN.findall(text)
    if not res["eligible"]:
        allowed = set(res["ints"]) | TEMPLATE_INTS | TEMPLATE_DECS
        for t in tokens:
            if t not in allowed:
                raise ValueError("계산값과 맞지 않는 숫자: " + t)
        return True
    ref_s, m_s = int(round(res["ref"])), int(round(res["m"]))
    needed = [mmss(res[k]) for k in ("pred", "lo", "hi", "m", "ref")]
    found = [t for t in tokens if ":" in t]
    if found != needed:
        raise ValueError("시간 표기가 계산값과 다름: %s != %s" % (found, needed))
    if int(round(res["a_term"] + res["b_term"])) != res["tot_int"]:
        raise ValueError("a 항 + 보정 항의 반올림 != 예상 변화")
    # 보정 항과 문구의 N(반올림한 직전-평소 차이)의 일치: 부호가 반대이고, 크기 차이가 반올림 오차(1초 x |b|) 이내
    gap = ref_s - m_s
    if abs(res["b_term"] - res["b_coef"] * gap) > abs(res["b_coef"]) * 1.0 + 1e-9:
        raise ValueError("보정 항이 문구의 N과 맞지 않음")
    if gap != 0 and res["b_term"] * gap * res["b_coef"] < 0:
        raise ValueError("보정 항의 방향이 N과 맞지 않음")
    dec = {_abs_text(signed1(res["a_term"])), _abs_text(signed1(res["b_term"])), cov_text(res).rstrip("%")}
    ints = {str(abs(gap)), _abs_text(signed(res["tot_int"])), _abs_text(signed(res["a_term"]))} | TEMPLATE_INTS
    for t in tokens:
        if ":" in t:
            continue
        if ("." in t and t not in dec) or ("." not in t and t not in ints):
            raise ValueError("계산값과 맞지 않는 숫자: " + t)
    expect = {"포함률": "실측 포함률 %s(cold-start test, 탐색 결과)" % cov_text(res), "예상 변화": "예상 변화 약 %s초" % signed(res["tot_int"]),
              "a 항": "기본 변화량 a(%s초)" % signed1(res["a_term"]), "보정 항": "b×(직전 − 평소)(%s초)" % signed1(res["b_term"]),
              "a 설명": "중앙값은 약 %s초였음" % signed(res["a_term"])}
    for k, frag in expect.items():
        if frag not in text:
            raise ValueError("문구에 계산값이 없음: " + k)
    if trend_text(res) not in text:
        raise ValueError("방향 문장이 계산값과 맞지 않음")
    if gap != 0 and (" %d초 " % abs(gap)) not in text:
        raise ValueError("직전-평소 차이(초)가 문구에 없음")
    return True


def check_goal_numbers(text, info):
    """목표 비교 블록의 숫자가 계산값과 일치하는지 검사한다. 불일치 시 ValueError."""
    tokens = _TOKEN.findall(text)
    times = [t for t in tokens if ":" in t]
    if times != [info["goal"]]:
        raise ValueError("목표 시간 표기가 계산값과 다름")
    for t in tokens:
        if ":" not in t and t not in info["ints"]:
            raise ValueError("계산값과 맞지 않는 숫자: " + t)
    for frag in (info["label"], info["s1"], info["s2"]):
        if frag not in text:
            raise ValueError("문구에 필요한 문장이 없음")
    return True


# ---------------------------------------------------------------- 뛴 다음의 피드백 (docs/demo_feedback_spec.md)
FB_HEAD = "뛴 다음의 피드백(기준 시점 이후 첫 5km 기록과 예측의 비교, 다음 7일 안에 5km가 관측된 경우에 한한 비교)"
FB_FIXED = ["범위 안·밖은 실력 변화의 증거가 아니며, 한 번의 기록에는 일상적인 흔들림이 포함됩니다.",
            "이 비교는 1회 관측이며 예측 성능의 근거가 아닙니다."]
FB_BASIS = "(train 폴드 밖 B2 잔차 기준)"
FB_POS = {"fast": "범위보다 빠름", "in": "범위 안", "slow": "범위보다 느림"}
FB_TEMPLATE_INTS = {"1", "2", "5", "7", "80"}     # 1회 관측·1% 미만, B2, 5km, 다음 7일, 명목 80%


def load_quantiles(path=QUANT_PATH):
    q = json.loads(Path(path).read_text(encoding="utf-8"))
    lv, v = q["levels_percent"], q["quantiles_sec"]
    if len(lv) != len(v) or len(v) < 2 or any(v[i] > v[i + 1] for i in range(len(v) - 1)):
        raise ValueError("분위수 표 형식 오류")
    return q


def residual_cdf(d, quantiles):
    """잔차(실제 - 예측, 초)의 누적비율 F(d)(%): 분위수 표 사이 선형 보간. 표의 범위(1%~99% 분위수) 밖이면 None."""
    lv, v = quantiles["levels_percent"], quantiles["quantiles_sec"]
    if d < v[0] or d > v[-1]:
        return None
    i = min(bisect.bisect_right(v, d) - 1, len(v) - 2)
    den = v[i + 1] - v[i]
    frac = (d - v[i]) / den if den > 0 else 0.0
    return lv[i] + frac * (lv[i + 1] - lv[i])


def find_actual(sessions, asof, now):
    """[asof, asof+7일)에 시작한 5.0~5.5km 달리기 중 t5_sec가 있는 가장 이른 기록. 극단(T5 > 5,000초)도 제외하지 않는다."""
    end = asof + 7 * DAY
    c5 = sorted((s for s in sessions if asof <= s["start"] < end and KM_LO <= s["km"] <= KM_HI), key=lambda s: s["start"])
    actual, skipped = None, 0
    for s in c5:
        if s["t5"] is not None:
            actual = s
            break
        skipped += 1
    return {"actual": actual, "skipped": skipped, "n_cands": len(c5), "elapsed": end <= now}


def format_feedback(res, fb, quantiles):
    """피드백 블록. 반환 (문구, 점검용 정보)."""
    info = {"ints": set(), "times": [], "dates": [], "share": None, "unobserved": False}
    lines = [FB_HEAD]
    summ = []
    if not res["eligible"]:
        lines.append("[입력 부족] 예측이 없어 비교하지 않습니다(위 부족한 조건이 충족되면 비교할 수 있습니다).")
        summ.append("[입력 부족] 예측이 없어 비교하지 않습니다")
    elif fb["actual"] is None:
        lines.append(("[결과 미관측] " if fb["elapsed"] else "") + "다음 7일 안에 비교할 5km 기록이 없습니다")
        if not fb["elapsed"]:
            lines.append("아직 7일이 지나지 않았습니다")
        info["unobserved"] = bool(fb["elapsed"])
        summ.append(("[결과 미관측] " if fb["elapsed"] else "") + "다음 7일 안에 비교할 5km 기록이 없습니다" + ("" if fb["elapsed"] else "(아직 7일이 지나지 않았습니다)"))
        if fb["n_cands"]:
            lines.append("5.0~5.5km 기록 %d개가 있으나 5km 시간(t5_sec)이 없어 비교하지 못했습니다" % fb["n_cands"])
            summ.append("5.0~5.5km 기록 %d개가 있으나 5km 시간(t5_sec)이 없어 비교하지 못했습니다" % fb["n_cands"])
            info["ints"].add(str(fb["n_cands"]))
    else:
        act = fb["actual"]
        act_s, pred_s, lo_s, hi_s = int(round(act["t5"])), int(round(res["pred"])), int(round(res["lo"])), int(round(res["hi"]))
        d = act_s - pred_s
        date = datetime.fromtimestamp(act["start"], tz=timezone.utc).strftime("%Y-%m-%d")
        lines.append("실제 기록: %s 5km 시간 %s" % (date, mmss(act["t5"])))
        if fb["skipped"]:
            lines.append("이보다 먼저 뛴 5.0~5.5km 기록 %d개는 5km 시간(t5_sec)이 없어 건너뛰었습니다" % fb["skipped"])
            info["ints"].add(str(fb["skipped"]))
        lines.append("예측: %s / 명목 80%% 범위 %s ~ %s" % (mmss(res["pred"]), mmss(res["lo"]), mmss(res["hi"])))
        lines.append("차이(실제 − 예측): %s초 (음수 = 예측보다 빠름)" % signed(d))
        pos = "fast" if act_s < lo_s else ("slow" if act_s > hi_s else "in")
        lines.append("명목 범위 대비 위치: " + FB_POS[pos])
        summ.append("실제 %s 5km %s(예측 %s, 차이 %s초) · 명목 범위 대비: %s" % (date, mmss(act["t5"]), mmss(res["pred"]), signed(d), FB_POS[pos]))
        if fb["skipped"]:
            summ.append("이보다 먼저 뛴 5.0~5.5km 기록 %d개는 t5_sec가 없어 건너뛰었습니다" % fb["skipped"])
        info["times"] = [mmss(act["t5"]), mmss(res["pred"]), mmss(res["lo"]), mmss(res["hi"])]
        info["dates"] = date.split("-")
        info["ints"].add(str(abs(d)))
        info["d"], info["pos"] = d, pos
        if d == 0:
            lines.append("예측과 같았습니다")
            summ.append("예측과 같았습니다")
        else:
            f = residual_cdf(d, quantiles)
            word, share = ("빨랐던", f) if d < 0 else ("느렸던", None if f is None else 100.0 - f)
            if share is None:
                share_txt = "1% 미만"
            else:
                share_txt = "%.1f%%" % share
                info["share"] = share_txt.rstrip("%")
            lines.append("학습 자료에서 예측보다 %d초 이상 %s 경우: %s %s" % (abs(d), word, share_txt, FB_BASIS))
            summ.append("학습 자료에서 예측보다 %d초 이상 %s 경우: %s %s" % (abs(d), word, share_txt, FB_BASIS))
    lines += FB_FIXED
    info["summary"] = [FB_SUMMARY_HEAD] + [IND + t for t in summ] + [IND + " ".join(FB_FIXED)]
    return "\n".join(lines), info


def check_feedback_numbers(text, info):
    """피드백 블록 속 숫자가 계산값(또는 템플릿 고정 숫자)과 일치하는지 검사한다. 불일치 시 ValueError."""
    tokens = _TOKEN.findall(text)
    if [t for t in tokens if ":" in t] != info["times"]:
        raise ValueError("피드백의 시간 표기가 계산값과 다름")
    allowed = set(info["ints"]) | set(info["dates"]) | FB_TEMPLATE_INTS | TEMPLATE_DECS | ({info["share"]} if info["share"] else set())
    for t in tokens:
        if ":" not in t and t not in allowed:
            raise ValueError("피드백에 계산값과 맞지 않는 숫자: " + t)
    return True


# ---------------------------------------------------------------- 예측 저장(--log)과 결과 연결(--resolve) (docs/prospective_log_spec.md)
REPO = Path(__file__).resolve().parent.parent
WARN_TAG = "[조건별 검증 부족: 개발 결과 기준 경고] train 폴드 밖 개발 결과 기준이며 새 자료로 확인되지 않았습니다"
ONE_PERSON_LINE = "1인 기록은 입력 연결·계산·표시 확인 사례이며 일반화 성능의 근거가 아님"
NOTICE_KIND = {("a", "1개(x = 0)"): "과거 5km 1개", ("a", "2~4개"): "과거 5km 2~4개", ("b", "120초 초과"): "평소와 2분 넘게 차이", ("e", "300초 초과"): "정지 5분 초과"}
NO_STOP_KIND = "정지 정보 없음"
LOG_STATUS_OK, LOG_STATUS_SHORT = "예측", "입력 부족"
RES_OBS, RES_NONE, RES_WAIT, RES_SHORT = "결과 관측", "결과 미관측", "7일 미경과", "입력 부족"


def inside_repo(path):
    try:
        Path(path).resolve().relative_to(REPO)
        return True
    except ValueError:
        return False


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    """(코드 커밋 해시, demo/ 미커밋 변경 유무). git이 없거나 실패하면 ('unknown', None)."""
    import subprocess
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "demo"], cwd=REPO, capture_output=True, text=True, timeout=20, check=True).stdout.strip())
        return h, dirty
    except Exception:
        return "unknown", None


def notice_kinds(matched, lines):
    kinds = [NOTICE_KIND.get((s["dim"], s["label"]), "%s %s" % (s["dim"], s["label"])) for s in matched]
    if NO_STOP_LINE in lines:
        kinds.append(NO_STOP_KIND)
    return kinds


def build_log_entry(csv_path, asof_text, asof_s, res, matched, note_lines, sessions, now, commit=None):
    """저장할 한 줄(dict). 좌표·개별 기록 값은 넣지 않는다(개수·해시·집계 값만)."""
    import datetime as _dt
    h, dirty = commit if commit is not None else git_commit()
    n_later = sum(1 for s in sessions if s["start"] >= asof_s)
    ok = res["eligible"]
    return {"logged_at": _dt.datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "asof": str(asof_text), "asof_epoch": asof_s,
            "csv_sha256": sha256_file(csv_path), "code_commit": h, "code_dirty": dirty,
            "params_sha256": sha256_file(PARAMS_PATH), "notes_sha256": sha256_file(NOTES_PATH),
            "status": LOG_STATUS_OK if ok else LOG_STATUS_SHORT,
            "pred_sec": int(round(res["pred"])) if ok else None, "range_lo_sec": int(round(res["lo"])) if ok else None, "range_hi_sec": int(round(res["hi"])) if ok else None,
            "notices": notice_kinds(matched, note_lines) if ok else [],
            "kind": "후향 점검" if n_later else "전향", "n_later_sessions": n_later}


def append_log(log_path, entry):
    """로그에 한 줄 추가. 같은 (asof, csv_sha256, code_commit) 줄이 있으면 추가하지 않는다. 반환 (추가 여부, 줄 번호)."""
    p = Path(log_path)
    if inside_repo(p):
        raise InputProblem("--log 파일이 저장소 폴더 안입니다(개인 기록이 커밋될 수 있음).", "저장소 밖 폴더의 경로를 지정하세요. 예: <저장소 밖 폴더>/my_runs/predictions.mine.jsonl")
    lines = []
    if p.exists():
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, l in enumerate(lines, start=1):
            try:
                e = json.loads(l)
            except ValueError:
                raise InputProblem("--log 파일 %d번째 줄을 읽을 수 없습니다." % i, "JSON Lines 형식(한 줄에 JSON 하나)인 파일을 지정하거나 새 파일 이름을 쓰세요.")
            if (e.get("asof_epoch"), e.get("csv_sha256"), e.get("code_commit")) == (entry["asof_epoch"], entry["csv_sha256"], entry["code_commit"]):
                return False, i
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        raise InputProblem("--log 파일에 쓸 수 없습니다(%s)." % type(e).__name__, "경로와 쓰기 권한을 확인하세요.")
    return True, len(lines) + 1


def load_log(path):
    p = Path(path)
    if not p.exists():
        raise InputProblem("저장 파일을 찾을 수 없습니다: %s" % path, "--log로 만든 .jsonl 경로를 확인하세요.")
    out = []
    for i, l in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not l.strip():
            continue
        try:
            e = json.loads(l)
            float(e["asof_epoch"])
            if e["status"] not in (LOG_STATUS_OK, LOG_STATUS_SHORT):
                raise ValueError("status")
            if e["status"] == LOG_STATUS_OK:
                int(e["range_lo_sec"]), int(e["range_hi_sec"])
            e["kind"]
        except (ValueError, KeyError, TypeError):
            raise InputProblem("%d번째 줄을 읽을 수 없습니다." % i, "--log로 저장한 줄이 아니거나 손상되었습니다. 해당 줄을 지우거나 새 파일을 쓰세요.")
        out.append(e)
    if not out:
        raise InputProblem("저장 파일에 예측이 없습니다.", "--log로 예측을 먼저 저장하세요.")
    return out


def classify_entries(entries, sessions, now):
    """저장된 줄마다 상태를 정한다. 반환 [dict(상태, 위치, 정답 세션 시작, 공유, 5km인데 t5 없음)]. 예측은 다시 계산하지 않는다."""
    rows = []
    for e in entries:
        r = {"kind": e["kind"], "asof_epoch": float(e["asof_epoch"]), "state": None, "pos": None, "actual_start": None, "shared": False, "n_no_t5": 0}
        if e["status"] != LOG_STATUS_OK:
            r["state"] = RES_SHORT
        else:
            fb = find_actual(sessions, r["asof_epoch"], now)
            if fb["actual"] is not None:
                act_s = int(round(fb["actual"]["t5"]))
                lo, hi = int(e["range_lo_sec"]), int(e["range_hi_sec"])
                r.update(state=RES_OBS, pos=("fast" if act_s < lo else ("slow" if act_s > hi else "in")), actual_start=fb["actual"]["start"])
            elif not fb["elapsed"]:
                r["state"] = RES_WAIT
            else:
                r["state"] = RES_NONE
                r["n_no_t5"] = fb["n_cands"]
        rows.append(r)
    cnt = {}
    for r in rows:
        if r["actual_start"] is not None:
            cnt[r["actual_start"]] = cnt.get(r["actual_start"], 0) + 1
    for r in rows:
        r["shared"] = r["actual_start"] is not None and cnt[r["actual_start"]] > 1
    return rows


def _count(rows):
    c = {RES_SHORT: 0, RES_OBS: 0, RES_NONE: 0, RES_WAIT: 0}
    pos = {"in": 0, "fast": 0, "slow": 0}
    for r in rows:
        c[r["state"]] += 1
        if r["pos"]:
            pos[r["pos"]] += 1
    indep = len({r["actual_start"] for r in rows if r["actual_start"] is not None})
    assert sum(c.values()) == len(rows)
    return c, pos, indep


def format_resolve(entries, rows, data_sha, detail=False):
    c, pos, indep = _count(rows)
    out = ["예측 저장 결과 연결 요약(저장된 예측 %d개, 데이터 sha256 %s…)" % (len(rows), data_sha[:12]),
           "각 예측은 [asof, asof+7일)에 시작한 첫 적격 5km(5.0~5.5km, t5_sec 있음, 극단 포함) 하나와만 연결했고 예측은 다시 계산하지 않았습니다.",
           "전체 예측 수: %d" % len(rows),
           "입력 부족: %d" % c[RES_SHORT],
           "결과 관측: %d (명목 범위 안 %d / 범위보다 빠름 %d / 범위보다 느림 %d), 독립 결과 수: %d" % (c[RES_OBS], pos["in"], pos["fast"], pos["slow"], indep),
           "결과 미관측: %d (7일이 지났으나 적격 5km 기록 없음; 실패나 적중으로 세지 않음)" % c[RES_NONE],
           "7일 미경과: %d" % c[RES_WAIT]]
    n_shared = sum(1 for r in rows if r["shared"])
    if n_shared:
        out.append("같은 미래 세션을 정답으로 공유한 예측: %d개(독립 결과 수는 서로 다른 정답 세션의 수)" % n_shared)
    n_no_t5 = sum(1 for r in rows if r["n_no_t5"])
    if n_no_t5:
        out.append("결과 미관측 중 5.0~5.5km 기록은 있으나 t5_sec가 없는 예측: %d개" % n_no_t5)
    for kind in ("전향", "후향 점검"):
        sub = [r for r in rows if r["kind"] == kind]
        if sub:
            cc, pp, ii = _count(sub)
            out.append("%s %d개: 입력 부족 %d, 결과 관측 %d(독립 결과 수 %d), 결과 미관측 %d, 7일 미경과 %d" % (kind, len(sub), cc[RES_SHORT], cc[RES_OBS], ii, cc[RES_NONE], cc[RES_WAIT]))
    out.append("후향 점검 = 입력 CSV에 asof 이후 기록이 이미 있는 상태에서 저장한 예측(결과를 모르는 전향 예측과 구분).")
    out.append(ONE_PERSON_LINE)
    if detail:
        out.append("")
        out.append("번호 | 종류 | asof 날짜 | 상태 | 공유")
        for i, r in enumerate(rows, start=1):
            st = r["state"] if r["pos"] is None else "%s(%s)" % (r["state"], FB_POS[r["pos"]])
            out.append("%d | %s | %s | %s | %s" % (i, r["kind"], datetime.fromtimestamp(r["asof_epoch"], tz=timezone.utc).strftime("%Y-%m-%d"), st, "공유" if r["shared"] else "-"))
    return "\n".join(out)


def resolve(log_path, data_path, now=None, detail=False):
    """저장된 예측과 데이터 CSV를 연결해 요약한다. 반환 (rows, 문구)."""
    now = time.time() if now is None else now
    entries = load_log(log_path)
    sessions, _notices = load_sessions(data_path, now)
    rows = classify_entries(entries, sessions, now)
    return rows, format_resolve(entries, rows, sha256_file(data_path), detail)


def run_full(csv_path, asof, params=None, goal=None, now=None, feedback=False, log=None, commit=None):
    """입력 검사 → 예측 → 문구와 숫자 검사. 입력·옵션 문제는 InputProblem. 반환 (res, 기존 전체 출력 문구). 요약 화면 도입 전의 출력이며 --detail의 '자세히 보기'와 같다."""
    params = params or load_params()
    now = time.time() if now is None else now
    asof_s = parse_asof(asof, now)
    goal_s = parse_goal(goal) if goal is not None else None
    sessions, notices = load_sessions(csv_path, now)
    res = assess(sessions, asof_s, params)
    main_text = format_report(res)
    check_numbers(main_text, res)
    parts = ["※ " + n for n in notices + res["notices"]] + [main_text]
    note_lines, matched = stratum_lines(res)
    check_stratum_lines(note_lines, matched, res)
    P = {"input_notices": list(notices + res["notices"]), "matched": matched, "note_lines": note_lines, "goal": None, "fb": None, "core": None, "log_line": None}
    res["_parts"] = P
    if matched:
        parts.append(WARN_TAG)              # [조건별 검증 부족: 개발 결과 기준 경고]
    parts += note_lines
    if goal_s is not None:
        if res["eligible"]:
            g_text, info = format_goal(res, goal_s, params)
            check_goal_numbers(g_text, info)
            parts.append("")
            parts.append(g_text)
            P["goal"] = (g_text, info)
        else:
            P["goal"] = "none"
            parts.append("")
            parts.append("목표 비교는 예측이 가능할 때만 표시합니다. 위 부족한 조건을 채운 뒤 다시 실행하세요.")
    if feedback:
        fb_text, fb_info = format_feedback(res, find_actual(sessions, asof_s, now), load_quantiles())
        check_feedback_numbers(fb_text, fb_info)
        res["feedback"] = fb_info
        P["fb"] = (fb_text, fb_info)
        parts.append("")
        parts.append(fb_text)
    P["core"] = "\n".join(parts)
    if log:
        entry = build_log_entry(csv_path, asof, asof_s, res, matched, note_lines, sessions, now, commit)
        added, n = append_log(log, entry)
        res["log_entry"] = entry
        P["log_line"] = "예측을 저장했습니다(%d번째 줄)." % n if added else "이미 저장되어 있습니다(%d번째 줄, 같은 기준 시점·입력 파일·코드 커밋)." % n
        parts.append("")
        parts.append(P["log_line"])
    return res, "\n".join(parts)


# ---------------------------------------------------------------- 요약 화면 + 자세히 보기 (docs/demo_screen_spec.md)
STATUS_SHORT = "[입력 부족]"
STATUS_WARN = "[조건별 검증 부족: 개발 결과 기준 경고]"
STATUS_UNOBS = "[결과 미관측]"
NO_SPECIAL = "특이 조건 없음"
CAUTION = "원인 분석·훈련 처방이 아닙니다."
SUMMARY_TAIL = "--detail로 계산 과정과 근거 문서를 볼 수 있습니다."
DETAIL_TAIL = "아래 자세히 보기에 계산 과정과 근거 문서가 있습니다."
DETAIL_HEAD = "───── 자세히 보기(--detail): 기존 전체 출력 ─────"
FIX_SUMMARY = "직전 28일 안의 달리기 기록(4회 이상)과 5.0~5.5km의 5km 기록 1회 이상(t5_sec 포함)을 CSV에 추가하거나 기록이 충분한 기준 시점(--asof)을 지정하면 예측할 수 있습니다."
IND = "       "
FB_SUMMARY_HEAD = "[뛴 다음의 피드백] 기준 시점 이후 첫 5km와 예측의 비교(다음 7일 안에 5km가 관측된 경우에 한함)"
GOAL_SUMMARY_NO = "[목표 비교] 예측이 가능할 때만 표시합니다(위 [안내]의 부족한 조건 참고)."


def _block(label, lines):
    """머리표 첫 줄 + 이어지는 줄은 들여쓰기."""
    return [label + " " + lines[0]] + [IND + l for l in lines[1:]]


def summary_lines(res, detail=False):
    """요약 화면(기본 출력). 계산값은 res에서만 가져온다. 보이는 순서와 묶음만 바꾼 것이다."""
    P = res["_parts"]
    ok = res["eligible"]
    if ok:
        L = _block("[결과]", ["다음 7일 안 5km 예상 기록: %s / 명목 80%% 범위 %s ~ %s" % (mmss(res["pred"]), mmss(res["lo"]), mmss(res["hi"])),
                             "다음 7일 안 적격 5km 기록이 관측되는 경우에 대한 예측입니다(명목 80% 범위)",
                             "실측 포함률 %s(cold-start test, 탐색 결과), 개인에게 80%%를 보장하지 않음" % cov_text(res)])
    else:
        L = _block("[결과]", ["예측할 수 없습니다(자격 미달)."])
    status = []
    if not ok:
        status.append(STATUS_SHORT)
    if P["matched"]:
        status.append(STATUS_WARN)
    if P["fb"] is not None and P["fb"][1].get("unobserved"):
        status.append(STATUS_UNOBS)
    L.append("[상태] " + (" ".join(status) if status else NO_SPECIAL))
    notes = []
    if not ok:
        notes += ["부족한 조건: " + s for s in res["shortfalls"]]
    else:
        notes += ["%s: 학습 자료 교차검증 포함률 %.1f%%(%d건/%d명, 개발 결과)" % (STRATUM_RULES[(s["dim"], s["label"])][1], s["cov"], s["n"], s["users"]) for s in P["matched"]]
        if NO_STOP_LINE in P["note_lines"]:
            notes.append(NO_STOP_LINE[2:])
    notes += P["input_notices"]
    L += _block("[안내]", notes or ["조건별 안내 없음"])
    if ok:
        L += _block("[근거]", ["평소 %s, 직전 %s → 예상 변화 약 %s초(기본 변화량 %s초 + 평소 대비 보정 %s초)" % (mmss(res["m"]), mmss(res["ref"]), signed(res["tot_int"]), signed1(res["a_term"]), signed1(res["b_term"]))])
    else:
        L += _block("[해결]", [FIX_SUMMARY])
    L.append("[주의] " + CAUTION + " " + (DETAIL_TAIL if detail else SUMMARY_TAIL))
    return L


def goal_summary_lines(goal_part):
    """목표 비교 블록(머리표 + 핵심 2~3줄). goal_part = (전체 문구, 점검 정보) 또는 'none'(예측 없음)."""
    if goal_part == "none":
        return [GOAL_SUMMARY_NO]
    text, info = goal_part
    ls = text.split("\n")
    head = "[목표 비교] 목표 %s: %s" % (info["goal"], info["label"])
    return [head, IND + ls[1], IND + info["s1"]]


def check_summary_numbers(summary, core, notes):
    """요약 화면의 숫자 토큰이 자세히 보기(기존 전체 출력)와 stratum_notes.json의 관측 수·사용자 수에 있는 값뿐인지 검사한다."""
    allowed = set(_TOKEN.findall(core)) | TEMPLATE_INTS | TEMPLATE_DECS
    for s in notes:
        allowed |= {str(s["n"]), str(s["users"])}
    bad = [t for t in _TOKEN.findall(summary) if t not in allowed]
    if bad:
        raise ValueError("요약 화면에 계산값과 맞지 않는 숫자: " + ", ".join(sorted(set(bad))))
    return True


def run(csv_path, asof, params=None, goal=None, now=None, feedback=False, log=None, commit=None, detail=False):
    """기본 출력 = 요약 화면(+ 목표·피드백 블록), detail=True면 뒤에 기존 전체 출력(run_full)을 그대로 붙인다. 반환 (res, 출력 문구)."""
    res, _full = run_full(csv_path, asof, params, goal, now, feedback, log, commit)
    P = res["_parts"]
    out = summary_lines(res, detail)
    if P["goal"] is not None:
        out += [""] + goal_summary_lines(P["goal"])
    if P["fb"] is not None:
        out += [""] + P["fb"][1]["summary"]
    summary = "\n".join(out)
    check_summary_numbers(summary, P["core"], load_notes())
    text = summary
    if detail:
        text += "\n\n" + DETAIL_HEAD + "\n" + P["core"]
    if P["log_line"]:
        text += "\n\n" + P["log_line"]
    return res, text


def main(argv=None):
    ap = argparse.ArgumentParser(description="다음 7일 5km 기록 예상(평균 회귀 규칙) 시연")
    ap.add_argument("csv", nargs="?", default=None, help="세션 CSV (열: " + ", ".join(REQUIRED_COLS) + ")")
    ap.add_argument("--asof", default=None, help="기준 시점(에포크 초 또는 ISO 일시)")
    ap.add_argument("--goal", default=None, help="목표 기록 mm:ss(10:00~83:20). 예측 범위와 비교해 '명목 범위보다 빠름/안/느림'을 보여 준다")
    ap.add_argument("--feedback", action="store_true", help="예측 뒤에 기준 시점 이후 7일 안 첫 5km 기록과 예측을 비교한 피드백을 붙인다(처방·원인·실력 판정 아님)")
    ap.add_argument("--log", default=None, help="예측을 이 JSON Lines 파일에 한 줄 추가한다(저장소 밖 경로; 좌표·개별 기록 값은 저장하지 않음)")
    ap.add_argument("--resolve", default=None, help="--log로 저장한 파일의 예측을 --data CSV의 실제 결과와 연결해 요약한다")
    ap.add_argument("--data", default=None, help="--resolve에서 결과를 찾을 세션 CSV")
    ap.add_argument("--detail", action="store_true", help="요약 화면 뒤에 기존 전체 출력(자세히 보기)을 붙인다. --resolve에서는 줄별 상태(날짜·상태·공유)를 덧붙인다")
    a = ap.parse_args(argv)
    if a.resolve:
        try:
            if not a.data:
                raise InputProblem("--resolve에는 --data CSV가 필요합니다.", "예: python demo/coach_demo.py --resolve 예측.jsonl --data 세션.csv")
            _rows, text = resolve(a.resolve, a.data, detail=a.detail)
        except InputProblem as e:
            print(problem_text(e))
            return 2
        except Exception as e:
            print("오류: 예상하지 못한 문제가 생겼습니다(%s)." % type(e).__name__)
            print("해결: 저장 파일과 데이터 CSV의 형식을 확인하고, 계속되면 파일과 명령을 함께 알려 주세요.")
            return 2
        print(text)
        return 0
    if a.csv is None or a.asof is None:
        ap.error("세션 CSV와 --asof가 필요합니다(결과 연결은 --resolve 파일 --data CSV).")
    try:
        _res, text = run(a.csv, a.asof, goal=a.goal, feedback=a.feedback, log=a.log, detail=a.detail)
    except InputProblem as e:
        print(problem_text(e))
        return 2
    except Exception as e:      # 예상하지 못한 문제도 추적 메시지 없이 안내한다
        print("오류: 예상하지 못한 문제가 생겼습니다(%s)." % type(e).__name__)
        print("해결: 입력 CSV의 열·값 형식을 확인하고, 계속되면 입력 파일과 명령을 함께 알려 주세요.")
        return 2
    print(text)
    return 0


if __name__ == "__main__":
    for _st in (sys.stdout, sys.stderr):
        if hasattr(_st, "reconfigure"):
            _st.reconfigure(encoding="utf-8")
    sys.exit(main())
