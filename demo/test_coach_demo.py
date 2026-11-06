# -*- coding: utf-8 -*-
"""coach_demo 단위 테스트. pytest 없이도 `python demo/test_coach_demo.py`로 실행된다(test_ 함수를 assert로 직접 호출)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coach_demo as cd  # noqa: E402

P = {"a": -15.0, "b": -0.8, "resid_q10": -200.0, "resid_q90": 280.0, "test_coverage_cold": 0.739}
ASOF = 100 * cd.DAY


def S(days_before_end, km=3.0, t5=None, start_gap=1800):
    end = ASOF - days_before_end * cd.DAY
    return {"start": end - start_gap, "end": end, "km": km, "t5": t5}


def test_eligibility_runs_and_5km():
    four = [S(d) for d in (1, 5, 9, 13)]
    r = cd.assess(four, ASOF, P)
    assert not r["eligible"] and r["n_runs"] == 4 and len(r["shortfalls"]) == 1   # 5km 없음
    three_5km = [S(1, 5.2, 1500), S(5), S(9)]
    r = cd.assess(three_5km, ASOF, P)
    assert not r["eligible"] and "3회" in r["shortfalls"][0] and len(r["shortfalls"]) == 1   # 횟수 부족
    ok = four[:3] + [S(2, 5.2, 1500)]
    assert cd.assess(ok, ASOF, P)["eligible"]


def test_window_boundaries():
    # 기준 시점 이후(미완료) 세션과 28일보다 오래된 세션은 창에 안 들어간다
    sess = [S(1), S(5), S(9), S(13, 5.2, 1500), S(-1, 5.2, 1400), S(29, 5.2, 1450)]
    r = cd.assess(sess, ASOF, P)
    assert r["n_runs"] == 4 and r["eligible"]
    assert r["ref"] == 1500   # 28일보다 오래된 5km나 미래 5km가 기준이 되지 않는다


def test_invalid_5km_excluded():
    # 극단(5000초 초과)·거리 범위 밖은 유효 5km가 아님
    sess = [S(1), S(5), S(9), S(13, 5.2, 5001), S(14, 4.9, 1500)]
    assert not cd.assess(sess, ASOF, P)["eligible"]


def test_b2_formula():
    assert abs(cd.predict_change(1500, 1400, P) - (-15.0 + -0.8 * 100)) < 1e-9
    sess = [S(1), S(5), S(9), S(20, 5.2, 1400), S(2, 5.1, 1500)]
    r = cd.assess(sess, ASOF, P)
    assert r["ref"] == 1500 and r["m"] == 1450          # m = 중앙값(1400, 1500)
    y = -15.0 - 0.8 * (1500 - 1450)
    assert abs(r["y_hat"] - y) < 1e-9
    assert abs(r["pred"] - (1500 + y)) < 1e-9
    assert abs(r["lo"] - (1500 + y - 200)) < 1e-9 and abs(r["hi"] - (1500 + y + 280)) < 1e-9


def test_single_past_5km_m_equals_ref():
    sess = [S(1), S(5), S(9), S(2, 5.1, 1500)]
    r = cd.assess(sess, ASOF, P)
    assert r["m"] == r["ref"] == 1500 and abs(r["y_hat"] - (-15.0)) < 1e-9


def test_reference_is_latest_start():
    sess = [S(1), S(5), S(9, 5.2, 1600), S(3, 5.2, 1450)]
    assert cd.assess(sess, ASOF, P)["ref"] == 1450


def test_mmss():
    assert cd.mmss(1499.6) == "25:00" and cd.mmss(61) == "01:01" and cd.mmss(0) == "00:00"


def test_number_check_passes_and_catches_mismatch():
    sess = [S(1), S(5), S(9), S(20, 5.2, 1400), S(2, 5.1, 1500)]
    r = cd.assess(sess, ASOF, P)
    text = cd.format_report(r)
    assert cd.check_numbers(text, r)
    assert "50초 느렸" in text                        # 1500 - 1450
    bad = text.replace("50초", "51초")
    try:
        cd.check_numbers(bad, r)
    except ValueError:
        pass
    else:
        raise AssertionError("숫자 불일치를 잡지 못함")
    bad2 = text.replace(cd.mmss(r["pred"]), "99:59", 1)
    try:
        cd.check_numbers(bad2, r)
    except ValueError:
        pass
    else:
        raise AssertionError("시간 불일치를 잡지 못함")


def test_two_term_split_and_coverage_text():
    sess = [S(1), S(5), S(9), S(20, 5.2, 1400), S(2, 5.1, 1500)]
    r = cd.assess(sess, ASOF, P)
    assert r["a_term"] == -15.0 and abs(r["b_term"] - (-40.0)) < 1e-9 and r["tot_int"] == -55      # -15 + (-0.8 x 50)
    text = cd.format_report(r)
    assert "기본 변화량 a(−15.0초)" in text and "b×(직전 − 평소)(−40.0초)" in text and "예상 변화 약 −55초" in text
    assert "테스트 포함률 73.9%(탐색 결과), 개인에게 80%를 보장하지 않음" in text and "명목 80% 범위 " in text
    assert "다음 7일 안에 5km 기록이 관측된 경우를 기준으로 한 예측입니다" in text
    assert "직전 기록이 평소보다 50초 느렸던 만큼 보정 항(−40.0초)이 평소 쪽(빠른 쪽)으로 움직였고" in text and "통계적 현상이며, 원인 효과가 아님" in text and "때문에" not in text
    assert "중앙값은 약 −15초였음(상관, 인과 아님)" in text
    assert "평균 변화량" not in text and "49c02f8" not in text and "최종 평가 결과(docs/final_results_5km.md)" in text
    # N과 보정 항 일치: 50초 느림 <-> 보정 -40.0 (= -0.8 x 50)
    assert "평소보다 50초 느렸" in text
    # 항·포함률·N을 바꾸면 검사에서 걸린다
    for old, new in [("(−40.0초)", "(−41.0초)"), ("73.9%", "74.0%"), ("약 −15초였음", "약 −16초였음"), ("(−15.0초)", "(−15.1초)")]:
        try:
            cd.check_numbers(text.replace(old, new), r)
        except ValueError:
            continue
        raise AssertionError("검사를 통과함: " + old)
    # 보정 항이 N과 어긋난 결과 객체는 검사에서 걸린다
    bad = dict(r, b_term=-10.0, tot_int=-25)
    try:
        cd.check_numbers(cd.format_report(bad), bad)
    except ValueError:
        pass
    else:
        raise AssertionError("보정 항-N 불일치를 잡지 못함")


def test_ineligible_text_lists_only_shortfalls():
    r = cd.assess([S(1), S(5)], ASOF, P)
    text = cd.format_report(r)
    assert cd.check_numbers(text, r)
    assert "2회" in text and "5km" in text and "예상 기록" not in text


def test_faster_and_equal_wording():
    sess = [S(1), S(5), S(9), S(20, 5.2, 1600), S(2, 5.1, 1500)]
    assert "빨랐" in cd.format_report(cd.assess(sess, ASOF, P))
    sess = [S(1), S(5), S(9), S(2, 5.1, 1500)]
    assert "같아" in cd.format_report(cd.assess(sess, ASOF, P))


# ---------------------------------------------------------------- P4: 예외 처리 (프로그램이 멈추지 않고 이유·해결 방법을 한국어로 안내)
import contextlib
import io
import tempfile

NOW = ASOF + 10 * cd.DAY          # 테스트용 현재 시각(기준 시점보다 10일 뒤)
HEADER = "start_time,end_time,distance_km,duration_sec,avg_hr,t5_sec"


def _write(tmp, name, text):
    p = Path(tmp) / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _csv(tmp, rows, name="s.csv"):
    """rows: (start, end, km, t5 또는 None) 목록 -> CSV 파일 경로(시각은 에포크 초 숫자)."""
    lines = [HEADER] + ["%s,%s,%s,1500,150,%s" % (s, e, km, "" if t5 is None else t5) for s, e, km, t5 in rows]
    return _write(tmp, name, "\n".join(lines) + "\n")


def _rows_ok(extra=()):
    """자격을 채우는 기본 기록: 4회 달리기(5km 1개 포함) + extra."""
    base = [(ASOF - d * cd.DAY - 1800, ASOF - d * cd.DAY, km, t5) for d, km, t5 in [(1, 3.0, None), (5, 3.5, None), (9, 4.0, None), (3, 5.2, 1500.0)]]
    return base + list(extra)


def _problem(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except cd.InputProblem as e:
        assert e.reason and e.fix, "이유와 해결 방법이 모두 있어야 함"
        return e
    raise AssertionError("InputProblem이 나와야 함")


def test_input_file_missing_empty_and_directory():
    with tempfile.TemporaryDirectory() as tmp:
        e = _problem(cd.load_sessions, str(Path(tmp) / "none.csv"), NOW)
        assert "찾을 수 없습니다" in e.reason and "경로" in e.fix
        e = _problem(cd.load_sessions, tmp, NOW)
        assert "폴더" in e.reason
        e = _problem(cd.load_sessions, _write(tmp, "empty.csv", ""), NOW)
        assert "비어 있습니다" in e.reason
        e = _problem(cd.load_sessions, _write(tmp, "header.csv", HEADER + "\n"), NOW)
        assert "데이터 행이 없습니다" in e.reason


def test_missing_columns_named():
    with tempfile.TemporaryDirectory() as tmp:
        e = _problem(cd.load_sessions, _write(tmp, "c.csv", "start_time,end_time,distance_km\n1,2,3\n"), NOW)
        assert "필요한 열이 없습니다" in e.reason and "duration_sec" in e.reason and "t5_sec" in e.reason and "헤더" in e.fix


def test_time_number_and_order_format_errors_with_line_numbers():
    with tempfile.TemporaryDirectory() as tmp:
        bad_time = _write(tmp, "t.csv", HEADER + "\n%d,%d,3.0,1500,150,\n어제,%d,3.0,1500,150,\n" % (ASOF - 100, ASOF - 50, ASOF - 10))
        e = _problem(cd.load_sessions, bad_time, NOW)
        assert "3번째 줄" in e.reason and "형식" in e.reason and "ISO" in e.fix
        bad_num = _write(tmp, "n.csv", HEADER + "\n%d,%d,abc,1500,150,\n" % (ASOF - 100, ASOF - 50))
        e = _problem(cd.load_sessions, bad_num, NOW)
        assert "2번째 줄" in e.reason and "숫자" in e.reason
        bad_order = _write(tmp, "o.csv", HEADER + "\n%d,%d,3.0,1500,150,\n" % (ASOF - 50, ASOF - 100))
        e = _problem(cd.load_sessions, bad_order, NOW)
        assert "end_time이 start_time보다 이릅니다" in e.reason


def test_iso_and_epoch_times_both_accepted():
    assert cd.parse_time("2025-03-03T00:00:00") == cd.parse_time("2025-03-03 00:00:00") == 1740960000.0
    assert cd.parse_time("1740960000") == 1740960000.0


def test_future_dated_records_excluded_with_notice_and_future_asof_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        rows = _rows_ok([(NOW + cd.DAY, NOW + cd.DAY + 1800, 5.2, 1400.0)])          # 현재 시각보다 미래
        sess, notes = cd.load_sessions(_csv(tmp, rows), NOW)
        assert len(sess) == 4 and len(notes) == 1 and "미래 날짜" in notes[0] and "1개" in notes[0]
    e = _problem(cd.parse_asof, str(NOW + cd.DAY), NOW)
    assert "미래" in e.reason and "오늘" in e.fix
    e = _problem(cd.parse_asof, "내일", NOW)
    assert "읽을 수 없습니다" in e.reason and "ISO" in e.fix
    assert cd.parse_asof(str(ASOF), NOW) == float(ASOF)


def test_duplicate_records_same_start_end_removed_with_notice():
    with tempfile.TemporaryDirectory() as tmp:
        rows = _rows_ok()
        sess, notes = cd.load_sessions(_csv(tmp, rows + [rows[0], rows[3]]), NOW)
        assert len(sess) == 4 and len(notes) == 1 and "중복 기록 2개" in notes[0] and "먼저 나온" in notes[0]
        sess, notes = cd.load_sessions(_csv(tmp, rows, "nodup.csv"), NOW)
        assert notes == []


def test_shortfall_shows_what_and_how_many_are_missing():
    r = cd.assess([S(1), S(5)], ASOF, P)
    assert not r["eligible"]
    assert "2회입니다(4회 이상 필요, 2회 부족)" in r["shortfalls"][0]
    assert "5km 기록이 없습니다(1회 이상 필요, 1회 부족)" in r["shortfalls"][1]
    text = cd.format_report(r)
    assert cd.check_numbers(text, r) and "해결:" in text
    r = cd.assess([S(1), S(5), S(9), S(13), S(2, 5.2, 1500)], ASOF, P)
    assert r["eligible"]                                                                 # 정확히 5회, 유효 5km 1개


def test_5km_outside_range_or_extreme_latest_falls_back_to_earlier_valid_record_and_says_so():
    # 가장 최근 5km 기록이 5.0~5.5km 범위 밖(4.8km)이면 그 이전 유효 기록(1500초)을 기준 기록으로 쓴다
    r = cd.assess([S(1), S(5), S(9), S(20, 5.2, 1500), S(2, 4.8, 1400)], ASOF, P)
    assert r["eligible"] and r["ref"] == 1500
    assert len(r["notices"]) == 1 and "5.0~5.5km 범위 밖 1개" in r["notices"][0] and "20일 전" in r["notices"][0] and "그 이전의 유효 기록" in r["notices"][0]
    # 극단(T5 > 5,000초)
    r = cd.assess([S(1), S(5), S(9), S(20, 5.2, 1500), S(2, 5.2, 6000)], ASOF, P)
    assert r["eligible"] and r["ref"] == 1500 and "극단(T5 > 5,000초) 1개" in r["notices"][0]
    # 유효 기록이 하나도 없으면 자격 미달이고 이유를 보여 준다
    r = cd.assess([S(1), S(5), S(9), S(13), S(2, 4.8, 1400), S(3, 5.2, 6000)], ASOF, P)
    assert not r["eligible"] and "5km 기록 2개가 있으나" in r["shortfalls"][0] and "범위 밖 1개" in r["shortfalls"][0] and "극단(T5 > 5,000초) 1개" in r["shortfalls"][0]
    assert cd.check_numbers(cd.format_report(r), r)
    # 가장 최근 기록이 유효하면 안내가 없다
    r = cd.assess([S(1), S(5), S(9), S(20, 4.8, 1400), S(2, 5.2, 1500)], ASOF, P)
    assert r["eligible"] and r["notices"] == [] and r["ref"] == 1500


def test_only_observable_inputs_are_used_records_after_asof_do_not_change_prediction():
    base = [S(1), S(5), S(9), S(2, 5.2, 1500)]
    r0 = cd.assess(base, ASOF, P)
    later = [{"start": ASOF + 1000, "end": ASOF + 2000, "km": 5.2, "t5": 1000.0}, {"start": ASOF - 100, "end": ASOF + 500, "km": 5.1, "t5": 900.0}]
    r1 = cd.assess(base + later, ASOF, P)
    assert (r1["pred"], r1["lo"], r1["hi"], r1["m"], r1["ref"]) == (r0["pred"], r0["lo"], r0["hi"], r0["m"], r0["ref"])
    assert len(r1["notices"]) == 1 and "기준 시점 이후에 끝난 기록 2개" in r1["notices"][0]


def _rows_clean():
    """층 안내가 없는 기본 기록: 달리기 6회, 유효 5km 5개(평소 1500초, 직전 1500초), 정지 시간 열 포함."""
    runs = [(ASOF - d * cd.DAY - 1800, ASOF - d * cd.DAY, 3.0, None, 0) for d in (1, 5, 9)]
    fives = [(ASOF - d * cd.DAY - 1800, ASOF - d * cd.DAY, 5.2, 1500.0, 40) for d in (3, 8, 12, 40, 50)]
    return runs + fives


def _csv_stop(tmp, rows, name="st.csv"):
    """rows: (start, end, km, t5 또는 None, stop 또는 None) -> stop_sec 열이 있는 CSV."""
    lines = [HEADER + ",stop_sec"] + ["%s,%s,%s,1500,150,%s,%s" % (s_, e, km, "" if t5 is None else t5, "" if st is None else st) for s_, e, km, t5, st in rows]
    return _write(tmp, name, "\n".join(lines) + "\n")


def test_run_prints_notices_before_prediction_and_no_notice_when_clean():
    with tempfile.TemporaryDirectory() as tmp:
        path = _csv_stop(tmp, _rows_clean())
        _res, clean = cd.run(path, str(ASOF), P, now=NOW)
        assert "※" not in clean and clean.startswith("다음 7일 안 5km 예상 기록")       # 안내 줄이 하나도 없다
        rows = _rows_clean()
        _res, noisy = cd.run(_csv_stop(tmp, rows + [rows[0]], "dup.csv"), str(ASOF), P, now=NOW)
        assert noisy.startswith("※ ") and noisy.endswith(clean)           # 안내만 앞에 붙고 예측 문구는 같다


def test_cli_input_problems_print_reason_and_fix_without_traceback():
    def call(argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cd.main(argv)
        return code, buf.getvalue()
    with tempfile.TemporaryDirectory() as tmp:
        code, out = call([str(Path(tmp) / "none.csv"), "--asof", str(ASOF)])
        assert code == 2 and out.startswith("오류: 입력 파일을 찾을 수 없습니다") and "\n해결: " in out and "Traceback" not in out
        code, out = call([_write(tmp, "e.csv", ""), "--asof", str(ASOF)])
        assert code == 2 and "비어 있습니다" in out
        code, out = call([_csv(tmp, _rows_ok()), "--asof", "내일"])
        assert code == 2 and "기준 시점" in out and "해결:" in out
        code, out = call([_csv(tmp, _rows_ok(), "ok.csv"), "--asof", "2025-03-03T00:00:00"])       # 기록이 1970년이라 자격 미달이지만 멈추지 않는다
        assert code == 0 and "예측할 수 없습니다" in out and "해결:" in out


# ---------------------------------------------------------------- P4: 목표 비교 (--goal mm:ss)
def test_goal_format_and_range_errors():
    assert cd.parse_goal("10:00") == 600 and cd.parse_goal("24:30") == 1470 and cd.parse_goal("83:20") == 5000
    for bad in ["abc", "24", "24:7", "24:60", "24:30:00", "-5:00", ""]:
        e = _problem(cd.parse_goal, bad)
        assert "형식" in e.reason and "mm:ss" in e.reason and "예:" in e.fix, bad
    e = _problem(cd.parse_goal, "9:59")
    assert "10분 미만" in e.reason and "10:00" in e.fix
    e = _problem(cd.parse_goal, "83:21")
    assert "5,000초" in e.reason and "83:20" in e.fix


def _res_with_range():
    # 기준 1500초, 평소 1450초 -> 예상 변화 -55, 예측 1445, 범위 [1245, 1725]
    return cd.assess([S(1), S(5), S(9), S(20, 5.2, 1400), S(2, 5.1, 1500)], ASOF, P)


def test_goal_three_outcomes_and_boundaries():
    r = _res_with_range()
    lo, hi, pred = int(round(r["lo"])), int(round(r["hi"])), int(round(r["pred"]))
    assert (lo, hi, pred) == (1245, 1725, 1445)
    cases = [(lo - 1, "명목 범위보다 빠름"), (lo, "명목 범위 안"), (pred, "명목 범위 안"), (hi, "명목 범위 안"), (hi + 1, "명목 범위보다 느림")]
    for goal, label in cases:
        text, info = cd.format_goal(r, goal, P)
        assert ("(목표 %s): %s" % (cd.mmss(goal), label)) in text, (goal, label)
        assert cd.check_goal_numbers(text, info)


def test_goal_difference_sign_and_words():
    r = _res_with_range()                                              # 예측 1445초
    text, _ = cd.format_goal(r, 1400, P)                              # 목표가 45초 빠름
    assert "예측값과 목표의 차이(예측 − 목표): +45초 (목표가 예측보다 45초 빠름)" in text
    text, _ = cd.format_goal(r, 1500, P)                              # 목표가 55초 느림
    assert "예측값과 목표의 차이(예측 − 목표): −55초 (목표가 예측보다 55초 느림)" in text
    text, _ = cd.format_goal(r, 1445, P)
    assert "차이(예측 − 목표): 0초 (목표와 예측이 같음)" in text


def test_goal_required_sentences_use_numbers_from_params():
    r = _res_with_range()
    text, info = cd.format_goal(r, 1445, P)
    assert "범위 안·밖은 달성 확률이나 달성 가능 여부를 뜻하지 않으며, 훈련 처방을 제공하지 않습니다." in text
    # P: q10 = -200, q90 = +280 -> 폭 480초 = 약 8분, 200초 = 3분 20초
    assert "현재 범위 폭은 약 8분(−200초 ~ +280초)으로 넓어, 예측보다 약 3분 20초 빠른 목표까지 범위 안으로 표시됩니다." in text
    p2 = dict(P, resid_q10=-202.1, resid_q90=277.9)                  # 현재 b2_params.json과 같은 크기의 값
    text2, _ = cd.format_goal(cd.assess([S(1), S(5), S(9), S(20, 5.2, 1400), S(2, 5.1, 1500)], ASOF, p2), 1445, p2)
    assert "현재 범위 폭은 약 8분(−202초 ~ +278초)으로 넓어, 예측보다 약 3분 22초 빠른 목표까지 범위 안으로 표시됩니다." in text2
    real = cd.load_params()                                            # 실제 b2_params.json의 값으로도 같은 문장이 나온다
    q10, q90 = real["resid_q10"], real["resid_q90"]
    s2 = cd.format_goal(cd.assess([S(1), S(5), S(9), S(20, 5.2, 1400), S(2, 5.1, 1500)], ASOF, real), 1445, real)[0]
    assert "(%s초 ~ %s초)" % (cd.signed(q10), cd.signed(q90)) in s2


def test_goal_numbers_check_catches_altered_numbers():
    r = _res_with_range()
    text, info = cd.format_goal(r, 1400, P)
    for old, new in [("+45초", "+46초"), ("약 8분", "약 9분"), ("약 3분 20초", "약 3분 21초"), ("목표 23:20", "목표 23:21")]:
        assert old in text, old
        try:
            cd.check_goal_numbers(text.replace(old, new), info)
        except ValueError:
            continue
        raise AssertionError("검사를 통과함: " + old)


def test_run_without_goal_is_unchanged_and_with_goal_only_appends():
    with tempfile.TemporaryDirectory() as tmp:
        path = _csv(tmp, _rows_ok())
        _r0, plain = cd.run(path, str(ASOF), P, now=NOW)
        _r1, none_goal = cd.run(path, str(ASOF), P, goal=None, now=NOW)
        assert plain == none_goal and "목표 비교" not in plain
        _r2, with_goal = cd.run(path, str(ASOF), P, goal="24:00", now=NOW)
        assert with_goal.startswith(plain + "\n") and "목표 비교(목표 24:00)" in with_goal


def test_goal_with_ineligible_input_and_goal_errors_do_not_crash():
    with tempfile.TemporaryDirectory() as tmp:
        short = _csv(tmp, [(ASOF - cd.DAY - 1800, ASOF - cd.DAY, 3.0, None)])
        _r, text = cd.run(short, str(ASOF), P, goal="24:00", now=NOW)
        assert "예측할 수 없습니다" in text and "목표 비교는 예측이 가능할 때만 표시합니다" in text and "명목 범위" not in text
        e = _problem(cd.run, _csv(tmp, _rows_ok(), "g.csv"), str(ASOF), P, goal="9:00", now=NOW)
        assert "10분 미만" in e.reason
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cd.main([_csv(tmp, _rows_ok(), "g2.csv"), "--asof", "2025-03-03T00:00:00", "--goal", "abc"])
        assert code == 2 and buf.getvalue().startswith("오류: 목표 형식이 올바르지 않습니다") and "\n해결: " in buf.getvalue()


def test_goal_numbers_come_from_prediction_not_future_values():
    # 목표 비교는 예측 범위(관측 가능한 입력으로 계산)와 목표만 쓴다: 기준 시점 이후 기록을 더해도 결과가 같다
    base = [S(1), S(5), S(9), S(2, 5.2, 1500)]
    later = [{"start": ASOF + 1000, "end": ASOF + 2000, "km": 5.2, "t5": 900.0}]
    t0, _ = cd.format_goal(cd.assess(base, ASOF, P), 1400, P)
    t1, _ = cd.format_goal(cd.assess(base + later, ASOF, P), 1400, P)
    assert t0 == t1


# ---------------------------------------------------------------- 방향 문장: 두 항과 최종 예측의 방향 (다섯 경우)
def _trend_case(past_t5, ref_t5=1500.0):
    r = cd.assess(_sess_x(past_t5, ref_t5, ref_stop=0), ASOF, P)
    text = cd.format_report(r)
    assert cd.check_numbers(text, r)
    line = [l for l in text.split(chr(10)) if "통계적 현상이며, 원인 효과가 아님" in l]
    assert len(line) == 1 and "때문에" not in line[0]
    return r, line[0]


def test_trend_sentence_five_cases_match_final_direction():
    # P: a = -15.0, b = -0.8
    r, l = _trend_case([1500.0])                                   # (가) x = 0
    assert r["x"] == 0 and r["pred"] < r["ref"] and "보정 항은 없고" in l and "기본 변화량(−15.0초)만 반영" in l and "직전 기록보다 15초 빠릅니다" in l
    r, l = _trend_case([1450.0, 1450.0])                           # (나) 직전이 평소보다 50초 느림
    assert r["pred"] < r["ref"] and "50초 느렸던 만큼 보정 항(−40.0초)이 평소 쪽(빠른 쪽)" in l and "기본 변화량(−15.0초)도 같은 방향" in l and "직전 기록보다 55초 빠릅니다" in l
    r, l = _trend_case([1510.0, 1510.0])                           # (다) x = -10: 보정 +8.0 < |a| 15.0
    assert r["pred"] < r["ref"] and abs(r["b_term"]) < abs(r["a_term"]) and "평소보다 10초 빨랐던 만큼 평소 쪽으로 +8.0초 움직였지만" in l and "기본 변화량(−15.0초)이 더 커서" in l and "직전 기록보다 7초 빠릅니다" in l
    r, l = _trend_case([1530.0, 1530.0])                           # (라) x = -30: 보정 +24.0 > |a|
    assert r["pred"] > r["ref"] and abs(r["b_term"]) > abs(r["a_term"]) and "평소 쪽으로 +24.0초 움직였고 기본 변화량(−15.0초)보다 커서" in l and "직전 기록보다 9초 느립니다(평소 쪽)" in l
    r, l = _trend_case([1518.75, 1518.75])                         # (마) 두 항이 상쇄 -> |예상 변화| < 0.5
    assert abs(r["y_hat"]) < 0.5 and "거의 상쇄되어 최종 예측은 직전 기록과 거의 같습니다" in l and "빠릅니다" not in l and "느립니다" not in l


def test_trend_sentence_direction_matches_prediction_for_many_x():
    for m in [1300.0, 1450.0, 1499.0, 1500.0, 1501.0, 1510.0, 1518.0, 1518.75, 1519.0, 1530.0, 1600.0]:
        r, l = _trend_case([m] if m == 1500.0 else [m, m])
        d = r["pred"] - r["ref"]
        word = "거의 같습니다" if (abs(r["y_hat"]) < 0.5 or r["tot_int"] == 0) else ("빠릅니다" if d < 0 else "느립니다")
        assert word in l, (m, d, l)
        assert ("빠릅니다" in l) == (word == "빠릅니다") and ("느립니다" in l) == (word == "느립니다")



# ---------------------------------------------------------------- 조건별 범위 안내 (docs/demo_stratum_note_spec.md)
def _sess_x(past_t5, ref_t5=1500.0, ref_stop=None, n_other_runs=3):
    """직전 기록(ref_t5, 2일 전)과 과거 5km(past_t5 목록, 40일 이상 전)로 만든 세션. 과거 개수 = len(past_t5) + 1."""
    sess = [S(d) for d in (1, 5, 9)][:n_other_runs]
    ref = S(2, 5.2, ref_t5)
    ref["stop"] = ref_stop
    return sess + [ref] + [S(40 + 10 * i, 5.2, t) for i, t in enumerate(past_t5)]


def _notes():
    return cd.load_notes()


def test_notes_json_matches_doc_train_rows_and_selection_rule():
    import make_stratum_notes as mk
    rows = mk.read_train_rows()
    assert len(rows) == 16
    sel = mk.select(rows)
    notes = _notes()
    assert [(n["dim"], n["label"]) for n in notes] == mk.SPEC_STRATA == [(r["dim"], r["label"]) for r in sel]
    for n, r in zip(notes, sel):
        assert (n["cov"], n["ci_lo"], n["ci_hi"], n["n"], n["users"]) == (r["cov"], r["ci_lo"], r["ci_hi"], r["n"], r["users"])
        assert r["n"] >= 30 and r["ci_hi"] < 80.0
    assert [n["cov"] for n in notes] == [70.6, 73.1, 67.8, 39.1]
    assert all(not (r["n"] >= 30 and r["ci_hi"] < 80.0) for r in rows if r not in sel)      # 선정되지 않은 층은 규칙을 만족하지 않는다
    doc = mk.DOC.read_text(encoding="utf-8")
    for n in notes:                                         # 문서에 같은 포함률 [CI]가 train 폴드 밖 행으로 실제 있음
        assert ("| train 폴드 밖 | %s |" % n["label"]) in doc and ("%.1f%% [%.1f%%, %.1f%%]" % (n["cov"], n["ci_lo"], n["ci_hi"])) in doc


def _lines(sess):
    r = cd.assess(sess, ASOF, P)
    return r, cd.stratum_lines(r, _notes())


def test_stratum_a_boundaries():
    for n_other, expect in [(0, "과거 유효 5km가 직전 기록 1개뿐"), (1, "과거 유효 5km가 2~4개뿐"), (3, "과거 유효 5km가 2~4개뿐"), (4, None)]:
        r, (lines, matched) = _lines(_sess_x([1500.0] * n_other, ref_stop=0))
        assert r["n_past"] == n_other + 1
        tag = [l for l in lines if "과거 유효 5km" in l]
        if expect:
            assert len(tag) == 1 and ("(%s)" % expect) in tag[0], (n_other, lines)
        else:
            assert tag == [] and lines == [], (n_other, lines)                    # 5개 이상이면 안내 없음


def test_stratum_b_boundary_120_seconds():
    r, (lines, _m) = _lines(_sess_x([1380.0, 1380.0], ref_stop=0))               # 중앙값 1380 -> x = 120.0 (120 초과 아님)
    assert r["x"] == 120.0 and not any("2분 넘게" in l for l in lines)
    r, (lines, _m) = _lines(_sess_x([1379.0, 1379.0], ref_stop=0))               # x = 121
    assert r["x"] == 121.0 and any("직전 기록이 평소와 2분 넘게 차이" in l and "67.8%" in l for l in lines)
    r, (lines, _m) = _lines(_sess_x([1621.0, 1621.0], ref_stop=0))               # 음수 방향도 절대값 121
    assert r["x"] == -121.0 and any("2분 넘게" in l for l in lines)


def test_stratum_e_boundary_and_missing_stop_info():
    ok = [1500.0] * 4
    r, (lines, _m) = _lines(_sess_x(ok, ref_stop=300))
    assert lines == []                                                             # 정확히 300초는 초과 아님
    r, (lines, _m) = _lines(_sess_x(ok, ref_stop=300.5))
    assert len(lines) == 2 and "직전 기록 세션에 5분 넘는 정지가 있음" in lines[0] and "39.1%" in lines[0] and lines[1] == cd.WIDTH_LINE
    r, (lines, _m) = _lines(_sess_x(ok, ref_stop=None))                           # 정보 없음: 안내 줄 없이 확인 불가 문구만
    assert lines == [cd.NO_STOP_LINE] and "정지 시간 정보가 없어 이 조건은 확인하지 않았습니다" in lines[0]
    assert not any("5분 넘는" in l for l in lines)


def test_multiple_strata_show_all_lines_in_order_and_width_line_once():
    r, (lines, matched) = _lines(_sess_x([1000.0, 1000.0], ref_stop=400))        # 과거 3개, x = 500, 정지 400
    assert [m["dim"] for m in matched] == ["a", "b", "e"]
    assert len(lines) == 4
    assert "2~4개" in lines[0] and "73.1%" in lines[0] and "2분 넘게" in lines[1] and "67.8%" in lines[1] and "5분 넘는" in lines[2] and "39.1%" in lines[2]
    assert lines[3] == cd.WIDTH_LINE and sum(1 for l in lines if l == cd.WIDTH_LINE) == 1
    assert cd.check_stratum_lines(lines, matched, r)
    try:
        cd.check_stratum_lines([l.replace("73.1%", "73.2%") for l in lines], matched, r)
    except ValueError:
        pass
    else:
        raise AssertionError("문구 불일치를 잡아야 함")


def test_stratum_notes_do_not_change_prediction_or_range():
    r0 = cd.assess(_sess_x([1500.0] * 4, ref_stop=0), ASOF, P)                    # 안내 없음
    r1 = cd.assess(_sess_x([1500.0] * 4, ref_stop=900), ASOF, P)                  # 정지 안내만 다름
    assert cd.stratum_lines(r0, _notes())[0] == [] and cd.stratum_lines(r1, _notes())[0] != []
    assert (r0["pred"], r0["lo"], r0["hi"], r0["y_hat"]) == (r1["pred"], r1["lo"], r1["hi"], r1["y_hat"])
    assert r1["hi"] - r1["lo"] == P["resid_q90"] - P["resid_q10"]
    assert abs(r1["lo"] - (r1["pred"] + P["resid_q10"])) < 1e-9 and abs(r1["hi"] - (r1["pred"] + P["resid_q90"])) < 1e-9
    real = cd.load_params()
    assert real["a"] == -15.287228077141389 and real["resid_q10"] == -202.11038214375517 and real["resid_q90"] == 277.9097030360434


def test_stop_sec_column_is_optional_and_validated():
    with tempfile.TemporaryDirectory() as tmp:
        _r, text = cd.run(_csv(tmp, _rows_ok()), str(ASOF), P, now=NOW)         # 열 없음: 기존 CSV 그대로 동작
        assert cd.NO_STOP_LINE in text and text.index(cd.NO_STOP_LINE) > text.index("다음 7일 안 5km 예상 기록")
        for bad in ["abc", "-5"]:
            rows = _rows_clean()
            p = _csv_stop(tmp, rows[:-1] + [rows[-1][:4] + (bad,)], "bad.csv")
            e = _problem(cd.run, p, str(ASOF), P, now=NOW)
            assert "stop_sec" in e.reason and "0 이상" in e.reason
        _r, with_blank = cd.run(_csv_stop(tmp, [r[:4] + (None,) for r in _rows_clean()], "blank.csv"), str(ASOF), P, now=NOW)
        assert with_blank.endswith(cd.NO_STOP_LINE)                              # 열은 있으나 기준 기록의 값이 비어 있음


def test_notes_come_after_prediction_before_goal_and_goal_block_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        path = _csv_stop(tmp, [r[:4] + (900,) if r[3] else r for r in _rows_clean()])
        _r, plain = cd.run(path, str(ASOF), P, now=NOW)
        _r, goal = cd.run(path, str(ASOF), P, goal="24:00", now=NOW)
        assert plain.index("이 예측은 원인 분석이 아니며") < plain.index("※ 이 조건") and plain.endswith(cd.WIDTH_LINE)
        assert goal.startswith(plain + "\n") and "목표 비교(목표 24:00)" in goal and goal.index("※ 이 조건") < goal.index("목표 비교(")


def test_example_outputs_match_live_runs_and_cover_each_stratum():
    import make_examples as mx
    ex = Path(mx.OUT)
    seen = {}
    for csv_name, goal, out_name in mx.OUTPUTS:
        argv = [str(ex / csv_name), "--asof", mx.ASOF_TEXT] + (["--goal", goal] if goal else [])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cd.main(argv) == 0
        assert (ex / out_name).read_text(encoding="utf-8") == buf.getvalue(), out_name
        seen[out_name] = buf.getvalue()
    keys = {"ex5_past1.out.txt": "직전 기록 1개뿐", "ex6_past3.out.txt": "2~4개뿐", "ex7_far_from_usual.out.txt": "2분 넘게", "ex8_long_stop.out.txt": "5분 넘는 정지"}
    for name, frag in keys.items():
        assert frag in seen[name] and cd.WIDTH_LINE in seen[name], name
    assert "※" not in seen["ex9_no_note.out.txt"]                                # 안내 줄이 하나도 없다
    assert "※ 이 조건" in seen["ex1_steady.out.txt"] and cd.NO_STOP_LINE in seen["ex1_steady.out.txt"]
    assert "평소 쪽으로 +5.5초 움직였지만" in seen["ex11_trend_smaller_a.out.txt"] and "직전 기록보다 10초 빠릅니다" in seen["ex11_trend_smaller_a.out.txt"]
    assert "직전 기록보다" in seen["ex12_trend_larger_a.out.txt"] and "느립니다(평소 쪽)" in seen["ex12_trend_larger_a.out.txt"]
    assert "※ 이 조건" in seen["ex10_multi.out.txt"] and seen["ex10_multi.out.txt"].count("※ 이 조건") == 3


def test_avg_hr_column_is_optional_and_does_not_change_prediction():
    with tempfile.TemporaryDirectory() as tmp:
        rows = _rows_clean()
        with_hr = _csv_stop(tmp, rows, "with_hr.csv")
        text = Path(with_hr).read_text(encoding="utf-8").split(chr(10))
        no_hr = [",".join(c for i, c in enumerate(l.split(",")) if i != 4) for l in text]            # avg_hr(5번째 열) 제거
        p_no = _write(tmp, "no_hr.csv", chr(10).join(no_hr))
        blank = [text[0]] + [",".join("" if i == 4 else c for i, c in enumerate(l.split(","))) for l in text[1:] if l]
        p_blank = _write(tmp, "blank_hr.csv", chr(10).join(blank) + chr(10))
        t0 = cd.run(with_hr, str(ASOF), P, now=NOW)[1]
        assert cd.run(p_no, str(ASOF), P, now=NOW)[1] == t0 and cd.run(p_blank, str(ASOF), P, now=NOW)[1] == t0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print("ok", n)
    print(len(tests), "tests passed")
