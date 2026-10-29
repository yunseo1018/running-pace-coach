# -*- coding: utf-8 -*-
"""coach_demo 단위 테스트. pytest 없이도 `python demo/test_coach_demo.py`로 실행된다(test_ 함수를 assert로 직접 호출)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coach_demo as cd  # noqa: E402

P = {"a": -15.0, "b": -0.8, "resid_q10": -200.0, "resid_q90": 280.0}
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


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print("ok", n)
    print(len(tests), "tests passed")
