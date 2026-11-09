# -*- coding: utf-8 -*-
"""docs/demo_cases.md 생성: demo/examples의 합성 예시 ①~⑰을 실제로 실행해 프로그램 반응을 표로 정리한다(+ 입력 오류 사례 5개).
사용(저장소 루트): python demo/make_demo_cases.py          (문서 쓰기)
                   python demo/make_demo_cases.py --check  (쓰지 않고 현재 문서와 같은지 확인, 다르면 종료 코드 1)
표의 반응 칸은 실행 출력(coach_demo.main과 같은 경로)에서 문구를 그대로 가져온다. 표준 라이브러리만 사용."""
import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOC = ROOT / "docs" / "demo_cases.md"
sys.path.insert(0, str(HERE))
import coach_demo as cd  # noqa: E402
import make_examples as mx  # noqa: E402

T_EX = "test_example_outputs_match_live_runs_and_cover_each_stratum"
T_FB = "test_feedback_example_outputs_match_live_runs"
# (번호, 예시 CSV, 분류, 옵션, 입력 조건 한 줄, 해당 테스트)
CASES = [
    ("①", "ex1_steady.csv", "정상 예측", [], "8주, 5km 4회(평소 약 25분), 마지막 5km도 평소 수준, stop_sec 열 없음", [T_EX, "test_two_term_split_and_coverage_text"]),
    ("②", "ex2_slower_last.csv", "정상 예측", [], "마지막 5km만 평소보다 약 150초 느림", [T_EX, "test_two_term_split_and_coverage_text"]),
    ("③", "ex3_no_5km.csv", "예측 불가(자격 미달)", [], "달리기는 충분하나 5km 기록 없음", ["test_shortfall_shows_what_and_how_many_are_missing"]),
    ("④", "ex4_few_runs.csv", "예측 불가(자격 미달)", [], "달리기 2회, 5km 기록이 5.0~5.5km 범위 밖(5.8km)", ["test_shortfall_shows_what_and_how_many_are_missing", "test_5km_outside_range_or_extreme_latest_falls_back_to_earlier_valid_record_and_says_so"]),
    ("⑤", "ex1_steady.csv", "목표 비교", ["--goal", "20:00"], "①과 같은 기록, 목표 20:00", ["test_goal_three_outcomes_and_boundaries"]),
    ("⑥", "ex1_steady.csv", "목표 비교", ["--goal", "24:00"], "①과 같은 기록, 목표 24:00", ["test_goal_three_outcomes_and_boundaries"]),
    ("⑦", "ex1_steady.csv", "목표 비교", ["--goal", "30:00"], "①과 같은 기록, 목표 30:00", ["test_goal_three_outcomes_and_boundaries"]),
    ("⑧", "ex5_past1.csv", "조건별 안내", [], "과거 유효 5km가 직전 기록 1개뿐(stop_sec 열 있음, 정지 40초)", ["test_stratum_a_boundaries", T_EX]),
    ("⑨", "ex6_past3.csv", "조건별 안내", [], "과거 유효 5km 3개", ["test_stratum_a_boundaries", T_EX]),
    ("⑩", "ex7_far_from_usual.csv", "조건별 안내", [], "직전 5km가 평소(1500초)보다 180초 느림, 과거 5km 5개", ["test_stratum_b_boundary_120_seconds", T_EX]),
    ("⑪", "ex8_long_stop.csv", "조건별 안내", [], "직전 기록 세션의 정지 420초, 과거 5km 5개", ["test_stratum_e_boundary_and_missing_stop_info", T_EX]),
    ("⑫", "ex10_multi.csv", "조건별 안내", [], "과거 5km 3개 + 직전이 평소보다 300초 느림 + 정지 400초", ["test_multiple_strata_show_all_lines_in_order_and_width_line_once", T_EX]),
    ("⑬", "ex9_no_note.csv", "정상 예측", [], "과거 5km 5개, 평소와 차이 0초, 정지 40초(안내 조건 해당 없음)", ["test_run_prints_notices_before_prediction_and_no_notice_when_clean", T_EX]),
    ("⑭", "ex11_trend_smaller_a.csv", "정상 예측", [], "직전 5km가 평소보다 약 7초 빠름(방향 문장 (다))", ["test_trend_sentence_five_cases_match_final_direction", T_EX]),
    ("⑮", "ex12_trend_larger_a.csv", "정상 예측", [], "직전 5km가 평소보다 약 40초 빠름(방향 문장 (라))", ["test_trend_sentence_five_cases_match_final_direction", T_EX]),
    ("⑯", "ex13_feedback_inside.csv", "뛴 다음의 피드백", ["--feedback"], "⑬과 같은 과거 기록 + 기준 시점 2일 뒤 5km 24:30", ["test_feedback_inside_range_slower_than_prediction", T_FB]),
    ("⑰", "ex14_feedback_faster.csv", "뛴 다음의 피드백", ["--feedback"], "⑬과 같은 과거 기록 + t5 없는 5km 1개 + 기준 시점 3일 뒤 5km 20:30", ["test_feedback_faster_and_slower_than_range", "test_feedback_skips_runs_without_t5_and_reports_count", T_FB]),
]
FORBIDDEN = ["훈련", "휴식", "쉬", "권장", "추천", "하세요", "해야", "늘리", "줄이", "늘었", "줄었", "향상", "좋아졌", "나빠졌", "나아졌", "퇴보", "저하",
             "확률", "달성", "때문", "원인", "덕분", "탓", "결과로"]


def run_cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cd.main(argv)
    return code, buf.getvalue().rstrip("\n")


def esc(s):
    return s.replace("|", "\\|")


def reaction(text):
    """실행 출력(기본 출력 = 요약 화면)에서 반응 요약 문구를 그대로 뽑는다(여러 줄은 <br>로 잇는다)."""
    if text.startswith("오류:"):
        return esc("오류 출력, 종료 코드 2: " + " / ".join(text.split("\n")))
    sec, buf = None, {}
    for l in text.split("\n"):
        m = re.match(r"\[(결과|상태|안내|근거|주의|목표 비교|뛴 다음의 피드백)\] ?(.*)", l)
        if m:
            sec = m.group(1)
            buf.setdefault(sec, []).append(m.group(2))
        elif l.startswith(" " * 7) and sec:
            buf[sec].append(l.strip())
        elif not l.strip():
            sec = None
    out = []
    first = buf["결과"][0]
    out.append(("**예측 불가**: " if first.startswith("예측할 수 없습니다") else "예측 가능: ") + first)
    out.append("상태 표시: " + buf["상태"][0])
    if first.startswith("예측할 수 없습니다"):
        out.append("부족한 조건: " + " ".join(x[len("부족한 조건: "):] for x in buf["안내"] if x.startswith("부족한 조건: ")))
    else:
        out += ["안내: " + x for x in buf["안내"]]
    if "목표 비교" in buf:
        out.append("목표 비교: " + " / ".join(buf["목표 비교"][:2]))
    if "뛴 다음의 피드백" in buf:
        out += ["피드백: " + x for x in buf["뛴 다음의 피드백"][1:3]]
    return esc("<br>".join(out))


def input_error_cases():
    """입력 오류 사례(합성 입력, 임시 파일): (이름, 입력 조건, 인자 목록, 해당 테스트)."""
    tmp = Path(tempfile.mkdtemp())
    good = "start_time,end_time,distance_km,duration_sec,avg_hr,t5_sec\n"
    p_cols = tmp / "no_cols.csv"
    p_cols.write_text("start_time,end_time,distance_km\n2025-02-01T07:00:00,2025-02-01T07:30:00,5.2\n", encoding="utf-8")
    p_time = tmp / "bad_time.csv"
    p_time.write_text(good + "3/3/2025 7:00,2025-02-01T07:30:00,5.2,1500,150,1500\n", encoding="utf-8")
    p_ok = tmp / "ok.csv"
    p_ok.write_text(good + "2025-02-01T07:00:00,2025-02-01T07:30:00,5.2,1500,150,1500\n", encoding="utf-8")
    asof = ["--asof", mx.ASOF_TEXT]
    return [
        ("E1", "CSV 파일이 없음", ["none.csv"] + asof, "test_input_file_missing_empty_and_directory"),
        ("E2", "필수 열(duration_sec, t5_sec)이 없음", [str(p_cols)] + asof, "test_missing_columns_named"),
        ("E3", "start_time이 엑셀식 날짜 형식(3/3/2025 7:00)", [str(p_time)] + asof, "test_time_number_and_order_format_errors_with_line_numbers"),
        ("E4", "--goal이 mm:ss 형식이 아님(24)", [str(p_ok)] + asof + ["--goal", "24"], "test_goal_format_and_range_errors"),
        ("E5", "--asof가 현재보다 미래(2999-01-01)", [str(p_ok), "--asof", "2999-01-01T00:00:00"], "test_future_dated_records_excluded_with_notice_and_future_asof_rejected"),
    ]


def build():
    L = []
    A = L.append
    A("# 합성 예시 사례표 (시연 `demo/coach_demo.py`)")
    A("")
    A("생성: `python demo/make_demo_cases.py`(저장소 루트). `demo/examples/`의 직접 만든 합성 예시 ①~⑰을 기준 시점 2025-03-03으로 **실제로 실행**해 프로그램 반응 칸의 문구를 그대로 옮겼다(기본 출력인 요약 화면 기준; `--detail`은 뒤에 기존 전체 출력을 그대로 붙인다, 사양 [`demo_screen_spec.md`](demo_screen_spec.md)). "
      "다시 생성한 결과가 이 파일과 같은지는 테스트(`test_demo_cases_doc_matches_regeneration`)와 `python demo/make_demo_cases.py --check`로 확인한다. 실제 데이터와 무관하다. "
      "B2 계산식·파라미터·명목 80% 범위는 이 정리에서 바꾸지 않았다. 모든 예측은 \"다음 7일 안에 5km 기록이 관측된 경우\"에 조건부다.")
    A("")
    A("분류: 정상 예측 / 조건별 안내 / 예측 불가(자격 미달) / 입력 오류 / 목표 비교 / 뛴 다음의 피드백. 한 예시가 여러 분류에 걸쳐도 대표 분류 하나를 적었다(실제 반응 칸에 안내가 함께 나온다).")
    A("")
    A("## 1. 예시 ①~⑰")
    A("")
    A("| 번호 | 예시 파일 | 분류 | 입력 조건 | 프로그램 반응(실제 출력의 문구) | 해당 테스트 |")
    A("|---|---|---|---|---|---|")
    for no, f, cat, opts, cond, tests in CASES:
        code, text = run_cli([str(mx.OUT / f), "--asof", mx.ASOF_TEXT] + opts)
        assert code == 0, (f, code)
        cmd = "`%s`" % f + ("<br>`%s`" % " ".join(opts) if opts else "")
        A("| %s | %s | %s | %s | %s | %s |" % (no, cmd, cat, cond, reaction(text), "<br>".join("`%s`" % t for t in tests)))
    A("")
    A("## 2. 입력 오류 (예시 파일 없음: 합성 입력을 임시로 만들어 실행)")
    A("")
    A("입력·옵션 문제는 멈추지 않고 \"오류: 이유 / 해결: 방법\" 두 줄과 종료 코드 2로 알린다(추적 메시지 없음).")
    A("")
    A("| 번호 | 입력 조건 | 프로그램 반응(실제 출력 전문) | 해당 테스트 |")
    A("|---|---|---|---|")
    for no, cond, argv, test in input_error_cases():
        code, text = run_cli(argv)
        assert code == 2 and text.startswith("오류:"), (no, code)
        A("| %s | %s | %s | `%s` |" % (no, cond, esc("<br>".join(text.split("\n"))), test))
    A("")
    A("## 3. 이 프로그램이 하지 않는 것")
    A("")
    A("- **처방**: 훈련·휴식을 권하거나 계획을 제시하지 않는다.")
    A("- **확률**: 목표 달성 확률이나 달성 가능 여부를 말하지 않는다(목표 비교와 피드백의 범위 안·밖은 확률이 아니다).")
    A("- **원인 설명**: 예측이나 범위 안·밖의 원인을 설명하지 않는다(상관이며 인과가 아니다).")
    A("- **실력 변화 판정**: 실력이 늘었다·줄었다고 판정하지 않는다. 범위 안·밖은 실력 변화의 증거가 아니며, 한 번의 기록에는 일상적인 흔들림이 포함된다.")
    A("")
    A("뛴 다음의 피드백 블록(`--feedback`)에는 아래 금지 표현이 없음을 테스트(`test_feedback_has_no_forbidden_expressions`)가 확인한다(사양 [`demo_feedback_spec.md`](demo_feedback_spec.md) 7절).")
    A("")
    A("- 훈련·휴식 권고: " + ", ".join("\"%s\"" % w for w in FORBIDDEN[:9]))
    A("- 실력 변화 판정: " + ", ".join("\"%s\"" % w for w in FORBIDDEN[9:17]))
    A("- 목표 달성 확률: " + ", ".join("\"%s\"" % w for w in FORBIDDEN[17:19]))
    A("- 원인 설명: " + ", ".join("\"%s\"" % w for w in FORBIDDEN[19:]))
    A("")
    return "\n".join(L)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    text = build()
    if "--check" in argv:
        same = DOC.exists() and DOC.read_text(encoding="utf-8") == text
        print("일치" if same else "다름")
        return 0 if same else 1
    DOC.write_text(text, encoding="utf-8", newline="\n")
    print("saved", DOC.name)
    return 0


if __name__ == "__main__":
    for _st in (sys.stdout, sys.stderr):
        if hasattr(_st, "reconfigure"):
            _st.reconfigure(encoding="utf-8")
    sys.exit(main())
