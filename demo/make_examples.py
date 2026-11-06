# -*- coding: utf-8 -*-
"""합성 예시 CSV 3개 생성(demo/examples/). 실제 데이터와 무관한 직접 만든 값이며 seed 고정. python demo/make_examples.py"""
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "examples"
ASOF = datetime(2025, 3, 3, 0, 0, tzinfo=timezone.utc)   # 가상의 월요일 기준 시점
COLS = ["start_time", "end_time", "distance_km", "duration_sec", "avg_hr", "t5_sec"]


def build(rng, five_km_weeks, usual, last_offset):
    """기준 시점 전 8주, 주 3~4회 일반 달리기(3.0~4.8km) + 지정한 주마다 5km 1회.
    five_km_weeks 중 가장 늦은 주의 5km만 평소 기록에 last_offset초를 더한다."""
    rows = []
    for w in range(8):
        week0 = ASOF - timedelta(days=7 * (8 - w))
        for d in rng.sample(range(7), 3 + (w % 2)):
            rows.append([week0 + timedelta(days=d, hours=7), round(rng.uniform(3.0, 4.8), 2), None])
        if w in five_km_weeks:
            off = last_offset if w == max(five_km_weeks) else rng.uniform(-25, 25)
            t5 = round(usual + off + rng.uniform(-5, 5), 1)
            rows.append([week0 + timedelta(days=rng.randint(0, 6), hours=18), round(rng.uniform(5.05, 5.4), 2), t5])
    out = []
    for start, km, t5 in sorted(rows, key=lambda r: r[0]):
        pace = (t5 / 5.0) if t5 else rng.uniform(290, 340)
        dur = int(pace * km)
        end = start + timedelta(seconds=dur)
        out.append([start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"), km, dur, rng.randint(140, 165),
                    "" if t5 is None else t5])
    return out


def write(name, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)


def main():
    rng = random.Random(42)
    # 1) 꾸준히 달린 사람: 평소 약 25분, 마지막 5km도 평소 수준
    write("ex1_steady.csv", build(rng, [1, 3, 5, 7], 1500.0, 0.0))
    # 2) 직전 5km가 평소보다 느렸던 사람: 마지막 5km만 평소보다 약 150초 느림
    write("ex2_slower_last.csv", build(rng, [0, 2, 4, 7], 1500.0, 150.0))
    # 3) 자격 미달: 달리기는 있으나 5km 기록 없음
    write("ex3_no_5km.csv", build(rng, [], 1500.0, 0.0))
    # 4) 자격 미달(달리기 2회 + 5km 기록이 5.0~5.5km 범위 밖): 부족한 개수와 제외 이유 표시용. 난수를 쓰지 않아 위 예시에 영향 없음
    write("ex4_few_runs.csv", [["2025-02-20T07:00:00", "2025-02-20T07:20:00", 3.5, 1200, 148, ""],
                               ["2025-02-26T07:00:00", "2025-02-26T07:31:00", 5.8, 1860, 155, 1560.0]])
    print("saved", sorted(p.name for p in OUT.glob("*.csv")))
    write_outputs()


# (CSV, 목표, 출력 파일): 기준 시점은 모두 2025-03-03. 목표를 넣지 않은 출력은 기존 예시 출력과 같다.
ASOF_TEXT = "2025-03-03T00:00:00"
OUTPUTS = [("ex1_steady.csv", None, "ex1_steady.out.txt"), ("ex2_slower_last.csv", None, "ex2_slower_last.out.txt"), ("ex3_no_5km.csv", None, "ex3_no_5km.out.txt"),
           ("ex1_steady.csv", "20:00", "ex1_goal_faster.out.txt"), ("ex1_steady.csv", "24:00", "ex1_goal_inside.out.txt"), ("ex1_steady.csv", "30:00", "ex1_goal_slower.out.txt"),
           ("ex4_few_runs.csv", None, "ex4_few_runs.out.txt")]


def write_outputs():
    """예시 CSV로 coach_demo를 실행해 출력 파일(.out.txt)을 만든다(명령줄 실행과 같은 출력)."""
    import contextlib
    import io
    import sys
    sys.path.insert(0, str(OUT.parent))
    import coach_demo
    for csv_name, goal, out_name in OUTPUTS:
        argv = [str(OUT / csv_name), "--asof", ASOF_TEXT] + (["--goal", goal] if goal else [])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = coach_demo.main(argv)
        assert code == 0, (csv_name, goal, code)
        (OUT / out_name).write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    print("saved outputs", [o for _c, _g, o in OUTPUTS])


if __name__ == "__main__":
    main()
