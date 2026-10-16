# -*- coding: utf-8 -*-
"""
한 번의 명령으로 결과 문서를 재생성하는 실행기 (10월 3주차 6-4).

  python run_all.py                       # 27 → 28 → 29 → 31 → 32 (기본, 순서 고정)
  python run_all.py --all                 # 25, 27, 28, 29, 30, 31, 32, 33, 34 전체(고정 순서)
  python run_all.py --only 28,29          # 일부만 (순서는 고정 순서를 유지)
  python run_all.py --check               # 각 단계 후 결과 문서를 git HEAD 문서와 대조
  python run_all.py --repeat 2            # 같은 명령을 N회 실행하고 docs/ 전체가 바이트 단위로 같은지 확인
  python run_all.py --save-predictions    # 31번이 data/predictions/에 가명화 예측값 저장(git 제외)

의존: 27번이 docs/mlp_selected_hyperparams.json(MLP 하이퍼파라미터만)을 저장하고 28·29·31·32·33번이 읽는다.
이 실행기는 JSON을 읽는 단계가 있는지 소스로 확인하고, 27번 실행 후 JSON 해시를 고정해 이후 단계 도중 바뀌면 중단한다.
sklearn 의존 없음. 단계별 로그: logs/run_all_<단계>.log, 소요 시간: logs/run_all_timing.log (logs/는 git 제외).
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import env_tools

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
LOGS = ROOT / "logs"

STAGES = {
    "25": ("25_tree_models.py", ["docs/tree_results.md"]),
    "27": ("27_mlp_torch.py", ["docs/mlp_results.md", "docs/mlp_selected_hyperparams.json"]),
    "28": ("28_model_comparison.py", ["docs/model_comparison.md"]),
    "29": ("29_coaching.py", ["docs/coaching_results.md"]),
    "30": ("30_ridge_discrepancy.py", ["docs/ridge_discrepancy.md"]),
    "31": ("31_same_scope_comparison.py", ["docs/same_scope_comparison.md"]),
    "32": ("32_coaching_abstain.py", ["docs/coaching_abstain_results.md"]),
    "33": ("33_extreme_session_map.py", ["docs/extreme_session_window_map.md"]),
    "34": ("34_multisession_availability.py", ["docs/multisession_coaching_design.md"]),
}
DEFAULT_ORDER = ["27", "28", "29", "31", "32"]
ALL_ORDER = ["25", "27", "28", "29", "30", "31", "32", "33", "34"]
JSON_READERS = ["28", "29", "31", "32", "33"]
EXCLUDED_FROM_BYTE_COMPARE = {"sklearn_import_issue.md"}  # 시점이 들어가는 문서


def sh(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash():
    p = DOCS / "mlp_selected_hyperparams.json"
    return sh(p) if p.exists() else None


def check_json_dependency(stage):
    """JSON을 읽는 단계(28 직접, 29/31/32/33은 28을 import)가 실제로 그 파일을 쓰는지 소스로 확인."""
    script = STAGES[stage][0]
    src = (ROOT / script).read_text(encoding="utf-8")
    if stage == "28":
        ok = "mlp_selected_hyperparams.json" in src
    else:
        ok = "28_model_comparison.py" in src or "29_coaching.py" in src  # 29번은 28번을 import(JSON 연동), 32·33은 28 또는 29를 import
    if not ok:
        raise SystemExit(f"[중단] {script}가 27번의 선택 결과(JSON)를 읽는 28번을 참조하지 않는다 — 하드코딩 의존 의심")


def head_text(rel):
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=str(ROOT))
    return None if r.returncode != 0 else r.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")


def compare_with_head(rel):
    """결과: (IDENTICAL | ENV_LINES_ONLY | TEXT_ONLY | NUMBERS_SUBSET | NUMBERS_CHANGED | NEW, 변경 줄 수)."""
    cur = (ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
    old = head_text(rel)
    if old is None:
        return "NEW", 0
    if cur == old:
        return "IDENTICAL", 0

    def flt(t):
        return [l for l in t.split("\n") if not re.search(r"sklearn|DLL load|애플리케이션 제어", l)]

    import collections
    import difflib
    n = sum(1 for l in difflib.unified_diff(old.split("\n"), cur.split("\n"), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    if flt(cur) == flt(old):
        return "ENV_LINES_ONLY", n
    tok = lambda t: collections.Counter(re.findall(r"[0-9]+[.][0-9]{2,}", t))
    tc, to = tok(cur), tok(old)
    if tc == to:
        return "TEXT_ONLY(숫자 토큰 동일)", n
    if set(tc) <= set(to):
        return "NUMBERS_SUBSET(새로운 숫자 없음, 일부 줄/행 삭제·중복 변화)", n
    return "NUMBERS_CHANGED", n


def docs_manifest():
    m = {}
    for p in sorted(DOCS.glob("*")):
        if p.is_file() and p.name not in EXCLUDED_FROM_BYTE_COMPARE:
            m[p.name] = sh(p)
    return m


def run_stage(stage, extra_args):
    script = STAGES[stage][0]
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / f"run_all_{stage}.log"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        r = subprocess.run([sys.executable, "-X", "utf8", "-u", script] + extra_args, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    if r.returncode != 0:
        tail = "".join(open(log_path, encoding="utf-8", errors="replace").readlines()[-15:])
        raise SystemExit(f"[중단] {script} 실패(exit {r.returncode}), 로그 {log_path.name} 끝부분:\n{tail}")
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="25,27,28,29,30,31,32,33,34 전체")
    ap.add_argument("--only", default="", help="쉼표로 구분한 단계 번호(예: 28,29)")
    ap.add_argument("--check", action="store_true", help="각 단계 후 결과 문서를 git HEAD 문서와 대조")
    ap.add_argument("--repeat", type=int, default=1, help="같은 실행을 N회 반복하고 docs/ 바이트 동일성 확인")
    ap.add_argument("--save-predictions", action="store_true", help="31번이 data/predictions/에 가명화 예측값 저장")
    args = ap.parse_args()

    order = ALL_ORDER if args.all else DEFAULT_ORDER
    if args.only:
        want = [x.strip() for x in args.only.split(",") if x.strip()]
        bad = [x for x in want if x not in STAGES]
        if bad:
            raise SystemExit(f"알 수 없는 단계: {bad} (가능: {sorted(STAGES)})")
        order = [s for s in ALL_ORDER if s in want]
    print(f"실행 순서: {' → '.join(order)} (반복 {args.repeat}회)")

    manifests, timing_rows, check_rows = [], [], []
    for rep in range(1, args.repeat + 1):
        print(f"\n===== 실행 {rep}/{args.repeat} =====")
        json_ref = json_hash()
        for st in order:
            if st in JSON_READERS:
                check_json_dependency(st)
                if json_ref is None:
                    raise SystemExit(f"[중단] {STAGES[st][0]}는 docs/mlp_selected_hyperparams.json이 필요하다 — 먼저 27번을 실행")
            extra = ["--save-predictions"] if (st == "31" and args.save_predictions) else []
            print(f"[{st}] {STAGES[st][0]} 실행 중...", flush=True)
            dt = run_stage(st, extra)
            if st == "27":
                json_ref = json_hash()
                print(f"    JSON sha256={json_ref[:16]}… (이후 단계에서 고정 확인)")
            elif json_ref is not None and json_hash() != json_ref:
                raise SystemExit(f"[중단] {STAGES[st][0]} 실행 중 docs/mlp_selected_hyperparams.json이 바뀌었다")
            timing_rows.append((rep, st, STAGES[st][0], dt))
            print(f"    완료 {dt / 60:.1f}분", flush=True)
            if args.check:
                for rel in STAGES[st][1]:
                    status, n = compare_with_head(rel)
                    check_rows.append((rep, st, rel, status, n))
                    print(f"    [--check] {rel}: {status}" + (f" (변경 {n}줄)" if n else ""))
        # 환경 문서 재생성 (결정적)
        subprocess.run([sys.executable, "-X", "utf8", "env_tools.py", "--environment"], cwd=str(ROOT), check=True)
        manifests.append(docs_manifest())

    # 타이밍 요약/기록
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / "run_all_timing.log", "a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} stages={order} repeat={args.repeat} =====\n")
        for rep, st, script, dt in timing_rows:
            f.write(f"run{rep} {st} {script} {dt:.1f}s\n")
    print("\n단계별 소요 시간(분):")
    for rep, st, script, dt in timing_rows:
        print(f"  run{rep} [{st}] {script}: {dt / 60:.1f}")
    for rep in range(1, args.repeat + 1):
        print(f"  run{rep} 합계: {sum(d for r, _, _, d in timing_rows if r == rep) / 60:.1f}분")

    if args.check:
        print("\n--check 요약 (HEAD 문서 대조):")
        for rep, st, rel, status, n in check_rows:
            print(f"  run{rep} {rel}: {status}" + (f" ({n}줄)" if n else ""))
        if any(r[3] == "NUMBERS_CHANGED" for r in check_rows):
            print("  ※ NUMBERS_CHANGED가 있다 — 수치가 바뀌었는지 확인 필요")
    if args.repeat >= 2:
        base = manifests[0]
        diffs = []
        for k, m in enumerate(manifests[1:], start=2):
            for name in sorted(set(base) | set(m)):
                if base.get(name) != m.get(name):
                    diffs.append((k, name))
        print(f"\n바이트 동일성(docs/ 전체, {len(base)}개 파일, {', '.join(sorted(EXCLUDED_FROM_BYTE_COMPARE))} 제외):", "모두 동일" if not diffs else f"차이 {diffs}")
        if diffs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
