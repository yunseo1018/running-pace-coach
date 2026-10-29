# -*- coding: utf-8 -*-
"""
데모(B2)와 참고용 GBM·MLP의 예측 1회 시간·메모리·모델 크기 측정 (PC CPU). 저장소 루트에서: python demo/bench_demo.py
결과: docs/demo_performance.md. 학습 자료는 data/5km_dataset_train.pkl(로컬, 비공개)이며 문서에는 집계 수치만 적는다.
- 시간: 예측 1회를 1,000번 반복(워밍업 50회 제외)한 중앙값·90% 분위. time.perf_counter_ns.
- 메모리: tracemalloc 최대값(파이썬 힙만, torch 네이티브 메모리는 포함되지 않음).
- 크기: B2 = 파라미터 JSON 파일 바이트, GBM = pickle 직렬화 바이트, MLP = state_dict 직렬화 바이트 + 스케일러 배열 바이트(배포 형식 최적화·INT8 아님).
- 시연은 B2만 사용한다. GBM·MLP는 49c02f8과 같은 설정(기본 입력 11, 선택된 하이퍼파라미터)으로 cold-start 학습 자료 전체에 다시 학습한 참고용이다.
"""
import importlib.util
import io
import os
import pickle
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
import coach_demo as cd  # noqa: E402

N_REP, N_WARM = 1000, 50
EXAMPLE = Path("demo") / "examples" / "ex2_slower_last.csv"
ASOF = "2025-03-03T00:00:00"
DOC = Path("docs") / "demo_performance.md"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_it(fn):
    for _ in range(N_WARM):
        fn()
    ts = np.empty(N_REP)
    for i in range(N_REP):
        t0 = time.perf_counter_ns()
        fn()
        ts[i] = (time.perf_counter_ns() - t0) / 1e6
    return float(np.median(ts)), float(np.percentile(ts, 90)), float(ts.max())


def peak_mem(fn):
    fn()   # 지연 초기화를 측정에서 제외
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or "확인 불가"


def main():
    import torch
    torch.set_num_threads(4)
    m42 = _load("42_final_eval_5km.py", "m42")
    m40, m25, m27 = m42.m40, m42.m40.m25, m42.m40.m27
    hp = m42.read_selected()
    tr = pd.read_pickle(Path("data") / "5km_dataset_train.pkl")
    cols = m40.SETS["기본(11)"]
    Xtr, ytr = tr[cols].to_numpy(float), tr.y.to_numpy(float)
    x1 = Xtr[:1].copy()

    params = cd.load_params()
    sessions = cd.read_sessions(EXAMPLE)
    asof = cd.parse_time(ASOF)

    def b2_core():
        res = cd.assess(sessions, asof, params)
        cd.check_numbers(cd.format_report(res), res)

    def b2_with_csv():
        cd.run(EXAMPLE, ASOF, params)

    def b2_formula_only():
        cd.predict_change(1650.0, 1500.0, params)

    gbm = m25.build_gbt(hp[("GBM", "기본(11)")], "absolute_error").fit(Xtr, ytr)
    n_ep = 200
    from_doc = (Path("docs") / "final_results_5km.md").read_text(encoding="utf-8")
    assert "MLP/기본(11) 200" in from_doc    # 49c02f8 cold-start에서 내부 검증으로 정해진 에폭 수
    p = hp[("MLP", "기본(11)")]
    mlp, scaler = m27.train_mlp_fixed_epochs(Xtr, ytr, p["n_hidden_layers"], p["width"], p["weight_decay"], m27.SELECTION_SEED, n_ep)

    def gbm_pred():
        gbm.predict(x1)

    def mlp_pred():
        m27.predict_mlp(mlp, scaler, x1)

    buf = io.BytesIO()
    torch.save(mlp.state_dict(), buf)
    scaler_bytes = sum(np.asarray(v).nbytes for v in (scaler.values() if isinstance(scaler, dict) else scaler))
    n_params = sum(t.numel() for t in mlp.parameters())
    sizes = {"B2 (데모)": Path(cd.PARAMS_PATH).stat().st_size,
             "GBM": len(pickle.dumps(gbm)),
             "MLP": buf.getbuffer().nbytes + int(scaler_bytes)}

    rows = []
    for name, fn, extra in [("B2 데모: 자격 판정·예측·문구·숫자 검사(CSV 읽기 제외)", b2_core, "B2 (데모)"),
                            ("B2 데모: 위 + CSV 파일 읽기·파싱", b2_with_csv, "B2 (데모)"),
                            ("B2 식만 (a + b·x)", b2_formula_only, "B2 (데모)"),
                            ("GBM 예측 1행 (기본 입력 11, 참고)", gbm_pred, "GBM"),
                            ("MLP 예측 1행 (기본 입력 11, 참고)", mlp_pred, "MLP")]:
        med, p90, mx = time_it(fn)
        rows.append((name, med, p90, mx, peak_mem(fn), sizes[extra]))

    # 프로세스 시작 포함 전체 명령 실행 시간(5회, 참고)
    wall = []
    for _ in range(5):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-I", "demo/coach_demo.py", str(EXAMPLE), "--asof", ASOF], capture_output=True, check=True)
        wall.append((time.perf_counter() - t0) * 1000)

    lines = ["# 데모 성능 측정 (PC CPU, 서버·모바일 아님)", "",
             "생성: `demo/bench_demo.py`. 시연은 B2(평균 회귀)만 사용한다. GBM·MLP는 비교 참고용이다(49c02f8과 같은 설정으로 cold-start 학습 자료에 다시 학습; 정확도 비교가 아님). 합성 예시 `" + str(EXAMPLE).replace("\\", "/") + "` 사용.", "",
             "## 환경", "",
             "- CPU: " + cpu_name(), "- OS: " + platform.platform(), "- Python: " + sys.version.split()[0],
             f"- numpy {np.__version__}, torch {torch.__version__}, torch 스레드 4", "- 측정 방법: 예측 1회를 %d번 반복(워밍업 %d회 제외), `time.perf_counter_ns`. 메모리 = `tracemalloc` 최대값(파이썬 힙만; torch 네이티브 메모리·인터프리터 기본 메모리는 포함되지 않음)." % (N_REP, N_WARM), "",
             "## 예측 1회 (%d회 반복)" % N_REP, "",
             "| 대상 | 중앙값(ms) | 90% 분위(ms) | 최대(ms) | tracemalloc 최대(바이트) | 크기(바이트) |", "|---|---|---|---|---|---|"]
    for name, med, p90, mx, pk, sz in rows:
        lines.append(f"| {name} | {med:.4f} | {p90:.4f} | {mx:.4f} | {pk:,} | {sz:,} |")
    lines += ["",
              "- 크기: B2 = `demo/b2_params.json` 파일 크기. GBM = pickle 직렬화 크기. MLP = `state_dict` 직렬화 + 스케일러 배열 크기(파라미터 %s개, 배포용 최적화·양자화 없음). 표의 크기 열은 대상 행이 아니라 해당 모델의 크기이며 B2 세 행은 같은 값이다." % f"{n_params:,}",
              "- 프로세스 시작을 포함한 `python demo/coach_demo.py ...` 한 번 실행(5회) 벽시계 시간: 최소 %.0f ms, 중앙값 %.0f ms, 최대 %.0f ms (인터프리터 시작 시간이 대부분)." % (min(wall), float(np.median(wall)), max(wall)),
              "- 이 수치는 이 PC 한 대에서의 값이며 다른 기기·모바일에는 일반화할 수 없다. 모바일·온디바이스(INT8 포함) 측정은 하지 않았다(미측정).",
              "- 값의 변동: 같은 명령을 다시 실행하면 중앙값이 달라질 수 있다(다른 프로세스·전원 상태에 따라).", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines[8:]))


if __name__ == "__main__":
    main()
