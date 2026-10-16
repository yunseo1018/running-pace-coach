# -*- coding: utf-8 -*-
"""
재현성 도구 (10월 3주차 6번).

- sklearn import 진단: 문서 본문에는 쓰지 않고 logs/sklearn_diagnostics.log(.gitignore 대상)에만 기록한다.
- docs/environment.md 자동 생성(타임스탬프 없음 -> 같은 환경에서 바이트 동일).
- docs/sklearn_import_issue.md 생성(재현 시도 기록; 홈·프로젝트 경로는 <USER>, <PROJECT>로 마스킹; 시점이 들어가므로 바이트 비교 대상 아님).
- 평가 행별 예측값 내보내기 보조(사용자·세션 ID는 고정 salt 해시로 대체; data/predictions/ 아래, git 제외).

CLI:
  python env_tools.py --environment        # docs/environment.md 생성
  python env_tools.py --sklearn-issue      # docs/sklearn_import_issue.md 생성 (+ logs 기록)
  python env_tools.py --sklearn-probe TAG  # logs/sklearn_diagnostics.log에만 기록
"""
import hashlib
import importlib.metadata as im
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
DOCS = ROOT / "docs"
PSEUDO_SALT = "running-pace-coach/pred-export/v1"  # 고정 salt(가명화일 뿐 암호학적 익명화가 아님 -> 파일은 git 제외)

SKLEARN_PROBES = [
    ("import sklearn", "import sklearn"),
    ("sklearn.linear_model.Ridge", "from sklearn.linear_model import Ridge"),
    ("sklearn.model_selection.GroupKFold", "from sklearn.model_selection import GroupKFold"),
    ("sklearn.preprocessing.StandardScaler", "from sklearn.preprocessing import StandardScaler"),
    ("sklearn.tree.DecisionTreeRegressor", "from sklearn.tree import DecisionTreeRegressor"),
    ("sklearn.ensemble.HistGradientBoostingRegressor", "from sklearn.ensemble import HistGradientBoostingRegressor"),
    ("sklearn.neural_network.MLPRegressor", "from sklearn.neural_network import MLPRegressor"),
    ("sklearn.linear_model.QuantileRegressor", "from sklearn.linear_model import QuantileRegressor"),
]


# ===================== 해시/마스킹 =====================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def files_key(paths):
    """파일 목록의 내용 해시를 하나로 묶은 키(캐시 무효화용). 존재하지 않는 파일은 'MISSING'."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        pp = Path(p)
        if not pp.is_absolute():
            pp = ROOT / pp
        h.update(p.encode("utf-8") + b":" + (sha256_file(pp).encode() if pp.exists() else b"MISSING") + b"\n")
    return h.hexdigest()


def mask_text(text):
    """홈/프로젝트 경로를 <USER>, <PROJECT>로 마스킹(슬래시·역슬래시·대소문자 모두)."""
    out = text
    for real, token in [(str(ROOT), "<PROJECT>"), (str(Path.home()), "<USER>")]:
        variants = {real, real.replace("\\", "/"), real.replace("/", "\\")}
        for v in sorted(variants, key=len, reverse=True):
            out = re.sub(re.escape(v), token, out, flags=re.IGNORECASE)
    # 마스킹 후에도 남을 수 있는 사용자 폴더 이름 패턴(C:\Users\<name>)
    out = re.sub(r"(?i)([A-Z]:[\\/]Users[\\/])[^\\/\s'\"]+", r"\1<USER>", out)
    return out


def write_log(name, text):
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / name, "a", encoding="utf-8") as f:
        f.write(text)


# ===================== sklearn 진단 (문서에는 쓰지 않음) =====================
def run_probe(stmt, timeout=180):
    """새 인터프리터에서 import를 시도하고 (성공 여부, 출력 전문)을 돌려준다."""
    try:
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", stmt], capture_output=True, timeout=timeout,
                           cwd=str(ROOT), text=True, encoding="utf-8", errors="replace")
        return r.returncode == 0, (r.stderr or "") + (r.stdout or "")
    except Exception as e:  # noqa: BLE001
        return False, f"probe 실행 실패: {type(e).__name__}: {e}"


def sklearn_probe_results():
    return [(name, *run_probe(stmt)) for name, stmt in SKLEARN_PROBES]


def log_sklearn_probes(tag):
    """진단 결과를 logs/sklearn_diagnostics.log(.gitignore)에만 기록한다. 결과 수치 경로와 무관."""
    now = datetime.now().isoformat(timespec="seconds")
    lines = [f"\n===== {now} [{tag}] python={sys.version.split()[0]} =====\n"]
    res = sklearn_probe_results()
    for name, ok, out in res:
        lines.append(f"[{'OK' if ok else 'FAIL'}] {name}\n")
        if not ok:
            lines.append(out.rstrip() + "\n")
    write_log("sklearn_diagnostics.log", "".join(lines))
    return res


def log_sklearn_ridge_comparison(tag):
    """(진단) ridge_lib vs sklearn Ridge/GroupKFold 대조 — 로그에만 기록."""
    code = r"""
import numpy as np, sys
sys.path.insert(0, '.')
import ridge_lib as rl
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
rng = np.random.default_rng(0)
n, p = 300, 22
X = rng.normal(size=(n, p)); y = X @ rng.normal(size=p) + 3.0 + rng.normal(scale=0.1, size=n)
b, i0 = rl.ridge_fit(X, y, alpha=1.0)[:2]
sk = Ridge(alpha=1.0).fit(X, y)
print('ridge coef max diff', float(np.abs(b - sk.coef_).max()), 'intercept diff', float(abs(i0 - sk.intercept_)))
uid = rng.integers(0, 50, size=n)
sk_sets = [set(uid[v].tolist()) for _, v in GroupKFold(n_splits=5).split(X, y, groups=uid)]
mn_sets = [set(uid[v].tolist()) for _, v in rl.group_kfold_indices(uid, 5, 42)]
print('group fold membership identical:', sorted(map(frozenset, sk_sets)) == sorted(map(frozenset, mn_sets)))
"""
    ok, out = run_probe(code)
    now = datetime.now().isoformat(timespec="seconds")
    write_log("sklearn_diagnostics.log", f"\n===== {now} [{tag}: ridge_lib vs sklearn 대조] =====\n[{'OK' if ok else 'FAIL'}]\n{out}")
    return ok


# ===================== docs/sklearn_import_issue.md =====================
def write_sklearn_issue_md(attempts=3):
    now = datetime.now().isoformat(timespec="seconds")
    all_runs = []
    for k in range(attempts):
        all_runs.append(sklearn_probe_results())
    n_fail = sum(1 for run in all_runs for (_, ok, _) in run if not ok)
    lines = ["> 실행 시점이 포함된 진단 기록이며 바이트 재현성 검증 대상이 아님.\n\n" + "# sklearn import 이슈 기록 (Windows 애플리케이션 제어 정책, `DLL load failed`)\n\n"]
    lines.append("> 이 문서는 `python env_tools.py --sklearn-issue`로 생성한다(시점이 들어가므로 실행마다 바이트가 달라지며 재현성 비교 대상이 아니다). "
                 "홈·프로젝트 경로는 `<USER>`, `<PROJECT>`로 마스킹했다. **이 파이프라인의 결과 수치는 sklearn에 의존하지 않는다**"
                 "(Ridge/GroupKFold/표준화=`ridge_lib`, GBM=`gbt_lib`, MLP=torch).\n\n")
    lines.append("## 재현 시도 방법\n")
    lines.append(f"- 시점: {now} (로컬 시각). Python {sys.version.split()[0]}, OS {platform.platform()}.\n")
    lines.append(f"- 방법: 서로 다른 새 인터프리터(`python -X utf8 -c <import 문>`)에서 아래 {len(SKLEARN_PROBES)}개 import를 각각 시도하고 {attempts}회 반복했다(총 {attempts * len(SKLEARN_PROBES)}회).\n")
    try:
        lines.append(f"- 설치된 scikit-learn 배포 버전(메타데이터 조회, import 없음): {im.version('scikit-learn')}\n")
    except Exception:  # noqa: BLE001
        lines.append("- scikit-learn 배포 메타데이터를 찾지 못했다.\n")
    lines.append("\n## 결과 요약\n")
    lines.append("| import | " + " | ".join(f"시도 {k + 1}" for k in range(attempts)) + " |\n|---|" + "---|" * attempts + "\n")
    for idx, (name, _) in enumerate(SKLEARN_PROBES):
        lines.append(f"| `{name}` | " + " | ".join("성공" if all_runs[k][idx][1] else "**실패**" for k in range(attempts)) + " |\n")
    lines.append(f"\n- 총 {attempts * len(SKLEARN_PROBES)}회 중 실패 {n_fail}회.\n")
    if n_fail:
        lines.append("\n## 재현된 오류의 traceback 전문 (경로 마스킹)\n")
        shown = set()
        for k, run in enumerate(all_runs):
            for name, ok, out in run:
                if not ok:
                    body = mask_text(out.rstrip())
                    key = (name, body)
                    if key in shown:
                        continue
                    shown.add(key)
                    lines.append(f"\n### 시도 {k + 1}: `{name}`\n```\n{body}\n```\n")
        lines.append("\n- 관찰: 실패하는 모듈 이름(예: `_minpack`, `_pava_pybind` 등 scipy/sklearn 컴파일 확장)이 실행마다 달라지는 간헐 현상이 이전 작업에서 반복 확인되었다. "
                     "원인은 Windows Application Control 정책 때문으로 **추정**되지만 정책 설정을 확인하지 못했다(미확인). 정책 우회는 시도하지 않았다.\n")
    else:
        lines.append("\n## 재현되지 않음\n이 시점에는 위 모든 import가 성공해 traceback을 확보하지 못했다. 이전 작업(10월 2~3주차)에서는 같은 환경에서 "
                     "`DLL load failed ... 애플리케이션 제어 정책에서 이 파일을 차단했습니다` 형태의 `ImportError`가 실행마다 다른 모듈에서 간헐적으로 발생했다. "
                     "재현되면 이 명령을 다시 실행해 traceback 전문을 기록한다.\n")
    (DOCS / "sklearn_import_issue.md").write_text("".join(lines), encoding="utf-8")
    write_log("sklearn_diagnostics.log", f"\n===== {now} [--sklearn-issue] 실패 {n_fail}/{attempts * len(SKLEARN_PROBES)} =====\n")
    return n_fail


# ===================== docs/environment.md =====================
def _const(path, name):
    src = (ROOT / path).read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}\s*=\s*(.+?)(?:\s*#.*)?$", src, re.M)
    return m.group(1).strip() if m else "미확인"


def write_environment_md():
    import numpy
    import pandas
    import scipy
    import torch
    data = ROOT / "data"
    data_files = ["run_sessions_deduped_merged.pkl", "run_sessions_deduped_merged_with_moving.pkl", "user_sequences.pkl",
                  "extended_train_dataset.pkl", "extended_test_dataset.pkl",
                  "extended_personalized_train_dataset.pkl", "extended_personalized_test_dataset.pkl"]
    lines = ["# 실행 환경 기록 (자동 생성: `python env_tools.py --environment`)\n\n"]
    lines.append("타임스탬프를 넣지 않아 같은 환경·같은 데이터에서는 바이트 단위로 같다. 사용자·세션 ID와 개인 경로는 쓰지 않는다.\n\n")
    lines.append("## 1. 소프트웨어\n")
    lines.append(f"- Python {sys.version.split()[0]}\n- OS: {platform.platform()}\n")
    lines.append(f"- numpy {numpy.__version__}, pandas {pandas.__version__}, scipy {scipy.__version__}, torch {torch.__version__} (CPU 스레드 {torch.get_num_threads()})\n")
    try:
        lines.append(f"- scikit-learn 배포 버전 {im.version('scikit-learn')}: **이 파이프라인은 sklearn에 의존하지 않는다**(진단 전용, import 실패해도 결과 불변). 실패 기록은 `docs/sklearn_import_issue.md`\n")
    except Exception:  # noqa: BLE001
        lines.append("- scikit-learn: 설치 안 됨(파이프라인은 sklearn에 의존하지 않음)\n")
    lines.append("- 직접 구현: `ridge_lib.py`(Ridge/GroupKFold/표준화), `gbt_lib.py`(GBM: 손실 absolute_error/squared_error, 격자·고정값은 `docs/tree_results.md` 2절), MLP는 torch(L1 손실)\n\n")
    lines.append("## 2. 실행 명령\n```\n"
                 "pip install -r requirements.txt\n"
                 "python run_all.py                      # 27 → 28 → 29 → 31 → 32 (기본)\n"
                 "python run_all.py --all                # 25, 27, 28, 29, 30, 31, 32, 33 전체\n"
                 "python run_all.py --only 28,29         # 일부만 (고정 순서 유지)\n"
                 "python run_all.py --check              # 각 단계 후 결과 문서를 git HEAD 문서와 대조\n"
                 "python run_all.py --repeat 2           # 같은 명령을 두 번 실행하고 docs/ 바이트 동일성 확인\n"
                 "python run_all.py --save-predictions   # data/predictions/에 가명화 예측값 저장(git 제외)\n"
                 "python env_tools.py --environment      # 이 문서 재생성\n"
                 "python env_tools.py --sklearn-issue    # docs/sklearn_import_issue.md 재생성\n"
                 "```\n\n")
    lines.append("## 3. 시드·분할\n")
    lines.append(f"- 사용자 분할(03단계): RANDOM_SEED={_const('03_train_test_split.py', 'RANDOM_SEED')}, TRAIN_RATIO={_const('03_train_test_split.py', 'TRAIN_RATIO')} (사용자 단위 80/20)\n")
    lines.append(f"- personalized: 사용자별 시간순 history/hold-out 비율 {_const('24_build_xy_dataset_extended.py', 'HISTORY_RATIO')}; 모델 선택용 내부 분할 비율 {_const('25_tree_models.py', 'HISTORY_INNER_RATIO')}\n")
    lines.append(f"- GroupKFold(ridge_lib): {_const('25_tree_models.py', 'N_FOLDS')}-fold, 시드 {_const('25_tree_models.py', 'CV_SEED')}\n")
    lines.append(f"- 사용자 단위 cluster bootstrap(21·28·31): {_const('21_baseline_integrated_report.py', 'N_BOOTSTRAP')}회, 시드 {_const('21_baseline_integrated_report.py', 'BOOTSTRAP_SEED')}\n")
    lines.append(f"- 코칭 bootstrap(29·32): {_const('29_coaching.py', 'N_BOOT')}회, 시드 {_const('29_coaching.py', 'BOOT_SEED')}; 32 무작위 보류 통제 {_const('32_coaching_abstain.py', 'N_CTRL')}회\n")
    lines.append(f"- MLP(27): 선택 시드 {_const('27_mlp_torch.py', 'SELECTION_SEED')}, 최종 시드 {_const('27_mlp_torch.py', 'FINAL_SEEDS')}, 대표 시드 42 (28·29·31·32는 `MLP_SEED`=42)\n")
    lines.append("- GBM(`gbt_lib`): 행·피처 서브샘플링이 없어 난수를 쓰지 않는다(결정적; `random_state`는 결과에 영향 없음)\n\n")
    lines.append("## 4. 데이터·분할·하이퍼파라미터 해시 (sha256)\n")
    lines.append("| 파일 | sha256 |\n|---|---|\n")
    for fn in data_files:
        p = data / fn
        lines.append(f"| `data/{fn}` | {sha256_file(p) if p.exists() else '(없음)'} |\n")
    for fn in ["train_user_ids.txt", "test_user_ids.txt"]:
        p = data / fn
        if p.exists():
            ids = sorted(x.strip() for x in p.read_text(encoding="utf-8").split("\n") if x.strip())
            lines.append(f"| 분할 ID 목록 `data/{fn}` (정렬된 ID {len(ids)}개를 개행으로 이은 문자열의 해시; ID 자체는 기록하지 않음) | {sha256_text(chr(10).join(ids))} |\n")
    jp = DOCS / "mlp_selected_hyperparams.json"
    lines.append(f"| `docs/mlp_selected_hyperparams.json` (27번이 저장, 28·29·31·32가 읽음) | {sha256_file(jp) if jp.exists() else '(없음)'} |\n\n")
    pred_dir = data / "predictions"
    lines.append("## 5. 평가 행별 예측값 파일 (선택 기능, git 제외)\n")
    files = sorted(pred_dir.glob("*.csv")) if pred_dir.exists() else []
    if files:
        lines.append(f"`python run_all.py --save-predictions`로 만든 파일. 사용자·세션 ID는 고정 salt 해시로 대체했다(가명화일 뿐 익명화가 아니므로 파일 자체는 커밋 금지; `data/`는 `.gitignore` 대상).\n\n| 파일 | sha256 |\n|---|---|\n")
        for fp in files:
            lines.append(f"| `data/predictions/{fp.name}` | {sha256_file(fp)} |\n")
    else:
        lines.append("현재 저장된 예측값 파일이 없다(`--save-predictions` 미사용). 파일을 만들면 이 절에 파일별 sha256이 기록된다.\n")
    (DOCS / "environment.md").write_text("".join(lines), encoding="utf-8")


# ===================== 예측값 내보내기 =====================
def pseudo_id(x):
    return hashlib.sha256((PSEUDO_SALT + "|" + str(x)).encode("utf-8")).hexdigest()[:16]


def save_predictions(split, user_ids, session_ids, y, preds):
    """data/predictions/<split>_predictions.csv. 사용자·세션 ID는 salt 해시. 값은 고정 형식으로 써 바이트 재현이 가능하다."""
    import pandas as pd
    out_dir = ROOT / "data" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"user_hash": [pseudo_id(u) for u in user_ids], "session_hash": [pseudo_id(s) for s in session_ids], "y": y})
    for k in sorted(preds):
        df[k] = preds[k]
    path = out_dir / f"{split.replace('-', '_')}_predictions.csv"
    df.to_csv(path, index=False, float_format="%.9g", lineterminator="\n")
    return path


if __name__ == "__main__":
    if "--environment" in sys.argv:
        write_environment_md()
        print("docs/environment.md 생성")
    elif "--sklearn-issue" in sys.argv:
        n = write_sklearn_issue_md()
        print(f"docs/sklearn_import_issue.md 생성 (실패 {n}회)")
    elif "--sklearn-probe" in sys.argv:
        tag = sys.argv[sys.argv.index("--sklearn-probe") + 1] if len(sys.argv) > sys.argv.index("--sklearn-probe") + 1 else "manual"
        log_sklearn_probes(tag)
        print("logs/sklearn_diagnostics.log에 기록")
    else:
        print(__doc__)
