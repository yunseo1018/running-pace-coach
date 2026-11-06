# -*- coding: utf-8 -*-
"""demo/feedback_quantiles.json 생성. 사양: docs/demo_feedback_spec.md 6절. 저장소 루트에서: python demo/make_feedback_quantiles.py
- 자료: 범위 보정(scripts/55_interval_calibration_5km.py)과 같은 train 사용자 단위 GroupKFold(5, seed 42) 폴드 밖 B2 잔차(실제 T5 - 예측 T5, 초). test 자료는 쓰지 않는다.
- 잔차 원값은 저장하지 않고 1%~99% 분위수 99개만 저장한다(사용자 ID·행 정보 없음). 난수 없음(재실행하면 같은 파일).
- 안전 점검: 10%·90% 분위수가 demo/b2_params.json의 resid_q10·resid_q90와 일치하지 않으면 파일을 쓰지 않는다.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "feedback_quantiles.json"
PARAMS = ROOT / "demo" / "b2_params.json"
LEVELS = list(range(1, 100))


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("m55", ROOT / "scripts" / "55_interval_calibration_5km.py")
    m55 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m55)
    tr = pd.read_pickle(ROOT / "data" / "5km_dataset_train.pkl")
    assert len(tr) == 1720 and tr.uid.nunique() == 278
    q10, q90, _coefs, res = m55.oof_quantiles(tr)
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    if abs(q10 - p["resid_q10"]) > 1e-9 or abs(q90 - p["resid_q90"]) > 1e-9:
        print("폴드 밖 잔차의 10%·90% 분위수가 b2_params.json과 다릅니다. 파일을 쓰지 않습니다.", q10, q90)
        return 1
    q = [round(float(v), 3) for v in np.percentile(res, LEVELS)]
    assert all(q[i] <= q[i + 1] for i in range(len(q) - 1))
    out = {"source": "train 사용자 단위 GroupKFold(5, seed 42) 폴드 밖 B2 잔차(실제 T5 - 예측 T5, 초); test 자료는 쓰지 않음",
           "n": int(len(res)), "levels_percent": LEVELS, "quantiles_sec": q}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("saved", OUT.name, "n =", len(res), "q01/q10/q50/q90/q99 =", q[0], q[9], q[49], q[89], q[98])
    return 0


if __name__ == "__main__":
    sys.exit(main())
