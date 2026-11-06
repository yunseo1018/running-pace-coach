# -*- coding: utf-8 -*-
"""docs/b2_failure_5km.md의 'train 폴드 밖' 층 표에서 조건별 범위 안내 대상 층을 골라 demo/stratum_notes.json을 만든다.
선정 규칙(docs/demo_stratum_note_spec.md 2절): 평가 대상 30개 이상이고 명목 80% 범위 포함률의 95% CI 상한이 80% 미만인 층.
선정 결과가 사양에 고정한 4개 층과 다르면 중단한다(사양을 먼저 갱신해야 함). test 숫자는 읽지 않는다. 표준 라이브러리만 사용. 사용: python demo/make_stratum_notes.py"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "b2_failure_5km.md"
OUT = Path(__file__).resolve().parent / "stratum_notes.json"
MIN_N = 30
NOMINAL = 80.0
SECTION_DIM = {"## 1. (a)": "a", "## 2. (b)": "b", "## 3. (c)": "c", "## 4. (d)": "d", "## 5. (e)": "e"}
# 사양에 고정한 선정 결과(차원, 문서의 층 이름)
SPEC_STRATA = [("a", "1개(x = 0)"), ("a", "2~4개"), ("b", "120초 초과"), ("e", "300초 초과")]
ROW = re.compile(r"^\| train 폴드 밖 \| (.+?) \| ([\d,]+) \| ([\d,]+) \| [\d.]+ \| [\d.]+ \| [^|]+ \| ([\d.]+)% \[([\d.]+)%, ([\d.]+)%\] \|")


def read_train_rows(text=None):
    """문서에서 train 폴드 밖 층 행을 읽는다. 반환 [{dim, label, n, users, cov, lo, hi}] (test 행은 읽지 않는다)."""
    text = DOC.read_text(encoding="utf-8") if text is None else text
    rows, dim = [], None
    for line in text.splitlines():
        for head, d in SECTION_DIM.items():
            if line.startswith(head):
                dim = d
        m = ROW.match(line)
        if m and dim:
            rows.append({"dim": dim, "label": m.group(1), "n": int(m.group(2).replace(",", "")), "users": int(m.group(3).replace(",", "")),
                         "cov": float(m.group(4)), "ci_lo": float(m.group(5)), "ci_hi": float(m.group(6))})
    return rows


def select(rows):
    return [r for r in rows if r["n"] >= MIN_N and r["ci_hi"] < NOMINAL]


def main():
    rows = read_train_rows()
    assert len(rows) == 16, len(rows)
    sel = select(rows)
    got = [(r["dim"], r["label"]) for r in sel]
    if got != SPEC_STRATA:
        print("선정 결과가 사양의 4개 층과 다릅니다. docs/demo_stratum_note_spec.md를 먼저 갱신하고 커밋하세요:", got)
        return 1
    out = {"source": "docs/b2_failure_5km.md (train 폴드 밖 행만; test 숫자는 쓰지 않음)",
           "rule": "평가 대상 30개 이상, 명목 80%% 범위 포함률의 95%% CI 상한 < 80%%".replace("%%", "%"),
           "strata": sel}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("saved", OUT.name, [(r["dim"], r["label"], r["cov"]) for r in sel])
    return 0


if __name__ == "__main__":
    sys.exit(main())
