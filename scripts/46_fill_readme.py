# -*- coding: utf-8 -*-
"""README 맨 위(5km 과제·데모 절)를 결과 문서와 데모 실행 출력에서 읽은 값으로 채운다. 이전 과제 부록(<!-- OLD_README_BELOW --> 아래)은 그대로 둔다.
수치 출처: docs/final_results_5km.md((c) 표, 평가 대상·사용자 수), docs/demo_interval_check.md(포함률), docs/demo_performance.md(성능), demo/coach_demo.py 실행 출력.
사용: python scripts/46_fill_readme.py          # README.md 갱신
      python scripts/46_fill_readme.py --check  # 쓰지 않고 현재 README와 같은지만 확인(다르면 종료 코드 1)
저장소 루트의 README.md만 읽고 쓴다. 같은 입력이면 항상 같은 결과(멱등)."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rd = ROOT / "README.md"
old = rd.read_text(encoding="utf-8")
MARK = "<!-- OLD_README_BELOW -->"
if MARK in old:
    old = old[old.index(MARK) + len(MARK):].lstrip("\n")
    while old.startswith("# 부록:"):     # 아래에서 다시 붙이는 부록 제목은 한 번만 남긴다(멱등)
        old = old.split("\n", 1)[1].lstrip("\n")
else:
    # 기존 README 제목 단계를 한 단계씩 낮춘다(코드 펜스 안의 # 주석은 건드리지 않음)
    out, fence = [], False
    for line in old.splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence and line.startswith("#"):
            line = "#" + line
        out.append(line)
    old = "\n".join(out) + "\n"


def txt(p):
    return (ROOT / p).read_text(encoding="utf-8")


# ---- 주 결론 수치 (final_results_5km.md (c) 표)
fin = txt("docs/final_results_5km.md")
diffs = {}
for name, a, b in [("cold", "## A.", "## B."), ("pers", "## B.", None)]:
    sec = fin[fin.index(a): fin.index(b) if b else len(fin)]
    sec = sec[sec.index("### (c)"): sec.index("### (d)")]
    for line in sec.splitlines():
        m = re.match(r"\| (Ridge|GBM|MLP) \| ([+-][\d.]+) \[([-\d.]+), ([-\d.]+)\] \|", line)
        if m:
            diffs[(name, m.group(1))] = (float(m.group(2)), float(m.group(4)))
assert len(diffs) == 6
lo_d, hi_d = min(v[0] for v in diffs.values()), max(v[0] for v in diffs.values())
assert all(v[1] >= 0 for v in diffs.values())      # CI 상한이 모두 0 이상 = 통계적 통과 0/6
n_cold = re.search(r"평가 기준 시점 (\d+)개, 사용자 (\d+)명", fin)
n_pers = re.search(r"평가 기준 시점 (\d+)개, 사용자 (\d+)명", fin[fin.index("## B."):])

# ---- 데모 출력
ex = [("ex1_steady", "① 꾸준히 달린 사람"), ("ex2_slower_last", "② 직전 기록이 평소보다 느렸던 사람"), ("ex3_no_5km", "③ 자격 미달(5km 기록 없음)")]
demo_blocks = []
for f, title in ex:
    r = subprocess.run([sys.executable, "-I", "demo/coach_demo.py", f"demo/examples/{f}.csv", "--asof", "2025-03-03T00:00:00"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", check=True)
    demo_blocks.append((f, title, r.stdout.strip()))

# ---- 포함률·성능
chk = txt("docs/demo_interval_check.md")
cov = re.findall(r"\| (cold-start test|personalized 평가) \| (\d+) \| (\d+) \| ([\d.]+)% \|", chk)
assert len(cov) == 2
perf = txt("docs/demo_performance.md")
prow = {}
for line in perf.splitlines():
    m = re.match(r"\| (.+?) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d,]+) \| ([\d,]+) \|", line)
    if m:
        prow[m.group(1)] = m.groups()[1:]
b2c = prow["B2 데모: 자격 판정·예측·문구·숫자 검사(CSV 읽기 제외)"]

L = []
A = L.append
A("# 5km 기록 변화 예측과 '정직한 기대치 코치'")
A("")
A("서강대학교 가상융합전문대학원 「AX 이론과 활용」 수업 프로젝트입니다(FitRec/Endomondo 러닝 기록 사용).")
A("")
A("**연구 질문**: 지난 4주 동안 몇 번, 얼마나 멀리, 얼마나 쉬면서 달렸는지로 다음 7일 안 5km 기록의 변화를 예측할 수 있는가(상관관계까지만 주장하며 인과가 아니다).")
A("")
A(f"**주 결론 (최종 평가 커밋 49c02f8, test 1회)**: 지난 4주의 거리·횟수·공백 정보를 넣은 Ridge·GBM·MLP는 cold-start test({n_cold.group(2)}명)와 personalized({n_pers.group(2)}명) 모두에서 "
  f"'직전 기록이 평소보다 느리면 평소 쪽으로 돌아온다'는 평균 회귀 규칙(B2)보다 사용자 평균 MAE가 통계적으로 낮지 않았고(6개 비교 모두 CI 상한이 0 이상, 모델 − B2 점추정 {lo_d:+.1f}~{hi_d:+.1f}초), "
  "이는 동등성의 증명이 아니라 이 자료에서 B2를 이긴다는 증거를 찾지 못했다는 뜻이다.")
A("")
A("![사용자 평균 MAE와 95% CI](docs/figures/fig1_mae_ci.png)")
A("")
A("그림 1: 사용자 평균 MAE(초, 낮을수록 좋음)와 사용자 단위 cluster bootstrap 95% CI. 회색 = 기준치(B0 변화 없음, B1, B2 평균 회귀), 파랑 = 기본 입력 모델. 상세: [`docs/final_results_5km.md`](docs/final_results_5km.md).")
A("")
A("![공백 코호트](docs/figures/fig3_gap_cohort.png)")
A("")
A("그림 3: **사후 분석**. 연속한 두 5km 사이의 관측된 러닝 기록의 공백 구간별 기록 변화(둘째 − 첫째) 중앙값과 95% CI. 쌍 30개 미만 구간은 흐리게 표시(참고, 표본 부족). 상관만 보며 인과·처방이 아니다. 상세: [`docs/within_person_results_5km.md`](docs/within_person_results_5km.md). 그림 2(정답 분포)·4(민감도 요약)는 [`docs/figures/`](docs/figures/).")
A("")
A("## 시연 프로토타입: 정직한 기대치 코치 (`demo/coach_demo.py`)")
A("")
A("위 결론에 따라 모델 대신 **평균 회귀 규칙(B2)만** 사용한다. 직전 28일 안 달리기 4회 이상과 유효 5km 기록 1회 이상이 있어야 예측하고, 부족하면 어떤 조건이 부족한지만 말한다. 예상 변화를 기본 변화량 a(LAD 절편, 평균이 아니라 '직전 기록이 평소와 같을 때'의 조건부 중앙값) 항과 평소 대비 보정 b×(직전 − 평소) 항으로 나눠 소수 1자리 초 값(합계만 정수 초)과 함께 보여주고, 80% 범위는 학습 자료의 B2 잔차 10%·90% 분위수로 정했다(문구에 테스트 실제 포함률을 함께 표기). 출력 문구 속 모든 숫자는 계산값과 일치하는지 자동 검사한다(불일치 시 오류). 표준 라이브러리만 쓴다.")
A("")
A("```bash")
A("python demo/coach_demo.py demo/examples/ex2_slower_last.csv --asof 2025-03-03T00:00:00")
A("python demo/test_coach_demo.py          # 단위 테스트(pytest 없이 실행 가능, 10개)")
A("python demo/make_examples.py            # 합성 예시 CSV 재생성(seed 고정)")
A("python demo/make_b2_params.py           # b2_params.json과 docs/demo_interval_check.md 재생성(로컬 data/ 필요)")
A("python demo/bench_demo.py               # docs/demo_performance.md 재생성")
A("```")
A("")
A("입력 CSV 열: `start_time, end_time, distance_km, duration_sec, avg_hr, t5_sec`(5km 세션만 t5_sec = 5.000km 도달 시간, 그 외 비움). 예시 3개는 **직접 만든 합성 데이터**이며 실제 데이터와 무관하다(데이터 재배포 금지).")
A("")
for f, title, out in demo_blocks:
    A(f"**{title}** (`demo/examples/{f}.csv`, 기준 시점 2025-03-03)")
    A("")
    A("```text")
    A(out)
    A("```")
    A("")
A(f"- **80% 범위 사후 점검**(범위는 학습 자료로만 정했고 점검 후 바꾸지 않음, [`docs/demo_interval_check.md`](docs/demo_interval_check.md)): 실제 포함률 cold-start test {cov[0][3]}%({cov[0][1]}개), personalized {cov[1][3]}%({cov[1][1]}개). 두 평가 모두 목표 80%보다 낮으며(범위가 새 자료에서 더 자주 벗어남) 원인은 미확인이다.")
A(f"- **성능(PC CPU 한 대, 모바일 아님)**: B2 예측 1회(CSV 읽기 제외) 중앙값 {b2c[0]} ms, tracemalloc 최대 {b2c[3]}바이트, 파라미터 파일 {b2c[4]}바이트. GBM·MLP 비교와 환경은 [`docs/demo_performance.md`](docs/demo_performance.md).")
A("- 이 데모는 원인 분석이나 훈련 처방이 아니며, 모델 정확도의 이점을 주장하지 않는다(B2가 모델과 구분되지 않아 가장 단순한 규칙을 택했다).")
A("")
A("## 실행 방법")
A("")
A("```bash")
A("pip install -r requirements.txt        # numpy, pandas, scipy, torch(CPU), matplotlib (Python 3.14.7에서 검증)")
A("# endomondoHR.json을 프로젝트 루트에 둔다(저장소에 포함되지 않음). 이후 단계는 data/ 파생 캐시가 필요하다.")
A("python 01_parse_run_sessions.py        # 러닝 세션 추출")
A("# 02~04, 06, 16~18 등 중복 제거·병합 단계(각 스크립트 docstring 참고, 아래 '이전 과제' 재현 방법과 같음)")
A("python run_all.py --all                # 25~34(이전 과제), 35~41(5km 데이터 점검·사전 선언 이후 단계), 43·44(사후 분석·민감도), 45(그래프)")
A("python run_all.py --only 43,44,45     # 5km 사후 분석·민감도·그래프만")
A("```")
A("")
A("- `42_final_eval_5km.py`(test 1회 최종 평가)는 `run_all`에 넣지 않았다. 최종 평가는 한 번만 실행했고(`data/final_eval_done.flag`가 있으면 중단) 결과는 [`docs/final_results_5km.md`](docs/final_results_5km.md)에 있다.")
A("- `run_all.py` 단계별 소요 시간·옵션은 아래 이전 과제 절의 표를 따른다. 그래프 45번은 결과 문서와 로컬 예측값을 읽기만 한다.")
A("- 데이터 캐시 `data/`와 행별 예측값은 커밋하지 않는다.")
A("")
A("## 데이터 안내")
A("")
A("- **FitRec/Endomondo 데이터셋** (UCSD McAuley Lab), `endomondoHR.json`(raw). 링크: https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html")
A("- 인용: Jianmo Ni, Larry Muhlstein, Julian McAuley, \"Modeling heart rate and activity data for personalized fitness recommendation\", The Web Conference (WWW), 2019.")
A("- **학술 용도 한정(academic use only), 재배포 금지.** 이 저장소에는 데이터 파일과 파생 사용자·세션 단위 표를 포함하지 않는다. 그래프·문서에는 집계 값만 있고, 데모 예시는 합성 데이터다. 사용하려면 출처에서 직접 내려받아야 한다.")
A("- 원본에는 날씨와 러닝 전 사용자가 입력한 계획 거리가 없다. `start_time`의 시각 기준(UTC 여부)은 판정 불가이며 기준 시점 격자의 가정으로만 쓴다.")
A("")
A("## 결과 문서와 커밋 순서")
A("")
A("저장소: https://github.com/yunseo1018/running-pace-coach. 아래 순서로 결과를 보기 전에 정의를 먼저 고정했다.")
A("")
A("| 순서 | 커밋 | 내용 | 문서 · 스크립트 |")
A("|---|---|---|---|")
A("| 1 | 3fad402 | 사전 선언 확정(타깃 T5, 기준치 B0/B1/B2, 통과 기준, 분할, 민감도 S1~S5 18항) | [`predeclaration_5km_final.md`](docs/predeclaration_5km_final.md), 초안 [`week4_predeclaration_5km.md`](docs/week4_predeclaration_5km.md) |")
A("| 2 | 337e6d9 | 5km 데이터셋 구축·개수·train 내부 기준치(test 정답 미계산) | [`dataset_counts_5km.md`](docs/dataset_counts_5km.md), [`baseline_results_train.md`](docs/baseline_results_train.md) · `38`, `39` |")
A("| 3 | 9fb0d8f | 사전 선언 보충 | [`predeclaration_5km_addendum.md`](docs/predeclaration_5km_addendum.md) |")
A("| 4 | 84670d6 | train 내부 모델 비교(Ridge/GBM/MLP, 하이퍼파라미터 선택) | [`model_cv_results_5km.md`](docs/model_cv_results_5km.md) · `40` |")
A("| 5 | f1857fb (태그 `freeze-5km-v1`) | **평가 동결**(방법·결론의 위계를 test 평가 전에 고정) | [`evaluation_freeze_5km.md`](docs/evaluation_freeze_5km.md) |")
A("| 6 | 70f10fe | 탐색적 E-MLP(train 내부, 주 결론에 쓰지 않음) | `41` |")
A("| 7 | **49c02f8** | **최종 평가**: cold-start test·personalized 각 1회 | [`final_results_5km.md`](docs/final_results_5km.md) · `42` |")
A("| 8 | 2e13216 | 결과 열람 기록 | [`results_viewed_log.md`](docs/results_viewed_log.md) |")
A("| 9 | 49b5f2b | 사후 분석·민감도 실행 사양 고정(분석 전) | [`posthoc_spec_5km.md`](docs/posthoc_spec_5km.md) |")
A("| 10 | 1174cec | **사후 분석**: 같은 사람 안 관계, 공백 코호트 | [`within_person_results_5km.md`](docs/within_person_results_5km.md) · `43` |")
A("| 11 | 106c0ce | **사전 지정 민감도** S1~S5(주 결론은 바뀌지 않음) | [`sensitivity_results_5km.md`](docs/sensitivity_results_5km.md) · `44` |")
A("| 12 | fd228c5 | 결과 그래프 4개 | [`docs/figures/`](docs/figures/) · `45` |")
A("| 13 | fa15c82 | 시연 프로토타입 | [`demo/`](demo/), [`demo_interval_check.md`](docs/demo_interval_check.md) |")
A("| 14 | b91bbd7 | 데모 성능 측정 | [`demo_performance.md`](docs/demo_performance.md) · `demo/bench_demo.py` |")
A("")
A("그 밖의 5km 과제 문서: [`week4_data_audit.md`](docs/week4_data_audit.md)(데이터 점검), [`week5_target_check.md`](docs/week5_target_check.md)(타깃 단위·포함/제외 단계). 이전 과제(다음 러닝 페이스 예측) 문서는 아래 부록과 [`docs/`](docs/)에 있다.")
A("")
A("## 외부 코드·라이브러리 출처")
A("")
A("- 이 저장소의 Ridge(`ridge_lib.py`), GBM(`gbt_lib.py`), haversine 거리, 5km 보간, 평가·bootstrap 코드는 직접 작성했다. 다른 저장소에서 복사·수정한 코드는 없다(저장소 안 검색으로 출처 표기가 없음을 확인).")
A("- 라이브러리: numpy, pandas, scipy(LAD는 `scipy.optimize.linprog`의 HiGHS), torch(MLP), matplotlib(그래프). 버전은 [`requirements.txt`](requirements.txt)와 [`docs/environment.md`](docs/environment.md). scikit-learn은 사용하지 않는다.")
A("- 데이터와 인용은 위 '데이터 안내'를 따른다.")
A("")
A("## 주요 한계 (5km 과제)")
A("")
A("- 비유의는 동등성의 증명이 아니다. 평가 대상은 cold-start test 352개, personalized 442개로 CI가 넓다.")
A("- 같은 사람 안 분석·공백 코호트는 사후 분석이며 상관만 보인다. 표본이 작은 공백 구간은 참고용이다.")
A("- `start_time`의 시각 기준은 미확인이고, T5는 경과 시간 기준이다. 휴식·신호 대기가 포함된다.")
A("- 80% 범위의 실제 포함률은 사후 점검 값이며 데모는 연구 결과의 시연일 뿐 코칭 효과 검증이 아니다. 실제 사용자 반응은 관찰하지 않았다.")
A("")
A("---")
A("")
A(MARK)
A("")
A("# 부록: 이전 과제 — 다음 러닝 페이스 예측과 목표 기반 코칭 (3주차까지)")
A("")
text = "\n".join(L) + "\n" + old
new = text.replace("\r\n", "\n").encode("utf-8")
cur = rd.read_bytes().replace(b"\r\n", b"\n")
if "--check" in sys.argv:
    print("README 일치" if cur == new else "README 불일치")
    sys.exit(0 if cur == new else 1)
rd.write_bytes(new)
print("README written", len(new), "bytes")
