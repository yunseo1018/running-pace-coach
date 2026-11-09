# -*- coding: utf-8 -*-
"""
보고용 정리 표 2(교수 피드백 11/9 2번) + 분할 구분 확인(3번): docs/interval_by_condition.md.
- 새 학습·새 분석 없음. 사후 분석 56번(scripts/56_b2_failure_5km.py, 사양 f5ba594)의 같은 층 정의·자료·bootstrap(2,000회, seed 42)을 그대로 불러 와
  관측 수, 사용자 수, 포함률과 사용자 단위 95% CI, 평균 폭, 평가 대상이 1개뿐인 사용자 수를 정리한다. 층 경계는 56번 사양과 같다(결과를 본 뒤 바꾸지 않음).
- 포함률·관측 수·사용자 수는 docs/b2_failure_5km.md의 값과 같음을 assert한다. 시연 범위·B2 식·파라미터는 바꾸지 않는다.
- 4절은 읽기 전용 확인이다(코드 줄 인용 + assert). 고치지 않는다.
ID·행 값은 싣지 않는다(집계만). 사용법(저장소 루트): python scripts/58_interval_by_condition_5km.py   (run_all에는 넣지 않음; 원본 JSON 필요, 약 4분)
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import ridge_lib  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m56 = _load("scripts/56_b2_failure_5km.py", "m56")
m55, m43, m42, feas = m56.m55, m56.m43, m56.m42, m56.feas
md_table = m42.md_table
DOC = Path("docs") / "interval_by_condition.md"
B2_DOC = Path("docs") / "b2_failure_5km.md"
WEEK = feas.WEEK
SELECTED = [("(a) 과거 유효 5km 개수", "1개(x = 0)"), ("(a) 과거 유효 5km 개수", "2~4개"), ("(b) |T5(기준) − m| (초)", "120초 초과"), ("(e) 기준 기록 세션 안 정지 시간 (초)", "300초 초과")]
WANT_DIMS = ["(a)", "(b)", "(e)"]
ALIAS = {"0초": "0초", "0초 초과 ~ 30초": "0 초과 ~ 30초", "30초 초과 ~ 120초": "30 초과 ~ 120초", "120초 초과": "120초 초과",
         "0~60초": "0~60초", "60초 초과 ~ 300초": "60 초과 ~ 300초", "300초 초과": "300초 초과"}


def src_line(path, needle):
    """소스 파일에서 needle을 포함한 첫 줄의 (줄 번호, 줄 내용)."""
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return i, line.strip()
    raise AssertionError(f"{path}에서 줄을 찾지 못함: {needle}")


def quote(path, needle):
    n, text = src_line(path, needle)
    return f"`{path}:{n}`: `{text}`"


ROW = re.compile(r"^\| (train 폴드 밖|test cold-start[^|]*?) \| (.+?) \| ([\d,]+) \| ([\d,]+) \| [\d.]+ \| [\d.]+ \| [^|]+ \| ([\d.]+)% \[([\d.]+)%, ([\d.]+)%\] \|")


def read_b2_doc():
    """b2_failure_5km.md의 층 행: {(자료, 차원 머리, 층): (n, users, cov, lo, hi)}."""
    out, dim = {}, None
    for line in B2_DOC.read_text(encoding="utf-8").splitlines():
        m = re.match(r"## \d+\. (\(\w\))", line)
        if m:
            dim = m.group(1)
        r = ROW.match(line)
        if r and dim:
            ds = "train" if r.group(1).startswith("train") else "test"
            out[(ds, dim, r.group(2))] = (int(r.group(3).replace(",", "")), int(r.group(4).replace(",", "")), float(r.group(5)), float(r.group(6)), float(r.group(7)))
    return out


def one_user_count(sub):
    return int((sub.groupby("uid").size() == 1).sum())


def build_frames():
    d, trs, te, grp, cache = m43.load_base()
    ds = m43.main_dataset(d, grp, cache)
    m43.verify_against_final(ds, trs, te)
    tg = m56.collect_targets(d, grp, cache)
    j = ds.merge(tg, on=["uid", "t"], how="left", validate="one_to_one")
    assert len(j) == 2072 and j.ref_sid.notna().all()
    ref_t5 = np.array([cache["t5"][int(s)] for s in j.ref_sid])
    assert np.array_equal(ref_t5, j.t5_ref.to_numpy()), "기준 기록 정합 실패"
    j["abs_x"] = j.t5_ref_minus_m.abs()
    j["days_ref"] = (j.t - j.ref_end) / m56.DAY
    parse = feas.import_parser()
    stops = m56.stop_seconds(parse, sorted(set(j.ref_sid.astype(int))))
    j["stop_sec"] = [stops[int(s)] for s in j.ref_sid]
    tr = j[j.uid.isin(trs)].sort_values(["uid", "t"]).reset_index(drop=True)
    tr_file = pd.read_pickle(Path("data") / "5km_dataset_train.pkl").sort_values(["uid", "t"]).reset_index(drop=True)
    assert len(tr) == 1720 and np.array_equal(tr.uid.to_numpy(), tr_file.uid.to_numpy()) and np.array_equal(tr.t.to_numpy(), tr_file.t.to_numpy()) and np.array_equal(tr.y.to_numpy(), tr_file.y.to_numpy())
    q10, q90, _coefs, res = m55.oof_quantiles(tr_file)
    p = json.loads((Path("demo") / "b2_params.json").read_text(encoding="utf-8"))
    assert abs(q10 - p["resid_q10"]) < 1e-9 and abs(q90 - p["resid_q90"]) < 1e-9, "시연 범위와 다름"
    y_tr = tr.y.to_numpy(float)
    tr["e2"] = np.abs(res)
    tr["e0"] = np.abs(y_tr)
    tr["inside"] = (res >= q10) & (res <= q90)
    ts_ = j[j.uid.isin(te)].sort_values(["uid", "t"]).reset_index(drop=True)
    assert len(ts_) == 352 and ts_.uid.nunique() == 71
    yhat = p["a"] + p["b"] * ts_.t5_ref_minus_m.to_numpy(float)
    y_te = ts_.y.to_numpy(float)
    ts_["e2"] = np.abs(y_te - yhat)
    ts_["e0"] = np.abs(y_te)
    ts_["inside"] = (y_te >= yhat + q10) & (y_te <= yhat + q90)
    return dict(d=d, cache=cache, ds=ds, tr=tr, te=ts_, q10=q10, q90=q90, params=p, tr_file=tr_file)


def condition_tables(F):
    q10, q90 = F["q10"], F["q90"]
    width = q90 - q10
    doc = read_b2_doc()
    sets = {"train": F["tr"], "test": F["te"]}
    out_tables, sel_stats = [], {}
    for dname, fn, labels in m56.DIMS:
        head = dname.split(" ")[0]
        if head not in WANT_DIMS:
            continue
        rows = []
        for lb in labels:
            r = {"조건": ALIAS.get(lb, lb)}
            for ds_name, ds_label in (("train", "train 폴드 밖 (개발)"), ("test", "test cold-start (탐색, 이미 열람)")):
                df = sets[ds_name]
                cat = fn(df)
                assert cat.notna().all() and sum(int((cat == x).sum()) for x in labels) == len(df)
                sub = df[cat == lb]
                if len(sub) == 0:
                    r.update({f"[{ds_label}] 관측": "0", f"[{ds_label}] 사용자": "0", f"[{ds_label}] 포함률 [95% CI]": "-", f"[{ds_label}] 1개뿐 사용자": "-"})
                    continue
                s = m56.stratum_stats(sub)
                dd = doc[(ds_name, head, lb)]
                assert (s["n"], s["users"]) == (dd[0], dd[1]) and abs(100 * s["cov"] - dd[2]) < 0.05 and abs(100 * s["cov_ci"][0] - dd[3]) < 0.05 and abs(100 * s["cov_ci"][1] - dd[4]) < 0.05, (ds_name, head, lb)
                low = " (표본 부족)" if s["n"] < m56.MIN_N else ""
                r.update({f"[{ds_label}] 관측": f"{s['n']:,}{low}", f"[{ds_label}] 사용자": f"{s['users']:,}", f"[{ds_label}] 포함률 [95% CI]": m56.read_cell(s, "cov"),
                          f"[{ds_label}] 1개뿐 사용자": f"{one_user_count(sub):,}"})
                sel_stats[(ds_name, head, lb)] = (s, one_user_count(sub))
            r["평균 폭(초)"] = f"{width:.1f}"
            rows.append(r)
        # 열 이름 정리: 사용자/포함률/1개뿐 키가 자료마다 달라 위 update에서 자료별 접두사를 붙였다
        out_tables.append((dname, rows))
    return out_tables, sel_stats, width


def split_section(F):
    """3절: 분할 구분 확인(읽기 전용). 반환 (마크다운 줄 목록, 요약 dict)."""
    ds, d, cache = F["ds"], F["d"], F["cache"]
    p = F["params"]
    final = pd.read_pickle(Path("data") / "final_preds_5km.pkl")
    p_tr, p_ev, p_users = m42.personalized_split(ds)
    assert (len(p_tr), len(p_ev), p_users) == (1531, 442, 250)
    # 저장된 최종 평가 예측과 행 순서가 같음
    assert np.array_equal(final["pers"]["uid"], p_ev.uid.to_numpy()) and np.array_equal(final["pers"]["t"], p_ev.t.to_numpy())
    # (a) B2 a·b를 어느 자료로 적합했는가
    a_p, b_p = m42.lad_fit(p_tr.t5_ref_minus_m.to_numpy(), p_tr.y.to_numpy())
    pred_b2 = a_p + b_p * p_ev.t5_ref_minus_m.to_numpy()
    assert np.allclose(final["pers"]["preds"]["B2"], pred_b2, atol=1e-9), "저장된 personalized B2가 p_tr 적합과 다름"
    assert (a_p, b_p) != (p["a"], p["b"]) and (abs(a_p - p["a"]) > 1e-6)
    assert set(zip(p_tr.uid, p_tr.t)).isdisjoint(set(zip(p_ev.uid, p_ev.t)))
    # (c) 학습 행의 정답 7일 구간이 첫 평가 시점 전에 끝나는가
    first_eval = {}
    for uid, g in ds.groupby("uid", sort=True):
        g = g.sort_values("t")
        if len(g) < 2:
            continue
        ne = max(1, int(round(0.2 * len(g))))
        first_eval[uid] = int(g.t.iloc[len(g) - ne])
    assert len(first_eval) == 250 and all(first_eval[u] == int(p_ev[p_ev.uid == u].t.min()) for u in list(first_eval)[:250])
    fe = p_tr.uid.map(first_eval).to_numpy()
    assert (p_tr.t.to_numpy() + WEEK <= fe).all(), "학습 행의 [t, t+7일)이 첫 평가 시점 뒤로 걸침"
    max_gap_ok = int((fe - (p_tr.t.to_numpy() + WEEK)).min())
    # 정답 세션 자체의 종료 시각(세션이 7일 구간 끝을 넘겨 끝날 수 있음)
    ans_end_after = 0
    sess = {int(u): g for u, g in d.groupby("user_id")}
    ext = cache["ext"]
    for uid, t, f0 in zip(p_tr.uid, p_tr.t, fe):
        g = sess[int(uid)]
        s = g["start_time"].to_numpy(np.int64)
        e = s + g["duration_sec"].to_numpy(np.int64)
        fut = np.flatnonzero(g["cand"].to_numpy() & (s >= t) & (s < t + WEEK))
        assert len(fut) > 0
        a = int(fut[0])
        assert s[a] < t + WEEK <= f0
        if e[a] > f0:
            ans_end_after += 1
    # (b) m 계산에 예측 시점 이후 기록이 들어가지 않았는가: 모든 평가 대상(2,072행)을 독립 구현으로 다시 계산
    n_checked = n_single = 0
    for r in ds.itertuples():
        g = sess[int(r.uid)]
        s = g["start_time"].to_numpy(np.int64)
        e = s + g["duration_sec"].to_numpy(np.int64)
        ok = g["cand"].to_numpy() & np.array([not ext.get(int(x), False) for x in g["session_id"]])
        sel = np.flatnonzero(ok & (e < r.t))                    # 예측 시점 t 전에 끝난 세션만
        t5 = np.array([cache["t5"][int(x)] for x in g["session_id"].to_numpy()[sel]])
        assert e[sel].max() < r.t
        m_ind = float(np.median(t5))
        if len(sel) == 1:
            m_ind = float(t5[0])
            n_single += 1
        assert len(sel) == r.n_past and abs(m_ind - r.m) < 1e-9, (r.uid, r.t)
        # 시작이 t 이후인 세션을 모두 지워도 같다(미래 기록 비의존)
        keep = s < r.t
        sel2 = np.flatnonzero((ok & (e < r.t)) & keep)
        assert np.array_equal(sel, sel2)
        n_checked += 1
    # 범위 분위수와 피드백 분위수의 분할
    tr_keys = set(zip(F["tr_file"].uid, F["tr_file"].t))
    ev_in_coldtrain = sum((u, t) in tr_keys for u, t in zip(p_ev.uid, p_ev.t))
    users_in_coldtrain = len(set(p_ev.uid) & set(F["tr_file"].uid))
    L = ["## 4. 분할 구분 확인 (읽기 전용: 코드 줄 인용과 assert 결과, 아무것도 고치지 않음)", "",
         "### 4-1. 범위 분위수와 피드백 분위수를 만든 교차검증은 cold-start용(사용자 단위)이다", "",
         "- 범위 분위수(q10·q90, `demo/b2_params.json`의 `resid_q10`·`resid_q90`)와 피드백 분위수(`demo/feedback_quantiles.json`)는 모두 `oof_quantiles`가 만든 **train 사용자 단위 GroupKFold(5, seed 42) 폴드 밖 잔차**에서 나온다.",
         "  - " + quote("scripts/55_interval_calibration_5km.py", "def oof_quantiles(tr):"),
         "  - " + quote("scripts/55_interval_calibration_5km.py", "for trm, vam in ridge_lib.group_kfold_indices(uid, N_FOLDS, SEED):"),
         "  - " + quote("scripts/55_interval_calibration_5km.py", "assert set(uid[trm]).isdisjoint(set(uid[vam]))"),
         "  - " + quote("ridge_lib.py", "def group_kfold_indices(user_ids, n_folds, seed):") + " (docstring: \"user_id 단위로 fold를 나눠서, 같은 사용자의 모든 행이 항상 같은 fold에 들어가게 한다\")",
         "  - " + quote("demo/make_feedback_quantiles.py", "q10, q90, _coefs, res = m55.oof_quantiles(tr)"),
         "  - " + quote("demo/make_b2_params.py", "m55.oof_quantiles(tr)"),
         "- 검증 폴드의 사용자는 학습 폴드에 없다(위 assert). 따라서 이 분위수는 \"학습에 없던 사용자\"의 잔차 분포, 즉 **cold-start 조건**의 잔차 분포다. 같은 사용자의 이전 기록으로 적합한 모델의 잔차(personalized 조건)가 아니다.", "",
         "### 4-2. personalized 평가에서 확인한 것 (assert 통과)", "",
         f"평가 {len(p_ev)}개·{p_users}명, 학습 {len(p_tr):,}개(사용자 안 시간순 앞쪽 80%, 단 [t, t+7일)이 첫 평가 시점 전에 끝나는 행만). 최종 평가 예측(`data/final_preds_5km.pkl`)의 행 순서와 같음을 확인했다.", "",
         "**(a) B2의 a·b를 어느 자료로 적합했는가**: personalized 학습 행(`p_tr`, 1,531개)만으로 적합했다.",
         "- " + quote("42_final_eval_5km.py", "a, b = lad_fit(tr.t5_ref_minus_m.to_numpy(), tr.y.to_numpy())") + " (`predict_all(tr, ev, hp)`의 `tr`)",
         "- " + quote("42_final_eval_5km.py", "def personalized_split(ds):"),
         f"- assert: `p_tr`로 다시 적합한 B2 예측이 저장된 personalized B2 예측과 1e-9 이내로 같다. 적합값 a = {a_p:.3f}초, b = {b_p:.4f} (시연의 cold-start 적합값 a = {p['a']:.3f}초, b = {p['b']:.4f}와 다르다). 학습 행과 평가 행에 겹치는 (사용자, 시점)은 없다(assert).", "",
         "**(b) m 계산에 예측 시점 이후 기록이 들어갔는가**: 들어가지 않았다.",
         "- " + quote("42_final_eval_5km.py", "past = np.flatnonzero(iscand & (e < t) & ~ext)") + " (m = 이 세션들의 T5 중앙값, 1개뿐이면 기준 기록)",
         f"- assert: 모든 평가 대상 {n_checked:,}행(cold-start·personalized 공통 자료)에 대해 독립 구현으로 m·과거 5km 개수를 다시 계산해 데이터셋 값과 같고(과거 5km 1개뿐인 {n_single:,}행 포함), 사용된 세션의 종료 시각이 모두 t보다 이르며, 시작이 t 이후인 세션을 모두 지워도 같은 세션이 선택된다.", "",
         "**(c) 학습 행의 정답 7일 구간이 첫 평가 시점 전에 끝나는가**: 끝난다.",
         "- " + quote("42_final_eval_5km.py", "tr_idx += list(pre.index[(pre.t + WEEK) <= first_eval])"),
         f"- assert: 학습 {len(p_tr):,}행 모두 t + 7일 ≤ 그 사용자의 첫 평가 시점이다(가장 작은 여유 {max_gap_ok:,}초). 정답 세션의 시작도 모두 [t, t+7일) 안이라 첫 평가 시점 전이다.",
         f"- 참고(고치지 않음): 정답 세션은 7일 구간 안에서 *시작*하면 되므로, 정답 세션의 *종료* 시각이 첫 평가 시점을 넘는 학습 행이 {ans_end_after}개 있다(정답 T5는 세션 시작 후 5km 도달 시간이라 종료 시각과 무관하게 정해지지만, 엄격한 \"종료 전\" 기준으로는 {ans_end_after}개가 걸친다).", "",
         "### 4-3. 시연의 cold-start 범위를 personalized에 그대로 적용한 것은 맞는가", "",
         "**판정: 분할 불일치가 있다(고치지 않고 보고한다).**", "",
         "- 시연의 B2 a·b와 범위 분위수는 cold-start 자료(train 평가 대상 1,720개·278명 전체)에서 정했고 그 잔차는 사용자 단위 교차검증(학습에 없던 사용자)의 잔차다(4-1).",
         f"- personalized 평가 {len(p_ev)}개 중 **{ev_in_coldtrain}개**({100 * ev_in_coldtrain / len(p_ev):.1f}%)가 시연의 a·b와 범위를 만든 train 평가 대상 1,720개에 **그대로 들어 있다**(평가 사용자 {p_users}명 중 {users_in_coldtrain}명이 train 사용자). 따라서 `docs/interval_calibration_5km.md`·`docs/demo_interval_check.md`의 personalized 포함률은 학습 자료와 독립이 아니다(55번도 같은 한계를 적었다).",
         f"- 최종 평가(49c02f8)의 personalized B2는 `p_tr` 적합(a = {a_p:.3f}, b = {b_p:.4f})이지만, 범위 포함률 점검의 personalized 열은 시연의 cold-start 적합(a = {p['a']:.3f}, b = {p['b']:.4f})으로 ŷ를 만들었다. 같은 \"personalized\"라는 이름 아래 B2가 두 가지다.",
         "- personalized 조건에 맞는 범위를 따로 정하려면 사용자 안 시간순 분할로 잔차를 모아야 한다. **이번 정리에서는 계산하지 않았고**(새 분석·새 사양 필요), 시연 범위도 바꾸지 않았다. cold-start 열의 포함률은 이 불일치의 영향을 받지 않는다.", ""]
    return L, dict(overlap=ev_in_coldtrain, a_p=a_p, b_p=b_p, ans_end_after=ans_end_after)


def main():
    F = build_frames()
    tables, sel_stats, width = condition_tables(F)
    s_e = sel_stats[("train", "(e)", "300초 초과")]
    t_e = sel_stats[("test", "(e)", "300초 초과")]
    out = ["# 보고용 정리 표 2: 조건별 명목 80% 범위 포함률 + 분할 구분 확인", "",
           "생성: `scripts/58_interval_by_condition_5km.py`. 새 학습·새 분석 없이 사후 분석 `scripts/56_b2_failure_5km.py`(사양 `docs/posthoc_spec_b2_failure_5km.md`, 커밋 f5ba594)의 층 정의·자료·bootstrap을 그대로 불러 와 보고용 열을 맞췄다. "
           "층 경계는 결과를 본 뒤 바꾸지 않았고, **범위를 좁히거나 바꾸지 않는다**(시연의 B2 식·파라미터·범위 불변). 관측 수·사용자 수·포함률·CI가 `docs/b2_failure_5km.md`의 값과 같음을 assert로 확인했다. 시각은 start_time 원값(기준 미확인)이며 상관관계까지만 주장한다. "
           "모든 평가는 \"다음 7일 안에 적격 5km 기록이 관측된 경우\"에 조건부다. ID·행 값은 싣지 않았다.", "",
           "## 0. 먼저 읽을 것", "",
           f"- **39.1%(정지 300초 초과, train 폴드 밖)**: 관측 **{s_e[0]['n']}개**, 사용자 **{s_e[0]['users']}명**, 포함률 {m56.read_cell(s_e[0], 'cov')}, 평가 대상이 1개뿐인 사용자 {s_e[1]}명. "
           f"같은 층의 test cold-start(탐색, 이미 열람): 관측 {t_e[0]['n']}개, 사용자 {t_e[0]['users']}명, 포함률 {m56.read_cell(t_e[0], 'cov')}"
           + (" — 평가 대상 30개 미만이라 표본 부족으로 해석하지 않는다." if t_e[0]["n"] < m56.MIN_N else "") + " 관측 수가 적은 층이고, 한 사용자의 여러 기준 시점이 함께 들어 있을 수 있다.",
           f"- **현재 범위는 고정 폭**(ŷ + [{F['q10']:.1f}, +{F['q90']:.1f}]초 = 폭 {width:.1f}초)이라 모든 조건에서 **평균 폭이 같다**. 표의 \"평균 폭\" 열은 모두 {width:.1f}초이고, 조건별로 폭을 바꾸는 규칙은 없다.",
           "- **개발 결과 vs 탐색 결과**: train 폴드 밖 열은 범위·조건별 안내 4개 층을 정하는 데 쓴 **개발 결과**이고, test cold-start 열은 이미 열람한 자료의 **탐색 결과**다(새 자료가 아니다).",
           "- **시연의 조건별 안내 4개 층**(과거 유효 5km 1개, 2~4개, 직전 기록이 평소와 120초 넘게 차이, 직전 기록 세션 정지 300초 초과)은 train 폴드 밖 결과(평가 대상 30개 이상이고 포함률 95% CI 상한이 80% 미만)를 **보고 고른 개발 결과**이며, **새 자료로 확인되지 않았다.** 여러 층을 보고 고르므로 우연히 걸린 층이 있을 수 있다(다중 비교 보정 없음). 이 표는 안내 층이나 범위를 바꾸는 근거로 쓰지 않는다.",
           f"- 포함률 = 정답 y가 명목 80% 범위(ŷ + [q10_oof, q90_oof]) 안에 든 평가 대상의 비율(층 안 풀링). 95% CI는 사용자 단위 bootstrap(2,000회, seed 42, 층마다 그 층의 사용자를 복원추출). CI가 80%를 포함하면 \"명목 80%에서 벗어났다고 확정할 수 없음\"이고, 상한이 80% 미만일 때만 \"명목 80%보다 낮음\"이라고 쓴다. 관측 수 30개 미만은 표본 부족이다.",
           "- \"1개뿐 사용자\" = 그 조건 안에서 평가 대상이 1개뿐인 사용자 수(그 사용자의 층 안 비율은 0% 또는 100%라 사용자별 해석에 한계가 있다).", ""]
    for i, (dname, rows) in enumerate(tables, start=1):
        out += [f"## {i}. {dname}", "", md_table(rows), ""]
    # 판독 목록
    out += ["### 판독 (위 규칙 적용: 관측 30개 이상, 포함률 CI가 80%와 겹치는지)", ""]
    for (ds_name, head, lb), (s, _one) in sel_stats.items():
        if s["n"] < m56.MIN_N:
            continue
        lo, hi = 100 * s["cov_ci"][0], 100 * s["cov_ci"][1]
        read = "명목 80%보다 낮음(CI 상한 < 80%)" if hi < 80 else ("명목 80%보다 높음(CI 하한 > 80%)" if lo > 80 else "명목 80%에서 벗어났다고 확정할 수 없음")
        out.append(f"- {'train 폴드 밖(개발)' if ds_name == 'train' else 'test cold-start(탐색)'} {head} {ALIAS.get(lb, lb)}: {100 * s['cov']:.1f}% [{lo:.1f}%, {hi:.1f}%] → {read}")
    out += ["", "- test cold-start에서 CI가 80%를 포함하는 층이 있어도 그 층에서 범위가 맞는다는 증명은 아니다(표본이 작고 CI가 넓다). 조건별 안내를 바꾸지 않는다.", ""]
    L, _info = split_section(F)
    out += L
    DOC.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("saved", DOC.name)


if __name__ == "__main__":
    main()
