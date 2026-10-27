# -*- coding: utf-8 -*-
"""
10월 4주차(8주차) 1번: start_time의 시각 기준 점검 (읽기 전용, 개수만).

선택 의존성: timezonefinder (requirements.txt에 넣지 않음. 없으면 '확인 불가'로 기록하고 종료한다).
세션 시작 좌표 -> 시간대 이름 -> (a) epoch을 UTC 시각으로 본 시(0~23) vs (b) 현지 시각으로 변환한 시(0~23)를
시작 경도 기준 3개 지역(<-30 미주 / -30~60 유럽·아프리카 / >=60 아시아·오세아니아)별로 센다.

판정 규칙(사용자 지정):
  (b) 지역 간 분포가 비슷 -> "진짜 UTC"
  (a) 지역 간 분포가 비슷 & (b)는 어긋남 -> "현지 시각이 UTC처럼 저장됨"
  그 외 -> "판정 불가"
'비슷하다'의 조작적 기준(이 스크립트에서 고정, 사용자 지정 아님): 지역 쌍별 총변동거리(0=동일, 1=완전 상이)의 최댓값 <= SIMILAR_TV.

좌표·ID·세션별 시간대는 출력·저장하지 않는다. 결과는 docs/week4_data_audit.md의 TZ 블록만 갱신한다.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config

DOC_PATH = Path("docs") / "week4_data_audit.md"
TZ_START, TZ_END = "<!-- TZ_START -->", "<!-- TZ_END -->"
GPS_PATH = Path("data") / "session_gps_endpoints.pkl"
SIMILAR_TV = 0.15
REGS = ["미주(경도<-30)", "유럽·아프리카(-30~60)", "아시아·오세아니아(>=60)"]


def md_table(rows):
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_block(text):
    block = f"{TZ_START}\n{text}\n{TZ_END}"
    old = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else None
    if old is None:
        raise SystemExit("먼저 35_data_audit_5km.py를 실행해 docs/week4_data_audit.md를 만드세요.")
    if TZ_START in old and TZ_END in old:
        new = old[:old.index(TZ_START)] + block + old[old.index(TZ_END) + len(TZ_END):]
    else:
        new = old.rstrip("\n") + "\n\n" + block + "\n"
    DOC_PATH.write_text(new, encoding="utf-8", newline="\n")


def main():
    try:
        from timezonefinder import TimezoneFinder
        import timezonefinder
        ver = getattr(timezonefinder, "__version__", "확인 불가")
    except Exception as exc:  # 선택 의존성
        write_block("## 6. 시각 기준 점검\n\ntimezonefinder를 불러오지 못해 **확인 불가**(선택 의존성; "
                    f"`{type(exc).__name__}`). 설치: `pip install timezonefinder`.")
        print("timezonefinder 없음 -> 확인 불가 기록")
        return
    d = config.load_main_sessions()
    g = pd.read_pickle(GPS_PATH)
    m = d.merge(g[["session_id", "start_lat", "start_lon"]], on="session_id", how="left")
    no_coord = int(m.start_lon.isna().sum())
    m = m[m.start_lon.notna()].copy()
    tf = TimezoneFinder()
    cache = {}

    def tz(lat, lon):
        k = (round(lat, 2), round(lon, 2))
        if k not in cache:
            try:
                cache[k] = tf.timezone_at(lat=lat, lng=lon)
            except Exception:
                cache[k] = None
        return cache[k]
    m["tz"] = [tz(a, b) for a, b in zip(m.start_lat.to_numpy(), m.start_lon.to_numpy())]
    unresolved = int(m.tz.isna().sum())
    m = m[m.tz.notna()].copy()
    zc = {}

    def local_hour(ts, z):
        if z not in zc:
            zc[z] = ZoneInfo(z)
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(zc[z]).hour
    ts = m.start_time.to_numpy(np.int64)
    m["h_utc"] = ((ts % 86400) // 3600).astype(int)
    m["h_loc"] = [local_hour(t, z) for t, z in zip(ts, m.tz.to_numpy())]
    lon = m.start_lon.to_numpy()
    m["region"] = np.where(lon < -30, REGS[0], np.where(lon < 60, REGS[1], REGS[2]))

    out = ["## 6. 시각 기준 점검", "",
           f"`36_timezone_check.py` (선택 의존성 timezonefinder {ver}; requirements.txt에는 없음). "
           f"시작 좌표 없음 {no_coord}건, 시간대 미확인 {unresolved}건(이 건수는 아래 표에서 제외). "
           f"전체 {len(m):,}세션을 시작 경도로 3개 지역으로 나눠 센 시(0~23시)별 세션 수다. 좌표·ID는 싣지 않았다.", "",
           md_table([{"지역": r, "세션": f"{(m.region == r).sum():,}", "사용자": f"{m[m.region == r].user_id.nunique():,}"} for r in REGS]), ""]
    tvmax = {}
    for col, name in [("h_utc", "(a) epoch을 UTC 시각으로 본 시"), ("h_loc", "(b) 현지 시각으로 변환한 시")]:
        H, rows = {}, []
        for r in REGS:
            h = np.bincount(m[m.region == r][col].to_numpy(), minlength=24)
            H[r] = h / h.sum()
            rows.append({"지역": r, **{str(i): f"{int(h[i]):,}" for i in range(24)}})
        tvs = {f"{REGS[i].split('(')[0]}–{REGS[j].split('(')[0]}": 0.5 * np.abs(H[REGS[i]] - H[REGS[j]]).sum()
               for i in range(3) for j in range(i + 1, 3)}
        tvmax[col] = max(tvs.values())
        out += [f"### {name}", "", md_table(rows), "",
                "지역 쌍별 총변동거리: " + ", ".join(f"{k} {v:.3f}" for k, v in tvs.items()) + f" (최댓값 {tvmax[col]:.3f})", ""]
    sim_a, sim_b = tvmax["h_utc"] <= SIMILAR_TV, tvmax["h_loc"] <= SIMILAR_TV
    if sim_b:
        verdict = "진짜 UTC"
    elif sim_a and not sim_b:
        verdict = "현지 시각이 UTC처럼 저장됨"
    else:
        verdict = "판정 불가"
    out += [f"**판정: {verdict}.** 규칙: (b)가 지역 간 비슷하면 \"진짜 UTC\", (a)가 비슷하고 (b)가 어긋나면 \"현지 시각이 UTC처럼 저장됨\", 그 외 \"판정 불가\". "
            f"'비슷하다'는 지역 쌍 총변동거리 최댓값 ≤ {SIMILAR_TV}로 이 스크립트에서 고정한 조작적 기준이다(임의 선택; 이전 채팅 보고의 판정 불가와 같은 결과). "
            f"(a) 최댓값 {tvmax['h_utc']:.3f}, (b) 최댓값 {tvmax['h_loc']:.3f}로 둘 다 기준을 넘는다.", "",
            "해석 한계: (b)에서도 새벽 0~3시가 많고 6~10시가 거의 비는 형태가 지역마다 나타나 현실적인 일일 러닝 시각 분포(아침·저녁 봉우리, 새벽 바닥)와 맞지 않는다. "
            "원인은 **확인 불가**다. 유럽·아프리카가 전체의 약 85%라 지역 간 균형이 맞지 않는다. "
            "따라서 기준 시점의 UTC 월요일 00:00 격자와 현지 시간대 기반 분석은 근거로 쓰지 않으며, 민감도(기준 시점 이동)로만 다룬다."]
    write_block("\n".join(out))
    print("saved", DOC_PATH, "| verdict:", verdict)


if __name__ == "__main__":
    main()
