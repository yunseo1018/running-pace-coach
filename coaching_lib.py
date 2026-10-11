# -*- coding: utf-8 -*-
"""
코칭 1단계 모듈 (10월 2주차 10단계). 예측기와 완전히 분리: 이 파일은 모델/데이터를 import하지 않고,
"현재 페이스, 목표 변화율, 주차, 예측 페이스"라는 숫자만 받아 체크포인트/격차/카테고리/문구를 계산한다.
LLM·RAG를 쓰지 않는 고정 템플릿 방식이다.

**모든 목표는 합성 시나리오다(데이터에 사용자가 입력한 실제 목표가 없음). 실제 사용자 목표 기반
코칭 성능이 아니다.**

임계값(카테고리 경계)은 사전에 정한 설계값이며 결과를 보고 튜닝하지 않았다.
"""
import math

SECONDS_PER_WEEK = 7 * 86400.0

# 카테고리 경계: 격차(예측-체크포인트)를 '현재 페이스'로 나눈 비율 기준 (사전 고정, 튜닝 안 함)
AHEAD_MAX = -0.03      # 비율 <= -3%        -> 앞서감
ON_TRACK_MAX = 0.03    # -3% < 비율 <= +3%  -> 계획대로
SLIGHT_MAX = 0.10      # +3% < 비율 <= +10% -> 조금 뒤처짐 ; > +10% -> 크게 뒤처짐

CATEGORIES = ["앞서감", "계획대로", "조금 뒤처짐", "크게 뒤처짐"]

TEMPLATES = {
    "앞서감": [
        "{week}주차 체크포인트({cp})보다 {gap_abs}초/km 빠를 것으로 예상돼요. 이 흐름을 유지하면서 {dist}km 거리에 익숙해져 보세요.",
        "예상 페이스 {pred}/km로 이번 주 목표({cp}/km)를 앞서고 있어요. 무리하지 말고 회복 러닝도 챙기세요.",
        "목표 일정보다 앞서가고 있어요(예상 {pred}/km, 체크포인트 {cp}/km). 이번 {dist}km는 여유 있게 달려도 좋아요.",
    ],
    "계획대로": [
        "{week}주차 체크포인트({cp}/km)와 비슷한 예상 페이스({pred}/km)예요. 지금 계획대로 {dist}km를 꾸준히 이어가세요.",
        "목표 일정에 맞게 가고 있어요(격차 {gap_signed}초/km). 이번 주도 같은 강도를 유지해 보세요.",
        "예상 {pred}/km, 체크포인트 {cp}/km로 계획 범위 안이에요. {dist}km 러닝을 이어가면 됩니다.",
    ],
    "조금 뒤처짐": [
        "체크포인트({cp}/km)보다 예상 페이스({pred}/km)가 {gap_abs}초/km 느려요. 이번 {dist}km에서 후반 페이스를 조금만 끌어올려 보세요.",
        "목표 일정보다 조금 뒤처져 있어요. 주 1회 약간 빠른 구간 러닝을 넣어 격차({gap_abs}초/km)를 줄여 보세요.",
        "{week}주차 기준 {gap_abs}초/km 느린 예상이에요. 컨디션을 살피며 {dist}km 중 일부 구간만 속도를 올려 보세요.",
    ],
    "크게 뒤처짐": [
        "체크포인트({cp}/km)와 예상 페이스({pred}/km)의 격차가 {gap_abs}초/km로 커요. 목표 일정이나 목표 페이스를 다시 조정해 볼까요?",
        "지금 페이스로는 {week}주차 목표를 맞추기 어려워 보여요(격차 {gap_abs}초/km). 거리보다 꾸준함에 먼저 집중해 보세요.",
        "목표 대비 많이 느린 예상이에요. 무리한 가속 대신 {dist}km를 안정적으로 완주하는 것부터 다시 시작해 보세요.",
    ],
}


def weeks_until_target(last_input_end_ts, target_start_ts):
    """(타깃 시작 - 마지막 입력 세션 종료)를 7일 단위로 올림한 주차(1부터). 간격이 0 이하면 1."""
    gap = target_start_ts - last_input_end_ts
    return max(1, int(math.ceil(gap / SECONDS_PER_WEEK)))


def goal_pace(current_pace, delta):
    """합성 목표: 현재 실력 x (1 - delta)."""
    return current_pace * (1.0 - delta)


def checkpoint(current_pace, delta, week, deadline_weeks):
    """현재 실력에서 목표 페이스까지 주 단위 선형 보간. week는 1..deadline_weeks."""
    goal = goal_pace(current_pace, delta)
    return current_pace + (goal - current_pace) * (week / float(deadline_weeks))


def gap_sec(pred_pace, current_pace, delta, week, deadline_weeks):
    """격차 = 예측 페이스 - 해당 주차 체크포인트 (초/km, 양수=뒤처짐)."""
    return pred_pace - checkpoint(current_pace, delta, week, deadline_weeks)


def categorize_ratio(ratio):
    """ratio = 격차 / 현재 페이스. 경계는 모듈 상단 상수(사전 고정)."""
    if ratio <= AHEAD_MAX:
        return "앞서감"
    if ratio <= ON_TRACK_MAX:
        return "계획대로"
    if ratio <= SLIGHT_MAX:
        return "조금 뒤처짐"
    return "크게 뒤처짐"


def categorize(pred_pace, current_pace, delta, week, deadline_weeks):
    g = gap_sec(pred_pace, current_pace, delta, week, deadline_weeks)
    return categorize_ratio(g / current_pace), g


def fmt_pace(sec_per_km):
    m, s = divmod(int(round(sec_per_km)), 60)
    return f"{m}:{s:02d}"


def make_message(category, pred_pace, current_pace, delta, week, deadline_weeks, dist_km, variant_key):
    """고정 템플릿 문구. variant_key(정수)로 카테고리 내 2~3개 중 결정적으로 선택."""
    cp = checkpoint(current_pace, delta, week, deadline_weeks)
    g = pred_pace - cp
    tpl = TEMPLATES[category]
    t = tpl[variant_key % len(tpl)]
    return t.format(week=week, cp=fmt_pace(cp), pred=fmt_pace(pred_pace), gap_abs=int(round(abs(g))),
                    gap_signed=f"{g:+.0f}", dist=f"{dist_km:g}")
