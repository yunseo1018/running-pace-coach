# -*- coding: utf-8 -*-
"""보고서 공통 문구 (3주차 문구 정정). 25/27/28번이 같은 문구를 쓰도록 한곳에 둔다."""

# "GBM/MLP가 median-5보다 유의하게 낫다"는 문장 옆에 병기하는 조건 설명.
# 근거: docs/same_scope_comparison.md 7-3절(median-5 대비 22피처 모델의 풀링/macro MAE 차이, 사용자 단위 bootstrap CI).
MACRO_NOTE = ("(윈도우 풀링 MAE 기준, 22피처 기준; 사용자별 MAE 평균(macro)에서는 일부 조건에서 CI가 0을 포함하고, "
              "cold-start GBM22처럼 점추정이 오히려 높은 경우가 있음 — `docs/same_scope_comparison.md` 7-3절)")
