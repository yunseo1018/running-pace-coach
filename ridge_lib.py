# -*- coding: utf-8 -*-
"""
Ridge/Huber 회귀 + 전처리 + GroupKFold를 numpy만으로 직접 구현한 라이브러리.

**왜 sklearn을 안 쓰는가 (중요, 선택이 아니라 환경 제약)**: 이 환경에서
`import sklearn`이 실패한다 — scipy의 컴파일된 확장 모듈
(`scipy.spatial.distance._distance_pybind`)이 Windows 보안 정책(Application
Control Policy)에 의해 차단되어 DLL을 못 불러온다. 여러 번 재시도해도
동일하게 실패해 일시적 문제가 아님을 확인했다. sklearn.linear_model이
내부적으로 scipy를 거쳐 이 모듈을 임포트하기 때문에 Ridge/HuberRegressor
자체를 쓸 수 없다. 그래서 수학적으로 동일한 알고리즘을 numpy로 직접
구현했다 (아래에서 alpha=0일 때 OLS와 일치하는지 검증).
"""

import numpy as np


class ManualStandardScaler:
    """sklearn.preprocessing.StandardScaler와 동일한 동작 (평균 0, 표준편차 1로
    표준화). fit은 학습 데이터에서만 하고, transform은 학습/평가 데이터 모두에
    같은 통계치로 적용한다."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def ridge_fit(X, y, alpha, sample_weight=None):
    """
    Ridge 회귀를 closed-form으로 직접 푼다 (sklearn.linear_model.Ridge와 동일한
    정의: intercept는 정규화하지 않음 — X,y를 평균으로 센터링한 뒤 풀고,
    intercept는 나중에 복원).

    beta = (Xc^T W Xc + alpha*I)^-1 Xc^T W yc   (W: sample_weight 대각행렬)
    intercept = y_mean - X_mean @ beta
    """
    if sample_weight is None:
        w = np.ones(len(y))
    else:
        w = np.asarray(sample_weight)

    w_sum = w.sum()
    X_mean = (w[:, None] * X).sum(axis=0) / w_sum
    y_mean = (w * y).sum() / w_sum
    Xc = X - X_mean
    yc = y - y_mean

    n_features = X.shape[1]
    Xw = Xc * w[:, None]
    A = Xw.T @ Xc + alpha * np.eye(n_features)
    b = Xw.T @ yc
    beta = np.linalg.solve(A, b)
    intercept = y_mean - X_mean @ beta
    return beta, intercept


def ridge_predict(X, beta, intercept):
    return X @ beta + intercept


def _validate_ridge_against_ols():
    """alpha=0인 Ridge(가중치 없음)가 numpy.linalg.lstsq로 구한 OLS와
    사실상 같은 예측을 내는지 검증 (수동 구현의 정확성 확인용)."""
    rng = np.random.default_rng(0)
    n, p = 200, 4
    X = rng.normal(size=(n, p))
    true_beta = np.array([1.0, -2.0, 0.5, 0.0])
    y = X @ true_beta + 3.0 + rng.normal(scale=0.1, size=n)

    beta_ridge, intercept_ridge = ridge_fit(X, y, alpha=0.0)

    X_design = np.hstack([X, np.ones((n, 1))])
    ols_coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    ols_beta, ols_intercept = ols_coef[:-1], ols_coef[-1]

    pred_ridge = ridge_predict(X, beta_ridge, intercept_ridge)
    pred_ols = X @ ols_beta + ols_intercept
    max_diff = np.abs(pred_ridge - pred_ols).max()
    assert max_diff < 1e-6, f"alpha=0 Ridge가 OLS와 일치하지 않습니다! max_diff={max_diff}"
    return max_diff


def huber_fit(X, y, alpha=1e-4, delta_scale=1.345, max_iter=30, tol=1e-6):
    """
    Huber 회귀를 IRLS(Iteratively Reweighted Least Squares)로 간소화해서
    구현한다 (sklearn.linear_model.HuberRegressor와 완전히 같지는 않음 —
    scale을 매 반복마다 재추정하는 완전한 M-estimation 대신, 초기 OLS 잔차의
    MAD로 한 번만 delta를 정하고 고정한다. "민감도 참고용 1행"이라는
    요청 범위에 맞춰 간소화했다 — 이 간소화 자체를 보고서에 명시한다).

    delta = delta_scale * MAD(초기 OLS 잔차) (MAD: median absolute deviation,
    정규분포 기준 표준편차로 스케일하려면 1.4826을 곱하는 게 보통이지만,
    sklearn 기본값 관례(epsilon=1.345)를 따라 delta_scale=1.345를 그대로
    MAD에 곱한다 — 정확한 표준편차 추정이 아니라 "이상치 판정 임계값"
    정도의 근사치로만 쓴다는 뜻).
    가중치: |잔차| <= delta면 1, 아니면 delta/|잔차| (큰 잔차의 영향을 줄임).
    """
    beta, intercept = ridge_fit(X, y, alpha=alpha)
    resid = y - ridge_predict(X, beta, intercept)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    delta = delta_scale * mad

    for _ in range(max_iter):
        resid = y - ridge_predict(X, beta, intercept)
        abs_resid = np.abs(resid) + 1e-12
        weights = np.where(abs_resid <= delta, 1.0, delta / abs_resid)
        new_beta, new_intercept = ridge_fit(X, y, alpha=alpha, sample_weight=weights)
        diff = np.abs(new_beta - beta).max()
        beta, intercept = new_beta, new_intercept
        if diff < tol:
            break
    return beta, intercept, delta


def group_kfold_indices(user_ids, n_folds, seed):
    """GroupKFold를 수동으로 구현: user_id 단위로 fold를 나눠서, 같은
    사용자의 모든 행이 항상 같은 fold에 들어가게 한다."""
    unique_users = np.unique(user_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_users)
    folds = np.array_split(shuffled, n_folds)
    for i in range(n_folds):
        val_users = set(folds[i].tolist())
        val_mask = np.isin(user_ids, list(val_users))
        yield ~val_mask, val_mask


def huber_objective(X, y, beta, intercept, alpha, delta):
    """Huber 손실(고정 delta) + L2 페널티(alpha, intercept 제외)의 실제 목적함수 값.
    IRLS가 이 목적함수를 매 반복마다 줄이는지 검증하는 데 쓴다."""
    resid = y - ridge_predict(X, beta, intercept)
    abs_r = np.abs(resid)
    loss = np.where(abs_r <= delta, 0.5 * resid ** 2, delta * (abs_r - 0.5 * delta))
    return float(loss.sum() + alpha * np.sum(beta ** 2))


def _validate_huber_irls():
    """2(a) 검증: 이상치를 포함한 합성 데이터에서
    (1) IRLS가 반복마다 실제 Huber 목적함수를 감소(또는 불변)시키는지,
    (2) IRLS 수렴해의 목적함수가, 같은 목적함수를 numpy 경사하강법으로
        충분히 수렴시켜 얻은 해보다 나쁘지 않은지 확인한다."""
    rng = np.random.default_rng(1)
    n, p = 300, 3
    X = rng.normal(size=(n, p))
    true_beta = np.array([2.0, -1.0, 0.5])
    y = X @ true_beta + 1.0 + rng.normal(scale=0.5, size=n)
    outlier_idx = rng.choice(n, size=15, replace=False)
    y = y.copy()
    y[outlier_idx] += rng.normal(scale=50, size=15)

    alpha = 1e-4
    max_iter, tol = 30, 1e-6

    beta, intercept = ridge_fit(X, y, alpha=alpha)
    resid = y - ridge_predict(X, beta, intercept)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    delta = 1.345 * mad

    objectives = []
    for _ in range(max_iter):
        resid = y - ridge_predict(X, beta, intercept)
        abs_resid = np.abs(resid) + 1e-12
        weights = np.where(abs_resid <= delta, 1.0, delta / abs_resid)
        new_beta, new_intercept = ridge_fit(X, y, alpha=alpha, sample_weight=weights)
        obj = huber_objective(X, y, new_beta, new_intercept, alpha, delta)
        objectives.append(obj)
        diff = np.abs(new_beta - beta).max()
        beta, intercept = new_beta, new_intercept
        if diff < tol:
            break

    increases = [objectives[i + 1] - objectives[i] for i in range(len(objectives) - 1)]
    max_increase = max(increases) if increases else 0.0
    monotonic_ok = max_increase <= 1e-6  # 부동소수점 오차 허용치
    irls_final_obj = objectives[-1]

    # 같은 목적함수를 numpy 경사하강법(충분히 수렴)으로 독립적으로 최소화
    beta_gd = np.zeros(p)
    intercept_gd = 0.0
    lr = 2e-3
    for _ in range(50000):
        resid_gd = y - (X @ beta_gd + intercept_gd)
        abs_r = np.abs(resid_gd)
        grad_r = np.where(abs_r <= delta, -resid_gd, -delta * np.sign(resid_gd))
        grad_beta = X.T @ grad_r / n + 2 * alpha * beta_gd
        grad_intercept = grad_r.sum() / n
        beta_gd -= lr * grad_beta
        intercept_gd -= lr * grad_intercept
    gd_final_obj = huber_objective(X, y, beta_gd, intercept_gd, alpha, delta)

    return {
        "n_iterations": len(objectives),
        "monotonic_decrease": monotonic_ok,
        "max_increase": max_increase,
        "irls_final_obj": irls_final_obj,
        "gd_final_obj": gd_final_obj,
        "irls_not_worse_than_gd": irls_final_obj <= gd_final_obj + 1e-3,
    }


def _validate_group_kfold_no_overlap(n_users=1000, n_rows=5000, n_folds=5, seed=42, data_seed=0):
    """2(b) 검증: GroupKFold의 fold 간 사용자(그룹) 겹침이 0인지 확인한다."""
    rng = np.random.default_rng(data_seed)
    user_ids = rng.integers(0, n_users, size=n_rows)
    folds = list(group_kfold_indices(user_ids, n_folds, seed))
    val_user_sets = []
    for _, val_mask in folds:
        val_user_sets.append(set(user_ids[val_mask].tolist()))
    max_overlap = 0
    for i in range(len(val_user_sets)):
        for j in range(i + 1, len(val_user_sets)):
            overlap = len(val_user_sets[i] & val_user_sets[j])
            max_overlap = max(max_overlap, overlap)
    assert max_overlap == 0, f"fold 간 그룹 겹침 발견: {max_overlap}"
    return max_overlap


if __name__ == "__main__":
    diff = _validate_ridge_against_ols()
    print(f"[검증] alpha=0 Ridge vs numpy OLS 예측값 최대 차이: {diff:.2e} (1e-6 미만이어야 통과, 통과)")

    huber_result = _validate_huber_irls()
    print(f"[검증] Huber IRLS: {huber_result['n_iterations']}회 반복, "
          f"목적함수 단조감소={huber_result['monotonic_decrease']} "
          f"(최대 증가분={huber_result['max_increase']:.2e}), "
          f"IRLS 최종 목적함수={huber_result['irls_final_obj']:.6f}, "
          f"경사하강 최종 목적함수={huber_result['gd_final_obj']:.6f}, "
          f"IRLS가 경사하강보다 나쁘지않음={huber_result['irls_not_worse_than_gd']}")

    overlap = _validate_group_kfold_no_overlap()
    print(f"[검증] GroupKFold(합성 데이터, 5-fold) fold간 사용자 겹침 최대값: {overlap} (0이어야 통과, 통과)")


class RidgePipeline:
    """표준화(ManualStandardScaler, 학습 데이터에서만 fit) + Ridge를 묶은 sklearn Pipeline 대체.
    7단계 이후 모든 Ridge는 sklearn import 성공 여부와 무관하게 이것을 쓴다(재현성 고정)."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        self.scaler_ = ManualStandardScaler().fit(X)
        self.beta_, self.intercept_ = ridge_fit(self.scaler_.transform(X), y, alpha=self.alpha)[:2]
        return self

    def predict(self, X):
        return ridge_predict(self.scaler_.transform(X), self.beta_, self.intercept_)


def group_kfold_splits(user_ids, n_folds, seed):
    """group_kfold_indices의 인덱스 배열 버전: (train_idx, val_idx)를 순서대로 돌려준다.
    sklearn GroupKFold와 fold 멤버십은 다르다(시드 셔플 후 균등분할)."""
    for tr_mask, val_mask in group_kfold_indices(np.asarray(user_ids), n_folds, seed):
        yield np.where(tr_mask)[0], np.where(val_mask)[0]
