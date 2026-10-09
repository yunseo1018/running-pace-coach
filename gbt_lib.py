# -*- coding: utf-8 -*-
"""
Gradient Boosting Regressor(절대오차/제곱오차 손실)를 numpy로 직접 구현한 라이브러리.

**왜 sklearn을 안 쓰는가**: 이 환경에서 `sklearn.tree`의 컴파일된 확장
(`sklearn.tree._criterion`, `_splitter`)이 Windows Application Control
Policy에 의해 차단되어 import에 실패한다 (여러 번 재시도해도 동일 — 25단계
실행 로그에서 최초 발견). 이 때문에 DecisionTree/RandomForest/
GradientBoosting/HistGradientBoosting 등 sklearn.tree 기반 모델 **전부**
쓸 수 없다 (반면 sklearn.linear_model.Ridge, sklearn.model_selection.
GroupKFold는 정상 동작 — ridge_lib.py 검증 참고). lightgbm/xgboost는
미설치 상태이고 설치는 금지됐다. 그래서 HistGradientBoostingRegressor와
같은 역할을 하는 gradient boosting regression tree 앙상블을
ridge_lib.py와 같은 원칙(sklearn 없이 직접 구현 + 자체 검증)으로 작성한다.

**알고리즘 (sklearn.ensemble.HistGradientBoostingRegressor를 단순화해 모방)**:
1. 각 피처를 분위수 기준 n_bins개 구간으로 미리 이산화(binning) — 매 노드마다
   정렬하는 대신 미리 만든 구간 경계만 재사용해 분할 탐색 비용을 깊이/노드
   수와 거의 무관하게 만든다 (HistGradientBoosting과 같은 설계 의도).
2. 매 반복(iteration)마다 "pseudo-residual" g를 계산한다
   (squared_error: g=residual, absolute_error: g=sign(residual)).
3. g를 target으로, 구간별 제곱오차 감소량이 최대인 분할을 반복해 트리를
   키운다(깊이 max_depth까지, min_samples_leaf 미만이면 중단).
4. 트리 구조가 정해지면, **리프 값은 pseudo-residual이 아니라 실제
   residual로 다시 계산**한다 (squared_error: 평균, absolute_error: 중앙값
   — Friedman의 TreeBoost 보정과 같은 방식).
5. F_m(x) = F_{m-1}(x) + learning_rate * tree_m(x), 초기값 F_0 = 전체
   median(absolute_error) 또는 mean(squared_error).

**sklearn.ensemble.HistGradientBoostingRegressor와 다른 점(간소화, 명시)**:
- `max_leaf_nodes`(best-first 리프 확장) 대신 `max_depth`(완전 이진트리 깊이
  제한)로 트리 크기를 제어한다 — 구현이 훨씬 단순하고 수렴 성질이 비슷하지만
  정확히 같은 트리 모양은 아니다.
- L2 정규화(l2_regularization), 2차 근사 기반 분할 이득(뉴턴 스텝) 등 HGB의
  세부 최적화는 생략했다 — 1차 근사(제곱오차 감소량) 기준의 표준 gradient
  boosting이다.
- 분할 탐색은 구간(bin) 경계만 후보로 삼는 근사이지 모든 데이터 포인트를
  보는 정확한(exact) 탐색이 아니다(HGB도 원래 근사다).

sklearn.tree가 막혀 sklearn 결과와 직접 대조할 수 없으므로(ridge_lib.py의
alpha=0 vs OLS 같은 방식의 "정답 비교"가 불가능), 아래 `_validate_gbt_sanity()`로
대신 (1) 학습 손실이 반복마다 단조감소하는지 (2) 분명한 계단함수 합성
데이터에서 첫 분할이 실제 경계를 찾는지를 검증한다.
"""

import numpy as np


def _compute_bin_edges(X, n_bins):
    n, p = X.shape
    edges = []
    for j in range(p):
        col = X[:, j]
        qs = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]))
        edges.append(qs)
    return edges


def _apply_bins(X, edges):
    n, p = X.shape
    X_binned = np.zeros((n, p), dtype=np.int32)
    for j in range(p):
        X_binned[:, j] = np.searchsorted(edges[j], X[:, j], side="right")
    return X_binned


class _Node:
    __slots__ = ("is_leaf", "value", "feature", "bin_threshold", "left", "right")

    def __init__(self):
        self.is_leaf = True
        self.value = 0.0
        self.feature = None
        self.bin_threshold = None
        self.left = None
        self.right = None


def _best_split_for_node(X_binned, g, row_idx, n_bins_per_feat, min_samples_leaf):
    g_node = g[row_idx]
    total_sum = g_node.sum()
    total_sq = (g_node ** 2).sum()
    n_node = len(row_idx)
    node_sse = total_sq - (total_sum ** 2) / n_node

    best_gain, best_feat, best_bin = 0.0, None, None
    p = X_binned.shape[1]
    for j in range(p):
        nb = n_bins_per_feat[j]
        if nb <= 1:
            continue
        bins_j = X_binned[row_idx, j]
        sum_per_bin = np.bincount(bins_j, weights=g_node, minlength=nb + 1)[: nb + 1]
        sq_per_bin = np.bincount(bins_j, weights=g_node ** 2, minlength=nb + 1)[: nb + 1]
        cnt_per_bin = np.bincount(bins_j, minlength=nb + 1)[: nb + 1]
        cum_sum = np.cumsum(sum_per_bin)
        cum_sq = np.cumsum(sq_per_bin)
        cum_cnt = np.cumsum(cnt_per_bin)

        left_n = cum_cnt[:-1]
        right_n = n_node - left_n
        valid = (left_n >= min_samples_leaf) & (right_n >= min_samples_leaf)
        if not valid.any():
            continue
        left_sum = cum_sum[:-1]
        left_sq = cum_sq[:-1]
        right_sum = total_sum - left_sum
        right_sq = total_sq - left_sq
        left_n_safe = np.maximum(left_n, 1)
        right_n_safe = np.maximum(right_n, 1)
        left_sse = left_sq - (left_sum ** 2) / left_n_safe
        right_sse = right_sq - (right_sum ** 2) / right_n_safe
        gain = node_sse - (left_sse + right_sse)
        gain = np.where(valid, gain, -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best_gain:
            best_gain, best_feat, best_bin = float(gain[k]), j, k
    return best_feat, best_bin, best_gain


def _build_tree(X_binned, g, residual_actual, row_idx, depth, max_depth, min_samples_leaf,
                 n_bins_per_feat, leaf_agg):
    node = _Node()
    n_node = len(row_idx)
    if depth >= max_depth or n_node < 2 * min_samples_leaf:
        node.value = float(leaf_agg(residual_actual[row_idx]))
        return node

    feat, bin_k, gain = _best_split_for_node(X_binned, g, row_idx, n_bins_per_feat, min_samples_leaf)
    if feat is None or gain <= 1e-12:
        node.value = float(leaf_agg(residual_actual[row_idx]))
        return node

    mask_left = X_binned[row_idx, feat] <= bin_k
    left_idx = row_idx[mask_left]
    right_idx = row_idx[~mask_left]
    if len(left_idx) < min_samples_leaf or len(right_idx) < min_samples_leaf:
        node.value = float(leaf_agg(residual_actual[row_idx]))
        return node

    node.is_leaf = False
    node.feature = feat
    node.bin_threshold = bin_k
    node.left = _build_tree(X_binned, g, residual_actual, left_idx, depth + 1, max_depth,
                             min_samples_leaf, n_bins_per_feat, leaf_agg)
    node.right = _build_tree(X_binned, g, residual_actual, right_idx, depth + 1, max_depth,
                              min_samples_leaf, n_bins_per_feat, leaf_agg)
    return node


def _predict_tree(node, X_binned):
    n = X_binned.shape[0]
    out = np.empty(n)

    def rec(nd, idx):
        if nd.is_leaf or len(idx) == 0:
            out[idx] = nd.value
            return
        mask_left = X_binned[idx, nd.feature] <= nd.bin_threshold
        rec(nd.left, idx[mask_left])
        rec(nd.right, idx[~mask_left])

    rec(node, np.arange(n))
    return out


class SimpleGradientBoostingRegressor:
    """HistGradientBoostingRegressor를 모방한 numpy 전용 구현.
    loss: "absolute_error" 또는 "squared_error"."""

    def __init__(self, loss="absolute_error", learning_rate=0.1, max_iter=100,
                 max_depth=3, min_samples_leaf=20, n_bins=32, random_state=None):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.random_state = random_state  # 현재 구현은 결정적(샘플링 없음) — 호환용 인자

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.edges_ = _compute_bin_edges(X, self.n_bins)
        self.n_bins_per_feat_ = [len(e) + 1 for e in self.edges_]
        X_binned = _apply_bins(X, self.edges_)

        if self.loss == "absolute_error":
            self.init_ = float(np.median(y))
            leaf_agg = np.median
        elif self.loss == "squared_error":
            self.init_ = float(np.mean(y))
            leaf_agg = np.mean
        else:
            raise ValueError(f"지원하지 않는 loss: {self.loss}")

        n = len(y)
        F = np.full(n, self.init_)
        row_idx_all = np.arange(n)
        self.trees_ = []
        self.train_loss_history_ = []
        for _ in range(self.max_iter):
            residual = y - F
            g = np.sign(residual) if self.loss == "absolute_error" else residual
            tree = _build_tree(X_binned, g, residual, row_idx_all, 0, self.max_depth,
                                self.min_samples_leaf, self.n_bins_per_feat_, leaf_agg)
            pred = _predict_tree(tree, X_binned)
            F = F + self.learning_rate * pred
            self.trees_.append(tree)
            if self.loss == "absolute_error":
                self.train_loss_history_.append(float(np.mean(np.abs(y - F))))
            else:
                self.train_loss_history_.append(float(np.mean((y - F) ** 2)))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_binned = _apply_bins(X, self.edges_)
        F = np.full(X.shape[0], self.init_)
        for tree in self.trees_:
            F = F + self.learning_rate * _predict_tree(tree, X_binned)
        return F


def _validate_gbt_sanity():
    """sklearn.tree가 막혀 직접 대조가 불가능하므로, 자체 정합성만 검증한다:
    (1) 학습 손실이 반복마다 단조감소(또는 불변)하는지 (absolute_error/squared_error 둘 다)
    (2) 뚜렷한 계단함수 합성 데이터에서 1번째 트리의 첫 분할이 실제 경계를 찾는지."""
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.uniform(0, 1, size=(n, 3))
    y = np.where(X[:, 0] > 0.5, 100.0, 20.0) + rng.normal(scale=1.0, size=n)

    results = {}
    for loss in ["squared_error", "absolute_error"]:
        model = SimpleGradientBoostingRegressor(loss=loss, learning_rate=0.3, max_iter=15,
                                                  max_depth=2, min_samples_leaf=10, n_bins=32)
        model.fit(X, y)
        hist = np.array(model.train_loss_history_)
        increases = np.diff(hist)
        max_increase = float(increases.max()) if len(increases) else 0.0
        monotonic_ok = max_increase <= 1e-9
        results[loss] = {
            "first_loss": float(hist[0]), "last_loss": float(hist[-1]),
            "monotonic_decrease": monotonic_ok, "max_increase": max_increase,
        }

    # 1번째 트리의 분할 지점이 실제 경계(X[:,0]=0.5) 근처인지 확인
    model0 = SimpleGradientBoostingRegressor(loss="squared_error", learning_rate=1.0, max_iter=1,
                                              max_depth=1, min_samples_leaf=10, n_bins=32)
    model0.fit(X, y)
    root = model0.trees_[0]
    split_feature_correct = (root.feature == 0)
    split_threshold_value = model0.edges_[0][root.bin_threshold] if split_feature_correct else None
    split_near_boundary = (split_threshold_value is not None and abs(split_threshold_value - 0.5) < 0.1)

    return {
        "loss_results": results,
        "split_feature_correct": split_feature_correct,
        "split_threshold_value": split_threshold_value,
        "split_near_boundary": split_near_boundary,
    }


if __name__ == "__main__":
    result = _validate_gbt_sanity()
    for loss, r in result["loss_results"].items():
        print(f"[검증] {loss}: 초기손실={r['first_loss']:.3f} -> 최종손실={r['last_loss']:.3f}, "
              f"단조감소={r['monotonic_decrease']} (최대증가분={r['max_increase']:.2e})")
    print(f"[검증] 계단함수 분할: feature 0 선택={result['split_feature_correct']}, "
          f"분할 임계값={result['split_threshold_value']}, 0.5 근처(±0.1)={result['split_near_boundary']}")
