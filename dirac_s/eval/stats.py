"""Statistical methods — ported near-verbatim from covtoken/eval/stats.py (same functions, same
implementation), per this vault's established cross-project convention (data/manifest.py's split_of() is
already a verbatim port from covtoken for the identical reason: DIRAC and covtoken are separate packages,
never mounted together in an HF Job, so a runtime cross-import isn't available — the code gets ported once,
not reinvented). Phase 5's own statistical-methods note (IMPLEMENTATION_SPEC.md Section 6) calls for exactly
this: "linear/nonlinear probe accuracy or AUROC, paired bootstrap CI... against the matched baseline's probe
on the same split" (H1), and this is the vault's single source of truth for that class of comparison.

- AUROC + DeLong test for AUROC differences, 95% CI.
- Paired bootstrap over cases (n=2000) for any paired per-case metric difference; CI must exclude 0.
- Spearman rho with permutation p-value — not used by H1, kept for H4's uncertainty-error correlation later.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ----------------------------- AUROC + DeLong --------------------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """DeLong covariance (Sun & Xu 2014 fast algorithm). Returns (aucs, cov)."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    sx = np.atleast_2d(sx); sy = np.atleast_2d(sy)
    cov = sx / m + sy / n
    return aucs, cov


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of `scores` against binary `labels` (1=positive)."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    order = np.argsort(-scores)
    s = labels[order]
    # rank-based AUC
    pos = labels.sum(); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def delong_auc_ci(scores: np.ndarray, labels: np.ndarray, alpha: float = 0.05):
    """AUROC with DeLong 95% CI. Returns (auc, (lo, hi)). NaN/(NaN,NaN) if either class is empty —
    _fast_delong divides by both the positive and negative counts, so a degenerate split (e.g. an
    unexpectedly imbalanced probing label) would otherwise raise ZeroDivisionError instead of reporting
    a meaningful "this split can't be scored" result. Mirrors auroc()'s own pos==0/neg==0 guard above,
    which this function was missing until a real degenerate split (H1 probing) hit it."""
    labels = np.asarray(labels, int); scores = np.asarray(scores, float)
    m = int(labels.sum())
    if m == 0 or m == len(labels):
        return float("nan"), (float("nan"), float("nan"))
    order = np.argsort(-labels, kind="mergesort")  # positives first
    lab = labels[order]; sc = scores[order]
    aucs, cov = _fast_delong(sc[np.newaxis, :], m)
    auc = float(aucs[0]); var = float(cov[0, 0]) if np.ndim(cov) else float(cov)
    se = np.sqrt(max(var, 0.0))
    z = stats.norm.ppf(1 - alpha / 2)
    return auc, (max(0.0, auc - z * se), min(1.0, auc + z * se))


def delong_auc_diff_test(scores_a, scores_b, labels, alpha: float = 0.05):
    """Test AUROC(a) - AUROC(b) via DeLong. Returns dict with diff, ci, p (paired). NaN fields if either
    class is empty — see delong_auc_ci's docstring for why this guard exists."""
    labels = np.asarray(labels, int)
    m = int(labels.sum())
    if m == 0 or m == len(labels):
        return {"auc_a": float("nan"), "auc_b": float("nan"), "diff": float("nan"),
                "ci95": [float("nan"), float("nan")], "p": float("nan")}
    order = np.argsort(-labels, kind="mergesort")
    preds = np.vstack([np.asarray(scores_a, float)[order],
                       np.asarray(scores_b, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = np.sqrt(max(var, 1e-12))
    z = diff / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
            "ci95": [diff - zc * se, diff + zc * se], "p": p}


# ----------------------------- paired bootstrap ------------------------------
def paired_bootstrap_diff(values_a, values_b, n: int = 2000, alpha: float = 0.05,
                          seed: int = 0):
    """Paired bootstrap over cases for mean(a)-mean(b). Returns dict with diff, ci, excludes0."""
    a = np.asarray(values_a, float); b = np.asarray(values_b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    N = len(a); diffs = np.empty(n)
    base = float(a.mean() - b.mean())
    for i in range(n):
        idx = rng.integers(0, N, N)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": base, "ci95": [float(lo), float(hi)],
            "excludes_0": bool(lo > 0 or hi < 0)}


# ----------------------------- partial Spearman (residualization) -----------
def partial_spearman(a, b, controlling_for, seed: int = 0):
    """Partial correlation of `a` with `b`, controlling for `controlling_for` -- ported verbatim (same
    algorithm, same two-step method) from the sibling sheaf project's gate2_sheaf_cohomology_job.py, which
    dirac_formalization.md Section 8 explicitly asks to reuse "for direct comparability across the two
    projects" (that project's own magnitude-conflation check: partial_spearman_D_error_given_entropy).
    NOT the closed-form partial-correlation formula -- linear-regression residualization (OLS, degree 1)
    on the controlling variable, for both `a` and `b` separately, then a Spearman rank correlation of the
    two residual vectors. Returns the correlation coefficient (float)."""
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(controlling_for, float)

    def resid(y_, x_):
        coef = np.polyfit(x_, y_, 1)
        return y_ - np.polyval(coef, x_)

    return float(stats.spearmanr(resid(a, c), resid(b, c)).statistic)


# ----------------------------- Spearman permutation --------------------------
def spearman_perm(x, y, n: int = 5000, seed: int = 0):
    """Spearman rho with a permutation p-value. Returns dict rho, p, monotone."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        if abs(stats.spearmanr(x, rng.permutation(y)).statistic) >= abs(rho):
            count += 1
    p = (count + 1) / (n + 1)
    return {"rho": rho, "p": float(p)}
