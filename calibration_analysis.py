#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GenProt-DSM calibration analysis on the held-out test set.

Input:
    F:\GenProt-DSM_Resubmit\data\test.csv

Required columns:
    label
    prob_ensemble

Outputs:
    calibration_curve.png
    calibration_curve.pdf
    calibration_metrics.csv
    calibration_bins.csv

Analysis:
1) Reliability / calibration curve with 10 quantile bins
2) Brier score
3) Expected Calibration Error (ECE)
4) Calibration intercept
5) Calibration slope
6) 1000 stratified bootstrap 95% CIs for the four scalar calibration metrics

Important:
- Calibration uses the raw predicted probabilities.
- DO NOT threshold prob_ensemble at 0.554393... for calibration analysis.
"""

from __future__ import annotations

import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Paths
# =============================================================================

INPUT_FILE = r"F:\GenProt-DSM_Resubmit\data\test.csv"

OUTPUT_DIR = r"F:\GenProt-DSM_Resubmit\calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PNG_FILE = os.path.join(
    OUTPUT_DIR,
    "GenProt_DSM_calibration_curve.png"
)

PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "GenProt_DSM_calibration_curve.pdf"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "GenProt_DSM_calibration_metrics.csv"
)

BINS_FILE = os.path.join(
    OUTPUT_DIR,
    "GenProt_DSM_calibration_bins.csv"
)


# =============================================================================
# 2. Configuration
# =============================================================================

LABEL_COL = "label"
PROB_COL = "prob_ensemble"

N_BINS = 10
BINNING = "quantile"   # recommended here: quantile bins
N_BOOTSTRAP = 1000
RANDOM_SEED = 2026

# Avoid logit(0) / logit(1)
EPS = 1e-6


# =============================================================================
# 3. Helpers
# =============================================================================

def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p / (1.0 - p)
    )


def sigmoid(x):
    x = np.asarray(x, dtype=float)

    # Numerically stable sigmoid
    out = np.empty_like(x, dtype=float)

    pos = x >= 0
    neg = ~pos

    out[pos] = 1.0 / (
        1.0 + np.exp(-x[pos])
    )

    exp_x = np.exp(x[neg])

    out[neg] = exp_x / (
        1.0 + exp_x
    )

    return out


def calibration_intercept_slope(
    y_true,
    prob,
):
    """
    Fit logistic recalibration:

        logit(P(Y=1)) = intercept + slope * logit(predicted_probability)

    Ideal:
        intercept = 0
        slope     = 1

    This is fitted by maximum likelihood with no regularization.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    x = logit(prob)

    def negative_log_likelihood(params):
        intercept, slope = params

        eta = (
            intercept
            + slope * x
        )

        p = sigmoid(eta)

        p = np.clip(
            p,
            EPS,
            1.0 - EPS,
        )

        ll = np.sum(
            y_true * np.log(p)
            +
            (1.0 - y_true) * np.log(1.0 - p)
        )

        return -ll

    result = minimize(
        negative_log_likelihood,
        x0=np.array([0.0, 1.0], dtype=float),
        method="BFGS",
    )

    if not result.success:
        # retry from a neutral initialization
        result = minimize(
            negative_log_likelihood,
            x0=np.array([0.0, 0.5], dtype=float),
            method="BFGS",
        )

    if not result.success:
        return np.nan, np.nan

    intercept = float(
        result.x[0]
    )

    slope = float(
        result.x[1]
    )

    return intercept, slope


def make_quantile_bins(
    y_true,
    prob,
    n_bins=10,
):
    """
    Create quantile-based calibration bins.

    Returns one row per non-empty bin:
        bin
        n
        mean_predicted_probability
        observed_pathogenic_fraction
        lower_95
        upper_95

    Wilson 95% CI is used for the observed pathogenic fraction.
    """

    tmp = pd.DataFrame(
        {
            "y": np.asarray(
                y_true,
                dtype=int,
            ),
            "p": np.asarray(
                prob,
                dtype=float,
            ),
        }
    )

    # duplicates='drop' avoids failure when many identical probabilities exist.
    tmp["bin"] = pd.qcut(
        tmp["p"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    rows = []

    z = 1.959963984540054

    for bin_id, g in tmp.groupby(
        "bin",
        observed=True,
        sort=True,
    ):

        n = len(g)

        mean_p = float(
            g["p"].mean()
        )

        k = int(
            g["y"].sum()
        )

        observed = (
            k / n
        )

        # Wilson CI for a binomial proportion
        denom = (
            1.0
            + z * z / n
        )

        center = (
            observed
            + z * z / (2.0 * n)
        ) / denom

        half = (
            z
            * math.sqrt(
                observed * (1.0 - observed) / n
                + z * z / (4.0 * n * n)
            )
            / denom
        )

        lower = max(
            0.0,
            center - half,
        )

        upper = min(
            1.0,
            center + half,
        )

        rows.append(
            {
                "bin": int(bin_id) + 1,
                "n": int(n),
                "mean_predicted_probability": mean_p,
                "observed_pathogenic_fraction": float(observed),
                "observed_95CI_lower": float(lower),
                "observed_95CI_upper": float(upper),
            }
        )

    return pd.DataFrame(
        rows
    )


def expected_calibration_error(
    y_true,
    prob,
    n_bins=10,
):
    """
    Weighted ECE:

        sum_k (n_k / N) * | observed_k - predicted_k |

    Uses the same quantile-based bins as the reliability curve.
    """

    bins = make_quantile_bins(
        y_true,
        prob,
        n_bins=n_bins,
    )

    total_n = bins["n"].sum()

    ece = np.sum(
        (
            bins["n"]
            / total_n
        )
        *
        np.abs(
            bins["observed_pathogenic_fraction"]
            -
            bins["mean_predicted_probability"]
        )
    )

    return float(ece)


def compute_calibration_metrics(
    y_true,
    prob,
):
    """
    Returns:
        Brier
        ECE
        calibration intercept
        calibration slope
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    prob = np.asarray(
        prob,
        dtype=float,
    )

    brier = float(
        brier_score_loss(
            y_true,
            prob,
        )
    )

    ece = expected_calibration_error(
        y_true,
        prob,
        n_bins=N_BINS,
    )

    intercept, slope = (
        calibration_intercept_slope(
            y_true,
            prob,
        )
    )

    return {
        "Brier_score": brier,
        "ECE": ece,
        "Calibration_intercept": intercept,
        "Calibration_slope": slope,
    }


def stratified_bootstrap_metrics(
    y_true,
    prob,
    n_bootstrap=1000,
    seed=2026,
):
    """
    Stratified bootstrap:
    - resample positives with replacement
    - resample negatives with replacement
    - preserve original class counts
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    prob = np.asarray(
        prob,
        dtype=float,
    )

    pos_idx = np.where(
        y_true == 1
    )[0]

    neg_idx = np.where(
        y_true == 0
    )[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(
            "Both positive and negative samples are required."
        )

    rng = np.random.default_rng(
        seed
    )

    records = []

    for b in range(
        n_bootstrap
    ):

        sampled_pos = rng.choice(
            pos_idx,
            size=len(pos_idx),
            replace=True,
        )

        sampled_neg = rng.choice(
            neg_idx,
            size=len(neg_idx),
            replace=True,
        )

        idx = np.concatenate(
            [
                sampled_pos,
                sampled_neg,
            ]
        )

        m = compute_calibration_metrics(
            y_true[idx],
            prob[idx],
        )

        records.append(
            m
        )

    return pd.DataFrame(
        records
    )


def percentile_ci(
    values,
):
    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    if len(values) == 0:
        return np.nan, np.nan

    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


# =============================================================================
# 4. Read data
# =============================================================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

for col in [
    LABEL_COL,
    PROB_COL,
]:
    if col not in df.columns:
        raise KeyError(
            f"Missing required column: {col}\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )


work = df[
    [
        LABEL_COL,
        PROB_COL,
    ]
].copy()


work[LABEL_COL] = pd.to_numeric(
    work[LABEL_COL],
    errors="coerce",
)

work[PROB_COL] = pd.to_numeric(
    work[PROB_COL],
    errors="coerce",
)


before_n = len(
    work
)

work = work.dropna(
    subset=[
        LABEL_COL,
        PROB_COL,
    ]
).copy()


work[LABEL_COL] = (
    work[LABEL_COL]
    .astype(int)
)


# =============================================================================
# 5. Data validation
# =============================================================================

labels = set(
    work[LABEL_COL]
    .unique()
)

if not labels.issubset(
    {0, 1}
):
    raise ValueError(
        f"label must be 0/1; observed: {labels}"
    )


if (
    (work[PROB_COL] < 0).any()
    or
    (work[PROB_COL] > 1).any()
):
    bad_min = float(
        work[PROB_COL].min()
    )

    bad_max = float(
        work[PROB_COL].max()
    )

    raise ValueError(
        f"{PROB_COL} must be a probability in [0,1]. "
        f"Observed range: [{bad_min}, {bad_max}]"
    )


y = work[
    LABEL_COL
].to_numpy(
    dtype=int
)

prob = work[
    PROB_COL
].to_numpy(
    dtype=float
)


N = len(
    work
)

N_POS = int(
    (y == 1).sum()
)

N_NEG = int(
    (y == 0).sum()
)


print("=" * 100)
print("GenProt-DSM Calibration Analysis")
print("=" * 100)

print(
    f"Input rows               : {before_n:,}"
)

print(
    f"Usable rows              : {N:,}"
)

print(
    f"Positive                 : {N_POS:,}"
)

print(
    f"Negative                 : {N_NEG:,}"
)

print(
    f"Predicted probability min: {np.min(prob):.6f}"
)

print(
    f"Predicted probability max: {np.max(prob):.6f}"
)


# =============================================================================
# 6. Point estimates
# =============================================================================

point = compute_calibration_metrics(
    y,
    prob,
)

print("\nPoint estimates:")

for k, v in point.items():
    print(
        f"  {k:24s} = {v:.6f}"
    )


# =============================================================================
# 7. Calibration bins
# =============================================================================

bins_df = make_quantile_bins(
    y,
    prob,
    n_bins=N_BINS,
)

bins_df.to_csv(
    BINS_FILE,
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 8. 1000 stratified bootstrap CIs
# =============================================================================

print(
    f"\nRunning {N_BOOTSTRAP} stratified bootstrap replicates..."
)

boot_df = stratified_bootstrap_metrics(
    y_true=y,
    prob=prob,
    n_bootstrap=N_BOOTSTRAP,
    seed=RANDOM_SEED,
)


metrics_rows = []

for metric_name in [
    "Brier_score",
    "ECE",
    "Calibration_intercept",
    "Calibration_slope",
]:

    ci_low, ci_high = percentile_ci(
        boot_df[
            metric_name
        ]
    )

    metrics_rows.append(
        {
            "Metric": metric_name,
            "Estimate": point[metric_name],
            "CI95_lower": ci_low,
            "CI95_upper": ci_high,
            "Ideal_value": (
                1.0
                if metric_name == "Calibration_slope"
                else 0.0
            ),
            "N": N,
            "Bootstrap_replicates": N_BOOTSTRAP,
        }
    )


metrics_df = pd.DataFrame(
    metrics_rows
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 9. Plot calibration curve
# =============================================================================

x = bins_df[
    "mean_predicted_probability"
].to_numpy(
    dtype=float
)

y_obs = bins_df[
    "observed_pathogenic_fraction"
].to_numpy(
    dtype=float
)

y_low = bins_df[
    "observed_95CI_lower"
].to_numpy(
    dtype=float
)

y_high = bins_df[
    "observed_95CI_upper"
].to_numpy(
    dtype=float
)


yerr = np.vstack(
    [
        y_obs - y_low,
        y_high - y_obs,
    ]
)


fig, ax = plt.subplots(
    figsize=(6.4, 5.4)
)


# Ideal calibration line
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Perfect calibration",
)


# Reliability curve
ax.errorbar(
    x,
    y_obs,
    yerr=yerr,
    marker="o",
    markersize=5,
    linewidth=1.8,
    capsize=3,
    label="GenProt-DSM",
)


ax.set_xlim(
    0.0,
    1.0,
)

ax.set_ylim(
    0.0,
    1.0,
)

ax.set_xlabel(
    "Mean predicted pathogenicity probability",
    fontsize=11,
)

ax.set_ylabel(
    "Observed pathogenic fraction",
    fontsize=11,
)

ax.set_title(
    "Calibration of GenProt-DSM",
    fontsize=12,
)

ax.legend(
    frameon=False,
    fontsize=9,
    loc="upper left",
)

ax.grid(
    alpha=0.25,
)


# Compact metric annotation
annotation = (
    f"Brier = {point['Brier_score']:.3f}\n"
    f"ECE = {point['ECE']:.3f}\n"
    f"Intercept = {point['Calibration_intercept']:.3f}\n"
    f"Slope = {point['Calibration_slope']:.3f}"
)

ax.text(
    0.98,
    0.03,
    annotation,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        alpha=0.85,
        edgecolor="0.7",
    ),
)


fig.tight_layout()


# High-resolution PNG
fig.savefig(
    PNG_FILE,
    dpi=600,
    bbox_inches="tight",
)


# Vector PDF
fig.savefig(
    PDF_FILE,
    bbox_inches="tight",
)


plt.close(
    fig
)


# =============================================================================
# 10. Final output
# =============================================================================

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

print(
    f"\nPNG:\n{PNG_FILE}"
)

print(
    f"\nPDF:\n{PDF_FILE}"
)

print(
    f"\nMetrics CSV:\n{METRICS_FILE}"
)

print(
    f"\nCalibration bins CSV:\n{BINS_FILE}"
)

print(
    "\nCalibration uses RAW prob_ensemble probabilities."
)

print(
    "The classification threshold 0.5543934838088572 is NOT used here."
)
