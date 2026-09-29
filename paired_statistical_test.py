#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paired statistical comparison on the MAXIMUM COMMON SUBSET of:
    GenProt-DSM
    AlphaMissense
    CADD
    PrimateAI
    MPC
    DEOGEN2

Tests:
1) AUROC: paired DeLong test
2) AUPR : paired stratified bootstrap (10,000 replicates)
3) MCC  : paired stratified bootstrap (10,000 replicates)

All six methods use exactly the same samples.
The same bootstrap indices are used for GenProt-DSM and its comparator.

Fixed thresholds:
- GenProt-DSM : 0.5543934838088572
- AlphaMissense: 0.564
- CADD PHRED  : 15
- PrimateAI   : 0.803
- MPC         : 2.0
- DEOGEN2     : 0.5

Multiple comparisons:
Holm correction is performed separately for the five AUROC tests,
five AUPR tests, and five MCC tests.

AUPR definition:
average_precision_score (AP), matching the existing comparison script.

IMPORTANT:
CADD=15 is valid only if the column contains CADD PHRED/scaled scores.
"""

import os
import math
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
)


# =============================================================================
# 1. Paths
# =============================================================================

INPUT_FILE = r"F:\GenProt-DSM_Resubmit\data\test_score.csv"

OUTPUT_DIR = r"F:\GenProt-DSM_Resubmit\metrics_result"

COMMON_SUBSET_FILE = os.path.join(
    OUTPUT_DIR,
    "paired_tests_six_methods_common_subset.csv"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "paired_statistical_tests_GenProtDSM_vs_baselines.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. Configuration
# =============================================================================

LABEL_COL = "label"
OURS = "GenProt-DSM"

BASELINES = [
    "AlphaMissense",
    "CADD",
    "PrimateAI",
    "MPC",
    "DEOGEN2",
]

METHODS = [
    OURS,
    *BASELINES,
]

# For these six score columns, higher = more pathogenic.
THRESHOLDS = {
    "GenProt-DSM": 0.5543934838088572,
    "AlphaMissense": 0.564,
    "CADD": 15.0,       # PHRED/scaled CADD only
    "PrimateAI": 0.803,
    "MPC": 2.0,
    "DEOGEN2": 0.5,
}

N_BOOTSTRAP = 10000
RANDOM_SEED = 2026


# =============================================================================
# 3. DeLong implementation
# =============================================================================

def compute_midrank(x):
    """
    1-based midranks.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)

    rank_sorted = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i + 1

        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1

        rank_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j

    ranks = np.empty(n, dtype=float)
    ranks[order] = rank_sorted

    return ranks


def fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong covariance calculation.

    predictions_sorted_transposed:
        shape (n_classifiers, n_samples)
        positive samples MUST appear first.
    """

    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    if m < 2 or n < 2:
        raise ValueError(
            "DeLong requires at least 2 positive and 2 negative samples."
        )

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(
            predictions_sorted_transposed[r, :]
        )

    aucs = (
        tz[:, :m].sum(axis=1) / m / n
        - (m + 1.0) / (2.0 * n)
    )

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    delong_cov = sx / m + sy / n

    return aucs, delong_cov


def paired_delong_test(y_true, score_a, score_b):
    """
    Returns paired AUROC comparison:
      score_a - score_b
    """

    y_true = np.asarray(y_true, dtype=int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)

    # Positive samples first.
    order = np.argsort(-y_true)

    y_sorted = y_true[order]

    predictions = np.vstack([
        score_a[order],
        score_b[order],
    ])

    n_pos = int(y_sorted.sum())

    aucs, covariance = fast_delong(
        predictions,
        n_pos,
    )

    auc_a = float(aucs[0])
    auc_b = float(aucs[1])

    diff = auc_a - auc_b

    contrast = np.array([1.0, -1.0], dtype=float)

    variance = float(
        contrast @ covariance @ contrast.T
    )

    variance = max(variance, 0.0)
    se = math.sqrt(variance)

    if se == 0.0:

        if diff == 0.0:
            z = 0.0
            p = 1.0
        else:
            z = np.sign(diff) * np.inf
            p = 0.0

        ci_low = diff
        ci_high = diff

    else:

        z = diff / se

        # two-sided normal p-value
        p = math.erfc(
            abs(z) / math.sqrt(2.0)
        )

        z975 = 1.959963984540054

        ci_low = diff - z975 * se
        ci_high = diff + z975 * se

    return {
        "ours": auc_a,
        "baseline": auc_b,
        "difference": float(diff),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "p_value": float(p),
        "z": float(z),
    }


# =============================================================================
# 4. Bootstrap
# =============================================================================

def classify_fixed(score, method):
    score = np.asarray(score, dtype=float)

    return (
        score >= THRESHOLDS[method]
    ).astype(int)


def paired_stratified_bootstrap(
    y_true,
    ours_score,
    baseline_score,
    baseline_name,
    n_bootstrap,
    seed,
):
    """
    Stratified paired bootstrap.

    Positive and negative samples are sampled separately with replacement,
    preserving the original class counts.

    Crucially, the SAME sampled row indices are applied to both methods.
    """

    y_true = np.asarray(y_true, dtype=int)
    ours_score = np.asarray(ours_score, dtype=float)
    baseline_score = np.asarray(baseline_score, dtype=float)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) < 2 or len(neg_idx) < 2:
        raise ValueError(
            "Bootstrap needs at least 2 positive and 2 negative samples."
        )

    rng = np.random.default_rng(seed)

    aupr_diff = np.empty(
        n_bootstrap,
        dtype=float,
    )

    mcc_diff = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for b in range(n_bootstrap):

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

        sampled_idx = np.concatenate([
            sampled_pos,
            sampled_neg,
        ])

        yb = y_true[sampled_idx]

        ours_b = ours_score[sampled_idx]
        base_b = baseline_score[sampled_idx]

        # AUPR / Average Precision
        ours_aupr = average_precision_score(
            yb,
            ours_b,
        )

        base_aupr = average_precision_score(
            yb,
            base_b,
        )

        aupr_diff[b] = (
            ours_aupr - base_aupr
        )

        # MCC with fixed, external/prespecified thresholds
        ours_pred = classify_fixed(
            ours_b,
            OURS,
        )

        base_pred = classify_fixed(
            base_b,
            baseline_name,
        )

        ours_mcc = matthews_corrcoef(
            yb,
            ours_pred,
        )

        base_mcc = matthews_corrcoef(
            yb,
            base_pred,
        )

        mcc_diff[b] = (
            ours_mcc - base_mcc
        )

    return aupr_diff, mcc_diff


def bootstrap_inference(
    observed_difference,
    bootstrap_differences,
):
    """
    95% percentile CI + two-sided null-centered bootstrap p-value.
    """

    d = np.asarray(
        bootstrap_differences,
        dtype=float,
    )

    d = d[np.isfinite(d)]

    if len(d) == 0:
        return {
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
        }

    ci_low = float(
        np.percentile(d, 2.5)
    )

    ci_high = float(
        np.percentile(d, 97.5)
    )

    # Null-centered bootstrap distribution:
    # shift the bootstrap distribution so H0 is difference=0.
    centered = (
        d - float(observed_difference)
    )

    p = (
        np.sum(
            np.abs(centered)
            >= abs(float(observed_difference))
        )
        + 1
    ) / (
        len(centered) + 1
    )

    return {
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "p_value": float(p),
    }


# =============================================================================
# 5. Holm correction
# =============================================================================

def holm_adjust(pvalues):
    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(pvalues)

    order = np.argsort(pvalues)
    sorted_p = pvalues[order]

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for i, p in enumerate(sorted_p):

        corrected = (
            (m - i) * p
        )

        running_max = max(
            running_max,
            corrected,
        )

        adjusted_sorted[i] = min(
            running_max,
            1.0,
        )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    adjusted[order] = adjusted_sorted

    return adjusted


# =============================================================================
# 6. Read data
# =============================================================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

required_columns = [
    LABEL_COL,
    *METHODS,
]

for col in required_columns:

    if col not in df.columns:
        raise KeyError(
            f"Missing required column: {col}\n"
            f"Available columns: {list(df.columns)}"
        )


# Numeric copies for matching/analysis.
numeric = pd.DataFrame(
    index=df.index
)

for col in required_columns:

    numeric[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


# =============================================================================
# 7. Exact maximum common subset
# =============================================================================

common_mask = pd.Series(
    True,
    index=df.index,
)

for col in required_columns:

    common_mask &= numeric[col].notna()


common_df = numeric.loc[
    common_mask,
    required_columns,
].copy()

common_df[LABEL_COL] = (
    common_df[LABEL_COL]
    .astype(int)
)

labels = set(
    common_df[LABEL_COL].unique()
)

if not labels.issubset({0, 1}):
    raise ValueError(
        f"label must be 0/1; observed: {labels}"
    )

if len(common_df) == 0:
    raise RuntimeError(
        "No common subset across the six methods."
    )

y = common_df[LABEL_COL].to_numpy(
    dtype=int
)

if len(np.unique(y)) < 2:
    raise RuntimeError(
        "Common subset contains only one class."
    )

N = len(common_df)
N_POS = int((y == 1).sum())
N_NEG = int((y == 0).sum())

print("=" * 100)
print("Paired statistical comparison")
print("=" * 100)

print(f"Original N      = {len(df):,}")
print(f"Common subset N = {N:,}")
print(f"Positive        = {N_POS:,}")
print(f"Negative        = {N_NEG:,}")


# =============================================================================
# 8. CADD score-scale safety check
# =============================================================================

cadd = common_df[
    "CADD"
].to_numpy(dtype=float)

print(
    "\n[CADD scale check] "
    f"min={np.min(cadd):.4f}, "
    f"median={np.median(cadd):.4f}, "
    f"max={np.max(cadd):.4f}"
)

if np.max(cadd) < 10:

    raise ValueError(
        "\nCADD values do not look like PHRED/scaled CADD scores "
        "(maximum < 10), but this script uses cutoff 15 for MCC.\n"
        "Please verify whether your CADD column is CADD_raw/rankscore."
    )


# Save exact original rows for audit.
df.loc[
    common_mask
].to_csv(
    COMMON_SUBSET_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(
    f"\nSaved common subset:\n"
    f"{COMMON_SUBSET_FILE}"
)


# =============================================================================
# 9. Run comparisons
# =============================================================================

ours_score = common_df[
    OURS
].to_numpy(dtype=float)

rows = []

for i, baseline in enumerate(
    BASELINES
):

    baseline_score = common_df[
        baseline
    ].to_numpy(dtype=float)

    print("\n" + "-" * 100)
    print(f"{OURS} vs {baseline}")
    print("-" * 100)


    # =========================================================================
    # AUROC: paired DeLong
    # =========================================================================

    delong = paired_delong_test(
        y_true=y,
        score_a=ours_score,
        score_b=baseline_score,
    )

    rows.append(
        {
            "baseline": baseline,
            "metric": "AUROC",
            "N": N,
            "GenProt-DSM": delong["ours"],
            "Baseline_value": delong["baseline"],
            "Difference_ours_minus_baseline": delong["difference"],
            "CI_lower": delong["ci_lower"],
            "CI_upper": delong["ci_upper"],
            "P_value": delong["p_value"],
            "Test": "Paired DeLong",
        }
    )

    print(
        f"AUROC: "
        f"ours={delong['ours']:.4f}, "
        f"baseline={delong['baseline']:.4f}, "
        f"diff={delong['difference']:.4f}, "
        f"95% CI=[{delong['ci_lower']:.4f}, {delong['ci_upper']:.4f}], "
        f"p={delong['p_value']:.6g}"
    )


    # =========================================================================
    # AUPR point estimates
    # =========================================================================

    ours_aupr = float(
        average_precision_score(
            y,
            ours_score,
        )
    )

    base_aupr = float(
        average_precision_score(
            y,
            baseline_score,
        )
    )

    aupr_diff_obs = (
        ours_aupr - base_aupr
    )


    # =========================================================================
    # MCC point estimates, using fixed thresholds
    # =========================================================================

    ours_pred = classify_fixed(
        ours_score,
        OURS,
    )

    base_pred = classify_fixed(
        baseline_score,
        baseline,
    )

    ours_mcc = float(
        matthews_corrcoef(
            y,
            ours_pred,
        )
    )

    base_mcc = float(
        matthews_corrcoef(
            y,
            base_pred,
        )
    )

    mcc_diff_obs = (
        ours_mcc - base_mcc
    )


    # =========================================================================
    # Paired stratified bootstrap
    # =========================================================================

    (
        aupr_boot_diff,
        mcc_boot_diff,
    ) = paired_stratified_bootstrap(
        y_true=y,
        ours_score=ours_score,
        baseline_score=baseline_score,
        baseline_name=baseline,
        n_bootstrap=N_BOOTSTRAP,
        seed=RANDOM_SEED + i,
    )


    aupr_inf = bootstrap_inference(
        observed_difference=aupr_diff_obs,
        bootstrap_differences=aupr_boot_diff,
    )

    mcc_inf = bootstrap_inference(
        observed_difference=mcc_diff_obs,
        bootstrap_differences=mcc_boot_diff,
    )


    rows.append(
        {
            "baseline": baseline,
            "metric": "AUPR",
            "N": N,
            "GenProt-DSM": ours_aupr,
            "Baseline_value": base_aupr,
            "Difference_ours_minus_baseline": aupr_diff_obs,
            "CI_lower": aupr_inf["ci_lower"],
            "CI_upper": aupr_inf["ci_upper"],
            "P_value": aupr_inf["p_value"],
            "Test": f"Paired stratified bootstrap ({N_BOOTSTRAP})",
        }
    )

    rows.append(
        {
            "baseline": baseline,
            "metric": "MCC",
            "N": N,
            "GenProt-DSM": ours_mcc,
            "Baseline_value": base_mcc,
            "Difference_ours_minus_baseline": mcc_diff_obs,
            "CI_lower": mcc_inf["ci_lower"],
            "CI_upper": mcc_inf["ci_upper"],
            "P_value": mcc_inf["p_value"],
            "Test": f"Paired stratified bootstrap ({N_BOOTSTRAP})",
        }
    )


    print(
        f"AUPR : "
        f"ours={ours_aupr:.4f}, "
        f"baseline={base_aupr:.4f}, "
        f"diff={aupr_diff_obs:.4f}, "
        f"95% CI=[{aupr_inf['ci_lower']:.4f}, {aupr_inf['ci_upper']:.4f}], "
        f"p={aupr_inf['p_value']:.6g}"
    )

    print(
        f"MCC  : "
        f"ours={ours_mcc:.4f}, "
        f"baseline={base_mcc:.4f}, "
        f"diff={mcc_diff_obs:.4f}, "
        f"95% CI=[{mcc_inf['ci_lower']:.4f}, {mcc_inf['ci_upper']:.4f}], "
        f"p={mcc_inf['p_value']:.6g}"
    )


# =============================================================================
# 10. Holm correction
# =============================================================================

result_df = pd.DataFrame(
    rows
)

result_df[
    "P_Holm"
] = np.nan

for metric_name in [
    "AUROC",
    "AUPR",
    "MCC",
]:

    mask = (
        result_df["metric"]
        == metric_name
    )

    pvals = result_df.loc[
        mask,
        "P_value"
    ].to_numpy(dtype=float)

    result_df.loc[
        mask,
        "P_Holm"
    ] = holm_adjust(
        pvals
    )


result_df[
    "Significant_Holm_0.05"
] = (
    result_df["P_Holm"]
    < 0.05
)


# =============================================================================
# 11. Sort and save
# =============================================================================

metric_order = {
    "AUROC": 0,
    "AUPR": 1,
    "MCC": 2,
}

baseline_order = {
    name: i
    for i, name in enumerate(BASELINES)
}

result_df[
    "_metric_order"
] = result_df[
    "metric"
].map(metric_order)

result_df[
    "_baseline_order"
] = result_df[
    "baseline"
].map(baseline_order)

result_df = (
    result_df
    .sort_values(
        by=[
            "_metric_order",
            "_baseline_order",
        ]
    )
    .drop(
        columns=[
            "_metric_order",
            "_baseline_order",
        ]
    )
    .reset_index(
        drop=True
    )
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 100)

print(
    f"Saved paired statistical tests:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nSaved exact common subset:\n"
    f"{COMMON_SUBSET_FILE}"
)

print("=" * 100)

print(
    result_df.to_string(
        index=False
    )
)
