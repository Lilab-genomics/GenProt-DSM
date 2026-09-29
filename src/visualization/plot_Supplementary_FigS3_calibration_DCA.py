#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supplementary Fig. S3
Calibration and decision-curve analysis of GenProt-DSM

Panels
------
(a) Calibration curve
    - Directly reads the EXISTING final calibration-bin output.
    - Does NOT regenerate the plotted calibration curve.
    - 10 quantile bins + Wilson 95% CI are therefore exactly inherited
      from the user's previous calibration analysis.

(b) Predicted pathogenicity-probability distribution
    - Uses final per-variant GenProt-DSM probabilities on VarGeneDisjointTest.
    - Shows Benign and Pathogenic distributions.
    - The pre-specified ensemble decision threshold is read from the final
      model summary JSON and is shown only for classification context.
    - The threshold is NOT used for calibration.

(c) Decision curve analysis
    - GenProt-DSM vs Treat all vs Treat none.
    - Net benefit:
          NB(pt) = TP/N - FP/N * pt/(1-pt)
    - Because VarGeneDisjointTest is a balanced benchmark, this DCA reflects
      benchmark-level net benefit rather than population-level clinical utility.

(d) Threshold-specific net-benefit gain
    - Delta NB = NB_GenProtDSM - max(NB_TreatAll, NB_TreatNone)
    - Positive values indicate improvement over the better default strategy.
    - The pre-specified ensemble threshold is marked.

Primary inputs
--------------
1) Existing calibration bins:
   F:\\GenProt-DSM_Resubmit\\calibration\\GenProt_DSM_calibration_bins.csv

   Required columns:
       bin
       n
       mean_predicted_probability
       observed_pathogenic_fraction
       observed_95CI_lower
       observed_95CI_upper

2) Existing calibration metrics:
   F:\\GenProt-DSM_Resubmit\\calibration\\GenProt_DSM_calibration_metrics.csv

   Required columns:
       Metric
       Estimate
       CI95_lower
       CI95_upper
       Ideal_value
       N
       Bootstrap_replicates

3) Final VarGeneDisjointTest prediction table:
   F:\\GenProt-DSM_Resubmit\\dataset\\data\\test_score.csv

   Required columns:
       label
       GenProt-DSM

4) Final model summary:
   F:\\GenProt-DSM_Resubmit\\result\\pred\\ablation\\GenProt-DSM\\ckpt\\test_ensemble_summary.json

   Expected key:
       ensemble_threshold
   A recursive search is used in case the key is nested.

Outputs
-------
F:\\GenProt-DSM_Resubmit\\fig_Results\\Supplementary_FigS3\\

    Supplementary_FigS3_calibration_DCA.png
    Supplementary_FigS3_calibration_DCA.pdf
    FigS3_decision_curve.csv
    FigS3_analysis_summary.csv

Scientific safeguards
---------------------
- Calibration panel (a) is plotted from the existing calibration output.
- The script independently verifies that those calibration outputs correspond
  to the current final test_score.csv by checking:
      * total N
      * class counts
      * Brier score
      * 10-bin quantile calibration coordinates
- If the existing calibration files do not match the final test predictions,
  the script stops instead of silently mixing different result versions.
- The classification threshold is NEVER used for calibration calculations.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss


# =============================================================================
# 1. PATHS
# =============================================================================

CALIBRATION_BINS_CSV = (
    r"F:\GenProt-DSM_Resubmit\result\calibration"
    r"\GenProt_DSM_calibration_bins.csv"
)

CALIBRATION_METRICS_CSV = (
    r"F:\GenProt-DSM_Resubmit\result\calibration"
    r"\GenProt_DSM_calibration_metrics.csv"
)

TEST_SCORE_CSV = (
    r"F:\GenProt-DSM_Resubmit\dataset\data"
    r"\test_score.csv"
)

MODEL_SUMMARY_JSON = (
    r"F:\GenProt-DSM_Resubmit\result\pred\ablation"
    r"\GenProt-DSM\ckpt\test_ensemble_summary.json"
)

OUTPUT_DIR = (
    r"F:\GenProt-DSM_Resubmit\fig_Results"
    r"\Supplementary_FigS3"
)

OUTPUT_BASENAME = "Supplementary_FigS3_calibration_DCA"

DECISION_CURVE_CSV = os.path.join(
    OUTPUT_DIR,
    "FigS3_decision_curve.csv",
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "FigS3_analysis_summary.csv",
)


# =============================================================================
# 2. DATA CONFIGURATION
# =============================================================================

LABEL_COL = "label"
PROB_COL = "GenProt-DSM"

EXPECTED_N = 786
EXPECTED_POS = 393
EXPECTED_NEG = 393

# Existing calibration analysis used 10 quantile bins.
N_CALIBRATION_BINS = 10

# Decision-curve range.
# Avoid thresholds extremely close to 0 or 1, where odds pt/(1-pt) become
# numerically/visually extreme and are not informative for this benchmark.
DCA_THRESHOLD_MIN = 0.05
DCA_THRESHOLD_MAX = 0.90
DCA_THRESHOLD_STEP = 0.005

# Consistency tolerance between the existing calibration files and current
# final test probabilities.
CALIBRATION_MATCH_ATOL = 1e-6


# =============================================================================
# 3. VISUAL STYLE
# =============================================================================

FIG_DPI = 600
BASE_FONT_SIZE = 9

# User's established manuscript palette.
TRAIN_COLOR = "#72BCD5"
TEST_COLOR = "#F7AA58"

BENIGN_COLOR = "#DEEAEA"
PATHOGENIC_COLOR = "#F7E474"

GENPROT_COLOR = "#D62728"
DEFAULT_GRAY = "0.45"
DARK_GRAY = "0.20"

matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": BASE_FONT_SIZE,
    "axes.labelsize": BASE_FONT_SIZE,
    "xtick.labelsize": BASE_FONT_SIZE,
    "ytick.labelsize": BASE_FONT_SIZE,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# =============================================================================
# 4. GENERAL HELPERS
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.105,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def save_figure(fig, output_base: str) -> None:
    png = output_base + ".png"
    pdf = output_base + ".pdf"

    fig.savefig(
        png,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")


def recursive_find_key(obj: Any, key: str) -> Optional[Any]:
    """
    Recursively search a nested JSON-like object for the first occurrence
    of `key`.
    """
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]

        for value in obj.values():
            found = recursive_find_key(value, key)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = recursive_find_key(value, key)
            if found is not None:
                return found

    return None


def load_ensemble_threshold(path: str) -> float:
    require_file(path)

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    value = recursive_find_key(
        obj,
        "ensemble_threshold",
    )

    if value is None:
        raise KeyError(
            "Could not find 'ensemble_threshold' anywhere in:\n"
            f"{path}"
        )

    threshold = float(value)

    if not (0.0 < threshold < 1.0):
        raise ValueError(
            f"Invalid ensemble_threshold={threshold}. "
            "Expected a probability strictly between 0 and 1."
        )

    return threshold


# =============================================================================
# 5. READ FINAL TEST PREDICTIONS
# =============================================================================

def load_test_predictions(path: str):
    require_file(path)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    for col in [LABEL_COL, PROB_COL]:
        if col not in df.columns:
            raise KeyError(
                f"Missing required column: {col}\n"
                f"Available columns:\n{list(df.columns)}"
            )

    work = df[
        [LABEL_COL, PROB_COL]
    ].copy()

    work[LABEL_COL] = pd.to_numeric(
        work[LABEL_COL],
        errors="coerce",
    )

    work[PROB_COL] = pd.to_numeric(
        work[PROB_COL],
        errors="coerce",
    )

    before_n = len(work)

    work = work.dropna(
        subset=[LABEL_COL, PROB_COL]
    ).copy()

    work[LABEL_COL] = (
        work[LABEL_COL]
        .astype(int)
    )

    labels = set(
        work[LABEL_COL].unique()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{LABEL_COL} must contain only 0/1; observed: {labels}"
        )

    if (
        (work[PROB_COL] < 0).any()
        or
        (work[PROB_COL] > 1).any()
    ):
        raise ValueError(
            f"{PROB_COL} must be within [0, 1]. "
            f"Observed range = "
            f"[{work[PROB_COL].min()}, {work[PROB_COL].max()}]"
        )

    y = work[LABEL_COL].to_numpy(
        dtype=int
    )

    prob = work[PROB_COL].to_numpy(
        dtype=float
    )

    n = len(work)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    print(f"[CHECK] Raw input rows : {before_n:,}")
    print(f"[CHECK] Usable rows    : {n:,}")
    print(f"[CHECK] Pathogenic     : {n_pos:,}")
    print(f"[CHECK] Benign         : {n_neg:,}")

    if n != EXPECTED_N:
        raise RuntimeError(
            f"Expected N={EXPECTED_N}, but found N={n}."
        )

    if n_pos != EXPECTED_POS or n_neg != EXPECTED_NEG:
        raise RuntimeError(
            "Final VarGeneDisjointTest class counts do not match the "
            f"expected {EXPECTED_POS}/{EXPECTED_NEG}. "
            f"Observed positive/negative = {n_pos}/{n_neg}."
        )

    return work, y, prob


# =============================================================================
# 6. READ EXISTING CALIBRATION OUTPUTS
# =============================================================================

def load_calibration_bins(path: str) -> pd.DataFrame:
    require_file(path)

    df = pd.read_csv(path)

    required = [
        "bin",
        "n",
        "mean_predicted_probability",
        "observed_pathogenic_fraction",
        "observed_95CI_lower",
        "observed_95CI_upper",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            "Calibration bins file is missing columns:\n"
            + "\n".join(missing)
        )

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Calibration bins file contains missing/non-numeric required values."
        )

    return (
        df[required]
        .sort_values("bin")
        .reset_index(drop=True)
    )


def load_calibration_metrics(path: str) -> pd.DataFrame:
    require_file(path)

    df = pd.read_csv(path)

    required = [
        "Metric",
        "Estimate",
        "CI95_lower",
        "CI95_upper",
        "Ideal_value",
        "N",
        "Bootstrap_replicates",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            "Calibration metrics file is missing columns:\n"
            + "\n".join(missing)
        )

    df["Metric"] = df["Metric"].astype(str)

    numeric_cols = [
        "Estimate",
        "CI95_lower",
        "CI95_upper",
        "Ideal_value",
        "N",
        "Bootstrap_replicates",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


def metric_value(metrics_df: pd.DataFrame, metric_name: str) -> float:
    sub = metrics_df.loc[
        metrics_df["Metric"] == metric_name,
        "Estimate",
    ]

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected exactly one '{metric_name}' row in calibration metrics; "
            f"found {len(sub)}."
        )

    return float(sub.iloc[0])


# =============================================================================
# 7. VERIFY THAT EXISTING CALIBRATION OUTPUTS MATCH FINAL TEST DATA
# =============================================================================

def make_quantile_bin_audit(
    y: np.ndarray,
    prob: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    """
    Recreate only the basic quantile-bin coordinates for consistency checking.
    The plotted panel still uses the EXISTING calibration output file.
    """
    tmp = pd.DataFrame({
        "y": y.astype(int),
        "p": prob.astype(float),
    })

    tmp["bin"] = pd.qcut(
        tmp["p"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    rows = []

    for bin_id, g in tmp.groupby(
        "bin",
        observed=True,
        sort=True,
    ):
        rows.append({
            "bin": int(bin_id) + 1,
            "n": int(len(g)),
            "mean_predicted_probability": float(g["p"].mean()),
            "observed_pathogenic_fraction": float(g["y"].mean()),
        })

    return pd.DataFrame(rows)


def verify_calibration_consistency(
    y: np.ndarray,
    prob: np.ndarray,
    bins_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:

    # ---- N audit from metrics output
    metric_n_values = (
        pd.to_numeric(
            metrics_df["N"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    if len(metric_n_values) == 0:
        raise RuntimeError(
            "No valid N value found in calibration metrics."
        )

    if any(
        int(v) != len(y)
        for v in metric_n_values
    ):
        raise RuntimeError(
            "Existing calibration metrics were not generated from the current "
            f"N={len(y)} test set. N values in metrics file: "
            f"{metric_n_values.tolist()}"
        )

    # ---- N audit from bins output
    bins_n = int(
        bins_df["n"].sum()
    )

    if bins_n != len(y):
        raise RuntimeError(
            "Calibration bins do not match current final test set size: "
            f"sum(bin n)={bins_n}, current N={len(y)}."
        )

    # ---- Brier audit
    current_brier = float(
        brier_score_loss(
            y,
            prob,
        )
    )

    stored_brier = metric_value(
        metrics_df,
        "Brier_score",
    )

    if not np.isclose(
        current_brier,
        stored_brier,
        atol=CALIBRATION_MATCH_ATOL,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Existing calibration outputs do not match current GenProt-DSM "
            "probabilities.\n"
            f"Current Brier = {current_brier:.10f}\n"
            f"Stored Brier  = {stored_brier:.10f}\n"
            "Please rerun the calibration analysis on the final "
            "VarGeneDisjointTest probabilities before creating Fig. S3."
        )

    # ---- Quantile-bin audit
    audit_bins = make_quantile_bin_audit(
        y,
        prob,
        N_CALIBRATION_BINS,
    )

    if len(audit_bins) != len(bins_df):
        raise RuntimeError(
            "Number of calibration bins differs between current predictions "
            f"and stored output: current={len(audit_bins)}, stored={len(bins_df)}."
        )

    compare_cols = [
        "n",
        "mean_predicted_probability",
        "observed_pathogenic_fraction",
    ]

    for col in compare_cols:
        a = audit_bins[col].to_numpy(dtype=float)
        b = bins_df[col].to_numpy(dtype=float)

        if not np.allclose(
            a,
            b,
            atol=CALIBRATION_MATCH_ATOL,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Existing calibration-bin output does not match current "
                f"final probabilities for column '{col}'.\n"
                "Please rerun the calibration analysis on the final test set."
            )

    print(
        "[CHECK] Existing calibration outputs match the current "
        "VarGeneDisjointTest probabilities."
    )


# =============================================================================
# 8. DECISION-CURVE ANALYSIS
# =============================================================================

def compute_decision_curve(
    y: np.ndarray,
    prob: np.ndarray,
) -> pd.DataFrame:

    n = len(y)
    prevalence = float(np.mean(y))

    thresholds = np.arange(
        DCA_THRESHOLD_MIN,
        DCA_THRESHOLD_MAX + DCA_THRESHOLD_STEP / 2.0,
        DCA_THRESHOLD_STEP,
    )

    rows = []

    for pt in thresholds:
        pred_positive = (
            prob >= pt
        )

        tp = int(
            np.sum(
                (pred_positive == 1)
                & (y == 1)
            )
        )

        fp = int(
            np.sum(
                (pred_positive == 1)
                & (y == 0)
            )
        )

        odds = (
            pt / (1.0 - pt)
        )

        nb_model = (
            tp / n
            -
            fp / n * odds
        )

        nb_none = 0.0

        nb_all = (
            prevalence
            -
            (1.0 - prevalence)
            * odds
        )

        best_default = max(
            nb_none,
            nb_all,
        )

        delta_nb = (
            nb_model
            -
            best_default
        )

        rows.append({
            "threshold_probability": float(pt),
            "TP": tp,
            "FP": fp,
            "net_benefit_GenProt_DSM": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": float(nb_none),
            "best_default_net_benefit": float(best_default),
            "delta_net_benefit_vs_best_default": float(delta_nb),
            "prevalence_in_benchmark": prevalence,
            "N": n,
        })

    return pd.DataFrame(rows)


def interpolate_at_threshold(
    dca_df: pd.DataFrame,
    threshold: float,
    column: str,
) -> float:
    x = dca_df[
        "threshold_probability"
    ].to_numpy(dtype=float)

    y = dca_df[
        column
    ].to_numpy(dtype=float)

    return float(
        np.interp(
            threshold,
            x,
            y,
        )
    )


# =============================================================================
# 9. PANEL A — CALIBRATION CURVE
# =============================================================================

def draw_panel_a(
    ax,
    bins_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:

    x = bins_df[
        "mean_predicted_probability"
    ].to_numpy(dtype=float)

    y = bins_df[
        "observed_pathogenic_fraction"
    ].to_numpy(dtype=float)

    low = bins_df[
        "observed_95CI_lower"
    ].to_numpy(dtype=float)

    high = bins_df[
        "observed_95CI_upper"
    ].to_numpy(dtype=float)

    yerr = np.vstack([
        y - low,
        high - y,
    ])

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.1,
        color=DEFAULT_GRAY,
        label="Perfect calibration",
        zorder=1,
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        markersize=4.8,
        linewidth=1.7,
        capsize=2.5,
        color=GENPROT_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="GenProt-DSM",
        zorder=3,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(
        "Mean predicted pathogenicity probability"
    )

    ax.set_ylabel(
        "Observed pathogenic fraction"
    )

    ax.set_title(
        "Calibration curve"
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.55,
        alpha=0.6,
    )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=8,
    )

    # Use the already generated calibration metrics only as compact,
    # unboxed supporting information.
    brier = metric_value(
        metrics_df,
        "Brier_score",
    )

    ece = metric_value(
        metrics_df,
        "ECE",
    )

    intercept = metric_value(
        metrics_df,
        "Calibration_intercept",
    )

    slope = metric_value(
        metrics_df,
        "Calibration_slope",
    )

    metric_text = (
        f"Brier = {brier:.3f}\n"
        f"ECE = {ece:.3f}\n"
        f"Intercept = {intercept:.3f}\n"
        f"Slope = {slope:.3f}"
    )

    ax.text(
        0.97,
        0.04,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=DARK_GRAY,
    )

    panel_label(
        ax,
        "a",
    )


# =============================================================================
# 10. PANEL B — PROBABILITY DISTRIBUTION
# =============================================================================

def draw_panel_b(
    ax,
    y: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> None:

    benign = prob[
        y == 0
    ]

    pathogenic = prob[
        y == 1
    ]

    bins = np.linspace(
        0.0,
        1.0,
        26,
    )

    ax.hist(
        benign,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.75,
        color=BENIGN_COLOR,
        edgecolor="white",
        linewidth=0.55,
        label=f"Benign (n={len(benign)})",
    )

    ax.hist(
        pathogenic,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.72,
        color=PATHOGENIC_COLOR,
        edgecolor="white",
        linewidth=0.55,
        label=f"Pathogenic (n={len(pathogenic)})",
    )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=1.25,
        color=GENPROT_COLOR,
        label=f"Decision threshold = {threshold:.3f}",
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        "Predicted pathogenicity probability"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Predicted probability distribution"
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.55,
        alpha=0.6,
    )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=7.5,
    )

    panel_label(
        ax,
        "b",
    )


# =============================================================================
# 11. PANEL C — DECISION CURVE
# =============================================================================

def draw_panel_c(
    ax,
    dca_df: pd.DataFrame,
) -> None:

    x = dca_df[
        "threshold_probability"
    ].to_numpy(dtype=float)

    nb_model = dca_df[
        "net_benefit_GenProt_DSM"
    ].to_numpy(dtype=float)

    nb_all = dca_df[
        "net_benefit_treat_all"
    ].to_numpy(dtype=float)

    nb_none = dca_df[
        "net_benefit_treat_none"
    ].to_numpy(dtype=float)

    ax.plot(
        x,
        nb_model,
        color=GENPROT_COLOR,
        linewidth=2.0,
        label="GenProt-DSM",
        zorder=3,
    )

    ax.plot(
        x,
        nb_all,
        color=DEFAULT_GRAY,
        linewidth=1.15,
        linestyle="--",
        label="Treat all",
        zorder=2,
    )

    ax.plot(
        x,
        nb_none,
        color=DARK_GRAY,
        linewidth=1.0,
        linestyle=":",
        label="Treat none",
        zorder=1,
    )

    ax.set_xlim(
        DCA_THRESHOLD_MIN,
        DCA_THRESHOLD_MAX,
    )

    # Focus the displayed range on the clinically interpretable/useful region.
    # Treat-all may fall below the lower visible limit at high thresholds.
    top = max(
        0.55,
        float(np.nanmax(nb_model)) + 0.04,
    )

    ax.set_ylim(
        -0.12,
        top,
    )

    ax.set_xlabel(
        "Threshold probability"
    )

    ax.set_ylabel(
        "Net benefit"
    )

    ax.set_title(
        "Decision curve analysis"
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.55,
        alpha=0.6,
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=7.5,
    )

    panel_label(
        ax,
        "c",
    )


# =============================================================================
# 12. PANEL D — NET-BENEFIT GAIN
# =============================================================================

def draw_panel_d(
    ax,
    dca_df: pd.DataFrame,
    threshold: float,
) -> None:

    x = dca_df[
        "threshold_probability"
    ].to_numpy(dtype=float)

    delta = dca_df[
        "delta_net_benefit_vs_best_default"
    ].to_numpy(dtype=float)

    ax.axhline(
        0.0,
        color=DEFAULT_GRAY,
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    ax.plot(
        x,
        delta,
        color=GENPROT_COLOR,
        linewidth=1.9,
        zorder=3,
    )

    if (
        DCA_THRESHOLD_MIN
        <= threshold
        <= DCA_THRESHOLD_MAX
    ):
        delta_at_threshold = interpolate_at_threshold(
            dca_df,
            threshold,
            "delta_net_benefit_vs_best_default",
        )

        ax.axvline(
            threshold,
            color=GENPROT_COLOR,
            linestyle=":",
            linewidth=1.15,
            zorder=2,
        )

        ax.scatter(
            [threshold],
            [delta_at_threshold],
            s=28,
            color=GENPROT_COLOR,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )

        ax.annotate(
            f"Threshold = {threshold:.3f}\n"
            f"ΔNB = {delta_at_threshold:.3f}",
            xy=(
                threshold,
                delta_at_threshold,
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=7.5,
            ha="left",
            va="bottom",
            color=DARK_GRAY,
        )

    ax.set_xlim(
        DCA_THRESHOLD_MIN,
        DCA_THRESHOLD_MAX,
    )

    ax.set_xlabel(
        "Threshold probability"
    )

    ax.set_ylabel(
        "Net-benefit gain over default strategy"
    )

    ax.set_title(
        "Threshold-specific net-benefit gain"
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.55,
        alpha=0.6,
    )

    panel_label(
        ax,
        "d",
    )


# =============================================================================
# 13. SAVE ANALYSIS SUMMARY
# =============================================================================

def build_summary(
    y: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    metrics_df: pd.DataFrame,
    dca_df: pd.DataFrame,
) -> pd.DataFrame:

    prevalence = float(
        np.mean(y)
    )

    rows = [
        {
            "item": "N",
            "value": len(y),
            "note": "VarGeneDisjointTest usable samples",
        },
        {
            "item": "N_pathogenic",
            "value": int((y == 1).sum()),
            "note": "",
        },
        {
            "item": "N_benign",
            "value": int((y == 0).sum()),
            "note": "",
        },
        {
            "item": "Benchmark_prevalence",
            "value": prevalence,
            "note": (
                "Balanced benchmark prevalence; not a population prevalence estimate"
            ),
        },
        {
            "item": "Ensemble_decision_threshold",
            "value": threshold,
            "note": (
                "Read from final model summary; not used for calibration"
            ),
        },
    ]

    for metric_name in [
        "Brier_score",
        "ECE",
        "Calibration_intercept",
        "Calibration_slope",
    ]:
        sub = metrics_df.loc[
            metrics_df["Metric"] == metric_name
        ]

        if len(sub) == 1:
            r = sub.iloc[0]

            rows.append({
                "item": metric_name,
                "value": float(r["Estimate"]),
                "note": (
                    f"95% CI [{float(r['CI95_lower']):.6f}, "
                    f"{float(r['CI95_upper']):.6f}]"
                ),
            })

    if (
        DCA_THRESHOLD_MIN
        <= threshold
        <= DCA_THRESHOLD_MAX
    ):
        nb = interpolate_at_threshold(
            dca_df,
            threshold,
            "net_benefit_GenProt_DSM",
        )

        delta = interpolate_at_threshold(
            dca_df,
            threshold,
            "delta_net_benefit_vs_best_default",
        )

        rows.extend([
            {
                "item": "Net_benefit_at_ensemble_threshold",
                "value": nb,
                "note": "Benchmark-level decision-curve estimate",
            },
            {
                "item": "Delta_net_benefit_at_ensemble_threshold",
                "value": delta,
                "note": "Gain over better of Treat all / Treat none",
            },
        ])

    return pd.DataFrame(rows)


# =============================================================================
# 14. MAIN
# =============================================================================

def main() -> None:
    ensure_dir(
        OUTPUT_DIR
    )

    print("=" * 96)
    print("SUPPLEMENTARY FIG. S3")
    print("Calibration and decision-curve analysis")
    print("=" * 96)

    for path in [
        CALIBRATION_BINS_CSV,
        CALIBRATION_METRICS_CSV,
        TEST_SCORE_CSV,
        MODEL_SUMMARY_JSON,
    ]:
        print(f"[INPUT] {path}")
        require_file(path)

    # -------------------------------------------------------------------------
    # Final per-variant probabilities
    # -------------------------------------------------------------------------

    work, y, prob = load_test_predictions(
        TEST_SCORE_CSV
    )

    # -------------------------------------------------------------------------
    # Existing calibration outputs
    # -------------------------------------------------------------------------

    bins_df = load_calibration_bins(
        CALIBRATION_BINS_CSV
    )

    metrics_df = load_calibration_metrics(
        CALIBRATION_METRICS_CSV
    )

    # Critical version-consistency audit.
    verify_calibration_consistency(
        y,
        prob,
        bins_df,
        metrics_df,
    )

    print("\n[CALIBRATION METRICS]")
    print(
        metrics_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Final model threshold
    # -------------------------------------------------------------------------

    ensemble_threshold = load_ensemble_threshold(
        MODEL_SUMMARY_JSON
    )

    print(
        f"\n[CHECK] Ensemble decision threshold = "
        f"{ensemble_threshold:.12f}"
    )

    print(
        "[CHECK] This threshold is shown only for classification/decision "
        "context and is NOT used to compute calibration."
    )

    # -------------------------------------------------------------------------
    # Decision curve
    # -------------------------------------------------------------------------

    dca_df = compute_decision_curve(
        y,
        prob,
    )

    dca_df.to_csv(
        DECISION_CURVE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[SAVE] {DECISION_CURVE_CSV}"
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary_df = build_summary(
        y,
        prob,
        ensemble_threshold,
        metrics_df,
        dca_df,
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[SAVE] {SUMMARY_CSV}"
    )

    # -------------------------------------------------------------------------
    # Plot 2 x 2
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.4, 8.4),
        dpi=FIG_DPI,
    )

    draw_panel_a(
        axes[0, 0],
        bins_df,
        metrics_df,
    )

    draw_panel_b(
        axes[0, 1],
        y,
        prob,
        ensemble_threshold,
    )

    draw_panel_c(
        axes[1, 0],
        dca_df,
    )

    draw_panel_d(
        axes[1, 1],
        dca_df,
        ensemble_threshold,
    )

    # Important interpretation note for the balanced benchmark.
    fig.text(
        0.5,
        0.014,
        (
            "Decision-curve results reflect the balanced VarGeneDisjointTest "
            "benchmark prevalence (393/786) and should not be interpreted as "
            "population-level clinical utility estimates."
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=DARK_GRAY,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.96,
        bottom=0.095,
        wspace=0.27,
        hspace=0.30,
    )

    output_base = os.path.join(
        OUTPUT_DIR,
        OUTPUT_BASENAME,
    )

    save_figure(
        fig,
        output_base,
    )

    plt.close(fig)

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)
    print("PNG:", output_base + ".png")
    print("PDF:", output_base + ".pdf")
    print("DCA CSV:", DECISION_CURVE_CSV)
    print("Summary CSV:", SUMMARY_CSV)


if __name__ == "__main__":
    main()
