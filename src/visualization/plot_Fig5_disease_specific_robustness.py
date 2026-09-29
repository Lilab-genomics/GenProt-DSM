#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig. 5 — Disease-specific evaluation and robustness analysis of GenProt-DSM

Panels
------
(a) Brain-specific vs General training
    - AUROC, AUPR, ACC, F1, and MCC
    - Metrics read directly from the root-level final_metrics block in two test_ensemble_summary.json files

(b) GenProt-DSM vs three disease-specific predictors
    - AUROC and AUPR
    - Metrics read from disease-specific comparison.csv

(c) Disease-stratified performance
    - AUROC and AUPR point estimates with bootstrap 95% CIs
    - Metrics read from disease_bootstrap_table.csv
    - Only disease strata present in the result file are plotted

(d) Rare-variant evaluation
    - AUROC and AUPR
    - VarRareTest is reconstructed from test_score.csv using:
          AF < 0.01
      with missing AF excluded
    - Tool-specific score directions are read from metrics_tools.csv when
      available, otherwise predefined directions are used
    - Each predictor is evaluated on rare variants for which that predictor
      has a valid score

Terminology
-----------
- Use "AUROC" and "AUPR" in all figure text.
- Display CADD as "CADD v1.7".
- GenProt-DSM is highlighted in red.

Output directory
----------------
F:\\GenProt-DSM_Resubmit\\fig_Results\\Fig5_disease_specific_robustness
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# =============================================================================
# 1. PATHS
# =============================================================================

BRAIN_JSON = (
    r"F:\GenProt-DSM_Resubmit\result\pred"
    r"\model\ckpt\test_ensemble_summary.json"
)

GENERAL_JSON = (
    r"F:\GenProt-DSM_Resubmit\result\pred"
    r"\General_model\ckpt\test_ensemble_summary.json"
)

DISEASE_SPECIFIC_CSV = (
    r"F:\GenProt-DSM_Resubmit\result"
    r"\disease-specific comparison.csv"
)

DISEASE_STRATIFIED_CSV = (
    r"F:\GenProt-DSM_Resubmit\result"
    r"\disease_bootstrap_table.csv"
)

TEST_SCORE_CSV = (
    r"F:\GenProt-DSM_Resubmit\dataset\data"
    r"\test_score.csv"
)

# Used for score directions in panel d.
METRICS_TOOLS_CSV = (
    r"F:\GenProt-DSM_Resubmit\result"
    r"\metrics_tools.csv"
)

OUTPUT_DIR = (
    r"F:\GenProt-DSM_Resubmit\fig_Results"
    r"\Fig5_disease_specific_robustness"
)

OUTPUT_BASENAME = "Fig5_disease_specific_robustness"


# =============================================================================
# 2. GLOBAL STYLE
# =============================================================================

FIG_DPI = 600
BASE_FONT = 9

matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": BASE_FONT,
    "axes.titlesize": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": BASE_FONT,
    "ytick.labelsize": BASE_FONT,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# =============================================================================
# 3. COLORS / DISPLAY NAMES
# =============================================================================

GENPROT_COLOR = "#D62728"
GENERAL_COLOR = "#7F7F7F"

AUROC_COLOR = "#4C78A8"
AUPR_COLOR = "#F58518"

DISEASE_TOOL_COLORS = {
    "GenProt-DSM": GENPROT_COLOR,
    "PathoPredictor-Epilepsy": "#4C78A8",
    "BAF-Wald": "#72B7B2",
    "PP-dist&evo": "#F2CF5B",
}

TOOL_COLUMNS: List[str] = [
    "SIFT",
    "Polyphen2",
    "FATHMM",
    "PROVEAN",
    "MPC",
    "PrimateAI",
    "DEOGEN2",
    "AlphaMissense",
    "CADD",
    "DANN",
    "GenoCanyon",
    "ESM1b",
    "GenProt-DSM",
]

DISPLAY_NAME: Dict[str, str] = {
    "SIFT": "SIFT",
    "Polyphen2": "PolyPhen2",
    "FATHMM": "FATHMM",
    "PROVEAN": "PROVEAN",
    "MPC": "MPC",
    "PrimateAI": "PrimateAI",
    "DEOGEN2": "DEOGEN2",
    "AlphaMissense": "AlphaMissense",
    "CADD": "CADD v1.7",
    "DANN": "DANN",
    "GenoCanyon": "GenoCanyon",
    "ESM1b": "ESM1b",
    "GenProt-DSM": "GenProt-DSM",
}

FALLBACK_DIRECTION: Dict[str, str] = {
    "SIFT": "low",
    "Polyphen2": "high",
    "FATHMM": "low",
    "PROVEAN": "low",
    "MPC": "high",
    "PrimateAI": "high",
    "DEOGEN2": "high",
    "AlphaMissense": "high",
    "CADD": "high",
    "DANN": "high",
    "GenoCanyon": "high",
    "ESM1b": "low",
    "GenProt-DSM": "high",
}


# =============================================================================
# 4. GENERAL UTILITIES
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.04):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def save_figure(fig, output_base: str):
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


def read_csv_flexible(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    return df


def normalize_key(x: Any) -> str:
    return (
        str(x)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("–", "")
    )


def resolve_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:

    normalized = {
        normalize_key(c): c
        for c in df.columns
    }

    for c in candidates:
        key = normalize_key(c)
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            "\nRequired column not found.\n"
            f"Candidates: {list(candidates)}\n"
            f"Existing columns: {list(df.columns)}"
        )

    return None


def metric_to_fraction(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    if not np.isfinite(v):
        return None

    if 1.0 < v <= 100.0:
        v /= 100.0

    return float(v)


# =============================================================================
# 5. PANEL a — DIRECT EXTRACTION OF ROOT-LEVEL final_metrics
# =============================================================================

def to_fraction(v: Any) -> float:
    """
    JSON final_metrics are stored as percentages in the current model outputs
    (e.g. 93.928...), whereas figures use the 0–1 scale.
    """
    value = float(v)

    if value > 1.0:
        value /= 100.0

    return value


def read_final_metrics(path: str) -> Dict[str, Any]:
    """
    Read ONLY the root-level `final_metrics` block.

    This is deliberate. The JSON also contains fold-level
    `test_metrics_single_model` blocks; those are NOT the final five-fold
    ensemble result and must not be used for Fig. 5a.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "final_metrics" not in obj:
        raise KeyError(
            f"'final_metrics' not found at the JSON root:\n{path}"
        )

    m = obj["final_metrics"]

    required = [
        "auc_roc",
        "auc_pr",
        "accuracy",
        "f1",
        "mcc",
    ]

    missing = [
        key
        for key in required
        if key not in m
    ]

    if missing:
        raise KeyError(
            f"Missing fields in root-level final_metrics: {missing}\n"
            f"JSON: {path}"
        )

    return {
        "AUROC": to_fraction(m["auc_roc"]),
        "AUPR": to_fraction(m["auc_pr"]),
        "ACC": to_fraction(m["accuracy"]),
        "F1": to_fraction(m["f1"]),
        "MCC": to_fraction(m["mcc"]),
        "Source_path": "final_metrics",
    }


def prepare_panel_a() -> pd.DataFrame:
    brain = read_final_metrics(
        BRAIN_JSON
    )

    general = read_final_metrics(
        GENERAL_JSON
    )

    rows = []

    for model_name, record in [
        ("Brain-specific", brain),
        ("General", general),
    ]:
        rows.append({
            "Model": model_name,
            "AUROC": float(record["AUROC"]),
            "AUPR": float(record["AUPR"]),
            "ACC": float(record["ACC"]),
            "F1": float(record["F1"]),
            "MCC": float(record["MCC"]),
            "Source_path": record["Source_path"],
        })

    return pd.DataFrame(
        rows
    )


def draw_panel_a(
    ax,
    df: pd.DataFrame,
):
    metrics = [
        "AUROC",
        "AUPR",
        "ACC",
        "F1",
        "MCC",
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.34

    brain = (
        df.loc[
            df["Model"] == "Brain-specific"
        ]
        .iloc[0]
    )

    general = (
        df.loc[
            df["Model"] == "General"
        ]
        .iloc[0]
    )

    brain_vals = [
        float(brain[m])
        for m in metrics
    ]

    general_vals = [
        float(general[m])
        for m in metrics
    ]

    # Use the same blue/orange palette as panel c, as requested.
    b1 = ax.bar(
        x - width / 2,
        brain_vals,
        width,
        label="Brain-specific",
        color=AUROC_COLOR,
        edgecolor="white",
        linewidth=0.7,
    )

    b2 = ax.bar(
        x + width / 2,
        general_vals,
        width,
        label="General",
        color=AUPR_COLOR,
        edgecolor="white",
        linewidth=0.7,
    )

    ax.set_ylim(
        0,
        1.03,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        metrics
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Brain-specific vs general training"
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        frameon=True,
        fontsize=8,
        loc="upper right",
    )

    offset = 0.012

    for bars in [
        b1,
        b2,
    ]:
        for bar in bars:
            value = float(
                bar.get_height()
            )

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + offset,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
            )

    panel_label(
        ax,
        "a",
    )


# =============================================================================
# 6. PANEL b — DISEASE-SPECIFIC PREDICTORS
# =============================================================================

def prepare_panel_b() -> pd.DataFrame:
    raw = read_csv_flexible(
        DISEASE_SPECIFIC_CSV
    )

    tool_col = resolve_column(
        raw,
        [
            "tool",
            "tool_name",
            "method",
            "predictor",
        ],
    )

    n_col = resolve_column(
        raw,
        [
            "mutation",
            "mutations",
            "mutation_count",
            "mutation number",
            "N",
            "n",
            "n_used",
        ],
        required=False,
    )

    auroc_col = resolve_column(
        raw,
        [
            "AUROC",
            "AUC_ROC",
        ],
    )

    aupr_col = resolve_column(
        raw,
        [
            "AUPR",
            "AUC_PR",
        ],
    )

    out = pd.DataFrame({
        "Tool": raw[tool_col].astype(str).str.strip(),
        "AUROC": pd.to_numeric(
            raw[auroc_col],
            errors="coerce",
        ),
        "AUPR": pd.to_numeric(
            raw[aupr_col],
            errors="coerce",
        ),
    })

    if n_col is not None:
        out["N"] = pd.to_numeric(
            raw[n_col],
            errors="coerce",
        )
    else:
        out["N"] = np.nan

    out = out.dropna(
        subset=[
            "AUROC",
            "AUPR",
        ]
    ).reset_index(
        drop=True
    )

    return out


def draw_panel_b(
    ax,
    df: pd.DataFrame,
):
    # Sort by AUROC ascending so the strongest AUROC appears toward the top.
    plot_df = (
        df.sort_values(
            "AUROC",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    y = np.arange(
        len(plot_df)
    )

    height = 0.34

    # Keep Fig. 5b visually clean: show predictor names only.
    # Sample sizes remain available in the CSV/output for reporting in text.
    labels = (
        plot_df["Tool"]
        .astype(str)
        .tolist()
    )

    auroc_colors = []
    aupr_colors = []

    for tool in plot_df["Tool"]:
        if str(tool) == "GenProt-DSM":
            auroc_colors.append(
                GENPROT_COLOR
            )
            aupr_colors.append(
                GENPROT_COLOR
            )
        else:
            base = DISEASE_TOOL_COLORS.get(
                str(tool),
                "#7F7F7F",
            )

            auroc_colors.append(
                base
            )
            aupr_colors.append(
                base
            )

    bars_auroc = ax.barh(
        y + height / 2,
        plot_df["AUROC"],
        height=height,
        color=auroc_colors,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.6,
        label="AUROC",
    )

    bars_aupr = ax.barh(
        y - height / 2,
        plot_df["AUPR"],
        height=height,
        color=aupr_colors,
        alpha=0.48,
        edgecolor="white",
        linewidth=0.6,
        label="AUPR",
    )

    values = np.concatenate([
        plot_df["AUROC"].to_numpy(
            dtype=float
        ),
        plot_df["AUPR"].to_numpy(
            dtype=float
        ),
    ])

    vmin = float(
        np.nanmin(values)
    )
    vmax = float(
        np.nanmax(values)
    )

    spread = max(
        vmax - vmin,
        0.02,
    )

    ax.set_xlim(
        max(
            0.0,
            vmin - spread * 0.30,
        ),
        min(
            1.0,
            vmax + spread * 0.35,
        ),
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        labels,
        fontsize=7.5,
    )

    ax.set_xlabel(
        "Score"
    )

    ax.set_title(
        "Comparison with disease-specific predictors"
    )

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=6,
            alpha=0.95,
            label="AUROC",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=6,
            alpha=0.48,
            label="AUPR",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=True,
        fontsize=8,
        loc="lower right",
    )

    text_offset = (
        ax.get_xlim()[1]
        - ax.get_xlim()[0]
    ) * 0.006

    for bars in [
        bars_auroc,
        bars_aupr,
    ]:
        for bar in bars:
            value = float(
                bar.get_width()
            )

            ax.text(
                value + text_offset,
                bar.get_y()
                + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=6.8,
            )

    panel_label(
        ax,
        "b",
    )


# =============================================================================
# 7. PANEL c — DISEASE-STRATIFIED BOOTSTRAP RESULTS
# =============================================================================

def parse_estimate_ci(
    x: Any,
) -> Tuple[float, float, float]:
    """
    Parse strings such as:
        0.927 (0.856 - 0.973)
        0.927 (0.856–0.973)

    Returns:
        estimate, lower, upper

    If only a single value exists:
        estimate, NaN, NaN
    """

    if pd.isna(x):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    s = str(x).strip()

    nums = re.findall(
        r"[-+]?(?:\d*\.\d+|\d+)",
        s,
    )

    vals = [
        float(v)
        for v in nums
    ]

    if len(vals) >= 3:
        return (
            vals[0],
            vals[1],
            vals[2],
        )

    if len(vals) == 1:
        return (
            vals[0],
            np.nan,
            np.nan,
        )

    return (
        np.nan,
        np.nan,
        np.nan,
    )


def canonical_disease_name(x: Any) -> str:
    s = str(x).strip()
    key = (
        s.lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    if key.startswith(
        "alzheimer"
    ):
        return "Alzheimer"

    if key.startswith(
        "attention"
    ):
        return "Attention Deficit"

    if key.startswith(
        "autism"
    ):
        return "Autism"

    if key.startswith(
        "intellect"
    ):
        return "Intellectual Disability"

    if key.startswith(
        "schizophren"
    ):
        return "Schizophrenia"

    if key.startswith(
        "language"
    ):
        return "Language Disorder"

    if key.startswith(
        "tourette"
    ):
        return "Tourette"

    return s


def prepare_panel_c() -> pd.DataFrame:
    raw = read_csv_flexible(
        DISEASE_STRATIFIED_CSV
    )

    disease_col = resolve_column(
        raw,
        [
            "Disease",
            "disease",
            "disease type",
        ],
    )

    n_col = resolve_column(
        raw,
        [
            "N",
            "n",
            "count",
        ],
    )

    auroc_col = resolve_column(
        raw,
        [
            "AUROC (95% CI)",
            "AUROC_95CI",
            "AUROC",
        ],
    )

    aupr_col = resolve_column(
        raw,
        [
            "AUPR (95% CI)",
            "AUPR_95CI",
            "AUPR",
        ],
    )

    rows = []

    for _, row in raw.iterrows():
        auroc, auroc_lo, auroc_hi = (
            parse_estimate_ci(
                row[auroc_col]
            )
        )

        aupr, aupr_lo, aupr_hi = (
            parse_estimate_ci(
                row[aupr_col]
            )
        )

        rows.append({
            "Disease": canonical_disease_name(
                row[disease_col]
            ),
            "N": pd.to_numeric(
                row[n_col],
                errors="coerce",
            ),
            "AUROC": auroc,
            "AUROC_low": auroc_lo,
            "AUROC_high": auroc_hi,
            "AUPR": aupr,
            "AUPR_low": aupr_lo,
            "AUPR_high": aupr_hi,
        })

    out = pd.DataFrame(
        rows
    )

    out = out.dropna(
        subset=[
            "AUROC",
            "AUPR",
        ],
        how="all",
    ).reset_index(
        drop=True
    )

    return out


def ci_to_xerr(
    estimate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Optional[np.ndarray]:

    if (
        np.all(np.isnan(lower))
        or np.all(np.isnan(upper))
    ):
        return None

    lo_err = (
        estimate
        - lower
    )

    hi_err = (
        upper
        - estimate
    )

    lo_err = np.where(
        np.isnan(lo_err),
        0.0,
        lo_err,
    )

    hi_err = np.where(
        np.isnan(hi_err),
        0.0,
        hi_err,
    )

    lo_err = np.maximum(
        lo_err,
        0.0,
    )

    hi_err = np.maximum(
        hi_err,
        0.0,
    )

    return np.vstack([
        lo_err,
        hi_err,
    ])


def draw_panel_c(
    ax,
    df: pd.DataFrame,
):
    plot_df = (
        df.sort_values(
            "AUROC",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    y = np.arange(
        len(plot_df)
    )

    offset = 0.13

    labels = []

    for _, row in plot_df.iterrows():
        if np.isfinite(
            row["N"]
        ):
            labels.append(
                f"{row['Disease']} (N={int(row['N'])})"
            )
        else:
            labels.append(
                str(row["Disease"])
            )

    auroc = plot_df[
        "AUROC"
    ].to_numpy(
        dtype=float
    )

    aupr = plot_df[
        "AUPR"
    ].to_numpy(
        dtype=float
    )

    auroc_low = plot_df[
        "AUROC_low"
    ].to_numpy(
        dtype=float
    )

    auroc_high = plot_df[
        "AUROC_high"
    ].to_numpy(
        dtype=float
    )

    aupr_low = plot_df[
        "AUPR_low"
    ].to_numpy(
        dtype=float
    )

    aupr_high = plot_df[
        "AUPR_high"
    ].to_numpy(
        dtype=float
    )

    ax.errorbar(
        auroc,
        y + offset,
        xerr=ci_to_xerr(
            auroc,
            auroc_low,
            auroc_high,
        ),
        fmt="o",
        markersize=4.8,
        capsize=2.2,
        linewidth=0.9,
        color=AUROC_COLOR,
        label="AUROC (95% CI)",
    )

    ax.errorbar(
        aupr,
        y - offset,
        xerr=ci_to_xerr(
            aupr,
            aupr_low,
            aupr_high,
        ),
        fmt="s",
        markersize=4.6,
        capsize=2.2,
        linewidth=0.9,
        color=AUPR_COLOR,
        label="AUPR (95% CI)",
    )

    finite_values = np.concatenate([
        auroc[
            np.isfinite(
                auroc
            )
        ],
        aupr[
            np.isfinite(
                aupr
            )
        ],
        auroc_low[
            np.isfinite(
                auroc_low
            )
        ],
        auroc_high[
            np.isfinite(
                auroc_high
            )
        ],
        aupr_low[
            np.isfinite(
                aupr_low
            )
        ],
        aupr_high[
            np.isfinite(
                aupr_high
            )
        ],
    ])

    if len(
        finite_values
    ) > 0:
        xmin = max(
            0.0,
            float(
                np.min(
                    finite_values
                )
            )
            - 0.035,
        )

        # Leave a small visual margin beyond 1.0 when a CI reaches the
        # theoretical ceiling. This prevents the right cap/marker/annotation
        # from being clipped by the axes border.
        xmax = min(
            1.04,
            float(
                np.max(
                    finite_values
                )
            )
            + 0.035,
        )

        if xmax <= 1.0:
            xmax = min(
                1.04,
                xmax + 0.02,
            )
    else:
        xmin = 0.0
        xmax = 1.04

    ax.set_xlim(
        xmin,
        xmax,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        labels,
        fontsize=7.4,
    )

    ax.set_xlabel(
        "Score"
    )

    ax.set_title(
        "Disease-stratified performance"
    )

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        frameon=True,
        fontsize=7.5,
        loc="lower right",
    )

    x_left, x_right = ax.get_xlim()
    text_offset = (
        x_right
        - x_left
    ) * 0.008

    def annotate_metric(values, y_positions):
        for yi, value in zip(
            y_positions,
            values,
        ):
            if not np.isfinite(
                value
            ):
                continue

            # Put labels inside the plot when the point lies close to the
            # right boundary; otherwise place them just to the right.
            if value > x_right - 0.045:
                x_text = value - text_offset
                ha = "right"
            else:
                x_text = value + text_offset
                ha = "left"

            ax.text(
                x_text,
                yi,
                f"{value:.3f}",
                fontsize=6.8,
                ha=ha,
                va="center",
                clip_on=False,
            )

    annotate_metric(
        auroc,
        y + offset,
    )

    annotate_metric(
        aupr,
        y - offset,
    )

    panel_label(
        ax,
        "c",
    )


# =============================================================================
# 8. PANEL d — RARE VARIANT EVALUATION
# =============================================================================

def canonical_tool_name(x: Any) -> str:
    s = str(x).strip()
    key = s.lower()

    aliases = {
        "polyphen2": "Polyphen2",
        "PolyPhen2": "Polyphen2",
        "primateai": "PrimateAI",
        "deogen2": "DEOGEN2",
        "alphamissense": "AlphaMissense",
        "genocanyon": "GenoCanyon",
        "esm1b": "ESM1b",
        "genprot-dsm": "GenProt-DSM",
        "genprot_dsm": "GenProt-DSM",
        "cadd": "CADD",
        "cadd v1.7": "CADD",
    }

    if key in aliases:
        return aliases[key]

    for tool in TOOL_COLUMNS:
        if key == tool.lower():
            return tool

    return s


def prepare_direction_table() -> pd.DataFrame:
    raw = read_csv_flexible(
        METRICS_TOOLS_CSV
    )

    tool_col = resolve_column(
        raw,
        [
            "tool_name",
            "tool",
        ],
    )

    subset_col = resolve_column(
        raw,
        [
            "subset_type",
            "subset",
        ],
        required=False,
    )

    direction_col = resolve_column(
        raw,
        [
            "score_direction",
            "score_dir",
            "direction",
        ],
    )

    out = pd.DataFrame({
        "Tool": raw[tool_col].map(
            canonical_tool_name
        ),
        "direction": (
            raw[direction_col]
            .astype(str)
            .str.strip()
            .str.lower()
        ),
    })

    if subset_col is not None:
        out["subset_type"] = (
            raw[subset_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        original = out.loc[
            out["subset_type"]
            == "original"
        ].copy()

        if not original.empty:
            out = original

    out = (
        out.drop_duplicates(
            subset=[
                "Tool",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return out


def get_direction(
    tool: str,
    direction_df: pd.DataFrame,
) -> str:

    row = direction_df.loc[
        direction_df["Tool"]
        == tool
    ]

    if not row.empty:
        raw_direction = str(
            row.iloc[0]["direction"]
        ).lower()

        if raw_direction.startswith(
            "high"
        ):
            return "high"

        if raw_direction.startswith(
            "low"
        ):
            return "low"

    return FALLBACK_DIRECTION[
        tool
    ]


def transform_for_metric(
    score: np.ndarray,
    direction: str,
) -> np.ndarray:

    if direction == "high":
        return score

    if direction == "low":
        return -score

    raise ValueError(
        f"Unknown direction: {direction}"
    )


def prepare_panel_d() -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_csv_flexible(
        TEST_SCORE_CSV
    )

    required = [
        "AF",
        "label",
        *TOOL_COLUMNS,
    ]

    missing = [
        c
        for c in required
        if c not in raw.columns
    ]

    if missing:
        raise KeyError(
            "test_score.csv is missing columns:\n"
            + "\n".join(
                missing
            )
        )

    df = raw.copy()

    df["AF"] = pd.to_numeric(
        df["AF"],
        errors="coerce",
    )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    for tool in TOOL_COLUMNS:
        df[tool] = pd.to_numeric(
            df[tool],
            errors="coerce",
        )

    # Manuscript definition:
    # VarRareTest = AF < 0.01; variants with missing AF are excluded.
    rare = df.loc[
        df["AF"].notna()
        &
        (
            df["AF"]
            < 0.01
        )
    ].copy()

    if rare.empty:
        raise RuntimeError(
            "No variants satisfy AF < 0.01."
        )

    direction_df = prepare_direction_table()

    rows = []

    for tool in TOOL_COLUMNS:
        direction = get_direction(
            tool,
            direction_df,
        )

        sub = rare.loc[
            rare[tool].notna()
        ].copy()

        y = sub[
            "label"
        ].to_numpy(
            dtype=int
        )

        raw_score = sub[
            tool
        ].to_numpy(
            dtype=float
        )

        if len(
            np.unique(
                y
            )
        ) < 2:
            auroc = np.nan
            aupr = np.nan
        else:
            transformed = transform_for_metric(
                raw_score,
                direction,
            )

            auroc = roc_auc_score(
                y,
                transformed,
            )

            aupr = average_precision_score(
                y,
                transformed,
            )

        rows.append({
            "Tool": tool,
            "Display_name": DISPLAY_NAME[
                tool
            ],
            "N_used": len(
                sub
            ),
            "Pathogenic": int(
                np.sum(
                    y
                    == 1
                )
            ),
            "Benign": int(
                np.sum(
                    y
                    == 0
                )
            ),
            "Direction": direction,
            "AUROC": auroc,
            "AUPR": aupr,
        })

    result = pd.DataFrame(
        rows
    )

    rare_summary = pd.DataFrame([
        {
            "Rare_definition": "AF < 0.01; missing AF excluded",
            "N_total": len(
                rare
            ),
            "Pathogenic": int(
                (
                    rare[
                        "label"
                    ]
                    == 1
                ).sum()
            ),
            "Benign": int(
                (
                    rare[
                        "label"
                    ]
                    == 0
                ).sum()
            ),
        }
    ])

    return (
        result,
        rare_summary,
    )


def rare_tool_color(
    tool: str,
    i: int,
):
    if tool == "GenProt-DSM":
        return GENPROT_COLOR

    cmap = plt.get_cmap(
        "tab20"
    )

    return cmap(
        i % 20
    )


def draw_horizontal_metric_bars(
    ax,
    plot_df: pd.DataFrame,
    metric: str,
    show_y_labels: bool,
):
    y = np.arange(
        len(
            plot_df
        )
    )

    values = plot_df[
        metric
    ].to_numpy(
        dtype=float
    )

    colors = [
        rare_tool_color(
            tool,
            i,
        )
        for i, tool in enumerate(
            plot_df[
                "Tool"
            ]
        )
    ]

    bars = ax.barh(
        y,
        values,
        height=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_yticks(
        y
    )

    if show_y_labels:
        ax.set_yticklabels(
            plot_df[
                "Display_name"
            ],
            fontsize=6.6,
        )
    else:
        ax.set_yticklabels([])

    finite = values[
        np.isfinite(
            values
        )
    ]

    vmin = float(
        np.nanmin(
            finite
        )
    )

    vmax = float(
        np.nanmax(
            finite
        )
    )

    spread = max(
        vmax - vmin,
        0.05,
    )

    xmin = max(
        0.0,
        vmin
        - spread * 0.28,
    )

    xmax = min(
        1.02,
        vmax
        + spread * 0.42,
    )

    # AUPR on this highly imbalanced subset can be close to 1.0; allow
    # a little extra right margin for the numeric labels.
    if metric == "AUPR":
        xmax = min(
            1.02,
            max(
                xmax,
                vmax + 0.025,
            ),
        )

    ax.set_xlim(
        xmin,
        xmax,
    )

    ax.set_xlabel(
        metric
    )

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.5,
        alpha=0.6,
    )

    offset = (
        ax.get_xlim()[1]
        - ax.get_xlim()[0]
    ) * 0.008

    for bar, value in zip(
        bars,
        values,
    ):
        if not np.isfinite(
            value
        ):
            continue

        ax.text(
            value + offset,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.3f}",
            fontsize=5.8,
            ha="left",
            va="center",
        )


def draw_panel_d(
    fig,
    subspec,
    df: pd.DataFrame,
    rare_summary: pd.DataFrame,
):
    container = fig.add_subplot(
        subspec
    )

    container.axis(
        "off"
    )

    panel_label(
        container,
        "d",
        x=-0.075,
        y=1.06,
    )

    row = rare_summary.iloc[0]

    # Put the panel-level title clearly above both metric subplots.
    container.text(
        0.50,
        1.055,
        "Rare-variant evaluation "
        f"(N={int(row['N_total'])}; "
        f"{int(row['Pathogenic'])} pathogenic, "
        f"{int(row['Benign'])} benign)",
        transform=container.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        clip_on=False,
    )

    # Same tool order in both subplots, sorted by AUROC.
    plot_df = (
        df.sort_values(
            "AUROC",
            ascending=True,
            na_position="first",
        )
        .reset_index(
            drop=True
        )
    )

    inner = subspec.subgridspec(
        1,
        2,
        wspace=0.10,
    )

    ax_auc = fig.add_subplot(
        inner[0, 0]
    )

    ax_aupr = fig.add_subplot(
        inner[0, 1]
    )

    draw_horizontal_metric_bars(
        ax_auc,
        plot_df,
        metric="AUROC",
        show_y_labels=True,
    )

    draw_horizontal_metric_bars(
        ax_aupr,
        plot_df,
        metric="AUPR",
        show_y_labels=False,
    )

    # Keep metric titles close to their own axes and well below the
    # panel-level title.
    ax_auc.set_title(
        "AUROC",
        pad=4,
    )

    ax_aupr.set_title(
        "AUPR",
        pad=4,
    )


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    ensure_dir(
        OUTPUT_DIR
    )

    paths = {
        "Brain model": BRAIN_JSON,
        "General model": GENERAL_JSON,
        "Disease-specific comparison": DISEASE_SPECIFIC_CSV,
        "Disease-stratified results": DISEASE_STRATIFIED_CSV,
        "Variant score table": TEST_SCORE_CSV,
        "Tool thresholds": METRICS_TOOLS_CSV,
    }

    print(
        "=" * 100
    )
    print(
        "FIG. 5 INPUT FILES"
    )
    print(
        "=" * 100
    )

    for name, path in paths.items():
        print(
            f"{name}: {path}"
        )

        if not os.path.isfile(
            path
        ):
            raise FileNotFoundError(
                path
            )

    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------

    panel_a_df = prepare_panel_a()
    panel_b_df = prepare_panel_b()
    panel_c_df = prepare_panel_c()
    panel_d_df, rare_summary = prepare_panel_d()

    # -------------------------------------------------------------------------
    # Console audit
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )
    print(
        "PANEL a — BRAIN-SPECIFIC VS GENERAL"
    )
    print(
        "=" * 100
    )

    print(
        panel_a_df.to_string(
            index=False
        )
    )

    if not (
        panel_a_df["Source_path"]
        .astype(str)
        .eq("final_metrics")
        .all()
    ):
        raise RuntimeError(
            "Fig. 5a metric source audit failed: "
            "both models must come from root-level final_metrics."
        )

    print(
        "[PASS] Fig. 5a uses root-level final_metrics for both models."
    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "PANEL b — DISEASE-SPECIFIC PREDICTORS"
    )
    print(
        "=" * 100
    )

    print(
        panel_b_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "PANEL c — DISEASE-STRATIFIED RESULTS"
    )
    print(
        "=" * 100
    )

    print(
        panel_c_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "PANEL d — RARE VARIANTS"
    )
    print(
        "=" * 100
    )

    print(
        rare_summary.to_string(
            index=False
        )
    )

    print()

    print(
        panel_d_df[
            [
                "Display_name",
                "N_used",
                "Pathogenic",
                "Benign",
                "AUROC",
                "AUPR",
                "Direction",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Save audit tables
    # -------------------------------------------------------------------------

    panel_a_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig5a_brain_vs_general.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    panel_b_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig5b_disease_specific_predictors.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    panel_c_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig5c_disease_stratified.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    panel_d_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig5d_rare_variant_metrics.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    rare_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig5d_rare_variant_summary.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Build figure
    # -------------------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            15.0,
            10.2,
        ),
        dpi=FIG_DPI,
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[
            1.0,
            1.05,
        ],
        height_ratios=[
            0.88,
            1.12,
        ],
        wspace=0.28,
        hspace=0.34,
    )

    ax_a = fig.add_subplot(
        gs[0, 0]
    )

    ax_b = fig.add_subplot(
        gs[0, 1]
    )

    ax_c = fig.add_subplot(
        gs[1, 0]
    )

    draw_panel_a(
        ax_a,
        panel_a_df,
    )

    draw_panel_b(
        ax_b,
        panel_b_df,
    )

    draw_panel_c(
        ax_c,
        panel_c_df,
    )

    draw_panel_d(
        fig,
        gs[1, 1],
        panel_d_df,
        rare_summary,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.96,
        bottom=0.07,
        wspace=0.28,
        hspace=0.34,
    )

    output_base = os.path.join(
        OUTPUT_DIR,
        OUTPUT_BASENAME,
    )

    save_figure(
        fig,
        output_base,
    )

    plt.close(
        fig
    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "DONE"
    )
    print(
        "=" * 100
    )

    print(
        "PNG:",
        output_base + ".png",
    )

    print(
        "PDF:",
        output_base + ".pdf",
    )


if __name__ == "__main__":
    main()
