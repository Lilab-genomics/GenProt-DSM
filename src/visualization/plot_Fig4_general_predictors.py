#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig. 4 — Performance comparison with general-purpose pathogenicity predictors
and protein language model baseline

Panels
------
(a) VarGeneDisjointTest ROC
(b) VarGeneDisjointTest PR
(c) maximal shared subset ROC
(d) maximal shared subset PR

Definitions
-----------
original:
    For each predictor, use all VarGeneDisjointTest variants for which that
    predictor has a valid score.

all_tools_common_subset:
    Use only variants with valid scores for ALL predictors. Every tool is
    evaluated on exactly the same variants.

Important manuscript terminology
--------------------------------
- Display "CADD v1.7" everywhere in the figure.
- Display metric terminology as "AUROC" and "AUPR".
- AUPR is computed with sklearn.metrics.average_precision_score.
- ROC/PR curves are calculated from per-variant scores.
- Low-direction predictors are transformed as -raw_score so that larger
  transformed scores always indicate greater pathogenicity.

Input files
-----------
1) F:\\GenProt-DSM_Resubmit\\result\\metrics_tools.csv

   Expected logical columns:
       tool_name
       tool_col
       subset_type
       n_used
       score_dir
       AUC_ROC
       AUC_PR

2) F:\\GenProt-DSM_Resubmit\\dataset\\data\\test_score.csv

   Expected columns:
       Chrom
       Position
       Reference
       Alternate
       SIFT
       Polyphen2
       FATHMM
       PROVEAN
       MPC
       PrimateAI
       DEOGEN2
       ESM1b
       AlphaMissense
       CADD
       DANN
       GenoCanyon
       AF
       GenProt-DSM
       label

Output directory
----------------
F:\\GenProt-DSM_Resubmit\\fig_Results\\Fig4_general_predictors

Outputs
-------
Fig4_general_predictor_comparison.png
Fig4_general_predictor_comparison.pdf

Audit tables are also exported.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)


# =============================================================================
# 1. PATHS
# =============================================================================

METRICS_CSV = r"F:\GenProt-DSM_Resubmit\result\metrics_tools.csv"

SCORE_CSV = r"F:\GenProt-DSM_Resubmit\dataset\data\test_score.csv"

OUTPUT_DIR = r"F:\GenProt-DSM_Resubmit\fig_Results\Fig4_general_predictors"

OUTPUT_BASENAME = "Fig4_general_predictor_comparison"


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
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# =============================================================================
# 3. PREDICTORS / DISPLAY NAMES
# =============================================================================

# Order used in the figure legend.
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
    "Polyphen2": "PolyPhen-2",
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

# Fallback directions if metrics_tools.csv lacks a usable score_dir entry.
# high: larger raw score = more pathogenic
# low : smaller raw score = more pathogenic
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

# Fixed predictor colors.
# GenProt-DSM is intentionally red and thicker than all baselines.
TOOL_COLORS: Dict[str, str] = {
    "SIFT": "#1F77B4",
    "Polyphen2": "#AEC7E8",
    "FATHMM": "#FF7F0E",
    "PROVEAN": "#FFBB78",
    "MPC": "#2CA02C",
    "PrimateAI": "#98DF8A",
    "DEOGEN2": "#9467BD",
    "AlphaMissense": "#C5B0D5",
    "CADD": "#8C564B",
    "DANN": "#E377C2",
    "GenoCanyon": "#7F7F7F",
    "ESM1b": "#17BECF",
    "GenProt-DSM": "#D62728",
}


# =============================================================================
# 4. GENERAL UTILITIES
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def normalize_col_key(x: str) -> str:
    return (
        str(x)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def resolve_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> str | None:

    direct = {
        str(c).strip(): c
        for c in df.columns
    }

    for c in candidates:
        if c in direct:
            return direct[c]

    normalized = {
        normalize_col_key(c): c
        for c in df.columns
    }

    for c in candidates:
        key = normalize_col_key(c)
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"Required column not found.\n"
            f"Candidates: {candidates}\n"
            f"Existing columns: {list(df.columns)}"
        )

    return None


def panel_label(ax, label: str):
    ax.text(
        -0.10,
        1.04,
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


# =============================================================================
# 5. STANDARDIZE metrics_tools.csv
# =============================================================================

def standardize_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    tool_name_col = resolve_column(
        df,
        ["tool_name", "Tool", "tool"],
    )

    tool_col_col = resolve_column(
        df,
        ["tool_col", "tool_column", "score_col", "column"],
        required=False,
    )

    subset_col = resolve_column(
        df,
        ["subset_type", "subset"],
    )

    n_col = resolve_column(
        df,
        ["n_used", "N", "n"],
    )

    score_dir_col = resolve_column(
        df,
        ["score_dir", "score_direction", "direction"],
    )

    auroc_col = resolve_column(
        df,
        ["AUC_ROC", "AUROC", "roc_auc"],
    )

    aupr_col = resolve_column(
        df,
        ["AUC_PR", "AUPR", "average_precision", "AP"],
    )

    out = pd.DataFrame({
        "tool_name": df[tool_name_col].astype(str).str.strip(),
        "subset_type": df[subset_col].astype(str).str.strip(),
        "n_used": pd.to_numeric(df[n_col], errors="coerce"),
        "score_dir": df[score_dir_col].astype(str).str.strip().str.lower(),
        "AUROC_summary": pd.to_numeric(df[auroc_col], errors="coerce"),
        "AUPR_summary": pd.to_numeric(df[aupr_col], errors="coerce"),
    })

    if tool_col_col is not None:
        out["tool_col"] = df[tool_col_col].astype(str).str.strip()
    else:
        out["tool_col"] = out["tool_name"]

    # Normalize tool naming to score-file column naming.
    aliases = {
        "polyphen2": "Polyphen2",
        "polyphen-2": "Polyphen2",
        "alphamissense": "AlphaMissense",
        "genprot-dsm": "GenProt-DSM",
        "genprot_dsm": "GenProt-DSM",
        "esm1b": "ESM1b",
        "genocanyon": "GenoCanyon",
        "primateai": "PrimateAI",
        "deogen2": "DEOGEN2",
        "cadd": "CADD",
        "cadd v1.7": "CADD",
    }

    def canonical_tool(x):
        s = str(x).strip()
        low = s.lower()

        if low in aliases:
            return aliases[low]

        for tool in TOOL_COLUMNS:
            if low == tool.lower():
                return tool

        return s

    out["tool_name"] = out["tool_name"].map(canonical_tool)
    out["tool_col"] = out["tool_col"].map(canonical_tool)

    # Normalize subset names.
    out["subset_type"] = (
        out["subset_type"]
        .str.strip()
        .str.lower()
    )

    return out


# =============================================================================
# 6. STANDARDIZE per-variant score table
# =============================================================================

def standardize_score_table(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Chrom",
        "Position",
        "Reference",
        "Alternate",
        *TOOL_COLUMNS,
        "label",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            "test_score.csv is missing required columns:\n"
            + "\n".join(missing)
        )

    out = df.copy()

    out["label"] = pd.to_numeric(
        out["label"],
        errors="raise",
    ).astype(int)

    if not out["label"].isin([0, 1]).all():
        raise ValueError(
            "label must contain only 0 and 1."
        )

    for tool in TOOL_COLUMNS:
        out[tool] = pd.to_numeric(
            out[tool],
            errors="coerce",
        )

    # Construct a stable variant ID for audit exports.
    out["variant_id"] = (
        out[
            [
                "Chrom",
                "Position",
                "Reference",
                "Alternate",
            ]
        ]
        .astype(str)
        .agg(":".join, axis=1)
    )

    if out["variant_id"].duplicated().any():
        n_dup = int(
            out.loc[
                out["variant_id"].duplicated(keep=False),
                "variant_id",
            ]
            .nunique()
        )

        raise ValueError(
            f"test_score.csv contains {n_dup} duplicated genomic variants."
        )

    return out


# =============================================================================
# 7. SCORE DIRECTION
# =============================================================================

def get_tool_direction(
    tool: str,
    metrics_df: pd.DataFrame,
) -> str:

    rows = metrics_df.loc[
        metrics_df["tool_name"] == tool
    ]

    if not rows.empty:
        values = (
            rows["score_dir"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
            .tolist()
        )

        for x in values:
            if x.startswith("high"):
                return "high"
            if x.startswith("low"):
                return "low"

    return FALLBACK_DIRECTION[tool]


def transform_score(
    raw_score: pd.Series,
    direction: str,
) -> np.ndarray:

    x = raw_score.to_numpy(dtype=float)

    if direction == "high":
        return x

    if direction == "low":
        return -x

    raise ValueError(
        f"Unknown score direction: {direction}"
    )


# =============================================================================
# 8. SUMMARY LOOKUP
# =============================================================================

def get_summary_row(
    metrics_df: pd.DataFrame,
    tool: str,
    subset_type: str,
) -> pd.Series | None:

    rows = metrics_df.loc[
        (metrics_df["tool_name"] == tool)
        &
        (metrics_df["subset_type"] == subset_type.lower())
    ]

    if rows.empty:
        return None

    if len(rows) > 1:
        print(
            f"[WARN] Multiple summary rows for {tool} / {subset_type}; "
            "using the first."
        )

    return rows.iloc[0]


# =============================================================================
# 9. BUILD original / common subsets
# =============================================================================

def build_original_subset(
    scores_df: pd.DataFrame,
    tool: str,
) -> pd.DataFrame:

    sub = scores_df.loc[
        scores_df[tool].notna()
        &
        scores_df["label"].notna()
    ].copy()

    return sub


def build_common_subset(
    scores_df: pd.DataFrame,
) -> pd.DataFrame:

    mask = (
        scores_df[TOOL_COLUMNS]
        .notna()
        .all(axis=1)
        &
        scores_df["label"].notna()
    )

    return scores_df.loc[
        mask
    ].copy()


# =============================================================================
# 10. COMPUTE CURVES + VALIDATE AGAINST metrics_tools.csv
# =============================================================================

def evaluate_subset(
    sub: pd.DataFrame,
    tool: str,
    direction: str,
) -> Dict:

    y = sub["label"].to_numpy(dtype=int)

    score = transform_score(
        sub[tool],
        direction,
    )

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"{tool}: subset contains only one class."
        )

    fpr, tpr, _ = roc_curve(
        y,
        score,
    )

    precision, recall, _ = precision_recall_curve(
        y,
        score,
    )

    auroc = roc_auc_score(
        y,
        score,
    )

    # Use average_precision_score for manuscript AUPR.
    aupr = average_precision_score(
        y,
        score,
    )

    prevalence = float(
        np.mean(y)
    )

    return {
        "n": int(len(sub)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "AUROC": float(auroc),
        "AUPR": float(aupr),
        "prevalence": prevalence,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }


def validate_against_summary(
    tool: str,
    subset_type: str,
    computed: Dict,
    metrics_df: pd.DataFrame,
    tolerance: float = 1e-4,
) -> Dict:

    row = get_summary_row(
        metrics_df,
        tool,
        subset_type,
    )

    audit = {
        "tool": tool,
        "display_name": DISPLAY_NAME[tool],
        "subset_type": subset_type,
        "n_computed": computed["n"],
        "n_summary": np.nan,
        "AUROC_computed": computed["AUROC"],
        "AUROC_summary": np.nan,
        "AUROC_abs_diff": np.nan,
        "AUPR_computed": computed["AUPR"],
        "AUPR_summary": np.nan,
        "AUPR_abs_diff": np.nan,
        "score_direction": get_tool_direction(tool, metrics_df),
    }

    if row is None:
        print(
            f"[WARN] No summary row found for {tool} / {subset_type}."
        )
        return audit

    n_summary = int(row["n_used"])
    auroc_summary = float(row["AUROC_summary"])
    aupr_summary = float(row["AUPR_summary"])

    audit["n_summary"] = n_summary
    audit["AUROC_summary"] = auroc_summary
    audit["AUPR_summary"] = aupr_summary
    audit["AUROC_abs_diff"] = abs(
        computed["AUROC"] - auroc_summary
    )
    audit["AUPR_abs_diff"] = abs(
        computed["AUPR"] - aupr_summary
    )

    if computed["n"] != n_summary:
        raise RuntimeError(
            f"\nN mismatch for {tool} / {subset_type}\n"
            f"computed N = {computed['n']}\n"
            f"summary N  = {n_summary}\n"
        )

    if audit["AUROC_abs_diff"] > tolerance:
        print(
            f"[WARN] {tool} / {subset_type}: "
            f"AUROC computed={computed['AUROC']:.8f}, "
            f"summary={auroc_summary:.8f}, "
            f"diff={audit['AUROC_abs_diff']:.8f}"
        )

    if audit["AUPR_abs_diff"] > tolerance:
        print(
            f"[WARN] {tool} / {subset_type}: "
            f"AUPR computed={computed['AUPR']:.8f}, "
            f"summary={aupr_summary:.8f}, "
            f"diff={audit['AUPR_abs_diff']:.8f}"
        )

    return audit


# =============================================================================
# 11. PLOTTING
# =============================================================================

def line_style(tool: str) -> Tuple[float, float]:
    """
    Returns linewidth, alpha.
    """
    if tool == "GenProt-DSM":
        return 2.6, 1.0

    return 1.15, 0.92


def legend_metric_value(
    tool: str,
    subset_type: str,
    metric_name: str,
    computed: Dict,
    metrics_df: pd.DataFrame,
) -> float:
    """
    Use the value already reported in metrics_tools.csv when available,
    so the figure and summary table remain numerically identical.
    """

    row = get_summary_row(
        metrics_df,
        tool,
        subset_type,
    )

    if row is not None:
        if metric_name == "AUROC":
            return float(
                row["AUROC_summary"]
            )
        if metric_name == "AUPR":
            return float(
                row["AUPR_summary"]
            )

    return float(
        computed[metric_name]
    )


def draw_roc_panel(
    ax,
    results: Dict[str, Dict],
    metrics_df: pd.DataFrame,
    subset_type: str,
    title: str,
    label: str,
):
    # Draw baselines first, GenProt-DSM last.
    draw_order = [
        x for x in TOOL_COLUMNS
        if x != "GenProt-DSM"
    ] + ["GenProt-DSM"]

    for tool in draw_order:
        r = results[tool]

        lw, alpha = line_style(
            tool
        )

        legend_auroc = legend_metric_value(
            tool,
            subset_type,
            "AUROC",
            r,
            metrics_df,
        )

        label_text = (
            f"{DISPLAY_NAME[tool]} "
            f"(AUROC={legend_auroc:.3f})"
        )

        ax.plot(
            r["fpr"],
            r["tpr"],
            color=TOOL_COLORS[tool],
            linewidth=lw,
            alpha=alpha,
            label=label_text,
            zorder=5 if tool == "GenProt-DSM" else 2,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=0.8,
        color="#8A8A8A",
        alpha=0.75,
        label="Random",
        zorder=1,
    )

    ax.set_xlim(
        0,
        1,
    )
    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_xlabel(
        "False positive rate"
    )
    ax.set_ylabel(
        "True positive rate"
    )
    ax.set_title(
        title
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        loc="lower right",
        frameon=True,
        ncol=2,
        fontsize=6.7,
        columnspacing=0.8,
        handlelength=2.2,
        borderpad=0.5,
        labelspacing=0.35,
    )

    panel_label(
        ax,
        label,
    )


def draw_pr_panel(
    ax,
    results: Dict[str, Dict],
    metrics_df: pd.DataFrame,
    subset_type: str,
    title: str,
    label: str,
):
    draw_order = [
        x for x in TOOL_COLUMNS
        if x != "GenProt-DSM"
    ] + ["GenProt-DSM"]

    for tool in draw_order:
        r = results[tool]

        lw, alpha = line_style(
            tool
        )

        legend_aupr = legend_metric_value(
            tool,
            subset_type,
            "AUPR",
            r,
            metrics_df,
        )

        label_text = (
            f"{DISPLAY_NAME[tool]} "
            f"(AUPR={legend_aupr:.3f})"
        )

        # sklearn returns recall in descending order; plotting is still valid.
        ax.plot(
            r["recall"],
            r["precision"],
            color=TOOL_COLORS[tool],
            linewidth=lw,
            alpha=alpha,
            label=label_text,
            zorder=5 if tool == "GenProt-DSM" else 2,
        )

    # In each panel the relevant prevalence baseline is the pathogenic fraction.
    # For original subsets prevalence may differ slightly by tool because missing
    # scores differ. To avoid drawing 13 almost-identical baselines, use the
    # GenProt-DSM subset prevalence as the reference and state this in the legend.
    baseline = float(
        results["GenProt-DSM"]["prevalence"]
    )

    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=0.8,
        color="#8A8A8A",
        alpha=0.75,
        label=f"Prevalence={baseline:.3f}",
        zorder=1,
    )

    ax.set_xlim(
        0,
        1,
    )
    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_xlabel(
        "Recall"
    )
    ax.set_ylabel(
        "Precision"
    )
    ax.set_title(
        title
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        loc="lower left",
        frameon=True,
        ncol=2,
        fontsize=6.7,
        columnspacing=0.8,
        handlelength=2.2,
        borderpad=0.5,
        labelspacing=0.35,
    )

    panel_label(
        ax,
        label,
    )


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    ensure_dir(
        OUTPUT_DIR
    )

    print("=" * 96)
    print("FIG. 4 INPUT FILES")
    print("=" * 96)
    print("Metrics summary:", METRICS_CSV)
    print("Variant scores :", SCORE_CSV)
    print("Output dir     :", OUTPUT_DIR)

    metrics_raw = read_csv_flexible(
        METRICS_CSV
    )

    scores_raw = read_csv_flexible(
        SCORE_CSV
    )

    metrics_df = standardize_metrics_table(
        metrics_raw
    )

    scores_df = standardize_score_table(
        scores_raw
    )

    print(
        f"\n[CHECK] VarGeneDisjointTest rows = {len(scores_df):,}"
    )

    print(
        "[CHECK] label distribution:"
    )
    print(
        scores_df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # -------------------------------------------------------------------------
    # Original subsets
    # -------------------------------------------------------------------------

    original_results: Dict[str, Dict] = {}
    original_audit = []

    print(
        "\n"
        + "=" * 96
    )
    print(
        "ORIGINAL SUBSETS"
    )
    print(
        "=" * 96
    )

    for tool in TOOL_COLUMNS:
        direction = get_tool_direction(
            tool,
            metrics_df,
        )

        sub = build_original_subset(
            scores_df,
            tool,
        )

        result = evaluate_subset(
            sub,
            tool,
            direction,
        )

        original_results[tool] = result

        audit = validate_against_summary(
            tool,
            "original",
            result,
            metrics_df,
        )

        original_audit.append(
            audit
        )

        print(
            f"{DISPLAY_NAME[tool]:18s} "
            f"N={result['n']:4d}  "
            f"AUROC={result['AUROC']:.6f}  "
            f"AUPR={result['AUPR']:.6f}  "
            f"direction={direction}"
        )

    # -------------------------------------------------------------------------
    # Maximal all-tools common subset
    # -------------------------------------------------------------------------

    common_df = build_common_subset(
        scores_df
    )

    print(
        "\n"
        + "=" * 96
    )
    print(
        "MAXIMAL ALL-TOOLS COMMON SUBSET"
    )
    print(
        "=" * 96
    )
    print(
        f"N = {len(common_df):,}"
    )
    print(
        f"Pathogenic = {(common_df['label'] == 1).sum():,}"
    )
    print(
        f"Benign     = {(common_df['label'] == 0).sum():,}"
    )

    common_results: Dict[str, Dict] = {}
    common_audit = []

    for tool in TOOL_COLUMNS:
        direction = get_tool_direction(
            tool,
            metrics_df,
        )

        result = evaluate_subset(
            common_df,
            tool,
            direction,
        )

        common_results[tool] = result

        audit = validate_against_summary(
            tool,
            "all_tools_common_subset",
            result,
            metrics_df,
        )

        common_audit.append(
            audit
        )

        print(
            f"{DISPLAY_NAME[tool]:18s} "
            f"N={result['n']:4d}  "
            f"AUROC={result['AUROC']:.6f}  "
            f"AUPR={result['AUPR']:.6f}"
        )

    # -------------------------------------------------------------------------
    # Audit exports
    # -------------------------------------------------------------------------

    pd.DataFrame(
        original_audit
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig4_original_subset_metric_audit.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        common_audit
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig4_common_subset_metric_audit.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    common_df[
        [
            "variant_id",
            "Chrom",
            "Position",
            "Reference",
            "Alternate",
            "label",
        ]
    ].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig4_maximal_common_subset_variants.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 9.0),
        dpi=FIG_DPI,
    )

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    draw_roc_panel(
        ax_a,
        original_results,
        metrics_df,
        subset_type="original",
        title="VarGeneDisjointTest ROC",
        label="a",
    )

    draw_pr_panel(
        ax_b,
        original_results,
        metrics_df,
        subset_type="original",
        title="VarGeneDisjointTest PR",
        label="b",
    )

    draw_roc_panel(
        ax_c,
        common_results,
        metrics_df,
        subset_type="all_tools_common_subset",
        title="maximal shared subset ROC",
        label="c",
    )

    draw_pr_panel(
        ax_d,
        common_results,
        metrics_df,
        subset_type="all_tools_common_subset",
        title="maximal shared subset PR",
        label="d",
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.96,
        bottom=0.075,
        wspace=0.22,
        hspace=0.26,
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
        + "=" * 96
    )
    print(
        "DONE"
    )
    print(
        "=" * 96
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
