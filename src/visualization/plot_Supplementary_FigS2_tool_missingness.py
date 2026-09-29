#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supplementary Fig. S2
Tool-specific prediction missingness on VarGeneDisjointTest

Purpose
-------
Reproduce the old missingness-bar logic on the revised final test set.

For each predictor:
    n_used  = number of variants with a valid numeric prediction
    missing = total variants - n_used

Predictors are sorted by missing count in descending order.
The missing count is annotated above each bar.

Required input
--------------
F:\\GenProt-DSM_Resubmit\\dataset\\data\\test_score.csv

Required predictor columns
--------------------------
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
GenProt-DSM

Other columns such as:
Chrom, Position, Reference, Alternate, AF, label
may exist but are not required for this figure.

Important display terminology
-----------------------------
CADD is displayed as:
    CADD v1.7

Outputs
-------
F:\\GenProt-DSM_Resubmit\\fig_Results\\Supplementary_FigS2\\
    Supplementary_FigS2_tool_missingness.png
    Supplementary_FigS2_tool_missingness.pdf
    Supplementary_FigS2_tool_missingness.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

INPUT_CSV = r"F:\GenProt-DSM_Resubmit\dataset\data\test_score.csv"

OUTPUT_DIR = (
    r"F:\GenProt-DSM_Resubmit\fig_Results"
    r"\Supplementary_FigS2"
)

OUTPUT_BASENAME = "Supplementary_FigS2_tool_missingness"


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

PLOT_FONT_SIZE = 9
FIG_DPI = 600

matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_FONT_SIZE,
    "axes.labelsize": PLOT_FONT_SIZE,
    "xtick.labelsize": PLOT_FONT_SIZE,
    "ytick.labelsize": PLOT_FONT_SIZE,
    "legend.fontsize": PLOT_FONT_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# =============================================================================
# 3. PREDICTOR CONFIGURATION
# =============================================================================

# Actual column names in test_score.csv
PREDICTOR_COLUMNS = [
    "SIFT",
    "Polyphen2",
    "FATHMM",
    "PROVEAN",
    "MPC",
    "PrimateAI",
    "DEOGEN2",
    "ESM1b",
    "AlphaMissense",
    "CADD",
    "DANN",
    "GenoCanyon",
    "GenProt-DSM",
]

# Display names in the manuscript/figure
DISPLAY_NAMES = {
    "SIFT": "SIFT",
    "Polyphen2": "PolyPhen2",
    "FATHMM": "FATHMM",
    "PROVEAN": "PROVEAN",
    "MPC": "MPC",
    "PrimateAI": "PrimateAI",
    "DEOGEN2": "DEOGEN2",
    "ESM1b": "ESM1b",
    "AlphaMissense": "AlphaMissense",
    "CADD": "CADD v1.7",
    "DANN": "DANN",
    "GenoCanyon": "GenoCanyon",
    "GenProt-DSM": "GenProt-DSM",
}

# Revised primary test set size
EXPECTED_TEST_N = 786


# =============================================================================
# 4. UTILITIES
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_csv_smart(path: str) -> pd.DataFrame:
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


def validate_columns(df: pd.DataFrame):
    missing_cols = [
        c for c in PREDICTOR_COLUMNS
        if c not in df.columns
    ]

    if missing_cols:
        raise KeyError(
            "The following required predictor columns are missing:\n"
            + "\n".join(missing_cols)
            + "\n\nExisting columns:\n"
            + ", ".join(map(str, df.columns))
        )


def save_figure(fig, output_base: str):
    png_path = output_base + ".png"
    pdf_path = output_base + ".pdf"

    fig.savefig(
        png_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {pdf_path}")


# =============================================================================
# 5. COMPUTE TOOL-SPECIFIC MISSINGNESS
# =============================================================================

def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    A valid prediction is defined as a finite numeric value.

    This is slightly safer than only using .notna(), because strings or
    malformed values are also treated as missing.
    """

    total = int(len(df))
    rows = []

    for col in PREDICTOR_COLUMNS:
        score = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        valid_mask = np.isfinite(
            score.to_numpy(dtype=float)
        )

        n_used = int(valid_mask.sum())
        missing = int(total - n_used)

        rows.append({
            "tool_column": col,
            "tool_name": DISPLAY_NAMES[col],
            "total": total,
            "n_used": n_used,
            "missing": missing,
            "coverage_percent": (
                100.0 * n_used / total
                if total > 0
                else np.nan
            ),
            "missing_percent": (
                100.0 * missing / total
                if total > 0
                else np.nan
            ),
        })

    out = pd.DataFrame(rows)

    # Old logic: sort by missing count descending.
    # Stable secondary ordering by tool name makes ties reproducible.
    out = (
        out
        .sort_values(
            ["missing", "tool_name"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return out


# =============================================================================
# 6. PLOT
# =============================================================================

def plot_missingness(
    stats: pd.DataFrame,
    out_base: str,
):
    tools = stats["tool_name"].astype(str).tolist()
    missing = stats["missing"].to_numpy(dtype=float)

    n_tools = len(tools)

    # Same old logic: one distinct color per bar.
    cmap = plt.get_cmap("tab20")
    colors = [
        cmap(i % 20)
        for i in range(n_tools)
    ]

    fig_width = max(
        8.5,
        0.55 * n_tools,
    )

    fig, ax = plt.subplots(
        figsize=(fig_width, 4.5),
        dpi=FIG_DPI,
    )

    x = np.arange(n_tools)

    bars = ax.bar(
        x,
        missing,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        width=0.72,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        tools,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Missing predictions"
    )

    ax.set_xlabel(
        "Predictor"
    )

    ax.set_title(
        "Tool-specific prediction missingness on VarGeneDisjointTest"
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.set_axisbelow(True)

    max_missing = float(
        np.max(missing)
    ) if len(missing) else 0.0

    # Keep enough headroom for annotations.
    upper = max(
        5.0,
        max_missing * 1.16 + 1.0,
    )

    ax.set_ylim(
        0,
        upper,
    )

    # Old logic: annotate exact missing count on top of every bar.
    for rect, val in zip(
        bars,
        missing,
    ):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + upper * 0.012,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    save_figure(
        fig,
        out_base,
    )

    plt.close(fig)


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    ensure_dir(
        OUTPUT_DIR
    )

    print("=" * 92)
    print("SUPPLEMENTARY FIG. S2")
    print("=" * 92)
    print("Input:", INPUT_CSV)

    df = read_csv_smart(
        INPUT_CSV
    )

    print(
        f"[CHECK] VarGeneDisjointTest rows: {len(df):,}"
    )

    if len(df) != EXPECTED_TEST_N:
        print(
            f"[WARN] Expected N={EXPECTED_TEST_N}, "
            f"but test_score.csv contains N={len(df)}."
        )

    validate_columns(
        df
    )

    stats = compute_missingness(
        df
    )

    print("\n[Tool-specific prediction missingness]")
    print(
        stats[
            [
                "tool_name",
                "total",
                "n_used",
                "missing",
                "coverage_percent",
            ]
        ].to_string(
            index=False,
            formatters={
                "coverage_percent": lambda x: f"{x:.2f}",
            },
        )
    )

    # -------------------------------------------------------------------------
    # Save numerical audit
    # -------------------------------------------------------------------------

    stats_csv = os.path.join(
        OUTPUT_DIR,
        "Supplementary_FigS2_tool_missingness.csv",
    )

    stats.to_csv(
        stats_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n[SAVE] {stats_csv}"
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    output_base = os.path.join(
        OUTPUT_DIR,
        OUTPUT_BASENAME,
    )

    plot_missingness(
        stats,
        output_base,
    )

    print("\n" + "=" * 92)
    print("DONE")
    print("=" * 92)
    print("PNG:", output_base + ".png")
    print("PDF:", output_base + ".pdf")


if __name__ == "__main__":
    main()
