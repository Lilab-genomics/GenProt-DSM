#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supplementary Fig. S1 — Detailed gene-level variant distribution

Final layout
------------
(a) Exact gene-level variant-count distribution
    - Full observed range is shown.
    - Both axes use logarithmic scaling.
    - Train/Test are connected with lines and markers.

(b) Long-tail gene-level variant distribution
    - Shows the percentage of genes with at least k variants.
    - Both axes use logarithmic scaling so the long tail is visible.
    - y-axis labels are shown directly as percentages:
      100%, 10%, 1%, 0.1%.

Required input
--------------
F:\\GenProt-DSM_Resubmit\\result\\GeneDistribution\\gene_partition_summary.csv

Required columns
----------------
Gene
Train_variant_count
Test_variant_count

Optional columns
----------------
Total_variant_count
Partition

Outputs
-------
F:\\GenProt-DSM_Resubmit\\fig_Results\\Supplementary_FigS1\\
    Supplementary_FigS1_gene_level_distribution.png
    Supplementary_FigS1_gene_level_distribution.pdf
    FigS1a_exact_distribution.csv
    FigS1b_long_tail_distribution.csv
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator


# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

GENE_PARTITION_CSV = (
    r"F:\GenProt-DSM_Resubmit\result\GeneDistribution"
    r"\gene_partition_summary.csv"
)

OUTPUT_DIR = (
    r"F:\GenProt-DSM_Resubmit\fig_Results"
    r"\Supplementary_FigS1"
)

OUTPUT_BASENAME = "Supplementary_FigS1_gene_level_distribution"


# =============================================================================
# 2. GLOBAL PLOTTING STYLE
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
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# =============================================================================
# 3. COLORS
# =============================================================================

# Same Train/Test colors used in the revised main Fig. 1
TRAIN_COLOR = "#72BCD5"
TEST_COLOR  = "#F7AA58"


# =============================================================================
# 4. GENERAL UTILITIES
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_table(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]
    return df


def resolve_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:

    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for c in candidates:
        key = c.lower().strip()
        if key in lower_map:
            return lower_map[key]

    if required:
        raise KeyError(
            "\nRequired column not found.\n"
            f"Candidates: {candidates}\n"
            f"Existing columns: {list(df.columns)}"
        )

    return None


def panel_label(ax, label: str):
    ax.text(
        -0.10,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
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
# 5. AXIS FORMATTERS
# =============================================================================

def percent_tick_formatter(y, pos):
    """
    Show log-scale percentage values directly:
      100 -> 100%
       10 -> 10%
        1 -> 1%
      0.1 -> 0.1%
    """
    if y >= 1:
        return f"{y:g}%"
    return f"{y:.1f}%"


def plain_integer_formatter(x, pos):
    if x >= 1:
        return f"{int(round(x))}"
    return f"{x:g}"


# =============================================================================
# 6. STANDARDIZE GENE PARTITION TABLE
# =============================================================================

def standardize_gene_partition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    gene_col = resolve_column(
        df,
        ["Gene", "gene"],
    )

    train_col = resolve_column(
        df,
        [
            "Train_variant_count",
            "train_variant_count",
            "Train_count",
            "train_count",
            "Train variants",
            "Train_variants",
        ],
    )

    test_col = resolve_column(
        df,
        [
            "Test_variant_count",
            "test_variant_count",
            "Test_count",
            "test_count",
            "Test variants",
            "Test_variants",
        ],
    )

    df = df.rename(columns={
        gene_col: "Gene",
        train_col: "Train_variant_count",
        test_col: "Test_variant_count",
    })

    df["Gene"] = (
        df["Gene"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for col in [
        "Train_variant_count",
        "Test_variant_count",
    ]:
        df[col] = (
            pd.to_numeric(
                df[col],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

        if (df[col] < 0).any():
            raise ValueError(
                f"{col} contains negative values."
            )

    if df["Gene"].duplicated().any():
        dup = (
            df.loc[
                df["Gene"].duplicated(keep=False),
                "Gene",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "gene_partition_summary contains duplicated Gene rows.\n"
            f"Examples: {dup[:20]}"
        )

    return df


# =============================================================================
# 7. PREPARE EXACT DISTRIBUTION
# =============================================================================

def prepare_exact_distribution(
    partition_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Only observed exact counts are retained.

    For each split and exact k:
      Gene_count   = number of genes with exactly k variants
      Gene_percent = Gene_count / total genes in split * 100
    """

    rows = []

    for split, count_col in [
        ("Train", "Train_variant_count"),
        ("Test", "Test_variant_count"),
    ]:

        counts = (
            partition_df.loc[
                partition_df[count_col] > 0,
                count_col,
            ]
            .astype(int)
            .to_numpy()
        )

        if len(counts) == 0:
            raise RuntimeError(
                f"No genes with {count_col} > 0."
            )

        total_genes = int(len(counts))

        vc = (
            pd.Series(counts)
            .value_counts()
            .sort_index()
        )

        for k, n_gene in vc.items():
            rows.append({
                "Split": split,
                "Variants_per_gene": int(k),
                "Gene_count": int(n_gene),
                "Gene_percent": (
                    100.0 * int(n_gene) / total_genes
                ),
                "Total_genes_in_split": total_genes,
            })

    return pd.DataFrame(rows)


# =============================================================================
# 8. PREPARE LONG-TAIL DISTRIBUTION
# =============================================================================

def prepare_long_tail_distribution(
    partition_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each threshold k:
      percentage of genes with at least k variants.
    """

    rows = []

    for split, count_col in [
        ("Train", "Train_variant_count"),
        ("Test", "Test_variant_count"),
    ]:

        counts = (
            partition_df.loc[
                partition_df[count_col] > 0,
                count_col,
            ]
            .astype(int)
            .to_numpy()
        )

        if len(counts) == 0:
            raise RuntimeError(
                f"No genes with {count_col} > 0."
            )

        total_genes = int(len(counts))
        max_count = int(counts.max())

        for k in range(1, max_count + 1):
            n_ge = int(
                np.sum(counts >= k)
            )

            rows.append({
                "Split": split,
                "Minimum_variants_per_gene": int(k),
                "Genes_with_at_least_k": n_ge,
                "Genes_with_at_least_k_percent": (
                    100.0 * n_ge / total_genes
                ),
                "Total_genes_in_split": total_genes,
            })

    return pd.DataFrame(rows)


# =============================================================================
# 9. PANEL A — EXACT DISTRIBUTION
# =============================================================================

def draw_panel_a(
    ax,
    exact_df: pd.DataFrame,
):
    """
    Full-range exact distribution.

    Both x and y axes use logarithmic scaling.
    Maximum observed counts are marked directly at the terminal points.
    """

    configs = [
        ("Train", TRAIN_COLOR, "o"),
        ("Test", TEST_COLOR, "s"),
    ]

    global_max_x = 1

    for split, color, marker in configs:
        sub = (
            exact_df.loc[
                exact_df["Split"] == split
            ]
            .sort_values("Variants_per_gene")
        )

        x = sub[
            "Variants_per_gene"
        ].to_numpy(dtype=float)

        y = sub[
            "Gene_percent"
        ].to_numpy(dtype=float)

        if len(x) == 0:
            continue

        global_max_x = max(
            global_max_x,
            int(np.max(x)),
        )

        n_genes = int(
            sub["Total_genes_in_split"].iloc[0]
        )

        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.15,
            alpha=0.95,
            zorder=2,
        )

        ax.scatter(
            x,
            y,
            s=25,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.95,
            label=f"{split} (n={n_genes:,} genes)",
            zorder=3,
        )


    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    # Readable x ticks across the full observed range.
    x_ticks = [
        1, 2, 3, 5,
        10, 20, 30, 50,
        100, 200,
    ]

    x_ticks = [
        x for x in x_ticks
        if x <= max(global_max_x * 1.08, 2)
    ]

    ax.xaxis.set_major_locator(
        FixedLocator(x_ticks)
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(plain_integer_formatter)
    )

    # Direct percentage labels rather than 10^n notation.
    y_ticks = [
        0.1,
        1,
        10,
        100,
    ]

    ax.yaxis.set_major_locator(
        FixedLocator(y_ticks)
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(percent_tick_formatter)
    )

    ax.set_xlim(
        0.9,
        max(global_max_x * 1.20, 2.2),
    )

    ax.set_ylim(
        0.05,
        120,
    )

    ax.set_xlabel(
        "Variants per gene"
    )

    ax.set_ylabel(
        "Genes with exactly this many variants (%)"
    )

    ax.set_title(
        "Exact gene-level variant-count distribution"
    )

    ax.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.5,
        alpha=0.55,
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=8,
    )

    panel_label(
        ax,
        "a",
    )


# =============================================================================
# 10. PANEL B — LONG-TAIL DISTRIBUTION
# =============================================================================

def draw_panel_b(
    ax,
    long_tail_df: pd.DataFrame,
):
    """
    Full long-tail distribution.

    x-axis: logarithmic
    y-axis: logarithmic
    y labels are shown directly as percentages.
    """

    configs = [
        ("Train", TRAIN_COLOR, "o"),
        ("Test", TEST_COLOR, "s"),
    ]

    global_max_x = 1

    for split, color, marker in configs:
        sub = (
            long_tail_df.loc[
                long_tail_df["Split"] == split
            ]
            .sort_values("Minimum_variants_per_gene")
        )

        x = sub[
            "Minimum_variants_per_gene"
        ].to_numpy(dtype=float)

        y = sub[
            "Genes_with_at_least_k_percent"
        ].to_numpy(dtype=float)

        if len(x) == 0:
            continue

        global_max_x = max(
            global_max_x,
            int(np.max(x)),
        )

        n_genes = int(
            sub["Total_genes_in_split"].iloc[0]
        )

        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.5,
            alpha=0.95,
            label=f"{split} (n={n_genes:,} genes)",
            zorder=2,
        )

        # Sparse markers to keep the tail readable.
        if len(x) <= 18:
            marker_idx = np.arange(len(x))
        else:
            marker_idx = np.unique(
                np.linspace(
                    0,
                    len(x) - 1,
                    18,
                )
                .round()
                .astype(int)
            )

        ax.scatter(
            x[marker_idx],
            y[marker_idx],
            s=20,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.95,
            zorder=3,
        )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    x_ticks = [
        1, 2, 5,
        10, 20, 50,
        100, 200,
    ]

    x_ticks = [
        x for x in x_ticks
        if x <= max(global_max_x * 1.05, 2)
    ]

    ax.xaxis.set_major_locator(
        FixedLocator(x_ticks)
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(plain_integer_formatter)
    )

    y_ticks = [
        0.1,
        1,
        10,
        100,
    ]

    ax.yaxis.set_major_locator(
        FixedLocator(y_ticks)
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(percent_tick_formatter)
    )

    ax.set_xlim(
        0.9,
        max(global_max_x * 1.12, 2.2),
    )

    ax.set_ylim(
        0.05,
        120,
    )

    ax.set_xlabel(
        "Minimum variants per gene"
    )

    ax.set_ylabel(
        "Genes with at least this many variants (%)"
    )

    ax.set_title(
        "Long-tail gene-level variant distribution"
    )

    ax.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.5,
        alpha=0.55,
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=8,
    )

    panel_label(
        ax,
        "b",
    )


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    ensure_dir(
        OUTPUT_DIR
    )

    print("=" * 92)
    print("SUPPLEMENTARY FIG. S1")
    print("=" * 92)
    print("Input:", GENE_PARTITION_CSV)

    raw = read_table(
        GENE_PARTITION_CSV
    )

    partition_df = standardize_gene_partition(
        raw
    )

    # -------------------------------------------------------------------------
    # Basic audit
    # -------------------------------------------------------------------------

    train_counts = (
        partition_df.loc[
            partition_df["Train_variant_count"] > 0,
            "Train_variant_count",
        ]
        .astype(int)
        .to_numpy()
    )

    test_counts = (
        partition_df.loc[
            partition_df["Test_variant_count"] > 0,
            "Test_variant_count",
        ]
        .astype(int)
        .to_numpy()
    )

    overlap_n = int(
        (
            (partition_df["Train_variant_count"] > 0)
            &
            (partition_df["Test_variant_count"] > 0)
        )
        .sum()
    )

    print(
        f"[CHECK] Train genes: {len(train_counts):,}"
    )

    print(
        f"[CHECK] Test genes : {len(test_counts):,}"
    )

    print(
        f"[CHECK] Shared genes: {overlap_n:,}"
    )

    print(
        f"[CHECK] Train max variants/gene: {int(train_counts.max()):,}"
    )

    print(
        f"[CHECK] Test max variants/gene : {int(test_counts.max()):,}"
    )

    if overlap_n != 0:
        raise RuntimeError(
            f"Gene-disjoint audit failed: shared genes = {overlap_n}"
        )

    # -------------------------------------------------------------------------
    # Prepare statistics
    # -------------------------------------------------------------------------

    exact_df = prepare_exact_distribution(
        partition_df
    )

    long_tail_df = prepare_long_tail_distribution(
        partition_df
    )

    # -------------------------------------------------------------------------
    # Save supporting data
    # -------------------------------------------------------------------------

    exact_csv = os.path.join(
        OUTPUT_DIR,
        "FigS1a_exact_distribution.csv",
    )

    long_tail_csv = os.path.join(
        OUTPUT_DIR,
        "FigS1b_long_tail_distribution.csv",
    )

    exact_df.to_csv(
        exact_csv,
        index=False,
        encoding="utf-8-sig",
    )

    long_tail_df.to_csv(
        long_tail_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[SAVE] {exact_csv}"
    )

    print(
        f"[SAVE] {long_tail_csv}"
    )

    # -------------------------------------------------------------------------
    # Build figure
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.6),
        dpi=FIG_DPI,
    )

    draw_panel_a(
        axes[0],
        exact_df,
    )

    draw_panel_b(
        axes[1],
        long_tail_df,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.91,
        bottom=0.15,
        wspace=0.25,
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

    print("\n" + "=" * 92)
    print("DONE")
    print("=" * 92)
    print("PNG:", output_base + ".png")
    print("PDF:", output_base + ".pdf")


if __name__ == "__main__":
    main()
