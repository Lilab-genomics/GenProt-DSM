#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig. 1 — Characteristics of the revised benchmark dataset

Panels
------
(a) Overall chromosomal distribution of Train + Test variants
    - Uses the original jitter-cloud plotting logic from the submitted version.
    - Train and Test are merged before plotting.
    - Each chromosome uses the original tab20 color sequence.
    - The number of points equals the number of variants on that chromosome.
    - Vertical point locations are jittered within [0, log10(n+1)] only for visualization.

(b) Composition of seven brain-disorder categories

(c) Binned distribution of variants per gene in Train/Test

(d) Train/Test dataset summary and gene-disjoint verification

IMPORTANT
---------
The final Train/Test CSV files do NOT need a Gene column.

Gene-level information is read from:
    gene_partition_summary

Optional audit files:
    gene_variant_count_bins
    gene_disjoint_summary

Run
---
python F:\\GenProt-DSM_Resubmit\\plot_Fig1_revised_dataset.py

Outputs
-------
PNG:
    Fig1_revised_dataset_characteristics.png

PDF:
    Fig1_revised_dataset_characteristics.pdf

Supporting statistics:
    Fig1a_chromosome_distribution.csv
    Fig1b_disease_distribution.csv
    Fig1c_gene_variant_bins_recomputed.csv
    Fig1d_dataset_summary.csv
"""

from __future__ import annotations

import os
import re
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

# Final Train/Test datasets
TRAIN_CSV = r"F:\GenProt-DSM_Resubmit\dataset\data\VarGeneDisjointTrain.csv"
TEST_CSV  = r"F:\GenProt-DSM_Resubmit\dataset\data\VarGeneDisjointTest.csv"

# Gene-distribution statistics directory
GENE_STATS_DIR = r"F:\GenProt-DSM_Resubmit\result\GeneDistribution"

GENE_PARTITION_STEM = "gene_partition_summary"
GENE_BINS_STEM      = "gene_variant_count_bins"
GENE_DISJOINT_STEM  = "gene_disjoint_summary"

# Output
OUTPUT_DIR = r"F:\GenProt-DSM_Resubmit\fig_Results\Fig1_Revised_Dataset"
OUTPUT_BASENAME = "Fig1_revised_dataset_characteristics"


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
    "legend.fontsize": PLOT_FONT_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# =============================================================================
# 3. COLORS
# =============================================================================

# Train/Test colors used in panels c and d
TRAIN_COLOR = "#72BCD5"
TEST_COLOR  = "#F7AA58"

TRAIN_LIGHT = "#E4F3F7"
TEST_LIGHT  = "#FEF0DF"

# Manuscript-wide binary class colors
COLOR_CLASS_NEG = "#DEEAEA"
COLOR_CLASS_POS = "#F7E474"

# Fixed seven-color disease palette
DISEASE_PALETTE = [
    "#AADCE0",  # Intellectual Disability
    "#FFD06F",  # Alzheimer
    "#72BCD5",  # Attention Deficit
    "#FFE6B7",  # Language Disorder
    "#528FAD",  # Autism
    "#F7AA58",  # Schizophrenia
    "#376795",  # Tourette
]


# =============================================================================
# 4. DISEASE CATEGORY CONFIGURATION
# =============================================================================

DISEASE_ORDER = [
    "Intellectual Disability",
    "Alzheimer",
    "Attention Deficit",
    "Language Disorder",
    "Autism",
    "Schizophrenia",
    "Tourette",
]


def normalize_disease_name(x) -> str:
    """
    Normalize raw disease-category names into the seven categories used
    in the revised manuscript.

    Also tolerates truncated strings such as:
        Intellectual Disabili
    """

    if pd.isna(x):
        return ""

    s = str(x).strip()
    if not s:
        return ""

    key = re.sub(r"[\s_\-]+", " ", s.lower()).strip()

    # Intellectual disability
    if (
        key == "intellectual disability"
        or key == "intellectual disabilities"
        or key.startswith("intellectual disabili")
    ):
        return "Intellectual Disability"

    # Alzheimer
    if (
        key == "alzheimer"
        or key == "alzheimer disease"
        or key == "alzheimer's disease"
        or key == "alzheimers disease"
        or key.startswith("alzheimer")
    ):
        return "Alzheimer"

    # Attention deficit / ADHD
    if (
        key == "attention deficit"
        or key == "attention deficit disorder"
        or key == "attention deficit hyperactivity disorder"
        or key == "adhd"
        or key.startswith("attention deficit")
    ):
        return "Attention Deficit"

    # Language disorder
    if (
        key == "language disorder"
        or key.startswith("language disord")
    ):
        return "Language Disorder"

    # Autism
    if (
        key == "autism"
        or key == "autism spectrum disorder"
        or key == "asd"
        or key.startswith("autism")
    ):
        return "Autism"

    # Schizophrenia
    if (
        key == "schizophrenia"
        or key.startswith("schizophren")
    ):
        return "Schizophrenia"

    # Tourette
    if (
        key == "tourette"
        or key == "tourette syndrome"
        or key.startswith("tourette")
    ):
        return "Tourette"

    # Obsessive-compulsive disorder is intentionally excluded from
    # the revised seven-category dataset figure. If it unexpectedly
    # appears with a positive count in the final dataset, QC should stop.
    if (
        key == "obsessive compulsive disorder"
        or key == "obsessive compulsive disorders"
        or key == "ocd"
        or key.startswith("obsessive compulsive")
    ):
        return "Obsessive-compulsive Disorder"

    # Keep unknown category unchanged so QC catches it.
    return s


# =============================================================================
# 5. GENERAL FILE UTILITIES
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_file_by_stem(
    directory: str,
    stem: str,
    required: bool = True,
) -> Optional[str]:
    """
    Search for:
        stem.csv
        stem.xlsx
        stem.xls
    """

    if not os.path.isdir(directory):
        if required:
            raise FileNotFoundError(f"Directory not found:\n{directory}")
        return None

    for ext in [".csv", ".xlsx", ".xls"]:
        candidate = os.path.join(directory, stem + ext)
        if os.path.isfile(candidate):
            return candidate

    stem_low = stem.lower()

    for fn in os.listdir(directory):
        root, ext = os.path.splitext(fn)
        if root.lower() == stem_low and ext.lower() in [".csv", ".xlsx", ".xls"]:
            return os.path.join(directory, fn)

    if required:
        raise FileNotFoundError(
            f"Cannot find '{stem}' under:\n{directory}\n"
            f"Supported extensions: .csv / .xlsx / .xls"
        )

    return None


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
        raise ValueError(f"Unsupported file type: {ext}")

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

    mapping = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for c in candidates:
        key = c.lower().strip()
        if key in mapping:
            return mapping[key]

    if required:
        raise KeyError(
            "\nRequired column not found.\n"
            f"Candidates: {candidates}\n"
            f"Existing columns: {list(df.columns)}"
        )

    return None


# =============================================================================
# 6. FIGURE UTILITIES
# =============================================================================

def panel_label(ax, label: str):
    ax.text(
        -0.075,
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
# 7. STANDARDIZE TRAIN / TEST
# =============================================================================

def standardize_variant_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gene is deliberately NOT required.
    """

    df = df.copy()

    chr_col = resolve_column(
        df,
        ["Chrom", "Chr", "chrom", "chr", "Chromosome"],
    )

    pos_col = resolve_column(
        df,
        ["Position", "Start", "position", "start"],
    )

    ref_col = resolve_column(
        df,
        ["Reference", "Ref", "reference", "ref"],
    )

    alt_col = resolve_column(
        df,
        ["Alternate", "Alt", "alternate", "alt"],
    )

    label_col = resolve_column(
        df,
        ["label", "Label"],
    )

    disease_col = resolve_column(
        df,
        ["disease type", "Disease type", "disease_type", "Disease_type"],
        required=False,
    )

    variant_id_col = resolve_column(
        df,
        ["variant_id", "Variant_ID", "variantid"],
        required=False,
    )

    rename_map = {
        chr_col: "Chrom",
        pos_col: "Position",
        ref_col: "Reference",
        alt_col: "Alternate",
        label_col: "label",
    }

    if disease_col is not None:
        rename_map[disease_col] = "disease type"

    if variant_id_col is not None:
        rename_map[variant_id_col] = "variant_id"

    df = df.rename(columns=rename_map)

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    if not df["label"].isin([0, 1]).all():
        raise ValueError("label contains values other than 0/1.")

    if "variant_id" in df.columns:
        df["_variant_key"] = (
            df["variant_id"]
            .astype(str)
            .str.strip()
        )
    else:
        df["_variant_key"] = (
            df[["Chrom", "Position", "Reference", "Alternate"]]
            .astype(str)
            .agg("|".join, axis=1)
        )

    duplicates = df["_variant_key"].duplicated(keep=False)

    if duplicates.any():
        n_dup = df.loc[duplicates, "_variant_key"].nunique()
        raise ValueError(
            f"Found {n_dup} duplicated variants in final Train/Test dataset."
        )

    return df


# =============================================================================
# 8. STANDARDIZE GENE PARTITION SUMMARY
# =============================================================================

def standardize_gene_partition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected logical structure:
        Gene
        Train_variant_count
        Test_variant_count

    Total_variant_count and Partition may also exist.
    """

    df = df.copy()

    gene_col = resolve_column(
        df,
        ["Gene", "gene"],
    )

    train_count_col = resolve_column(
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

    test_count_col = resolve_column(
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

    partition_col = resolve_column(
        df,
        ["Partition", "partition", "Split", "split"],
        required=False,
    )

    rename_map = {
        gene_col: "Gene",
        train_count_col: "Train_variant_count",
        test_count_col: "Test_variant_count",
    }

    if partition_col is not None:
        rename_map[partition_col] = "Partition"

    df = df.rename(columns=rename_map)

    df["Gene"] = (
        df["Gene"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Train_variant_count"] = (
        pd.to_numeric(
            df["Train_variant_count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["Test_variant_count"] = (
        pd.to_numeric(
            df["Test_variant_count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
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
# 9. CHROMOSOME UTILITIES
# =============================================================================

def normalize_chr(x) -> str:
    s = str(x).strip()
    s = re.sub(r"^chr", "", s, flags=re.IGNORECASE)
    s = s.upper()

    if s in {"M", "MT"}:
        return "MT"

    try:
        return str(int(float(s)))
    except Exception:
        return s


def chromosome_sort_key(chrom):
    chrom = str(chrom)

    if chrom.isdigit():
        return (0, int(chrom))

    special = {
        "X": 23,
        "Y": 24,
        "MT": 25,
    }

    if chrom in special:
        return (1, special[chrom])

    return (2, 1000, chrom)


# =============================================================================
# 10. PANEL A — OVERALL CHROMOSOME JITTER DISTRIBUTION
#     Uses the original submitted-version plotting logic.
# =============================================================================

def prepare_chromosome_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train and Test are merged before chromosome counting.
    """

    all_df = pd.concat(
        [
            train_df[["Chrom"]].rename(columns={"Chrom": "Chr"}),
            test_df[["Chrom"]].rename(columns={"Chrom": "Chr"}),
        ],
        axis=0,
        ignore_index=True,
    )

    all_df["Chr"] = all_df["Chr"].map(normalize_chr)

    counts = (
        all_df["Chr"]
        .value_counts()
    )

    chromosomes = sorted(
        counts.index.tolist(),
        key=chromosome_sort_key,
    )

    out = pd.DataFrame({
        "Chromosome": chromosomes,
        "Count": [int(counts.get(c, 0)) for c in chromosomes],
    })

    out["Percent"] = (
        100.0
        * out["Count"]
        / out["Count"].sum()
    )

    out["log10_count_plus_1"] = np.log10(
        out["Count"].astype(float) + 1.0
    )

    return out


def draw_panel_a(
    ax,
    data: pd.DataFrame,
    random_seed: int = 42,
):
    """
    Reproduces the old chromosome-distribution visual style:

      - one vertical jitter cloud per chromosome
      - number of points = number of variants on that chromosome
      - x = chromosome index + Uniform(-0.28, 0.28)
      - y = Uniform(0, log10(n+1))
      - colors = matplotlib tab20 sequence

    A fixed random seed is used so the publication figure is reproducible.
    """

    rng = np.random.default_rng(random_seed)
    cmap = plt.get_cmap("tab20")

    xs = []
    ys = []
    cs = []

    chromosomes = data["Chromosome"].astype(str).tolist()
    counts = data["Count"].astype(int).tolist()

    for i, (chrom, n) in enumerate(
        zip(chromosomes, counts),
        start=1,
    ):
        if n <= 0:
            continue

        ymax = float(np.log10(n + 1.0))

        xj = (
            i
            + rng.uniform(
                -0.28,
                0.28,
                size=n,
            )
        )

        yj = rng.uniform(
            0.0,
            max(ymax, 1e-6),
            size=n,
        )

        xs.append(xj)
        ys.append(yj)

        # Same tab20 color logic as the original code
        color = cmap((i - 1) % 20)
        cs.append(
            np.tile(
                color,
                (n, 1),
            )
        )

    if not xs:
        raise RuntimeError(
            "No chromosome points available for Fig. 1a."
        )

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    C = np.concatenate(cs, axis=0)

    ax.scatter(
        X,
        Y,
        s=6,
        alpha=0.65,
        c=C,
        edgecolors="none",
    )

    ax.set_xlim(
        0.5,
        len(chromosomes) + 0.5,
    )

    ax.set_xticks(
        np.arange(
            1,
            len(chromosomes) + 1,
        )
    )

    ax.set_xticklabels(
        chromosomes,
        fontsize=8,
    )

    ax.set_ylabel(
        "log10(count+1)"
    )

    ax.set_xlabel(
        "Chromosome"
    )

    ax.set_title(
        "Chromosomal distribution of variants"
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.6,
    )

    panel_label(
        ax,
        "a",
    )


# =============================================================================
# 11. PANEL B — SEVEN BRAIN-DISORDER CATEGORIES
# =============================================================================

def prepare_disease_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:

    combined = pd.concat(
        [train_df, test_df],
        ignore_index=True,
    )

    if "disease type" not in combined.columns:
        raise KeyError(
            "Fig. 1b requires column 'disease type' in Train/Test CSV."
        )

    # Disease-category composition is defined on pathogenic variants.
    pos = combined.loc[
        combined["label"] == 1
    ].copy()

    missing = (
        pos["disease type"].isna()
        |
        (
            pos["disease type"]
            .astype(str)
            .str.strip()
            == ""
        )
    )

    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} pathogenic rows have missing disease type."
        )

    print("\n[CHECK] Raw pathogenic disease labels:")
    print(
        pos["disease type"]
        .astype(str)
        .value_counts(dropna=False)
        .to_string()
    )

    pos["_Disease"] = (
        pos["disease type"]
        .map(normalize_disease_name)
    )

    # Revised Fig. 1 contains seven categories only.
    # Any positive OCD row is treated as a data-QC problem rather than silently omitted.
    ocd_n = int(
        (
            pos["_Disease"]
            == "Obsessive-compulsive Disorder"
        )
        .sum()
    )

    if ocd_n > 0:
        raise ValueError(
            "\nThe revised dataset is expected to contain seven disease categories, "
            "but Obsessive-compulsive Disorder is still present.\n"
            f"Pathogenic OCD rows found: {ocd_n}\n"
            "Please verify the final dataset rather than silently removing them from the figure."
        )

    unknown = sorted(
        set(pos["_Disease"])
        - set(DISEASE_ORDER)
    )

    if unknown:
        raise ValueError(
            "\nUnexpected disease categories after normalization:\n"
            + "\n".join(unknown)
            + "\n\nPlease add corresponding aliases to normalize_disease_name()."
        )

    counts = (
        pos["_Disease"]
        .value_counts()
        .reindex(
            DISEASE_ORDER,
            fill_value=0,
        )
    )

    data = pd.DataFrame({
        "Disease": DISEASE_ORDER,
        "Count": counts.values,
    })

    total = int(data["Count"].sum())

    data["Percent"] = (
        100.0
        * data["Count"]
        / total
    )

    return data


def draw_panel_b(
    ax,
    data: pd.DataFrame,
):
    values = data["Count"].to_numpy()

    wedges, _ = ax.pie(
        values,
        colors=DISEASE_PALETTE,
        startangle=90,
        counterclock=False,
        wedgeprops={
            "width": 0.38,
            "edgecolor": "white",
            "linewidth": 0.8,
        },
    )

    total = int(values.sum())

    ax.text(
        0,
        0.05,
        "Pathogenic",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    ax.text(
        0,
        -0.12,
        f"N = {total:,}",
        ha="center",
        va="center",
        fontsize=9,
    )

    legend_labels = [
        f"{row['Disease']} "
        f"(n={int(row['Count'])}, {row['Percent']:.1f}%)"
        for _, row in data.iterrows()
    ]

    ax.legend(
        wedges,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        fontsize=8,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.0,
    )

    ax.set_title(
        "Brain-disorder category composition"
    )

    panel_label(
        ax,
        "b",
    )


# =============================================================================
# 12. PANEL C — BINNED VARIANTS PER GENE
# =============================================================================

GENE_BIN_ORDER = [
    "1",
    "2–5",
    "6–10",
    "11–20",
    "21–50",
    ">50",
]


def gene_count_to_bin(n: int) -> str:
    if n == 1:
        return "1"
    if 2 <= n <= 5:
        return "2–5"
    if 6 <= n <= 10:
        return "6–10"
    if 11 <= n <= 20:
        return "11–20"
    if 21 <= n <= 50:
        return "21–50"
    return ">50"


def prepare_gene_bin_data_from_partition(
    partition_df: pd.DataFrame,
) -> pd.DataFrame:

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
        )

        bins = counts.map(
            gene_count_to_bin
        )

        vc = (
            bins
            .value_counts()
            .reindex(
                GENE_BIN_ORDER,
                fill_value=0,
            )
        )

        total_genes = int(
            len(counts)
        )

        for category in GENE_BIN_ORDER:
            gene_n = int(
                vc.loc[category]
            )

            pct = (
                100.0
                * gene_n
                / total_genes
                if total_genes > 0
                else 0.0
            )

            rows.append({
                "Split": split,
                "Variants_per_gene": category,
                "Gene_count": gene_n,
                "Gene_percent": pct,
            })

    return pd.DataFrame(
        rows
    )


def draw_panel_c(
    ax,
    data: pd.DataFrame,
):

    train = (
        data[data["Split"] == "Train"]
        .set_index("Variants_per_gene")
        .reindex(GENE_BIN_ORDER)
    )

    test = (
        data[data["Split"] == "Test"]
        .set_index("Variants_per_gene")
        .reindex(GENE_BIN_ORDER)
    )

    x = np.arange(
        len(GENE_BIN_ORDER)
    )

    width = 0.37

    bars_train = ax.bar(
        x - width / 2,
        train["Gene_percent"],
        width=width,
        color=TRAIN_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="Train",
    )

    bars_test = ax.bar(
        x + width / 2,
        test["Gene_percent"],
        width=width,
        color=TEST_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="Test",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(GENE_BIN_ORDER)

    ax.set_xlabel(
        "Variants per gene"
    )

    ax.set_ylabel(
        "Genes (%)"
    )

    ax.set_title(
        "Gene-level variant distribution"
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        fontsize=8,
        frameon=True,
        loc="upper right",
    )

    max_y = max(
        float(train["Gene_percent"].max()),
        float(test["Gene_percent"].max()),
    )

    offset = max(
        0.4,
        max_y * 0.018,
    )

    # Numbers above bars are actual gene counts.
    for bars, counts in [
        (bars_train, train["Gene_count"]),
        (bars_test, test["Gene_count"]),
    ]:
        for bar, n in zip(
            bars,
            counts,
        ):
            n = int(n)

            if n <= 0:
                continue

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height()
                + offset,
                str(n),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylim(
        0,
        max_y * 1.16
        if max_y > 0
        else 1,
    )

    panel_label(
        ax,
        "c",
    )


# =============================================================================
# 13. OPTIONAL CROSS-CHECK — gene_variant_count_bins
# =============================================================================

def try_crosscheck_gene_bins(
    official_bins_path: Optional[str],
):
    if not official_bins_path:
        return

    try:
        df = read_table(
            official_bins_path
        )

        print(
            "\n[INFO] gene_variant_count_bins found:"
        )
        print(
            official_bins_path
        )
        print(
            "[INFO] Columns:"
        )
        print(
            list(df.columns)
        )

        out_copy = os.path.join(
            OUTPUT_DIR,
            "gene_variant_count_bins_input_snapshot.csv",
        )

        df.to_csv(
            out_copy,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "[INFO] Snapshot exported:"
        )
        print(
            out_copy
        )

    except Exception as e:
        print(
            f"[WARN] Could not inspect gene_variant_count_bins: {e}"
        )


# =============================================================================
# 14. PANEL D — DATASET + GENE-DISJOINT SUMMARY
# =============================================================================

def prepare_dataset_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    partition_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:

    train_gene_mask = (
        partition_df["Train_variant_count"] > 0
    )

    test_gene_mask = (
        partition_df["Test_variant_count"] > 0
    )

    overlap_mask = (
        train_gene_mask
        & test_gene_mask
    )

    n_train_genes = int(
        train_gene_mask.sum()
    )

    n_test_genes = int(
        test_gene_mask.sum()
    )

    n_overlap = int(
        overlap_mask.sum()
    )

    if n_overlap != 0:
        overlap_genes = (
            partition_df.loc[
                overlap_mask,
                "Gene",
            ]
            .tolist()
        )

        raise RuntimeError(
            "\nGene-disjoint validation FAILED.\n"
            f"Shared genes = {n_overlap}\n"
            f"Examples: {overlap_genes[:20]}"
        )

    summary = pd.DataFrame([
        {
            "Split": "Train",
            "Variants": len(train_df),
            "Pathogenic": int((train_df["label"] == 1).sum()),
            "Benign": int((train_df["label"] == 0).sum()),
            "Unique_genes": n_train_genes,
        },
        {
            "Split": "Test",
            "Variants": len(test_df),
            "Pathogenic": int((test_df["label"] == 1).sum()),
            "Benign": int((test_df["label"] == 0).sum()),
            "Unique_genes": n_test_genes,
        },
    ])

    return summary, n_overlap


def draw_panel_d(
    ax,
    summary: pd.DataFrame,
    overlap_n: int,
):
    """
    Panel d — compact bar-chart summary with non-overlapping layout.

    Top:
        Train/Test variant composition (Pathogenic + Benign)

    Bottom:
        Train/Test unique-gene counts

    Footer:
        Gene overlap = 0

    NOTE:
        Only panel d is changed. Panels a/b/c and all data logic remain unchanged.
    """

    # Outer panel only provides the title / section labels / footer.
    ax.axis("off")

    ax.set_title(
        "Gene-disjoint dataset summary",
        pad=8,
    )

    panel_label(
        ax,
        "d",
    )

    # -------------------------------------------------------------------------
    # Extract statistics
    # -------------------------------------------------------------------------
    train_row = (
        summary.loc[
            summary["Split"] == "Train"
        ]
        .iloc[0]
    )

    test_row = (
        summary.loc[
            summary["Split"] == "Test"
        ]
        .iloc[0]
    )

    splits = ["Train", "Test"]

    pathogenic = np.array([
        int(train_row["Pathogenic"]),
        int(test_row["Pathogenic"]),
    ])

    benign = np.array([
        int(train_row["Benign"]),
        int(test_row["Benign"]),
    ])

    totals = pathogenic + benign

    unique_genes = np.array([
        int(train_row["Unique_genes"]),
        int(test_row["Unique_genes"]),
    ])

    # =========================================================================
    # D1. VARIANT COMPOSITION
    # =========================================================================

    # Section heading placed in the OUTER panel so it cannot collide with legend.
    ax.text(
        0.12,
        0.885,
        "Variant composition",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

    ax_var = ax.inset_axes(
        [
            0.14,   # left
            0.565,  # bottom
            0.80,   # width
            0.255,  # height
        ]
    )

    y = np.array([1, 0])
    bar_height = 0.46

    ax_var.barh(
        y,
        pathogenic,
        height=bar_height,
        color=COLOR_CLASS_POS,
        edgecolor="white",
        linewidth=0.7,
        label="Pathogenic",
        zorder=3,
    )

    ax_var.barh(
        y,
        benign,
        left=pathogenic,
        height=bar_height,
        color=COLOR_CLASS_NEG,
        edgecolor="white",
        linewidth=0.7,
        label="Benign",
        zorder=3,
    )

    ax_var.set_yticks(y)
    ax_var.set_yticklabels(
        splits,
        fontsize=8,
    )

    max_total = float(totals.max())

    ax_var.set_xlim(
        0,
        max_total * 1.13,
    )

    # No internal title here — avoids collision with legend.
    ax_var.set_xlabel(
        "Variants",
        fontsize=8,
        labelpad=3,
    )

    ax_var.grid(
        axis="x",
        linestyle=":",
        linewidth=0.5,
        alpha=0.50,
        zorder=0,
    )

    ax_var.set_axisbelow(True)

    # Cleaner journal-style frame.
    ax_var.spines["top"].set_visible(False)
    ax_var.spines["right"].set_visible(False)
    ax_var.spines["left"].set_visible(False)

    # Keep bottom spine subtle.
    ax_var.spines["bottom"].set_linewidth(0.7)

    # Segment counts + total N.
    for i in range(2):
        ax_var.text(
            pathogenic[i] / 2,
            y[i],
            f"{pathogenic[i]:,}",
            ha="center",
            va="center",
            fontsize=7,
            zorder=4,
        )

        ax_var.text(
            pathogenic[i] + benign[i] / 2,
            y[i],
            f"{benign[i]:,}",
            ha="center",
            va="center",
            fontsize=7,
            zorder=4,
        )

        ax_var.text(
            totals[i] + max_total * 0.018,
            y[i],
            f"N={totals[i]:,}",
            ha="left",
            va="center",
            fontsize=7,
            fontweight="bold",
            zorder=4,
        )

    # Legend is placed ABOVE the top inset, to the right of the section heading.
    ax_var.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.04),
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=1.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    # =========================================================================
    # D2. UNIQUE GENES
    # =========================================================================

    ax.text(
        0.12,
        0.445,
        "Unique genes",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

    ax_gene = ax.inset_axes(
        [
            0.14,
            0.205,
            0.80,
            0.185,
        ]
    )

    y_gene = np.array([1, 0])

    gene_colors = [
        TRAIN_COLOR,
        TEST_COLOR,
    ]

    bars = ax_gene.barh(
        y_gene,
        unique_genes,
        height=0.46,
        color=gene_colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    ax_gene.set_yticks(y_gene)
    ax_gene.set_yticklabels(
        splits,
        fontsize=8,
    )

    max_gene = float(unique_genes.max())

    ax_gene.set_xlim(
        0,
        max_gene * 1.13,
    )

    ax_gene.set_xlabel(
        "Genes",
        fontsize=8,
        labelpad=3,
    )

    ax_gene.grid(
        axis="x",
        linestyle=":",
        linewidth=0.5,
        alpha=0.50,
        zorder=0,
    )

    ax_gene.set_axisbelow(True)

    ax_gene.spines["top"].set_visible(False)
    ax_gene.spines["right"].set_visible(False)
    ax_gene.spines["left"].set_visible(False)
    ax_gene.spines["bottom"].set_linewidth(0.7)

    for bar, value in zip(
        bars,
        unique_genes,
    ):
        ax_gene.text(
            bar.get_width() + max_gene * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            ha="left",
            va="center",
            fontsize=7,
            fontweight="bold",
            zorder=4,
        )

    # =========================================================================
    # D3. GENE OVERLAP FOOTER
    # =========================================================================

    # Footer separator is clearly below the gene x-axis label.

    ax.text(
        0.50,
        0.050,
        f"Gene overlap = {overlap_n}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )


# =============================================================================
# 15. OPTIONAL CROSS-CHECK — gene_disjoint_summary
# =============================================================================

def try_crosscheck_disjoint_summary(
    disjoint_summary_path: Optional[str],
):
    if not disjoint_summary_path:
        return

    try:
        df = read_table(
            disjoint_summary_path
        )

        print(
            "\n[INFO] gene_disjoint_summary found:"
        )
        print(
            disjoint_summary_path
        )
        print(
            "[INFO] gene_disjoint_summary:"
        )
        print(
            df.to_string(
                index=False
            )
        )

        out_copy = os.path.join(
            OUTPUT_DIR,
            "gene_disjoint_summary_input_snapshot.csv",
        )

        df.to_csv(
            out_copy,
            index=False,
            encoding="utf-8-sig",
        )

    except Exception as e:
        print(
            f"[WARN] Could not inspect gene_disjoint_summary: {e}"
        )


# =============================================================================
# 16. MAIN
# =============================================================================

def main():
    ensure_dir(
        OUTPUT_DIR
    )

    # -------------------------------------------------------------------------
    # Locate gene-statistics files
    # -------------------------------------------------------------------------

    gene_partition_path = find_file_by_stem(
        GENE_STATS_DIR,
        GENE_PARTITION_STEM,
        required=True,
    )

    gene_bins_path = find_file_by_stem(
        GENE_STATS_DIR,
        GENE_BINS_STEM,
        required=False,
    )

    gene_disjoint_path = find_file_by_stem(
        GENE_STATS_DIR,
        GENE_DISJOINT_STEM,
        required=False,
    )

    print("=" * 88)
    print("INPUT FILES")
    print("=" * 88)
    print("Train:", TRAIN_CSV)
    print("Test:", TEST_CSV)
    print("Gene partition:", gene_partition_path)
    print("Gene bins:", gene_bins_path)
    print("Gene disjoint summary:", gene_disjoint_path)

    # Prevent the path mistake encountered previously.
    if os.path.abspath(TRAIN_CSV).lower() == os.path.abspath(TEST_CSV).lower():
        raise RuntimeError(
            "\nTRAIN_CSV and TEST_CSV point to the SAME FILE.\n"
            "Please check the path configuration."
        )

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    train_df = standardize_variant_df(
        read_table(
            TRAIN_CSV
        )
    )

    test_df = standardize_variant_df(
        read_table(
            TEST_CSV
        )
    )

    gene_partition = standardize_gene_partition(
        read_table(
            gene_partition_path
        )
    )

    print(
        f"\n[CHECK] Train rows: {len(train_df):,}"
    )
    print(
        f"[CHECK] Test rows : {len(test_df):,}"
    )

    print(
        "\n[CHECK] Train disease types:"
    )

    if "disease type" in train_df.columns:
        print(
            train_df[
                "disease type"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[CHECK] Test disease types:"
    )

    if "disease type" in test_df.columns:
        print(
            test_df[
                "disease type"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # -------------------------------------------------------------------------
    # Prepare panel data
    # -------------------------------------------------------------------------

    chrom_data = prepare_chromosome_data(
        train_df,
        test_df,
    )

    disease_data = prepare_disease_data(
        train_df,
        test_df,
    )

    gene_bin_data = prepare_gene_bin_data_from_partition(
        gene_partition
    )

    dataset_summary, overlap_n = prepare_dataset_summary(
        train_df,
        test_df,
        gene_partition,
    )

    # -------------------------------------------------------------------------
    # Gene-disjoint audit
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 88
    )
    print(
        "GENE-DISJOINT AUDIT"
    )
    print(
        "=" * 88
    )

    train_gene_n = int(
        (
            gene_partition[
                "Train_variant_count"
            ]
            > 0
        )
        .sum()
    )

    test_gene_n = int(
        (
            gene_partition[
                "Test_variant_count"
            ]
            > 0
        )
        .sum()
    )

    print(
        f"Train unique genes : {train_gene_n:,}"
    )
    print(
        f"Test unique genes  : {test_gene_n:,}"
    )
    print(
        f"Shared genes       : {overlap_n:,}"
    )

    if overlap_n != 0:
        raise RuntimeError(
            "Gene overlap is not zero."
        )

    print(
        "[PASS] Train/Test are gene-disjoint."
    )

    # -------------------------------------------------------------------------
    # Optional cross-checks
    # -------------------------------------------------------------------------

    try_crosscheck_gene_bins(
        gene_bins_path
    )

    try_crosscheck_disjoint_summary(
        gene_disjoint_path
    )

    # -------------------------------------------------------------------------
    # Save supporting statistics
    # -------------------------------------------------------------------------

    chrom_data.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig1a_chromosome_distribution.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    disease_data.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig1b_disease_distribution.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    gene_bin_data.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig1c_gene_variant_bins_recomputed.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    summary_export = (
        dataset_summary.copy()
    )

    summary_export[
        "Gene_overlap"
    ] = overlap_n

    summary_export.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig1d_dataset_summary.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Build Fig. 1
    # -------------------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            13.5,
            8.2,
        ),
        dpi=FIG_DPI,
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[
            1.0,
            1.0,
        ],
        height_ratios=[
            1.0,
            1.0,
        ],
        wspace=0.22,
        hspace=0.36,
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
    ax_d = fig.add_subplot(
        gs[1, 1]
    )

    draw_panel_a(
        ax_a,
        chrom_data,
        random_seed=42,
    )

    draw_panel_b(
        ax_b,
        disease_data,
    )

    draw_panel_c(
        ax_c,
        gene_bin_data,
    )

    draw_panel_d(
        ax_d,
        dataset_summary,
        overlap_n,
    )

    # Slightly tighter than the previous draft.
    fig.subplots_adjust(
        left=0.065,
        right=0.975,
        top=0.95,
        bottom=0.08,
        wspace=0.22,
        hspace=0.36,
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

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 88
    )
    print(
        "FIG. 1 FINAL STATISTICS"
    )
    print(
        "=" * 88
    )

    print(
        "\n[a] Overall chromosome distribution "
        "(Train + Test merged)"
    )
    print(
        chrom_data.to_string(
            index=False
        )
    )

    print(
        "\n[b] Disease distribution"
    )
    print(
        disease_data.to_string(
            index=False
        )
    )

    print(
        "\n[c] Gene-level binned distribution"
    )
    print(
        gene_bin_data.to_string(
            index=False
        )
    )

    print(
        "\n[d] Dataset summary"
    )
    print(
        dataset_summary.to_string(
            index=False
        )
    )

    print(
        f"\nGene overlap = {overlap_n}"
    )

    print(
        "\n"
        + "=" * 88
    )
    print(
        "DONE"
    )
    print(
        "=" * 88
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
