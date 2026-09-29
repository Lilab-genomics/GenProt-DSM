#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supplementary Fig. S4
Additional analyses of prediction characteristics and substitution-specific
GenProt-DSM scores.

Panels
------
(a) Spearman correlation matrix (pairwise overlap)
    - Pairwise-complete observations.
    - All tools are first aligned to the same pathogenicity direction:
        higher score = more pathogenic.
    - Lower triangle: colored bubbles, with bubble size proportional to |rho|.
    - Upper triangle: exact Spearman rho values.

(b) Prediction agreement matrix (pairwise overlap)
    - Pairwise-complete observations.
    - Each tool is binarized using its predefined score direction and
      fixed threshold from metrics_tools.csv.
    - Lower triangle: colored bubbles, with bubble size proportional to
      pairwise agreement.
    - Upper triangle: exact agreement values.

(c) Prediction score distribution (GenProt-DSM)
    - Histogram of final GenProt-DSM probabilities on VarGeneDisjointTest.
    - Bars are colored from low-score blue to high-score red.

(d) Substitution-specific score heatmap (GenProt-DSM)
    - For each of the 12 possible SNV substitutions, GenProt-DSM scores are
      sorted from low to high and displayed as a vertical score strip.
    - Different strip lengths reflect different numbers of variants per
      substitution class.
    - Missing/padded cells are white.

Required inputs
---------------
1) Final per-variant prediction table:
   F:\\GenProt-DSM_Resubmit\\dataset\\data\\test_score.csv

   Required columns:
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
       GenProt-DSM

2) Tool metric/threshold table:
   F:\\GenProt-DSM_Resubmit\\result\\metrics_tools.csv

   Required columns:
       tool_column
       score_direction
       fixed_threshold

   The table may contain multiple rows per tool (e.g., different evaluation
   subsets). Threshold/direction metadata are checked for consistency and a
   single tool-level definition is extracted.

Display terminology
-------------------
- CADD is displayed as "CADD v1.7".
- AUROC/AUPR terminology is not used directly in this figure.
- "pairwise overlap" means that each pair of tools is evaluated only on
  variants for which both tools have valid predictions.

Outputs
-------
F:\\GenProt-DSM_Resubmit\\fig_Results\\Supplementary_FigS4\\

    Supplementary_FigS4_additional_prediction_analyses.png
    Supplementary_FigS4_additional_prediction_analyses.pdf
    FigS4a_spearman_matrix.csv
    FigS4a_pairwise_n.csv
    FigS4b_agreement_matrix.csv
    FigS4b_pairwise_n.csv
    FigS4d_substitution_counts.csv
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

TEST_SCORE_CSV = (
    r"F:\GenProt-DSM_Resubmit\dataset\data"
    r"\test_score.csv"
)

METRICS_TOOLS_CSV = (
    r"F:\GenProt-DSM_Resubmit\result"
    r"\metrics_tools.csv"
)

OUTPUT_DIR = (
    r"F:\GenProt-DSM_Resubmit\fig_Results"
    r"\Supplementary_FigS4"
)

OUTPUT_BASENAME = (
    "Supplementary_FigS4_additional_prediction_analyses"
)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

FIG_DPI = 600
BASE_FONT_SIZE = 9

matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": BASE_FONT_SIZE,
    "axes.labelsize": BASE_FONT_SIZE,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# =============================================================================
# 3. TOOL CONFIGURATION
# =============================================================================

# Keep the old manuscript ordering and add ESM1b as the revised pLM baseline.
TOOL_COLUMNS = [
    "GenProt-DSM",
    "SIFT",
    "Polyphen2",
    "FATHMM",
    "PROVEAN",
    "MPC",
    "DEOGEN2",
    "ESM1b",
    "AlphaMissense",
    "CADD",
    "DANN",
    "GenoCanyon",
    "PrimateAI",
]

DISPLAY_NAMES = {
    "GenProt-DSM": "GenProt-DSM",
    "SIFT": "SIFT",
    "Polyphen2": "PolyPhen2",
    "FATHMM": "FATHMM",
    "PROVEAN": "PROVEAN",
    "MPC": "MPC",
    "DEOGEN2": "DEOGEN2",
    "ESM1b": "ESM1b",
    "AlphaMissense": "AlphaMissense",
    "CADD": "CADD v1.7",
    "DANN": "DANN",
    "GenoCanyon": "GenoCanyon",
    "PrimateAI": "PrimateAI",
}

# Fallback direction map is used ONLY if score_direction text in
# metrics_tools.csv cannot be parsed. Thresholds are never guessed.
FALLBACK_DIRECTION = {
    "GenProt-DSM": "higher",
    "SIFT": "lower",
    "Polyphen2": "higher",
    "FATHMM": "lower",
    "PROVEAN": "lower",
    "MPC": "higher",
    "DEOGEN2": "higher",
    "ESM1b": "lower",
    "AlphaMissense": "higher",
    "CADD": "higher",
    "DANN": "higher",
    "GenoCanyon": "higher",
    "PrimateAI": "higher",
}

EXPECTED_N = 786


# =============================================================================
# 4. BASIC UTILITIES
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.09,
        1.035,
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


# =============================================================================
# 5. READ FINAL TEST DATA
# =============================================================================

def load_test_score(path: str) -> pd.DataFrame:
    require_file(path)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    required = [
        "Reference",
        "Alternate",
        *TOOL_COLUMNS,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            "test_score.csv is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + ", ".join(map(str, df.columns))
        )

    if len(df) != EXPECTED_N:
        print(
            f"[WARN] Expected VarGeneDisjointTest N={EXPECTED_N}, "
            f"but test_score.csv has N={len(df)}."
        )

    for col in TOOL_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["Reference"] = (
        df["Reference"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Alternate"] = (
        df["Alternate"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return df


# =============================================================================
# 6. READ PREDEFINED THRESHOLDS / SCORE DIRECTIONS
# =============================================================================

def normalize_direction(value, tool: str) -> str:
    """
    Convert common direction strings into:
        "higher" : larger score means more pathogenic
        "lower"  : smaller score means more pathogenic
    """
    if pd.isna(value):
        return FALLBACK_DIRECTION[tool]

    s = str(value).strip().lower()

    higher_tokens = [
        "higher",
        "high",
        "greater",
        "larger",
        "increase",
        "positive",
        "pathogenic_high",
        "higher_is_pathogenic",
        "higher is pathogenic",
        "1",
    ]

    lower_tokens = [
        "lower",
        "low",
        "less",
        "smaller",
        "decrease",
        "negative",
        "pathogenic_low",
        "lower_is_pathogenic",
        "lower is pathogenic",
        "-1",
    ]

    if any(token in s for token in higher_tokens):
        return "higher"

    if any(token in s for token in lower_tokens):
        return "lower"

    print(
        f"[WARN] Could not parse score_direction={value!r} for {tool}; "
        f"using predefined fallback direction={FALLBACK_DIRECTION[tool]!r}."
    )

    return FALLBACK_DIRECTION[tool]


def extract_tool_metadata(
    metrics_path: str,
) -> Dict[str, Dict[str, float | str]]:

    require_file(metrics_path)

    metrics = pd.read_csv(
        metrics_path,
        low_memory=False,
    )

    metrics.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in metrics.columns
    ]

    required = [
        "tool_column",
        "score_direction",
        "fixed_threshold",
    ]

    missing = [
        c for c in required
        if c not in metrics.columns
    ]

    if missing:
        raise KeyError(
            "metrics_tools.csv is missing required columns:\n"
            + "\n".join(missing)
        )

    metrics["tool_column"] = (
        metrics["tool_column"]
        .astype(str)
        .str.strip()
    )

    metadata = {}

    for tool in TOOL_COLUMNS:
        sub = metrics.loc[
            metrics["tool_column"] == tool
        ].copy()

        if sub.empty:
            raise RuntimeError(
                f"No metadata row found for tool_column={tool!r} "
                f"in {metrics_path}"
            )

        # Direction: tolerate duplicate rows, but ensure normalized directions
        # do not conflict.
        dirs = [
            normalize_direction(v, tool)
            for v in sub["score_direction"].tolist()
        ]

        dirs_unique = sorted(
            set(dirs)
        )

        if len(dirs_unique) != 1:
            raise RuntimeError(
                f"Conflicting score directions for {tool}: {dirs_unique}"
            )

        direction = dirs_unique[0]

        # Threshold: require at least one numeric fixed threshold.
        thresholds = (
            pd.to_numeric(
                sub["fixed_threshold"],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(thresholds) == 0:
            raise RuntimeError(
                f"No numeric fixed_threshold found for {tool}."
            )

        # Multiple subset rows are allowed only if the threshold is identical
        # apart from numerical precision.
        first = float(thresholds[0])

        if not np.allclose(
            thresholds,
            first,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Conflicting fixed thresholds for {tool}: "
                f"{thresholds.tolist()}"
            )

        metadata[tool] = {
            "direction": direction,
            "threshold": first,
        }

    return metadata


# =============================================================================
# 7. PATHOGENICITY-DIRECTION ALIGNMENT
# =============================================================================

def aligned_continuous_scores(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, float | str]],
) -> pd.DataFrame:
    """
    For Spearman correlation:
    - higher-is-pathogenic: keep score
    - lower-is-pathogenic : multiply score by -1

    Spearman correlation is rank based, so sign reversal is sufficient and
    does not require arbitrary min-max normalization.
    """
    out = pd.DataFrame(
        index=df.index
    )

    for tool in TOOL_COLUMNS:
        score = pd.to_numeric(
            df[tool],
            errors="coerce",
        )

        direction = metadata[tool]["direction"]

        if direction == "higher":
            out[tool] = score
        elif direction == "lower":
            out[tool] = -score
        else:
            raise RuntimeError(
                f"Unexpected direction={direction!r} for {tool}"
            )

    return out


def binary_predictions(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, float | str]],
) -> pd.DataFrame:
    """
    Convert each tool's continuous score to a pathogenic/benign call using
    its predefined fixed threshold and score direction.

    Missing continuous predictions remain missing.
    """
    out = pd.DataFrame(
        index=df.index
    )

    for tool in TOOL_COLUMNS:
        score = pd.to_numeric(
            df[tool],
            errors="coerce",
        )

        threshold = float(
            metadata[tool]["threshold"]
        )

        direction = metadata[tool]["direction"]

        pred = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        valid = score.notna()

        if direction == "higher":
            pred.loc[valid] = (
                score.loc[valid] >= threshold
            ).astype(int)

        elif direction == "lower":
            pred.loc[valid] = (
                score.loc[valid] <= threshold
            ).astype(int)

        else:
            raise RuntimeError(
                f"Unexpected direction={direction!r} for {tool}"
            )

        out[tool] = pred

    return out


# =============================================================================
# 8. PAIRWISE MATRICES
# =============================================================================

def pairwise_spearman(
    score_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    n_tools = len(TOOL_COLUMNS)

    corr = pd.DataFrame(
        np.eye(n_tools),
        index=TOOL_COLUMNS,
        columns=TOOL_COLUMNS,
        dtype=float,
    )

    nmat = pd.DataFrame(
        np.zeros((n_tools, n_tools), dtype=int),
        index=TOOL_COLUMNS,
        columns=TOOL_COLUMNS,
    )

    for i, tool_i in enumerate(TOOL_COLUMNS):
        for j, tool_j in enumerate(TOOL_COLUMNS):
            pair = score_df[
                [tool_i, tool_j]
            ].dropna()

            n = len(pair)
            nmat.loc[tool_i, tool_j] = n

            if i == j:
                corr.loc[tool_i, tool_j] = 1.0
                continue

            if n < 3:
                corr.loc[tool_i, tool_j] = np.nan
                continue

            rho = pair[
                [tool_i, tool_j]
            ].corr(
                method="spearman"
            ).iloc[0, 1]

            corr.loc[
                tool_i,
                tool_j,
            ] = float(rho)

    return corr, nmat


def pairwise_agreement(
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    n_tools = len(TOOL_COLUMNS)

    agree = pd.DataFrame(
        np.eye(n_tools),
        index=TOOL_COLUMNS,
        columns=TOOL_COLUMNS,
        dtype=float,
    )

    nmat = pd.DataFrame(
        np.zeros((n_tools, n_tools), dtype=int),
        index=TOOL_COLUMNS,
        columns=TOOL_COLUMNS,
    )

    for i, tool_i in enumerate(TOOL_COLUMNS):
        for j, tool_j in enumerate(TOOL_COLUMNS):
            pair = pred_df[
                [tool_i, tool_j]
            ].dropna()

            n = len(pair)
            nmat.loc[
                tool_i,
                tool_j,
            ] = n

            if i == j:
                agree.loc[
                    tool_i,
                    tool_j,
                ] = 1.0
                continue

            if n == 0:
                agree.loc[
                    tool_i,
                    tool_j,
                ] = np.nan
                continue

            value = np.mean(
                pair[tool_i].to_numpy(dtype=int)
                ==
                pair[tool_j].to_numpy(dtype=int)
            )

            agree.loc[
                tool_i,
                tool_j,
            ] = float(value)

    return agree, nmat


# =============================================================================
# 9. TRIANGULAR BUBBLE MATRIX PLOT
# =============================================================================

def draw_triangular_matrix(
    ax,
    matrix: pd.DataFrame,
    *,
    title: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    cmap_name: str,
    value_format: str,
    bubble_mode: str,
):
    """
    Reproduce the old plotting logic:
    - lower triangle = colored circles
    - upper triangle = exact text values
    - diagonal omitted

    bubble_mode:
        "abs"   -> size proportional to absolute value (Spearman)
        "value" -> size proportional to value (agreement)
    """

    values = matrix.loc[
        TOOL_COLUMNS,
        TOOL_COLUMNS,
    ].to_numpy(dtype=float)

    n = len(TOOL_COLUMNS)

    cmap = plt.get_cmap(
        cmap_name
    )

    norm = Normalize(
        vmin=vmin,
        vmax=vmax,
    )

    # Fixed matrix coordinates.
    ax.set_xlim(
        -0.5,
        n - 0.5,
    )

    ax.set_ylim(
        n - 0.5,
        -0.5,
    )

    # Draw lower triangle bubbles and upper triangle text.
    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            val = values[i, j]

            if not np.isfinite(val):
                continue

            if i > j:
                if bubble_mode == "abs":
                    size_fraction = min(
                        1.0,
                        abs(val),
                    )
                elif bubble_mode == "value":
                    if vmax == vmin:
                        size_fraction = 1.0
                    else:
                        size_fraction = (
                            (val - vmin)
                            / (vmax - vmin)
                        )
                    size_fraction = np.clip(
                        size_fraction,
                        0.0,
                        1.0,
                    )
                else:
                    raise ValueError(
                        bubble_mode
                    )

                # Similar visual scale to the old figure.
                marker_size = (
                    38
                    + 280 * size_fraction
                )

                ax.scatter(
                    j,
                    i,
                    s=marker_size,
                    c=[cmap(norm(val))],
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=3,
                )

            elif i < j:
                ax.text(
                    j,
                    i,
                    format(
                        val,
                        value_format,
                    ),
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color="black",
                )

    labels = [
        DISPLAY_NAMES[c]
        for c in TOOL_COLUMNS
    ]

    ax.set_xticks(
        np.arange(n)
    )

    ax.set_yticks(
        np.arange(n)
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_yticklabels(
        labels,
    )

    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        pad=2,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(
            0.7
        )

    ax.set_title(
        title,
        pad=7,
    )

    # Colorbar tied to the matrix axis.
    sm = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    sm.set_array([])

    cbar = ax.figure.colorbar(
        sm,
        ax=ax,
        fraction=0.045,
        pad=0.035,
    )

    cbar.set_label(
        colorbar_label,
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7.5
    )


# =============================================================================
# 10. PANEL C — GENPROT-DSM SCORE HISTOGRAM
# =============================================================================

def draw_panel_c(
    ax,
    df: pd.DataFrame,
) -> None:

    score = (
        pd.to_numeric(
            df["GenProt-DSM"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    bins = np.linspace(
        0.0,
        1.0,
        21,
    )

    counts, edges = np.histogram(
        score,
        bins=bins,
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    widths = np.diff(
        edges
    )

    cmap = plt.get_cmap(
        "coolwarm"
    )

    norm = Normalize(
        vmin=0.0,
        vmax=1.0,
    )

    colors = [
        cmap(norm(c))
        for c in centers
    ]

    ax.bar(
        edges[:-1],
        counts,
        width=widths,
        align="edge",
        color=colors,
        edgecolor="white",
        linewidth=0.55,
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        "Prediction score"
    )

    ax.set_ylabel(
        "Variant count"
    )

    ax.set_title(
        "Prediction score distribution (GenProt-DSM)"
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.55,
        alpha=0.6,
    )

    ax.set_axisbelow(
        True
    )

    panel_label(
        ax,
        "c",
    )


# =============================================================================
# 11. PANEL D — SUBSTITUTION-SPECIFIC SCORE HEATMAP
# =============================================================================

SUBSTITUTIONS = [
    "A→C",
    "A→G",
    "A→T",
    "C→A",
    "C→G",
    "C→T",
    "G→A",
    "G→C",
    "G→T",
    "T→A",
    "T→C",
    "T→G",
]


def substitution_label(
    ref: str,
    alt: str,
) -> str:
    return f"{ref}→{alt}"


def make_substitution_score_matrix(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:

    work = df[
        [
            "Reference",
            "Alternate",
            "GenProt-DSM",
        ]
    ].copy()

    work["GenProt-DSM"] = pd.to_numeric(
        work["GenProt-DSM"],
        errors="coerce",
    )

    work = work.dropna(
        subset=[
            "Reference",
            "Alternate",
            "GenProt-DSM",
        ]
    ).copy()

    valid_bases = {
        "A",
        "C",
        "G",
        "T",
    }

    work = work.loc[
        work["Reference"].isin(valid_bases)
        &
        work["Alternate"].isin(valid_bases)
        &
        (work["Reference"] != work["Alternate"])
    ].copy()

    work["substitution"] = [
        substitution_label(r, a)
        for r, a in zip(
            work["Reference"],
            work["Alternate"],
        )
    ]

    scores_by_sub = {}

    count_rows = []

    for sub in SUBSTITUTIONS:
        arr = (
            work.loc[
                work["substitution"] == sub,
                "GenProt-DSM",
            ]
            .sort_values(
                ascending=True
            )
            .to_numpy(dtype=float)
        )

        scores_by_sub[sub] = arr

        count_rows.append({
            "Substitution": sub,
            "N": int(len(arr)),
            "Mean_score": (
                float(np.mean(arr))
                if len(arr)
                else np.nan
            ),
            "Median_score": (
                float(np.median(arr))
                if len(arr)
                else np.nan
            ),
        })

    max_n = max(
        [
            len(v)
            for v in scores_by_sub.values()
        ]
        + [1]
    )

    matrix = np.full(
        (
            max_n,
            len(SUBSTITUTIONS),
        ),
        np.nan,
        dtype=float,
    )

    for j, sub in enumerate(
        SUBSTITUTIONS
    ):
        arr = scores_by_sub[
            sub
        ]

        if len(arr):
            matrix[
                :len(arr),
                j,
            ] = arr

    counts_df = pd.DataFrame(
        count_rows
    )

    return matrix, counts_df


def draw_panel_d(
    ax,
    matrix: np.ndarray,
) -> None:

    cmap = plt.get_cmap(
        "viridis"
    ).copy()

    cmap.set_bad(
        "white"
    )

    masked = np.ma.masked_invalid(
        matrix
    )

    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(
        np.arange(
            len(SUBSTITUTIONS)
        )
    )

    ax.set_xticklabels(
        SUBSTITUTIONS,
        rotation=55,
        ha="right",
        rotation_mode="anchor",
    )

    # Rows are sorted-score positions, not a biological quantity, so the
    # y-axis is intentionally unlabeled to retain the old visual logic.
    ax.set_yticks([])

    ax.set_xlabel(
        "Base substitution"
    )

    ax.set_title(
        "Substitution-specific score heatmap (GenProt-DSM)"
    )

    for spine in ax.spines.values():
        spine.set_linewidth(
            0.7
        )

    cbar = ax.figure.colorbar(
        im,
        ax=ax,
        fraction=0.045,
        pad=0.035,
    )

    cbar.set_label(
        "Prediction score",
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7.5
    )

    panel_label(
        ax,
        "d",
    )


# =============================================================================
# 12. MAIN
# =============================================================================

def main() -> None:
    ensure_dir(
        OUTPUT_DIR
    )

    print("=" * 96)
    print("SUPPLEMENTARY FIG. S4")
    print("Additional prediction analyses")
    print("=" * 96)

    print(
        f"[INPUT] {TEST_SCORE_CSV}"
    )

    print(
        f"[INPUT] {METRICS_TOOLS_CSV}"
    )

    df = load_test_score(
        TEST_SCORE_CSV
    )

    metadata = extract_tool_metadata(
        METRICS_TOOLS_CSV
    )

    print(
        f"[CHECK] Test rows = {len(df):,}"
    )

    print("\n[Tool metadata]")
    for tool in TOOL_COLUMNS:
        print(
            f"  {DISPLAY_NAMES[tool]:16s} "
            f"direction={metadata[tool]['direction']:6s} "
            f"threshold={float(metadata[tool]['threshold']):.8g}"
        )

    # -------------------------------------------------------------------------
    # Panel a
    # -------------------------------------------------------------------------

    aligned = aligned_continuous_scores(
        df,
        metadata,
    )

    spearman_df, spearman_n_df = pairwise_spearman(
        aligned
    )

    # -------------------------------------------------------------------------
    # Panel b
    # -------------------------------------------------------------------------

    pred_binary = binary_predictions(
        df,
        metadata,
    )

    agreement_df, agreement_n_df = pairwise_agreement(
        pred_binary
    )

    # -------------------------------------------------------------------------
    # Panel d
    # -------------------------------------------------------------------------

    substitution_matrix, substitution_counts_df = (
        make_substitution_score_matrix(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Save numerical outputs
    # -------------------------------------------------------------------------

    spearman_out = spearman_df.rename(
        index=DISPLAY_NAMES,
        columns=DISPLAY_NAMES,
    )

    spearman_n_out = spearman_n_df.rename(
        index=DISPLAY_NAMES,
        columns=DISPLAY_NAMES,
    )

    agreement_out = agreement_df.rename(
        index=DISPLAY_NAMES,
        columns=DISPLAY_NAMES,
    )

    agreement_n_out = agreement_n_df.rename(
        index=DISPLAY_NAMES,
        columns=DISPLAY_NAMES,
    )

    output_files = {
        "FigS4a_spearman_matrix.csv": spearman_out,
        "FigS4a_pairwise_n.csv": spearman_n_out,
        "FigS4b_agreement_matrix.csv": agreement_out,
        "FigS4b_pairwise_n.csv": agreement_n_out,
        "FigS4d_substitution_counts.csv": substitution_counts_df,
    }

    for filename, table in output_files.items():
        path = os.path.join(
            OUTPUT_DIR,
            filename,
        )

        table.to_csv(
            path,
            encoding="utf-8-sig",
        )

        print(
            f"[SAVE] {path}"
        )

    # -------------------------------------------------------------------------
    # Plot 2 x 2
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.2, 10.0),
        dpi=FIG_DPI,
    )

    # Panel a
    draw_triangular_matrix(
        axes[0, 0],
        spearman_df,
        title="Spearman correlation (pairwise overlap)",
        colorbar_label="Spearman ρ",
        vmin=-1.0,
        vmax=1.0,
        cmap_name="RdBu_r",
        value_format=".2f",
        bubble_mode="abs",
    )

    panel_label(
        axes[0, 0],
        "a",
    )

    # Panel b
    draw_triangular_matrix(
        axes[0, 1],
        agreement_df,
        title="Prediction agreement (pairwise overlap)",
        colorbar_label="Agreement",
        vmin=0.0,
        vmax=1.0,
        cmap_name="RdBu_r",
        value_format=".2f",
        bubble_mode="value",
    )

    panel_label(
        axes[0, 1],
        "b",
    )

    # Panel c
    draw_panel_c(
        axes[1, 0],
        df,
    )

    # Panel d
    draw_panel_d(
        axes[1, 1],
        substitution_matrix,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        top=0.95,
        bottom=0.10,
        wspace=0.30,
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

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)
    print("PNG:", output_base + ".png")
    print("PDF:", output_base + ".pdf")


if __name__ == "__main__":
    main()
