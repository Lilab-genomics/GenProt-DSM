#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final rare-variant evaluation for Supplementary Table S9.

Rare-variant definition:
    AF < 0.01
Missing AF values are excluded.

Methodological rules:
1) No separate VarRareTest.csv is required.
2) No threshold is optimized on the rare subset.
3) Each predictor uses the predefined score direction and fixed threshold
   already used in the main benchmark (read from metrics_tools.csv).
4) AUROC/AUPR use continuous scores.
5) AUPR uses sklearn.average_precision_score.
6) MCC uses the predictor-specific predefined fixed threshold.
7) 95% CIs use 1,000 stratified bootstrap replicates.
8) Predictors with missing scores are evaluated on their available rare variants.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

warnings.filterwarnings("once")


# =============================================================================
# 1. PATHS
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
    r"F:\GenProt-DSM_Resubmit\result"
    r"\rare_variant_evaluation"
)

DETAILED_OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "rare_variant_metrics_detailed.csv",
)

PAPER_OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "Supplementary_Table_S9_ready.csv",
)

RARE_AUDIT_CSV = os.path.join(
    OUTPUT_DIR,
    "rare_variant_subset_audit.csv",
)


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

AF_COL = "AF"
LABEL_COL = "label"

RARE_AF_THRESHOLD = 0.01

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 2026
CI_ALPHA = 0.95

# QC only: warnings, not hard failures
EXPECTED_TEST_N = 786
EXPECTED_RARE_POS = 161
EXPECTED_RARE_NEG = 13

TOOL_COLUMNS = [
    "GenProt-DSM",
    "ESM1b",
    "PrimateAI",
    "SIFT",
    "Polyphen2",
    "FATHMM",
    "PROVEAN",
    "MPC",
    "DEOGEN2",
    "AlphaMissense",
    "CADD",
    "DANN",
    "GenoCanyon",
]

DISPLAY_NAMES = {
    "GenProt-DSM": "GenProt-DSM",
    "ESM1b": "ESM1b",
    "PrimateAI": "PrimateAI",
    "SIFT": "SIFT",
    "Polyphen2": "PolyPhen2",
    "FATHMM": "FATHMM",
    "PROVEAN": "PROVEAN",
    "MPC": "MPC",
    "DEOGEN2": "DEOGEN2",
    "AlphaMissense": "AlphaMissense",
    "CADD": "CADD v1.7",
    "DANN": "DANN",
    "GenoCanyon": "GenoCanyon",
}

FALLBACK_DIRECTION = {
    "GenProt-DSM": "higher",
    "ESM1b": "lower",
    "PrimateAI": "higher",
    "SIFT": "lower",
    "Polyphen2": "higher",
    "FATHMM": "lower",
    "PROVEAN": "lower",
    "MPC": "higher",
    "DEOGEN2": "higher",
    "AlphaMissense": "higher",
    "CADD": "higher",
    "DANN": "higher",
    "GenoCanyon": "higher",
}


# =============================================================================
# 3. HELPERS
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


def read_csv_clean(path: str) -> pd.DataFrame:
    require_file(path)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df.columns = [
        str(c)
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .strip()
        for c in df.columns
    ]

    return df


def normalize_direction(value, tool: str) -> str:
    if pd.isna(value):
        print(
            f"[WARN] {tool}: score_direction missing; "
            f"using fallback={FALLBACK_DIRECTION[tool]}"
        )
        return FALLBACK_DIRECTION[tool]

    s = str(value).strip().lower()

    higher_patterns = [
        "higher",
        "greater",
        "larger",
        "higher_is_pathogenic",
        "higher is pathogenic",
        "pathogenic_high",
    ]

    lower_patterns = [
        "lower",
        "less",
        "smaller",
        "lower_is_pathogenic",
        "lower is pathogenic",
        "pathogenic_low",
    ]

    if any(x in s for x in higher_patterns):
        return "higher"

    if any(x in s for x in lower_patterns):
        return "lower"

    if s in {"1", "+1"}:
        return "higher"

    if s == "-1":
        return "lower"

    print(
        f"[WARN] {tool}: could not parse score_direction={value!r}; "
        f"using fallback={FALLBACK_DIRECTION[tool]}"
    )

    return FALLBACK_DIRECTION[tool]


def extract_tool_metadata(
    metrics_df: pd.DataFrame,
) -> Dict[str, Dict[str, float | str]]:

    required = [
        "tool_column",
        "score_direction",
        "fixed_threshold",
    ]

    missing = [
        c for c in required
        if c not in metrics_df.columns
    ]

    if missing:
        raise KeyError(
            "metrics_tools.csv is missing required columns:\n"
            + "\n".join(missing)
        )

    tmp = metrics_df.copy()

    tmp["tool_column"] = (
        tmp["tool_column"]
        .astype(str)
        .str.strip()
    )

    metadata: Dict[str, Dict[str, float | str]] = {}

    for tool in TOOL_COLUMNS:

        sub = tmp.loc[
            tmp["tool_column"] == tool
        ].copy()

        if sub.empty:
            raise RuntimeError(
                f"No metadata row found for tool_column={tool!r} "
                f"in {METRICS_TOOLS_CSV}"
            )

        directions = [
            normalize_direction(v, tool)
            for v in sub["score_direction"].tolist()
        ]

        unique_directions = sorted(
            set(directions)
        )

        if len(unique_directions) != 1:
            raise RuntimeError(
                f"Conflicting score directions for {tool}: "
                f"{unique_directions}"
            )

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
                f"No numeric fixed_threshold found for {tool}. "
                "The script will NOT optimize a threshold on the rare subset."
            )

        threshold0 = float(
            thresholds[0]
        )

        if not np.allclose(
            thresholds,
            threshold0,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Conflicting fixed thresholds for {tool}: "
                f"{thresholds.tolist()}"
            )

        metadata[tool] = {
            "direction": unique_directions[0],
            "threshold": threshold0,
        }

    return metadata


# =============================================================================
# 4. RARE SUBSET
# =============================================================================

def make_rare_subset(
    test_score_df: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        AF_COL,
        LABEL_COL,
        *TOOL_COLUMNS,
    ]

    missing = [
        c for c in required
        if c not in test_score_df.columns
    ]

    if missing:
        raise KeyError(
            "test_score.csv is missing required columns:\n"
            + "\n".join(missing)
        )

    work = test_score_df.copy()

    work[AF_COL] = pd.to_numeric(
        work[AF_COL],
        errors="coerce",
    )

    work[LABEL_COL] = pd.to_numeric(
        work[LABEL_COL],
        errors="coerce",
    )

    labels = set(
        work[LABEL_COL]
        .dropna()
        .astype(int)
        .unique()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{LABEL_COL} must contain only 0/1. Observed={labels}"
        )

    n_total = len(work)
    n_af_missing = int(
        work[AF_COL]
        .isna()
        .sum()
    )

    rare = work.loc[
        work[AF_COL].notna()
        &
        (work[AF_COL] < RARE_AF_THRESHOLD)
    ].copy()

    rare[LABEL_COL] = (
        rare[LABEL_COL]
        .astype(int)
    )

    n_rare = len(rare)
    n_pos = int(
        (rare[LABEL_COL] == 1)
        .sum()
    )
    n_neg = int(
        (rare[LABEL_COL] == 0)
        .sum()
    )

    print("\n" + "=" * 96)
    print("RARE-VARIANT SUBSET")
    print("=" * 96)
    print(f"Final test rows      : {n_total:,}")
    print(f"Rows with missing AF : {n_af_missing:,}")
    print(f"Rare definition      : AF < {RARE_AF_THRESHOLD}")
    print(f"Rare subset N        : {n_rare:,}")
    print(f"Rare pathogenic      : {n_pos:,}")
    print(f"Rare benign          : {n_neg:,}")

    if n_total != EXPECTED_TEST_N:
        print(
            f"[WARN] Expected final test N={EXPECTED_TEST_N}, "
            f"but observed N={n_total}."
        )

    if (
        n_pos != EXPECTED_RARE_POS
        or
        n_neg != EXPECTED_RARE_NEG
    ):
        print(
            "[INFO] Rare-subset counts differ from the previous "
            f"{EXPECTED_RARE_POS}/{EXPECTED_RARE_NEG} reference. "
            "Use the current final-dataset counts."
        )

    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            "Rare subset must contain both pathogenic and benign variants."
        )

    return rare


# =============================================================================
# 5. METRICS
# =============================================================================

def prepare_tool_arrays(
    rare_df: pd.DataFrame,
    tool: str,
    metadata: Dict[str, Dict[str, float | str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    raw = pd.to_numeric(
        rare_df[tool],
        errors="coerce",
    )

    y = pd.to_numeric(
        rare_df[LABEL_COL],
        errors="coerce",
    )

    mask = (
        raw.notna()
        &
        y.notna()
        &
        np.isfinite(
            raw.to_numpy(dtype=float)
        )
    )

    raw = raw.loc[
        mask
    ].to_numpy(
        dtype=float
    )

    y = (
        y.loc[
            mask
        ]
        .astype(int)
        .to_numpy()
    )

    direction = str(
        metadata[tool]["direction"]
    )

    threshold = float(
        metadata[tool]["threshold"]
    )

    if direction == "higher":

        ranking_score = raw

        pred = (
            raw >= threshold
        ).astype(int)

    elif direction == "lower":

        ranking_score = -raw

        pred = (
            raw <= threshold
        ).astype(int)

    else:
        raise RuntimeError(
            f"Unexpected direction={direction!r} for {tool}"
        )

    return (
        y,
        ranking_score,
        pred,
    )


def compute_metrics(
    y_true: np.ndarray,
    ranking_score: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "AUROC": np.nan,
            "AUPR": np.nan,
            "MCC": np.nan,
        }

    return {
        "AUROC": float(
            roc_auc_score(
                y_true,
                ranking_score,
            )
        ),

        "AUPR": float(
            average_precision_score(
                y_true,
                ranking_score,
            )
        ),

        "MCC": float(
            matthews_corrcoef(
                y_true,
                pred,
            )
        ),
    }


# =============================================================================
# 6. STRATIFIED BOOTSTRAP
# =============================================================================

def stratified_bootstrap(
    y_true: np.ndarray,
    ranking_score: np.ndarray,
    pred: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    ranking_score = np.asarray(
        ranking_score,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=int,
    )

    pos_idx = np.where(
        y_true == 1
    )[0]

    neg_idx = np.where(
        y_true == 0
    )[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(
            "Both classes are required for bootstrap."
        )

    rng = np.random.default_rng(
        seed
    )

    records = []

    for _ in range(
        n_bootstrap
    ):

        boot_pos = rng.choice(
            pos_idx,
            size=len(pos_idx),
            replace=True,
        )

        boot_neg = rng.choice(
            neg_idx,
            size=len(neg_idx),
            replace=True,
        )

        idx = np.concatenate(
            [
                boot_pos,
                boot_neg,
            ]
        )

        m = compute_metrics(
            y_true[idx],
            ranking_score[idx],
            pred[idx],
        )

        records.append(
            m
        )

    return pd.DataFrame(
        records
    )


def percentile_ci(
    values,
    alpha: float = 0.95,
) -> Tuple[float, float]:

    arr = (
        pd.to_numeric(
            pd.Series(values),
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(arr) == 0:
        return np.nan, np.nan

    tail = (
        1.0
        - alpha
    ) / 2.0

    low = float(
        np.quantile(
            arr,
            tail,
        )
    )

    high = float(
        np.quantile(
            arr,
            1.0 - tail,
        )
    )

    return low, high


# =============================================================================
# 7. EVALUATE ALL TOOLS
# =============================================================================

def evaluate_all_tools(
    rare_df: pd.DataFrame,
    metadata: Dict[str, Dict[str, float | str]],
) -> pd.DataFrame:

    rows = []

    print("\n" + "=" * 96)
    print("RARE-VARIANT PERFORMANCE")
    print("=" * 96)

    for tool_index, tool in enumerate(
        TOOL_COLUMNS
    ):

        (
            y,
            score,
            pred,
        ) = prepare_tool_arrays(
            rare_df,
            tool,
            metadata,
        )

        n_used = int(
            len(y)
        )

        n_pos = int(
            (y == 1).sum()
        )

        n_neg = int(
            (y == 0).sum()
        )

        point = compute_metrics(
            y,
            score,
            pred,
        )

        if n_pos > 0 and n_neg > 0:

            boot = stratified_bootstrap(
                y,
                score,
                pred,
                n_bootstrap=N_BOOTSTRAP,
                seed=(
                    BOOTSTRAP_SEED
                    + tool_index * 1000
                ),
            )

            auroc_lo, auroc_hi = percentile_ci(
                boot["AUROC"],
                alpha=CI_ALPHA,
            )

            aupr_lo, aupr_hi = percentile_ci(
                boot["AUPR"],
                alpha=CI_ALPHA,
            )

            mcc_lo, mcc_hi = percentile_ci(
                boot["MCC"],
                alpha=CI_ALPHA,
            )

        else:

            auroc_lo = auroc_hi = np.nan
            aupr_lo = aupr_hi = np.nan
            mcc_lo = mcc_hi = np.nan

        rows.append(
            {
                "Model":
                    DISPLAY_NAMES[tool],

                "tool_column":
                    tool,

                "N_used":
                    n_used,

                "Pathogenic_N":
                    n_pos,

                "Benign_N":
                    n_neg,

                "Missing_in_rare_subset":
                    int(
                        len(rare_df)
                        - n_used
                    ),

                "Score_direction":
                    metadata[tool][
                        "direction"
                    ],

                "Fixed_threshold":
                    float(
                        metadata[tool][
                            "threshold"
                        ]
                    ),

                "AUROC":
                    point[
                        "AUROC"
                    ],

                "AUROC_CI_lower":
                    auroc_lo,

                "AUROC_CI_upper":
                    auroc_hi,

                "AUPR":
                    point[
                        "AUPR"
                    ],

                "AUPR_CI_lower":
                    aupr_lo,

                "AUPR_CI_upper":
                    aupr_hi,

                "MCC":
                    point[
                        "MCC"
                    ],

                "MCC_CI_lower":
                    mcc_lo,

                "MCC_CI_upper":
                    mcc_hi,

                "Bootstrap_replicates":
                    N_BOOTSTRAP,
            }
        )

        print(
            f"{DISPLAY_NAMES[tool]:16s} "
            f"N={n_used:3d} "
            f"(P={n_pos:3d}, B={n_neg:3d}) | "
            f"AUROC={point['AUROC']:.4f} "
            f"({auroc_lo:.4f}–{auroc_hi:.4f}) | "
            f"AUPR={point['AUPR']:.4f} | "
            f"MCC={point['MCC']:.4f} "
            f"({mcc_lo:.4f}–{mcc_hi:.4f})"
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 8. PAPER TABLE
# =============================================================================

def fmt_estimate_ci(
    value,
    low,
    high,
) -> str:

    if (
        pd.isna(value)
        or
        pd.isna(low)
        or
        pd.isna(high)
    ):
        return "NA"

    return (
        f"{float(value):.4f} "
        f"({float(low):.4f}–{float(high):.4f})"
    )


def build_paper_table(
    detailed_df: pd.DataFrame,
) -> pd.DataFrame:

    out = pd.DataFrame()

    out["Model"] = (
        detailed_df[
            "Model"
        ]
    )

    out["AUROC (95% CI)"] = (
        detailed_df.apply(
            lambda r: fmt_estimate_ci(
                r["AUROC"],
                r["AUROC_CI_lower"],
                r["AUROC_CI_upper"],
            ),
            axis=1,
        )
    )

    out["AUPR"] = (
        detailed_df[
            "AUPR"
        ]
        .map(
            lambda x:
                "NA"
                if pd.isna(x)
                else f"{float(x):.4f}"
        )
    )

    out["MCC (95% CI)"] = (
        detailed_df.apply(
            lambda r: fmt_estimate_ci(
                r["MCC"],
                r["MCC_CI_lower"],
                r["MCC_CI_upper"],
            ),
            axis=1,
        )
    )

    return out


# =============================================================================
# 9. AUDIT
# =============================================================================

def save_rare_subset_audit(
    rare_df: pd.DataFrame,
) -> None:

    preferred = [
        "Chrom",
        "Position",
        "Reference",
        "Alternate",
        AF_COL,
        LABEL_COL,
    ]

    cols = [
        c
        for c in preferred
        if c in rare_df.columns
    ]

    audit = rare_df[
        cols
    ].copy()

    audit.to_csv(
        RARE_AUDIT_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6g",
    )


# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> None:

    ensure_dir(
        OUTPUT_DIR
    )

    print("=" * 96)
    print("FINAL RARE-VARIANT EVALUATION FOR SUPPLEMENTARY TABLE S9")
    print("=" * 96)

    print(
        f"[INPUT] {TEST_SCORE_CSV}"
    )

    print(
        f"[INPUT] {METRICS_TOOLS_CSV}"
    )

    test_score = read_csv_clean(
        TEST_SCORE_CSV
    )

    metrics_tools = read_csv_clean(
        METRICS_TOOLS_CSV
    )

    metadata = extract_tool_metadata(
        metrics_tools
    )

    print("\n[FIXED TOOL METADATA]")

    for tool in TOOL_COLUMNS:

        print(
            f"  {DISPLAY_NAMES[tool]:16s} "
            f"direction={str(metadata[tool]['direction']):6s} | "
            f"threshold={float(metadata[tool]['threshold']):.10g}"
        )

    rare_df = make_rare_subset(
        test_score
    )

    save_rare_subset_audit(
        rare_df
    )

    print(
        f"[SAVE] {RARE_AUDIT_CSV}"
    )

    detailed = evaluate_all_tools(
        rare_df,
        metadata,
    )

    detailed.to_csv(
        DETAILED_OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )

    print(
        f"\n[SAVE] {DETAILED_OUTPUT_CSV}"
    )

    paper = build_paper_table(
        detailed
    )

    paper.to_csv(
        PAPER_OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[SAVE] {PAPER_OUTPUT_CSV}"
    )

    print("\n" + "=" * 96)
    print("SUPPLEMENTARY TABLE S9 READY")
    print("=" * 96)

    print(
        paper.to_string(
            index=False
        )
    )

    print("\nMethod summary:")
    print(
        "- Rare variants: AF < 0.01; missing AF excluded."
    )
    print(
        "- No threshold is optimized on the rare subset."
    )
    print(
        "- MCC uses each method's predefined fixed threshold from metrics_tools.csv."
    )
    print(
        "- AUROC/AUPR use direction-aligned continuous scores."
    )
    print(
        "- AUPR uses average_precision_score."
    )
    print(
        "- 95% CIs use 1,000 stratified bootstrap replicates."
    )


if __name__ == "__main__":
    main()
