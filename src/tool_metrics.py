#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Corrected benchmark evaluation with THREE evaluation scopes:

1) original
   - Each tool is evaluated on its own available non-missing subset.

2) all_tools_common_subset
   - ALL benchmark tools must simultaneously have valid scores.
   - This reproduces the common-subset logic of the user's original script.

3) six_method_common_subset
   - Only the six reviewer-focused methods must simultaneously have valid scores:
       GenProt-DSM, AlphaMissense, CADD, PrimateAI, MPC, DEOGEN2.

Statistical/methodological corrections:
- NO test-label-driven automatic score flipping.
- NO test-set F1 optimization to choose thresholds.
- Score direction and binary thresholds are fixed a priori.
- GenProt-DSM threshold = 0.5543934838088572, obtained from training-set
  5-fold validation and kept fixed on test.
- AUPR uses average_precision_score (AP), consistent with the user's prior script.

Important assumptions:
- CADD threshold 15 assumes CADD PHRED/scaled score, NOT raw/rankscore.
- PolyPhen2 threshold 0.5 assumes the user's PolyPhen2 column is compatible with
  a 0.5 damaging operating threshold.
- DANN uses 0.5 as an operational literature-supported binary threshold;
  the original DANN paper did not establish a universal clinical cutoff.
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)


# =============================================================================
# 1. Paths
# =============================================================================

INPUT_FILE = r"F:\GenProt-DSM_Resubmit\data\test_score.csv"

OUTPUT_DIR = r"F:\GenProt-DSM_Resubmit\metrics_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "metrics_tools_corrected_with_two_common_subsets.csv"
)

ALL_COMMON_SUBSET_FILE = os.path.join(
    OUTPUT_DIR,
    "all_tools_common_subset.csv"
)

SIX_COMMON_SUBSET_FILE = os.path.join(
    OUTPUT_DIR,
    "reviewer_six_methods_common_subset.csv"
)


# =============================================================================
# 2. Tool definitions
# =============================================================================

LABEL_COL = "label"

# direction:
#   high = larger raw score -> more pathogenic
#   low  = smaller raw score -> more pathogenic
#
# threshold:
#   fixed threshold on RAW score
#
# comparator:
#   ge -> pathogenic if score >= threshold
#   le -> pathogenic if score <= threshold

TOOL_SPECS = {
    "SIFT": {
        "column": "SIFT",
        "direction": "low",
        "threshold": 0.05,
        "comparator": "le",
        "threshold_note": "SIFT deleterious cutoff <=0.05",
    },

    "Polyphen2": {
        "column": "Polyphen2",
        "direction": "high",
        "threshold": 0.5,
        "comparator": "ge",
        "threshold_note": (
            "Operational cutoff >=0.5. Verify the exact PolyPhen2 score type "
            "used in the input column."
        ),
    },

    "FATHMM": {
        "column": "FATHMM",
        "direction": "low",
        "threshold": -1.5,
        "comparator": "le",
        "threshold_note": "FATHMM damaging cutoff <=-1.5",
    },

    "PROVEAN": {
        "column": "PROVEAN",
        "direction": "low",
        "threshold": -2.5,
        "comparator": "le",
        "threshold_note": "PROVEAN default deleterious cutoff <=-2.5",
    },

    "MPC": {
        "column": "MPC",
        "direction": "high",
        "threshold": 2.0,
        "comparator": "ge",
        "threshold_note": "MPC high-deleterious/MisB cutoff >=2",
    },

    "PrimateAI": {
        "column": "PrimateAI",
        "direction": "high",
        "threshold": 0.803,
        "comparator": "ge",
        "threshold_note": "PrimateAI pathogenic operating cutoff >=0.803",
    },

    "DEOGEN2": {
        "column": "DEOGEN2",
        "direction": "high",
        "threshold": 0.5,
        "comparator": "ge",
        "threshold_note": "DEOGEN2 prediction cutoff >=0.5",
    },

    "AlphaMissense": {
        "column": "AlphaMissense",
        "direction": "high",
        "threshold": 0.564,
        "comparator": "ge",
        "threshold_note": "AlphaMissense likely-pathogenic boundary >=0.564",
    },

    "CADD": {
        "column": "CADD",
        "direction": "high",
        "threshold": 15.0,
        "comparator": "ge",
        "threshold_note": (
            "CADD PHRED/scaled only. CADD has no natural universal binary cutoff; "
            "15 is used as the prespecified operational threshold."
        ),
    },

    "DANN": {
        "column": "DANN",
        "direction": "high",
        "threshold": 0.5,
        "comparator": "ge",
        "threshold_note": (
            "Operational cutoff >=0.5. The original DANN paper did not define "
            "a universal clinical cutoff."
        ),
    },

    "GenoCanyon": {
        "column": "GenoCanyon",
        "direction": "high",
        "threshold": 0.5,
        "comparator": "ge",
        "threshold_note": (
            "Operational functional-potential cutoff >=0.5; GenoCanyon measures "
            "functional potential rather than missense pathogenicity specifically."
        ),
    },

    "ESM1b": {
        "column": "ESM1b",
        "direction": "low",
        "threshold": -7.5,
        "comparator": "le",
        "threshold_note": (
            "Lower score = more pathogenic; operational cutoff <=-7.5."
        ),
    },

    "GenProt-DSM": {
        "column": "GenProt-DSM",
        "direction": "high",
        "threshold": 0.5543934838088572,
        "comparator": "ge",
        "threshold_note": (
            "Fixed ensemble threshold from training-set 5-fold validation."
        ),
    },
}

ALL_METHODS = list(TOOL_SPECS.keys())

REVIEWER_METHODS = [
    "GenProt-DSM",
    "AlphaMissense",
    "CADD",
    "PrimateAI",
    "MPC",
    "DEOGEN2",
]


# =============================================================================
# 3. Helper functions
# =============================================================================

def orient_score(raw_score, direction):
    raw_score = np.asarray(raw_score, dtype=float)

    if direction == "high":
        return raw_score

    if direction == "low":
        return -raw_score

    raise ValueError(f"Unknown direction: {direction}")


def predict_with_fixed_threshold(
    raw_score,
    threshold,
    comparator,
):
    raw_score = np.asarray(raw_score, dtype=float)

    if threshold is None or comparator is None:
        return None

    if comparator == "ge":
        return (raw_score >= threshold).astype(int)

    if comparator == "le":
        return (raw_score <= threshold).astype(int)

    raise ValueError(
        f"Unknown comparator: {comparator}"
    )


def calculate_metrics(
    y_true,
    raw_score,
    tool_name,
    subset_type,
):
    """
    Continuous-score metrics:
        AUROC, AUPR/AP
    Threshold-dependent metrics:
        ACC, F1, precision, recall, MCC
    """

    spec = TOOL_SPECS[tool_name]

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    raw_score = np.asarray(
        raw_score,
        dtype=float,
    )

    n_used = len(y_true)

    result = {
        "tool_name": tool_name,
        "tool_column": spec["column"],
        "subset_type": subset_type,
        "n_used": n_used,

        "score_direction": spec["direction"],
        "fixed_threshold": spec["threshold"],
        "threshold_rule": spec["comparator"],
        "threshold_note": spec["threshold_note"],

        "AUC_ROC": np.nan,
        "AUC_PR": np.nan,
        "ACC": np.nan,
        "F1": np.nan,
        "PREC": np.nan,
        "REC": np.nan,
        "MCC": np.nan,
    }

    if n_used == 0:
        return result

    if len(np.unique(y_true)) < 2:
        return result

    # -------------------------------------------------------------
    # Fixed score direction.
    # Never infer direction from test labels.
    # -------------------------------------------------------------
    score = orient_score(
        raw_score,
        spec["direction"],
    )

    result["AUC_ROC"] = float(
        roc_auc_score(
            y_true,
            score,
        )
    )

    result["AUC_PR"] = float(
        average_precision_score(
            y_true,
            score,
        )
    )

    # -------------------------------------------------------------
    # Fixed threshold.
    # -------------------------------------------------------------
    pred = predict_with_fixed_threshold(
        raw_score,
        spec["threshold"],
        spec["comparator"],
    )

    if pred is not None:

        result["ACC"] = float(
            accuracy_score(
                y_true,
                pred,
            )
        )

        result["F1"] = float(
            f1_score(
                y_true,
                pred,
                zero_division=0,
            )
        )

        result["PREC"] = float(
            precision_score(
                y_true,
                pred,
                zero_division=0,
            )
        )

        result["REC"] = float(
            recall_score(
                y_true,
                pred,
                zero_division=0,
            )
        )

        result["MCC"] = float(
            matthews_corrcoef(
                y_true,
                pred,
            )
        )

    return result


def build_common_mask(
    df,
    y_all,
    methods,
):
    """
    Maximum common subset:
    label and EVERY requested tool score must be numeric/non-missing.
    """

    mask = y_all.notna().copy()

    for tool_name in methods:

        col = TOOL_SPECS[
            tool_name
        ]["column"]

        if col not in df.columns:

            print(
                f"[WARNING] Common-subset required column missing: "
                f"{tool_name} -> {col}"
            )

            return pd.Series(
                False,
                index=df.index,
            )

        s = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        mask &= s.notna()

    return mask


def print_subset_summary(
    name,
    mask,
    y_all,
):
    n = int(mask.sum())

    if n == 0:
        print(
            f"\n{name}: N=0"
        )
        return

    y = (
        y_all[mask]
        .astype(int)
        .to_numpy()
    )

    print(
        f"\n{name}:"
    )
    print(
        f"  N        = {n:,}"
    )
    print(
        f"  label=1  = {(y == 1).sum():,}"
    )
    print(
        f"  label=0  = {(y == 0).sum():,}"
    )


# =============================================================================
# 4. Read input
# =============================================================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

if LABEL_COL not in df.columns:

    raise KeyError(
        f"Missing label column '{LABEL_COL}'.\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )


y_all = pd.to_numeric(
    df[LABEL_COL],
    errors="coerce",
).astype("Int64")


print("=" * 110)
print("Corrected benchmark evaluation")
print("=" * 110)

print(
    f"Input rows = {len(df):,}"
)


# =============================================================================
# 5. Print per-tool valid counts
# =============================================================================

print("\nPer-tool valid prediction counts:")

for tool_name in ALL_METHODS:

    col = TOOL_SPECS[
        tool_name
    ]["column"]

    if col not in df.columns:

        print(
            f"  {tool_name:15s}: COLUMN MISSING ({col})"
        )

        continue

    n_valid = int(
        (
            y_all.notna()
            &
            pd.to_numeric(
                df[col],
                errors="coerce",
            ).notna()
        ).sum()
    )

    print(
        f"  {tool_name:15s}: {n_valid:,}"
    )


# =============================================================================
# 6. CADD score-scale check
# =============================================================================

if "CADD" in df.columns:

    cadd_numeric = pd.to_numeric(
        df["CADD"],
        errors="coerce",
    ).dropna()

    if len(cadd_numeric) > 0:

        cadd_min = float(
            cadd_numeric.min()
        )

        cadd_median = float(
            cadd_numeric.median()
        )

        cadd_max = float(
            cadd_numeric.max()
        )

        print(
            "\n[CADD scale check] "
            f"min={cadd_min:.4f}, "
            f"median={cadd_median:.4f}, "
            f"max={cadd_max:.4f}"
        )

        if cadd_max < 10:

            raise ValueError(
                "\nCADD column does not look like CADD PHRED/scaled score "
                "(maximum < 10), but threshold-dependent metrics use cutoff 15.\n"
                "Please verify whether this column is CADD_raw/rankscore."
            )


# =============================================================================
# 7. Build TWO maximum common subsets
# =============================================================================

# -------------------------------------------------------------
# A. All-tool common subset:
#    reproduces user's original common-subset idea.
# -------------------------------------------------------------

all_common_mask = build_common_mask(
    df=df,
    y_all=y_all,
    methods=ALL_METHODS,
)

all_common_n = int(
    all_common_mask.sum()
)

print_subset_summary(
    "ALL-TOOLS maximum common subset",
    all_common_mask,
    y_all,
)


# -------------------------------------------------------------
# B. Reviewer six-method common subset
# -------------------------------------------------------------

six_common_mask = build_common_mask(
    df=df,
    y_all=y_all,
    methods=REVIEWER_METHODS,
)

six_common_n = int(
    six_common_mask.sum()
)

print_subset_summary(
    "SIX-METHOD reviewer maximum common subset",
    six_common_mask,
    y_all,
)


# Save exact original rows for audit.
if all_common_n > 0:

    df.loc[
        all_common_mask
    ].to_csv(
        ALL_COMMON_SUBSET_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n[Saved all-tools common subset]\n"
        f"{ALL_COMMON_SUBSET_FILE}"
    )


if six_common_n > 0:

    df.loc[
        six_common_mask
    ].to_csv(
        SIX_COMMON_SUBSET_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n[Saved six-method common subset]\n"
        f"{SIX_COMMON_SUBSET_FILE}"
    )


# =============================================================================
# 8. Scope 1: ORIGINAL performance
# =============================================================================

rows_original = []

print("\n" + "=" * 110)
print("SCOPE 1: ORIGINAL — each tool uses its own available subset")
print("=" * 110)

for tool_name in ALL_METHODS:

    spec = TOOL_SPECS[
        tool_name
    ]

    col = spec[
        "column"
    ]

    if col not in df.columns:

        result = calculate_metrics(
            y_true=np.array(
                [],
                dtype=int,
            ),
            raw_score=np.array(
                [],
                dtype=float,
            ),
            tool_name=tool_name,
            subset_type="original",
        )

        rows_original.append(
            result
        )

        print(
            f"[Missing] {tool_name}: {col}"
        )

        continue


    raw = pd.to_numeric(
        df[col],
        errors="coerce",
    )

    mask = (
        y_all.notna()
        &
        raw.notna()
    )

    y_true = (
        y_all[mask]
        .astype(int)
        .to_numpy()
    )

    raw_score = (
        raw[mask]
        .astype(float)
        .to_numpy()
    )

    result = calculate_metrics(
        y_true=y_true,
        raw_score=raw_score,
        tool_name=tool_name,
        subset_type="original",
    )

    rows_original.append(
        result
    )

    print(
        f"{tool_name:15s} "
        f"N={result['n_used']:4d} "
        f"AUROC={result['AUC_ROC']:.4f} "
        f"AUPR={result['AUC_PR']:.4f} "
        f"MCC={result['MCC']:.4f}"
    )


# =============================================================================
# 9. Scope 2: ALL-TOOLS maximum common subset
# =============================================================================

rows_all_common = []

print("\n" + "=" * 110)
print("SCOPE 2: ALL-TOOLS MAXIMUM COMMON SUBSET")
print("=" * 110)

if all_common_n > 0:

    y_all_common = (
        y_all[
            all_common_mask
        ]
        .astype(int)
        .to_numpy()
    )

    for tool_name in ALL_METHODS:

        col = TOOL_SPECS[
            tool_name
        ]["column"]

        raw_score = pd.to_numeric(
            df.loc[
                all_common_mask,
                col,
            ],
            errors="coerce",
        ).astype(float).to_numpy()

        result = calculate_metrics(
            y_true=y_all_common,
            raw_score=raw_score,
            tool_name=tool_name,
            subset_type="all_tools_common_subset",
        )

        rows_all_common.append(
            result
        )

        # Every method MUST use the exact same N here.
        if result["n_used"] != all_common_n:
            raise RuntimeError(
                f"All-tools common subset N mismatch for {tool_name}"
            )

        print(
            f"{tool_name:15s} "
            f"N={result['n_used']:4d} "
            f"AUROC={result['AUC_ROC']:.4f} "
            f"AUPR={result['AUC_PR']:.4f} "
            f"MCC={result['MCC']:.4f}"
        )

else:

    print(
        "All-tools common subset is empty. "
        "Check missing columns/non-numeric values."
    )


# =============================================================================
# 10. Scope 3: SIX-METHOD reviewer maximum common subset
# =============================================================================

rows_six_common = []

print("\n" + "=" * 110)
print("SCOPE 3: SIX-METHOD REVIEWER MAXIMUM COMMON SUBSET")
print("=" * 110)

if six_common_n > 0:

    y_six_common = (
        y_all[
            six_common_mask
        ]
        .astype(int)
        .to_numpy()
    )

    for tool_name in REVIEWER_METHODS:

        col = TOOL_SPECS[
            tool_name
        ]["column"]

        raw_score = pd.to_numeric(
            df.loc[
                six_common_mask,
                col,
            ],
            errors="coerce",
        ).astype(float).to_numpy()

        result = calculate_metrics(
            y_true=y_six_common,
            raw_score=raw_score,
            tool_name=tool_name,
            subset_type="six_method_common_subset",
        )

        rows_six_common.append(
            result
        )

        if result["n_used"] != six_common_n:

            raise RuntimeError(
                f"Six-method common subset N mismatch for {tool_name}"
            )

        print(
            f"{tool_name:15s} "
            f"N={result['n_used']:4d} "
            f"AUROC={result['AUC_ROC']:.4f} "
            f"AUPR={result['AUC_PR']:.4f} "
            f"MCC={result['MCC']:.4f}"
        )

else:

    print(
        "Six-method common subset is empty."
    )


# =============================================================================
# 11. Combine and save
# =============================================================================

all_rows = (
    rows_original
    +
    rows_all_common
    +
    rows_six_common
)

out_df = pd.DataFrame(
    all_rows
)


column_order = [
    "tool_name",
    "tool_column",
    "subset_type",
    "n_used",

    "score_direction",
    "fixed_threshold",
    "threshold_rule",
    "threshold_note",

    "AUC_ROC",
    "AUC_PR",
    "ACC",
    "F1",
    "PREC",
    "REC",
    "MCC",
]

out_df = out_df[
    column_order
]


# Convenient ordering:
# original -> all-tools common -> reviewer six-method common
scope_order = {
    "original": 0,
    "all_tools_common_subset": 1,
    "six_method_common_subset": 2,
}

tool_order = {
    tool_name: i
    for i, tool_name in enumerate(
        ALL_METHODS
    )
}

out_df["_scope_order"] = (
    out_df["subset_type"]
    .map(scope_order)
)

out_df["_tool_order"] = (
    out_df["tool_name"]
    .map(tool_order)
)

out_df = (
    out_df
    .sort_values(
        by=[
            "_scope_order",
            "_tool_order",
        ]
    )
    .drop(
        columns=[
            "_scope_order",
            "_tool_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


out_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 12. Final summary
# =============================================================================

print("\n" + "=" * 110)
print("DONE")
print("=" * 110)

print(
    f"Original test N                 = {len(df):,}"
)

print(
    f"All-tools common subset N       = {all_common_n:,}"
)

print(
    f"Six-method common subset N      = {six_common_n:,}"
)

print(
    f"\nMetrics saved:\n"
    f"{OUTPUT_FILE}"
)

if all_common_n > 0:

    print(
        f"\nAll-tools exact common rows:\n"
        f"{ALL_COMMON_SUBSET_FILE}"
    )

if six_common_n > 0:

    print(
        f"\nSix-method exact common rows:\n"
        f"{SIX_COMMON_SUBSET_FILE}"
    )
