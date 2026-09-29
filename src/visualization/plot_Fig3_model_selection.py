#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig. 3 — Model configuration selection using Gene-grouped CV

Panels
------
(a) Protein encoder comparison
(b) DNA encoder comparison
(c) Fusion strategy comparison
(d) Representation dimension comparison
    Left : equal representation dimensions
    Right: fixed branch dimensions = 128, varying fused dimension

Panels a/b/c retain the submitted-version visualization logic:
    - horizontal bars
    - AUROC and AUPR in two vertically stacked axes
    - zoomed x-axis
    - different method colors
    - exact values annotated

Panel d:
    - two compact line plots
    - AUROC and AUPR lines
    - exact values annotated

Special rules
-------------
1) Gate is missing from fusion/sweep_summary.json.
   Its AUROC/AUPR are taken from the GRU row in
   GPNMSA_enhanced/sweep_summary.json because
   Protein=BiLSTM + DNA=GRU + Gate is the selected configuration.

2) d=128 is missing from both dimension sweep files.
   The same selected GRU configuration result is inserted as d=128.

No mean ± SD is plotted. The JSON parser explicitly reads validation_mean and ignores validation_std.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# =============================================================================
# 1. PATHS
# =============================================================================

PROTEIN_JSON = (
    r"F:\GenProt-DSM_Resubmit\result\pred"
    r"\protT5_enhanced\sweep_summary.json"
)

DNA_JSON = (
    r"F:\GenProt-DSM_Resubmit\result\pred"
    r"\GPNMSA_enhanced\sweep_summary.json"
)

FUSION_JSON = (
    r"F:\GenProt-DSM_Resubmit\result\pred"
    r"\fusion\sweep_summary.json"
)

DIM_EQUAL_JSON = (
    r"F:\GenProt-DSM_Resubmit\result\pred"
    r"\fusion_d\dim_sweep_summary.json"
)

DIM_FUSED_JSON = (
    r"F:\GenProt-DSM_Resubmit\result\pred"
    r"\fusion_dd\dim_sweep_summary.json"
)

OUTPUT_DIR = r"F:\GenProt-DSM_Resubmit\fig_Results\Fig3_model_selection"
OUTPUT_BASENAME = "Fig3_model_configuration_selection"


# =============================================================================
# 2. GLOBAL STYLE
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
# 3. METHOD NAME NORMALIZATION
# =============================================================================

METHOD_ALIASES = {
    "transformer": "Transformer",
    "trans": "Transformer",
    "tcn": "TCN",
    "bilstm": "BiLSTM",
    "bi-lstm": "BiLSTM",
    "bi_lstm": "BiLSTM",
    "cnn": "CNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn-bilstm": "CNN-BiLSTM",
    "cnn_bilstm": "CNN-BiLSTM",
    "cnnbilstm": "CNN-BiLSTM",
    "tcn-lstm": "TCN-LSTM",
    "tcn_lstm": "TCN-LSTM",
    "tcnlstm": "TCN-LSTM",
    "gate": "Gate",
    "gated": "Gate",
    "gated-fusion": "Gate",
    "gated_fusion": "Gate",
    "concat": "Concat",
    "concatenate": "Concat",
    "concatenation": "Concat",
    "bilinear": "Bilinear",
    "xattn": "Xattn",
    "x-attn": "Xattn",
    "cross-attention": "Xattn",
    "cross_attention": "Xattn",
    "crossattention": "Xattn",
    "avg": "Avg",
    "average": "Avg",
    "mean": "Avg",
}

PROTEIN_EXPECTED = [
    "Transformer", "TCN", "BiLSTM", "CNN",
    "LSTM", "GRU", "CNN-BiLSTM", "TCN-LSTM",
]

DNA_EXPECTED = PROTEIN_EXPECTED.copy()

FUSION_EXPECTED = [
    "Gate", "Concat", "Bilinear", "Xattn", "Avg",
]


# =============================================================================
# 4. BASIC UTILITIES
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def panel_label(ax, label: str):
    ax.text(
        -0.075, 1.06, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def normalize_method_token(x: Any) -> str:
    s = str(x).strip()
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "-")
    return s


def canonical_method_name(x: Any) -> str:
    token = normalize_method_token(x)

    if token in METHOD_ALIASES:
        return METHOD_ALIASES[token]

    for alias in sorted(METHOD_ALIASES.keys(), key=len, reverse=True):
        if alias in token:
            return METHOD_ALIASES[alias]

    return str(x).strip()


def metric_to_fraction(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    if not np.isfinite(v):
        return None

    if 1.0 < v <= 100.0:
        v /= 100.0

    if v < 0:
        return None

    return float(v)


# =============================================================================
# 5. FLEXIBLE JSON EXTRACTION
# =============================================================================

AUROC_KEYS = {
    "auc_roc", "auroc", "roc_auc", "rocauc",
    "aucroc", "auc-roc", "val_auc_roc", "val_auroc",
    "test_auc_roc", "test_auroc",
}

AUPR_KEYS = {
    "auc_pr", "aupr", "auprc", "auc_prc",
    "pr_auc", "prauc", "aucpr", "average_precision",
    "ap", "val_auc_pr", "val_aupr",
    "test_auc_pr", "test_aupr",
}

NAME_KEYS = [
    "name", "method", "module", "encoder", "backbone",
    "architecture", "fusion", "fusion_method", "model",
    "config", "config_name", "setting", "experiment",
    "experiment_name", "label",
]


def normalized_key(k: Any) -> str:
    return str(k).strip().lower().replace(" ", "_").replace("-", "_")


def find_metric_in_dict(d: Dict[str, Any], wanted: set) -> Optional[float]:
    wanted_norm = {normalized_key(x) for x in wanted}

    for k, v in d.items():
        if normalized_key(k) in wanted_norm:
            value = metric_to_fraction(v)
            if value is not None:
                return value

    for container_key in [
        "final_metrics", "metrics", "result", "results",
        "scores", "performance", "best_metrics", "summary",
    ]:
        for k, v in d.items():
            if (
                normalized_key(k) == normalized_key(container_key)
                and isinstance(v, dict)
            ):
                ans = find_metric_in_dict(v, wanted)
                if ans is not None:
                    return ans

    return None


def infer_name_from_dict(d: Dict[str, Any], path: Sequence[str]) -> str:
    """
    Infer the actual experiment/method name.

    Typical sweep_summary.json structure:
        candidate_results/
            tcn/
                validation_mean/
                validation_std/

    In that case the current metric node is "validation_mean", but the
    method name is the parent path component "tcn".
    """

    for key in NAME_KEYS:
        for actual_key, value in d.items():
            if normalized_key(actual_key) == normalized_key(key):
                if isinstance(value, (str, int, float)):
                    return str(value)

    generic = {
        "results", "result", "metrics", "summary", "experiments",
        "configs", "configurations", "sweep", "runs", "items",
        "models", "encoders", "methods", "candidate_results",
        "validation_mean", "validation_std",
        "train_mean", "train_std",
        "test_mean", "test_std",
        "mean", "std",
        "fold_mean", "fold_std",
    }

    # Walk backward through the JSON path and return the nearest
    # non-generic parent node. For
    # candidate_results/tcn/validation_mean -> "tcn".
    for p in reversed(path):
        p2 = str(p).strip()
        if not p2:
            continue

        nk = normalized_key(p2)

        if nk in generic:
            continue

        return p2

    return "unknown"

def recursive_metric_records(
    obj: Any,
    path: Tuple[str, ...] = (),
) -> List[Dict[str, Any]]:
    """
    Recursively collect metric dictionaries.

    IMPORTANT:
    sweep_summary.json contains both validation_mean and validation_std.
    Only mean/performance nodes are eligible for plotting; *_std nodes are
    deliberately excluded.
    """

    records: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        last_node = normalized_key(path[-1]) if path else ""

        # Never interpret standard-deviation dictionaries as actual performance.
        is_std_node = (
            last_node == "std"
            or last_node.endswith("_std")
            or "validation_std" in last_node
            or "train_std" in last_node
            or "test_std" in last_node
        )

        if not is_std_node:
            auc = find_metric_in_dict(obj, AUROC_KEYS)
            aupr = find_metric_in_dict(obj, AUPR_KEYS)

            if auc is not None and aupr is not None:
                records.append({
                    "raw_name": infer_name_from_dict(obj, path),
                    "AUROC": auc,
                    "AUPR": aupr,
                    "_path": "/".join(path),
                    "_dict": obj,
                })

        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                records.extend(
                    recursive_metric_records(v, path + (str(k),))
                )

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                records.extend(
                    recursive_metric_records(item, path + (str(i),))
                )

    return records

def deduplicate_metric_records(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    prepared = []

    for r in records:
        rr = dict(r)
        rr["Method"] = canonical_method_name(rr["raw_name"])
        prepared.append(rr)

    prepared = sorted(
        prepared,
        key=lambda x: len(x.get("_path", "")),
    )

    seen = set()
    out = []

    for r in prepared:
        key = (
            r["Method"],
            round(float(r["AUROC"]), 10),
            round(float(r["AUPR"]), 10),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(r)

    return out


def _record_priority(record: Dict[str, Any]) -> Tuple[int, int]:
    """
    Lower is better.

    Prefer validation_mean, because Fig. 3 is model selection performed
    within the training set. Fall back to other mean/performance nodes only
    when validation_mean is unavailable.
    """
    p = normalized_key(record.get("_path", ""))

    if "validation_mean" in p:
        rank = 0
    elif "val_mean" in p:
        rank = 1
    elif "mean" in p and "std" not in p:
        rank = 2
    elif "final_metrics" in p or "metrics" in p:
        rank = 3
    else:
        rank = 4

    # Prefer the more specific/deeper record when otherwise tied.
    return (rank, -len(record.get("_path", "")))


def extract_method_table(
    json_path: str,
    expected_names: Sequence[str],
    panel_name: str,
) -> pd.DataFrame:

    records = deduplicate_metric_records(
        recursive_metric_records(load_json(json_path))
    )

    if not records:
        raise RuntimeError(
            f"No valid mean AUROC/AUPR records could be extracted from:\n{json_path}"
        )

    rows = []

    for expected in expected_names:
        matches = [
            r for r in records
            if canonical_method_name(r["Method"]) == expected
        ]

        if not matches:
            continue

        # Select the validation_mean record when available.
        matches = sorted(matches, key=_record_priority)
        chosen = matches[0]

        rows.append({
            "Method": expected,
            "AUROC": float(chosen["AUROC"]),
            "AUPR": float(chosen["AUPR"]),
            "Source_path": chosen.get("_path", ""),
        })

    if not rows:
        detected = [
            (
                r["raw_name"], r["Method"], r["AUROC"],
                r["AUPR"], r["_path"]
            )
            for r in records
        ]
        raise RuntimeError(
            f"\nCould not match expected methods for {panel_name}.\n"
            f"JSON: {json_path}\n"
            f"Detected valid records:\n{detected}"
        )

    df = pd.DataFrame(rows)

    missing = [
        m for m in expected_names
        if m not in set(df["Method"])
    ]

    if missing:
        print(
            f"[WARN] {panel_name}: methods not found in JSON: {missing}"
        )

    return df


# =============================================================================
# 6. SELECTED GRU RESULT
# =============================================================================

def get_selected_gru_result(
    dna_df: pd.DataFrame,
) -> Tuple[float, float]:

    row = dna_df.loc[
        dna_df["Method"] == "GRU"
    ]

    if row.empty:
        raise RuntimeError(
            "\nCannot reconstruct Gate / d=128 because the GRU result "
            "was not found in GPNMSA_enhanced/sweep_summary.json."
        )

    row = row.iloc[0]

    return (
        float(row["AUROC"]),
        float(row["AUPR"]),
    )


# =============================================================================
# 7. DIMENSION EXTRACTION
# =============================================================================

DIM_KEYS = [
    "d", "dim", "dimension", "hidden_dim",
    "representation_dim", "fused_d", "fusion_d", "d_fused",
]

PROT_DIM_KEYS = [
    "prot_d", "protein_d", "prott5_d", "t5_d", "d_prot",
]

DNA_DIM_KEYS = [
    "gpn_d", "gpnmsa_d", "dna_d", "d_gpn", "d_dna",
]

FUSED_DIM_KEYS = [
    "fused_d", "fusion_d", "d_fused", "output_d",
]


def find_numeric_field(
    d: Dict[str, Any],
    candidate_keys: Sequence[str],
) -> Optional[int]:

    normalized_candidates = {
        normalized_key(x) for x in candidate_keys
    }

    for k, v in d.items():
        if normalized_key(k) in normalized_candidates:
            try:
                vv = int(float(v))
                if vv > 0:
                    return vv
            except Exception:
                pass

    return None


def numeric_from_path(path: str) -> Optional[int]:
    tokens = re.findall(
        r"(?<!\d)(\d{2,4})(?!\d)",
        str(path),
    )

    allowed = {
        16, 32, 64, 128, 256, 512,
        1024, 2048, 4096,
    }

    values = []

    for t in tokens:
        try:
            v = int(t)
            if v in allowed:
                values.append(v)
        except Exception:
            pass

    return values[-1] if values else None


def extract_dimension_table(
    json_path: str,
    kind: str,
) -> pd.DataFrame:

    records = deduplicate_metric_records(
        recursive_metric_records(load_json(json_path))
    )

    rows = []

    for r in records:
        dct = r.get("_dict", {})
        dim = None

        if isinstance(dct, dict):
            if kind == "fused":
                dim = find_numeric_field(
                    dct,
                    FUSED_DIM_KEYS + DIM_KEYS,
                )
            else:
                dim = find_numeric_field(
                    dct,
                    DIM_KEYS + PROT_DIM_KEYS + DNA_DIM_KEYS + FUSED_DIM_KEYS,
                )

        if dim is None:
            dim = numeric_from_path(r.get("_path", ""))

        if dim is None:
            dim = numeric_from_path(str(r.get("raw_name", "")))

        if dim is None:
            continue

        rows.append({
            "Dimension": int(dim),
            "AUROC": float(r["AUROC"]),
            "AUPR": float(r["AUPR"]),
            "Source_path": r.get("_path", ""),
        })

    if not rows:
        raise RuntimeError(
            f"\nCould not extract any dimension records from:\n{json_path}"
        )

    df = pd.DataFrame(rows)

    # Prefer validation_mean records for each dimension.
    df["_priority"] = df["Source_path"].astype(str).map(
        lambda p: (
            0 if "validation_mean" in normalized_key(p)
            else 1 if ("mean" in normalized_key(p) and "std" not in normalized_key(p))
            else 2
        )
    )

    df = (
        df.sort_values(
            ["Dimension", "_priority"],
            ascending=[True, True],
        )
        .drop_duplicates(
            subset=["Dimension"],
            keep="first",
        )
        .drop(columns=["_priority"])
        .sort_values("Dimension")
        .reset_index(drop=True)
    )

    return df


def insert_selected_128(
    df: pd.DataFrame,
    selected_auroc: float,
    selected_aupr: float,
) -> pd.DataFrame:

    out = df.copy()

    if 128 not in set(out["Dimension"].astype(int)):
        out = pd.concat(
            [
                out,
                pd.DataFrame([{
                    "Dimension": 128,
                    "AUROC": selected_auroc,
                    "AUPR": selected_aupr,
                    "Source_path": (
                        "Reconstructed from selected "
                        "GPNMSA_enhanced GRU configuration"
                    ),
                }]),
            ],
            ignore_index=True,
        )

    return (
        out.sort_values("Dimension")
        .reset_index(drop=True)
    )


# =============================================================================
# 8. PANELS a/b/c — OLD HORIZONTAL-BAR STYLE
# =============================================================================

def method_color_map(
    methods: Iterable[str],
) -> Dict[str, Any]:

    methods = list(dict.fromkeys(methods))
    cmap = plt.get_cmap("tab20")

    return {
        m: cmap(i % 20)
        for i, m in enumerate(methods)
    }


def zoom_xlim(
    values: np.ndarray,
) -> Tuple[float, float, float]:

    values = np.asarray(values, dtype=float)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    spread = vmax - vmin

    pad = max(0.002, spread * 0.25)

    left = max(0.0, vmin - pad)
    right = min(1.0, vmax + pad)

    if right - vmax < pad * 0.40:
        right = min(1.0, vmax + pad * 0.60)

    return left, right, pad


def draw_bar_metric(
    ax,
    df_ordered: pd.DataFrame,
    metric: str,
    color_map: Dict[str, Any],
):

    names = df_ordered["Method"].astype(str).tolist()
    values = df_ordered[metric].astype(float).to_numpy()
    y = np.arange(len(df_ordered))

    colors = [
        color_map[n]
        for n in names
    ]

    ax.barh(
        y,
        values,
        height=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(metric)

    left, right, pad = zoom_xlim(values)

    ax.set_xlim(left, right)

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=5)
    )

    ax.grid(
        True,
        axis="x",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    text_offset = max(0.00015, pad * 0.06)

    for yi, v in zip(y, values):
        ax.text(
            v + text_offset,
            yi,
            f"{v:.4f}",
            va="center",
            ha="left",
            fontsize=7,
        )


def draw_module_panel(
    fig,
    subspec,
    df: pd.DataFrame,
    title: str,
    label: str,
):
    container = fig.add_subplot(subspec)
    container.axis("off")

    panel_label(container, label)
    container.set_title(title, pad=4)

    inner = subspec.subgridspec(
        2, 1,
        hspace=0.34,
    )

    ax_auc = fig.add_subplot(inner[0, 0])
    ax_aupr = fig.add_subplot(inner[1, 0])

    # Same order for AUROC and AUPR, sorted by AUROC.
    ordered = (
        df.sort_values("AUROC", ascending=True)
        .reset_index(drop=True)
    )

    colors = method_color_map(
        ordered["Method"]
    )

    draw_bar_metric(
        ax_auc,
        ordered,
        metric="AUROC",
        color_map=colors,
    )

    draw_bar_metric(
        ax_aupr,
        ordered,
        metric="AUPR",
        color_map=colors,
    )


# =============================================================================
# 9. PANEL d — DIMENSION LINE PLOTS
# =============================================================================

def draw_dimension_axis(
    ax,
    df: pd.DataFrame,
    title: str,
):

    df = (
        df.sort_values("Dimension")
        .reset_index(drop=True)
    )

    x = df["Dimension"].astype(int).to_numpy()
    auroc = df["AUROC"].astype(float).to_numpy()
    aupr = df["AUPR"].astype(float).to_numpy()

    ax.plot(
        x,
        auroc,
        marker="o",
        markersize=4.5,
        linewidth=1.3,
        label="AUROC",
    )

    ax.plot(
        x,
        aupr,
        marker="s",
        markersize=4.5,
        linewidth=1.3,
        label="AUPR",
    )

    ax.set_title(title)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Score")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])

    all_values = np.concatenate([auroc, aupr])

    ymin = float(np.min(all_values))
    ymax = float(np.max(all_values))
    spread = ymax - ymin
    pad = max(0.004, spread * 0.25)

    ax.set_ylim(
        max(0.0, ymin - pad),
        min(1.0, ymax + pad),
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        frameon=True,
        fontsize=8,
        loc="best",
    )

    if 128 in set(x):
        ax.axvline(
            128,
            linestyle=":",
            linewidth=0.8,
            alpha=0.55,
        )

    y_range = max(
        ax.get_ylim()[1] - ax.get_ylim()[0],
        1e-6,
    )
    offset = y_range * 0.018

    for xi, yi in zip(x, auroc):
        ax.text(
            xi,
            yi - offset,
            f"{yi:.4f}",
            fontsize=7,
            ha="center",
            va="top",
        )

    for xi, yi in zip(x, aupr):
        ax.text(
            xi,
            yi + offset,
            f"{yi:.4f}",
            fontsize=7,
            ha="center",
            va="bottom",
        )


def draw_dimension_panel(
    fig,
    subspec,
    equal_df: pd.DataFrame,
    fused_df: pd.DataFrame,
    label: str = "d",
):
    container = fig.add_subplot(subspec)
    container.axis("off")

    panel_label(container, label)
    container.set_title("Representation dimension", pad=4)

    inner = subspec.subgridspec(
        1, 2,
        wspace=0.28,
    )

    ax_left = fig.add_subplot(inner[0, 0])
    ax_right = fig.add_subplot(inner[0, 1])

    draw_dimension_axis(
        ax_left,
        equal_df,
        title="Equal dimensions",
    )

    draw_dimension_axis(
        ax_right,
        fused_df,
        title="Fixed branches = 128",
    )


# =============================================================================
# 10. MAIN
# =============================================================================

def main():

    ensure_dir(OUTPUT_DIR)

    input_paths = {
        "Protein encoder": PROTEIN_JSON,
        "DNA encoder": DNA_JSON,
        "Fusion": FUSION_JSON,
        "Equal-dimension sweep": DIM_EQUAL_JSON,
        "Fused-dimension sweep": DIM_FUSED_JSON,
    }

    print("=" * 92)
    print("FIG. 3 INPUT FILES")
    print("=" * 92)

    for name, path in input_paths.items():
        print(f"{name}: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    # a/b
    protein_df = extract_method_table(
        PROTEIN_JSON,
        expected_names=PROTEIN_EXPECTED,
        panel_name="Protein encoder",
    )

    dna_df = extract_method_table(
        DNA_JSON,
        expected_names=DNA_EXPECTED,
        panel_name="DNA encoder",
    )

    # Selected GRU result
    selected_auroc, selected_aupr = get_selected_gru_result(
        dna_df
    )

    print("\n[SELECTED CONFIGURATION RESULT]")
    print(f"GRU / selected configuration AUROC = {selected_auroc:.6f}")
    print(f"GRU / selected configuration AUPR  = {selected_aupr:.6f}")

    # c: fusion + reconstructed Gate
    fusion_without_gate = extract_method_table(
        FUSION_JSON,
        expected_names=[
            m for m in FUSION_EXPECTED
            if m != "Gate"
        ],
        panel_name="Fusion strategy",
    )

    gate_row = pd.DataFrame([{
        "Method": "Gate",
        "AUROC": selected_auroc,
        "AUPR": selected_aupr,
        "Source_path": (
            "Reconstructed from selected "
            "GPNMSA_enhanced GRU configuration"
        ),
    }])

    fusion_df = pd.concat(
        [gate_row, fusion_without_gate],
        ignore_index=True,
    )

    fusion_df["_order"] = fusion_df["Method"].map({
        name: i
        for i, name in enumerate(FUSION_EXPECTED)
    })

    fusion_df = (
        fusion_df.sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    # d
    dim_equal_df = extract_dimension_table(
        DIM_EQUAL_JSON,
        kind="equal",
    )

    dim_fused_df = extract_dimension_table(
        DIM_FUSED_JSON,
        kind="fused",
    )

    dim_equal_df = insert_selected_128(
        dim_equal_df,
        selected_auroc,
        selected_aupr,
    )

    dim_fused_df = insert_selected_128(
        dim_fused_df,
        selected_auroc,
        selected_aupr,
    )

    # Save exact values used
    protein_df.to_csv(
        os.path.join(OUTPUT_DIR, "Fig3a_protein_encoder_values.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    dna_df.to_csv(
        os.path.join(OUTPUT_DIR, "Fig3b_DNA_encoder_values.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    fusion_df.to_csv(
        os.path.join(OUTPUT_DIR, "Fig3c_fusion_strategy_values.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    dim_equal_df.to_csv(
        os.path.join(OUTPUT_DIR, "Fig3d_equal_dimension_values.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    dim_fused_df.to_csv(
        os.path.join(OUTPUT_DIR, "Fig3d_fixed_branch_dimension_values.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Print values
    print("\n" + "=" * 92)
    print("VALUES USED IN FIG. 3")
    print("=" * 92)

    print("\n[a] Protein encoder")
    print(
        protein_df[["Method", "AUROC", "AUPR"]]
        .to_string(index=False)
    )

    print("\n[b] DNA encoder")
    print(
        dna_df[["Method", "AUROC", "AUPR"]]
        .to_string(index=False)
    )

    print("\n[c] Fusion strategy")
    print(
        fusion_df[["Method", "AUROC", "AUPR"]]
        .to_string(index=False)
    )

    print("\n[d-left] Equal dimensions")
    print(
        dim_equal_df[["Dimension", "AUROC", "AUPR"]]
        .to_string(index=False)
    )

    print("\n[d-right] Fixed branches = 128")
    print(
        dim_fused_df[["Dimension", "AUROC", "AUPR"]]
        .to_string(index=False)
    )

    # Build figure
    fig = plt.figure(
        figsize=(14.5, 9.0),
        dpi=FIG_DPI,
    )

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 0.92],
        wspace=0.27,
        hspace=0.31,
    )

    draw_module_panel(
        fig,
        gs[0, 0],
        protein_df,
        title="Protein encoder",
        label="a",
    )

    draw_module_panel(
        fig,
        gs[0, 1],
        dna_df,
        title="DNA encoder",
        label="b",
    )

    draw_module_panel(
        fig,
        gs[1, 0],
        fusion_df,
        title="Fusion strategy",
        label="c",
    )

    draw_dimension_panel(
        fig,
        gs[1, 1],
        equal_df=dim_equal_df,
        fused_df=dim_fused_df,
        label="d",
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.96,
        bottom=0.07,
        wspace=0.27,
        hspace=0.31,
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

    print("\n[DONE]")
    print("PNG:", output_base + ".png")
    print("PDF:", output_base + ".pdf")


if __name__ == "__main__":
    main()
