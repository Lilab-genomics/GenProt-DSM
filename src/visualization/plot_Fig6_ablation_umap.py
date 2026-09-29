#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig. 6 — Ablation and UMAP visualization of GenProt-DSM

This version intentionally follows the PREVIOUS manuscript layout:

(a) Ablation scatter:
      x = AUROC
      y = AUPR
      one point per ablation setting
      legend at lower right

(b) Six-panel UMAP:
      top row:
          ProtT5 Input
          GPN-MSA Input
          Functional Annotation
      bottom row:
          Protein Encoder Output
          DNA Encoder Output
          Gate Fusion Output

      All six UMAPs are colored by the FINAL five-fold ensemble
      pathogenicity probability.

Important metric source rule
----------------------------
For every ablation experiment, metrics are read ONLY from:
    <ablation_folder>/ckpt/test_ensemble_summary.json
        -> root-level "final_metrics"

Fold-level "test_metrics_single_model" values are never used.

Representation rule
-------------------
To reproduce the old visualization logic, the DNA encoder output,
protein encoder output, and fused output are averaged across the five
trained folds before UMAP.

Outputs
-------
F:\\GenProt-DSM_Resubmit\\fig_Results\\Fig6_ablation_umap
    Fig6_ablation_umap.png
    Fig6_ablation_umap.pdf
    Fig6a_ablation_metrics.csv
    Fig6b_umap_coordinates.csv
    representation_cache\\...

Requirements
------------
pip install numpy pandas torch matplotlib scikit-learn umap-learn
"""

from __future__ import annotations

import json
import os
import re
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError as e:
    raise ImportError(
        "umap-learn is required.\n"
        "Install with:\n"
        "    pip install umap-learn"
    ) from e


# =============================================================================
# 1. PATHS
# =============================================================================

ABLATION_ROOT = r"F:\GenProt-DSM_Resubmit\result\pred\ablation"

FULL_MODEL_DIR = os.path.join(
    ABLATION_ROOT,
    "GenProt-DSM",
)

# Fallback formal-model directory if a file is absent in the ablation baseline.
FORMAL_MODEL_DIR = r"F:\GenProt-DSM_Resubmit\result\pred\model"

TOOL_SCORE_CSV = (
    r"F:\GenProt-DSM_Resubmit\dataset\data\test_score.csv"
)

# Used only if checkpoint cfg["test_csv"] cannot be found.
TEST_META_FALLBACK = (
    r"F:\GenProt-DSM_Resubmit\dataset\data\VarGeneDisjointTest.csv"
)

# Used only if checkpoint cfg["memmap_dir"] cannot be found.
MEMMAP_DIR_FALLBACK = (
    r"F:\GenProt-DSM_Resubmit\dataset\data\feature\merged_memmap_raw"
)

OUTPUT_DIR = (
    r"F:\GenProt-DSM_Resubmit\fig_Results"
    r"\Fig6_ablation_umap"
)

CACHE_DIR = os.path.join(
    OUTPUT_DIR,
    "representation_cache",
)

OUTPUT_BASENAME = "Fig6_ablation_umap"

FIG_DPI = 600
RANDOM_STATE = 42
BATCH_SIZE = 64


# =============================================================================
# 2. ABLATION SETTINGS — MATCH OLD DISPLAY STYLE
# =============================================================================

ABLATION_SPECS = [
    {
        "folder": "GenProt-DSM",
        "display": "GenProt-DSM",
        "color": "#1f77b4",  # blue
    },
    {
        "folder": "w_o_protT5",
        "display": "w/o ProtT5-XL",
        "color": "#d62728",  # red
    },
    {
        "folder": "w_o_GPN_MSA",
        "display": "w/o GPN-MSA",
        "color": "#ffbb78",  # light orange
    },
    {
        "folder": "w_o_BiLSTM",
        "display": "w/o BiLSTM",
        "color": "#ff7f0e",  # orange
    },
    {
        "folder": "w_o_GRU",
        "display": "w/o GRU",
        "color": "#2ca02c",  # green
    },
    {
        "folder": "w_o_BiLSTM_GRU",
        "display": "w/o BiLSTM+GRU",
        "color": "#9467bd",  # purple
    },
    {
        "folder": "w_o_gate_fusion_avg",
        "display": "w/o gated fusion",
        "color": "#98df8a",  # light green
    },
]

# =============================================================================
# 3. STYLE
# =============================================================================

PLOT_FONT_SIZE = 9

matplotlib.rcParams.update({
    "font.size": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_FONT_SIZE,
    "axes.labelsize": PLOT_FONT_SIZE,
    "xtick.labelsize": PLOT_FONT_SIZE,
    "ytick.labelsize": PLOT_FONT_SIZE,
    "legend.fontsize": PLOT_FONT_SIZE,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# =============================================================================
# 4. GENERAL UTILITIES
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_csv_smart(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
        )

    df.columns = [
        str(c)
        .lstrip("\ufeff")
        .strip()
        for c in df.columns
    ]

    return df


def first_existing_file(
    candidates: List[str],
    label: str,
) -> str:
    for p in candidates:
        if os.path.isfile(p):
            print(f"[FOUND] {label}: {p}")
            return p

    raise FileNotFoundError(
        f"Could not find {label}.\nChecked:\n"
        + "\n".join(
            f"  - {p}"
            for p in candidates
        )
    )


def to_fraction(v: Any) -> float:
    x = float(v)

    if x > 1.0:
        x /= 100.0

    return x


def panel_label(
    ax,
    label: str,
    x: float = -0.10,
    y: float = 1.04,
):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


# =============================================================================
# 5. PANEL a — ABLATION SCATTER, OLD LAYOUT
# =============================================================================

def read_one_ablation(
    spec: Dict[str, str],
) -> Dict[str, Any]:

    json_path = os.path.join(
        ABLATION_ROOT,
        spec["folder"],
        "ckpt",
        "test_ensemble_summary.json",
    )

    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Ablation summary missing:\n{json_path}"
        )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    # CRITICAL: use root final_metrics only.
    if "final_metrics" not in obj:
        raise KeyError(
            f"'final_metrics' missing at JSON root:\n{json_path}"
        )

    fm = obj["final_metrics"]

    if "auc_roc" not in fm or "auc_pr" not in fm:
        raise KeyError(
            f"auc_roc/auc_pr missing in final_metrics:\n{json_path}"
        )

    return {
        "folder": spec["folder"],
        "display": spec["display"],
        "color": spec["color"],
        "AUROC": to_fraction(
            fm["auc_roc"]
        ),
        "AUPR": to_fraction(
            fm["auc_pr"]
        ),
        "metric_source": "root/final_metrics",
        "json_path": json_path,
    }


def load_ablation_table() -> pd.DataFrame:
    rows = [
        read_one_ablation(spec)
        for spec in ABLATION_SPECS
    ]

    df = pd.DataFrame(rows)

    if not df[
        "metric_source"
    ].eq(
        "root/final_metrics"
    ).all():
        raise RuntimeError(
            "Ablation metric-source audit failed."
        )

    return df


def plot_ablation_scatter(
    ax,
    df: pd.DataFrame,
):
    for _, row in df.iterrows():
        ax.scatter(
            float(row["AUROC"]),
            float(row["AUPR"]),
            s=70,
            alpha=0.90,
            c=[
                row["color"]
            ],
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=row["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=8,
            label=row["display"],
        )
        for _, row in df.iterrows()
    ]

    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        fontsize=8,
    )

    x = df[
        "AUROC"
    ].to_numpy(
        dtype=float
    )

    y = df[
        "AUPR"
    ].to_numpy(
        dtype=float
    )

    x_span = max(
        float(
            np.max(x)
            - np.min(x)
        ),
        0.005,
    )

    y_span = max(
        float(
            np.max(y)
            - np.min(y)
        ),
        0.005,
    )

    ax.set_xlim(
        max(
            0.0,
            float(
                np.min(x)
            )
            - 0.10
            * x_span,
        ),
        min(
            1.0,
            float(
                np.max(x)
            )
            + 0.10
            * x_span,
        ),
    )

    ax.set_ylim(
        max(
            0.0,
            float(
                np.min(y)
            )
            - 0.10
            * y_span,
        ),
        min(
            1.0,
            float(
                np.max(y)
            )
            + 0.10
            * y_span,
        ),
    )

    ax.set_xlabel(
        "AUROC"
    )

    ax.set_ylabel(
        "AUPR"
    )

    ax.set_title(
        "Ablation study (AUROC vs AUPR)"
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.6,
    )


# =============================================================================
# 6. MINIMAL FINAL-MODEL DEFINITIONS FOR C CHECKPOINT
# =============================================================================

def pseudo_pos(
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
) -> torch.Tensor:

    delta = (
        b_raw
        - a_raw
    )

    norm = torch.norm(
        delta,
        dim=-1,
    )

    return torch.argmax(
        norm,
        dim=1,
    )


class ModalityDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(
        self,
        gpn: torch.Tensor,
        t5: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if (
            not self.training
            or self.p <= 0
        ):
            return (
                gpn,
                t5,
            )

        B = gpn.size(0)

        u = torch.rand(
            (
                B,
                1,
            ),
            device=gpn.device,
            dtype=gpn.dtype,
        )

        drop_gpn = (
            u
            < (
                self.p
                / 2
            )
        ).float()

        drop_t5 = (
            (
                u
                >= (
                    self.p
                    / 2
                )
            )
            & (
                u
                < self.p
            )
        ).float()

        return (
            gpn
            * (
                1.0
                - drop_gpn
            ),
            t5
            * (
                1.0
                - drop_t5
            ),
        )


class BiLSTMEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d = int(
            cfg.d_model
        )

        h = d // 2

        if 2 * h != d:
            raise ValueError(
                "d_model must be even."
            )

        n_layers = int(
            getattr(
                cfg,
                "bilstm_layers",
                2,
            )
        )

        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=h,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                float(
                    cfg.dropout
                )
                if n_layers > 1
                else 0.0
            ),
        )

        self.ln = nn.LayerNorm(
            d
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        y, _ = self.lstm(
            x
        )

        return self.ln(
            y
        )


class GRUEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d = int(
            cfg.d_model
        )

        n_layers = int(
            getattr(
                cfg,
                "bigru_layers",
                2,
            )
        )

        self.gru = nn.GRU(
            input_size=d,
            hidden_size=d,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=(
                float(
                    cfg.dropout
                )
                if n_layers > 1
                else 0.0
            ),
        )

        self.ln = nn.LayerNorm(
            d
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        y, _ = self.gru(
            x
        )

        return self.ln(
            y
        )


class HeadProj(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d_in = int(
            cfg.d_model
        )

        d_out = int(
            getattr(
                cfg,
                "branch_d",
                cfg.fused_d,
            )
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                d_in,
                d_in,
            ),
            nn.GELU(),
            nn.Dropout(
                float(
                    cfg.dropout
                )
            ),
            nn.Linear(
                d_in,
                d_out,
            ),
        )

        self.ln = nn.LayerNorm(
            d_out
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.ln(
            self.mlp(
                x
            )
        )


class Scheme2PairEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        seq_len: int,
        cfg,
        backbone: str,
    ):
        super().__init__()

        self.seq_len = int(
            seq_len
        )

        self.cfg = cfg

        self.proj = nn.Linear(
            int(
                d_in
            ),
            int(
                cfg.d_model
            ),
        )

        self.seg_emb = nn.Embedding(
            2,
            int(
                cfg.d_model
            ),
        )

        self.abs_pos = nn.Embedding(
            2
            * self.seq_len,
            int(
                cfg.d_model
            ),
        )

        self.rel_max = int(
            cfg.relpos_max
        )

        self.rel_emb = nn.Embedding(
            2
            * self.rel_max
            + 1,
            int(
                cfg.d_model
            ),
        )

        self.site_emb = nn.Parameter(
            torch.zeros(
                1,
                1,
                int(
                    cfg.d_model
                ),
            )
        )

        b = str(
            backbone
        ).lower()

        if b == "bilstm":
            self.backbone = BiLSTMEncoder(
                cfg
            )
        elif b == "gru":
            self.backbone = GRUEncoder(
                cfg
            )
        else:
            raise ValueError(
                "This Fig. 6 script expects the final model:\n"
                "ProtT5=BiLSTM and GPN-MSA=GRU.\n"
                f"Got backbone={backbone}"
            )

        self.head = HeadProj(
            cfg
        )

    def _build_tokens(
        self,
        a_raw: torch.Tensor,
        b_raw: torch.Tensor,
    ) -> torch.Tensor:

        B, L, _ = a_raw.shape

        if L != self.seq_len:
            raise ValueError(
                f"Sequence length mismatch: {L} vs {self.seq_len}"
            )

        pos_idx = pseudo_pos(
            a_raw,
            b_raw,
        )

        a = self.proj(
            a_raw
        )

        b = self.proj(
            b_raw
        )

        base = torch.arange(
            L,
            device=a.device,
        ).unsqueeze(
            0
        ).expand(
            B,
            -1,
        )

        rel = (
            base
            - pos_idx.unsqueeze(
                1
            )
        )

        rel = torch.clamp(
            rel,
            -self.rel_max,
            self.rel_max,
        ) + self.rel_max

        rel_emb = self.rel_emb(
            rel
        )

        mask = torch.zeros(
            (
                B,
                L,
                1,
            ),
            device=a.device,
            dtype=a.dtype,
        )

        mask[
            torch.arange(
                B,
                device=a.device,
            ),
            pos_idx,
            0,
        ] = 1.0

        a = (
            a
            + rel_emb
            + mask
            * self.site_emb
        )

        b = (
            b
            + rel_emb
            + mask
            * self.site_emb
        )

        x = torch.cat(
            [
                a,
                b,
            ],
            dim=1,
        )

        seg = torch.cat(
            [
                torch.zeros(
                    (
                        B,
                        L,
                    ),
                    dtype=torch.long,
                    device=x.device,
                ),
                torch.ones(
                    (
                        B,
                        L,
                    ),
                    dtype=torch.long,
                    device=x.device,
                ),
            ],
            dim=1,
        )

        x = (
            x
            + self.seg_emb(
                seg
            )
        )

        pos = torch.arange(
            0,
            2 * L,
            device=x.device,
        ).unsqueeze(
            0
        ).expand(
            B,
            -1,
        )

        x = (
            x
            + self.abs_pos(
                pos
            )
        )

        x = self.backbone(
            x
        )

        return x

    def forward(
        self,
        a_raw: torch.Tensor,
        b_raw: torch.Tensor,
    ) -> torch.Tensor:

        x = self._build_tokens(
            a_raw,
            b_raw,
        )

        pooled = x.mean(
            dim=1
        )

        return self.head(
            pooled
        )


class DualBranchFusionModel(nn.Module):
    def __init__(
        self,
        t5_enc: nn.Module,
        gpn_enc: nn.Module,
        cfg,
    ):
        super().__init__()

        self.t5_enc = t5_enc
        self.gpn_enc = gpn_enc

        self.moddrop = (
            ModalityDropout(
                float(
                    cfg.moddrop_p
                )
            )
            if float(
                getattr(
                    cfg,
                    "moddrop_p",
                    0.0,
                )
            ) > 0
            else None
        )

        branch_d = int(
            getattr(
                cfg,
                "branch_d",
                cfg.fused_d,
            )
        )

        fused_d = int(
            cfg.fused_d
        )

        self.gate_ln_g = nn.LayerNorm(
            branch_d
        )

        self.gate_ln_t = nn.LayerNorm(
            branch_d
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(
                2
                * branch_d,
                int(
                    cfg.gate_hidden
                ),
            ),
            nn.GELU(),
            nn.Dropout(
                float(
                    cfg.dropout
                )
            ),
            nn.Linear(
                int(
                    cfg.gate_hidden
                ),
                1,
            ),
        )

        self._gate_tau = float(
            getattr(
                cfg,
                "gate_tau_start",
                4.0,
            )
        )

        if fused_d == branch_d:
            self.fuse_proj = nn.Identity()
        else:
            self.fuse_proj = nn.Sequential(
                nn.Linear(
                    branch_d,
                    fused_d,
                ),
                nn.GELU(),
                nn.Dropout(
                    float(
                        cfg.dropout
                    )
                ),
            )

        self.fuse_ln = nn.LayerNorm(
            fused_d
        )

        self.head_fused = nn.Linear(
            fused_d,
            1,
        )

        self.head_t5 = (
            nn.Linear(
                branch_d,
                1,
            )
            if float(
                getattr(
                    cfg,
                    "aux_t5_w",
                    0.0,
                )
            ) > 0
            else None
        )

        self.head_gpn = (
            nn.Linear(
                branch_d,
                1,
            )
            if float(
                getattr(
                    cfg,
                    "aux_gpn_w",
                    0.0,
                )
            ) > 0
            else None
        )

    def set_gate_tau(
        self,
        tau: float,
    ):
        self._gate_tau = float(
            tau
        )

    def _gate_fuse(
        self,
        g_in: torch.Tensor,
        t_in: torch.Tensor,
    ) -> torch.Tensor:

        tau = max(
            self._gate_tau,
            1e-6,
        )

        g_gate = self.gate_ln_g(
            g_in
        )

        t_gate = self.gate_ln_t(
            t_in
        )

        xcat = torch.cat(
            [
                g_gate,
                t_gate,
            ],
            dim=1,
        )

        alpha = torch.sigmoid(
            self.gate_mlp(
                xcat
            )
            / tau
        )

        return (
            alpha
            * g_in
            + (
                1.0
                - alpha
            )
            * t_in
        )

    def forward(
        self,
        t5_wt,
        t5_mut,
        gpn_ref,
        gpn_alt,
    ):
        t5_vec = self.t5_enc(
            t5_wt,
            t5_mut,
        )

        gpn_vec = self.gpn_enc(
            gpn_ref,
            gpn_alt,
        )

        g_in = gpn_vec
        t_in = t5_vec

        if self.moddrop is not None:
            g_in, t_in = self.moddrop(
                g_in,
                t_in,
            )

        fused = self._gate_fuse(
            g_in,
            t_in,
        )

        fused = self.fuse_proj(
            fused
        )

        fused = self.fuse_ln(
            fused
        )

        logits_fused = self.head_fused(
            fused
        )

        lt5 = (
            self.head_t5(
                t5_vec
            )
            if self.head_t5 is not None
            else None
        )

        lgpn = (
            self.head_gpn(
                gpn_vec
            )
            if self.head_gpn is not None
            else None
        )

        return (
            fused,
            logits_fused,
            lt5,
            lgpn,
        )


# =============================================================================
# 7. REPRESENTATION EXTRACTION — REPRODUCE OLD LOGIC
# =============================================================================

def autodetect_folds(
    model_dir: str,
) -> List[int]:

    root = os.path.join(
        model_dir,
        "fold_artifacts",
    )

    if not os.path.isdir(
        root
    ):
        return []

    folds = []

    for name in os.listdir(
        root
    ):
        m = re.fullmatch(
            r"fold(\d+)",
            name,
        )

        if m:
            folds.append(
                int(
                    m.group(1)
                )
            )

    return sorted(
        folds
    )


def resolve_full_model_dir() -> str:
    for d in [
        FULL_MODEL_DIR,
        FORMAL_MODEL_DIR,
    ]:
        if autodetect_folds(
            d
        ):
            print(
                f"[FOUND] full model directory: {d}"
            )
            return d

    raise FileNotFoundError(
        "Could not find fold_artifacts for the full model in either:\n"
        f"  {FULL_MODEL_DIR}\n"
        f"  {FORMAL_MODEL_DIR}"
    )


def resolve_memmap_dir(
    ckpt_cfg: Dict[str, Any],
) -> str:

    p = ckpt_cfg.get(
        "memmap_dir",
        None,
    )

    if (
        p
        and os.path.isdir(
            p
        )
    ):
        return p

    if os.path.isdir(
        MEMMAP_DIR_FALLBACK
    ):
        return MEMMAP_DIR_FALLBACK

    raise FileNotFoundError(
        "Raw test memmap directory not found.\n"
        f"checkpoint cfg memmap_dir = {p}\n"
        f"fallback = {MEMMAP_DIR_FALLBACK}"
    )


def load_test_memmaps(
    memmap_dir: str,
):
    paths = {
        "t5_wt": os.path.join(
            memmap_dir,
            "test_t5_wt.npy",
        ),
        "t5_mut": os.path.join(
            memmap_dir,
            "test_t5_mut.npy",
        ),
        "gpn_ref": os.path.join(
            memmap_dir,
            "test_gpn_ref.npy",
        ),
        "gpn_alt": os.path.join(
            memmap_dir,
            "test_gpn_alt.npy",
        ),
    }

    for p in paths.values():
        if not os.path.isfile(
            p
        ):
            raise FileNotFoundError(
                p
            )

    t5_wt = np.load(
        paths["t5_wt"],
        mmap_mode="r",
    )

    t5_mut = np.load(
        paths["t5_mut"],
        mmap_mode="r",
    )

    gpn_ref = np.load(
        paths["gpn_ref"],
        mmap_mode="r",
    )

    gpn_alt = np.load(
        paths["gpn_alt"],
        mmap_mode="r",
    )

    if t5_wt.shape != t5_mut.shape:
        raise ValueError(
            "ProtT5 WT/MUT shape mismatch."
        )

    if gpn_ref.shape != gpn_alt.shape:
        raise ValueError(
            "GPN-MSA REF/ALT shape mismatch."
        )

    if (
        t5_wt.shape[0]
        != gpn_ref.shape[0]
    ):
        raise ValueError(
            "ProtT5/GPN-MSA N mismatch."
        )

    if (
        t5_wt.shape[1]
        != gpn_ref.shape[1]
    ):
        raise ValueError(
            "ProtT5/GPN-MSA sequence-length mismatch."
        )

    return (
        t5_wt,
        t5_mut,
        gpn_ref,
        gpn_alt,
    )


def build_c_model(
    ckpt: Dict[str, Any],
    t5_dim: int,
    gpn_dim: int,
    seq_len: int,
):
    if "cfg" not in ckpt:
        raise KeyError(
            "C checkpoint missing cfg."
        )

    cfg = SimpleNamespace(
        **dict(
            ckpt["cfg"]
        )
    )

    if (
        str(
            cfg.t5_backbone
        ).lower()
        != "bilstm"
    ):
        raise ValueError(
            f"Expected t5_backbone=bilstm, got {cfg.t5_backbone}"
        )

    if (
        str(
            cfg.gpn_backbone
        ).lower()
        != "gru"
    ):
        raise ValueError(
            f"Expected gpn_backbone=gru, got {cfg.gpn_backbone}"
        )

    t5_enc = Scheme2PairEncoder(
        d_in=t5_dim,
        seq_len=seq_len,
        cfg=cfg,
        backbone="bilstm",
    )

    gpn_enc = Scheme2PairEncoder(
        d_in=gpn_dim,
        seq_len=seq_len,
        cfg=cfg,
        backbone="gru",
    )

    model = DualBranchFusionModel(
        t5_enc,
        gpn_enc,
        cfg,
    )

    state = ckpt.get(
        "model_state_dict",
        ckpt,
    )

    state = {
        re.sub(
            r"^module\.",
            "",
            k,
        ): v
        for k, v in state.items()
    }

    model.load_state_dict(
        state,
        strict=True,
    )

    # tau is not in state_dict; old interpretability code restored end tau.
    model.set_gate_tau(
        float(
            getattr(
                cfg,
                "gate_tau_end",
                2.0,
            )
        )
    )

    model.eval()

    return (
        model,
        cfg,
    )


@torch.no_grad()
def extract_one_fold(
    c_model: nn.Module,
    t5_wt,
    t5_mut,
    gpn_ref,
    gpn_alt,
    device: torch.device,
) -> Dict[str, np.ndarray]:

    n = int(
        t5_wt.shape[0]
    )

    prot_vec_all = []
    dna_vec_all = []
    prot_input_all = []
    dna_input_all = []

    for start in range(
        0,
        n,
        BATCH_SIZE,
    ):
        end = min(
            start
            + BATCH_SIZE,
            n,
        )

        bt5w = torch.from_numpy(
            np.asarray(
                t5_wt[
                    start:end
                ]
            ).copy()
        ).float().to(
            device
        )

        bt5m = torch.from_numpy(
            np.asarray(
                t5_mut[
                    start:end
                ]
            ).copy()
        ).float().to(
            device
        )

        bgref = torch.from_numpy(
            np.asarray(
                gpn_ref[
                    start:end
                ]
            ).copy()
        ).float().to(
            device
        )

        bgalt = torch.from_numpy(
            np.asarray(
                gpn_alt[
                    start:end
                ]
            ).copy()
        ).float().to(
            device
        )

        prot_vec = c_model.t5_enc(
            bt5w,
            bt5m,
        )

        dna_vec = c_model.gpn_enc(
            bgref,
            bgalt,
        )

        # OLD CODE:
        # mean pooled mutation-minus-reference delta over sequence
        prot_input_delta = torch.nanmean(
            bt5m
            - bt5w,
            dim=1,
        )

        dna_input_delta = torch.nanmean(
            bgalt
            - bgref,
            dim=1,
        )

        prot_vec_all.append(
            prot_vec
            .cpu()
            .numpy()
        )

        dna_vec_all.append(
            dna_vec
            .cpu()
            .numpy()
        )

        prot_input_all.append(
            prot_input_delta
            .cpu()
            .numpy()
        )

        dna_input_all.append(
            dna_input_delta
            .cpu()
            .numpy()
        )

    return {
        "prot_vec": np.concatenate(
            prot_vec_all,
            axis=0,
        ),
        "dna_vec": np.concatenate(
            dna_vec_all,
            axis=0,
        ),
        "prot_input_delta": np.concatenate(
            prot_input_all,
            axis=0,
        ),
        "dna_input_delta": np.concatenate(
            dna_input_all,
            axis=0,
        ),
    }


def resolve_prediction_csv(
    model_dir: str,
) -> str:

    return first_existing_file(
        [
            os.path.join(
                model_dir,
                "pred",
                "pred_test_ensemble.csv",
            ),
            os.path.join(
                FORMAL_MODEL_DIR,
                "pred",
                "pred_test_ensemble.csv",
            ),
        ],
        "pred_test_ensemble.csv",
    )


def resolve_test_metadata_csv(
    first_ckpt_cfg: Dict[str, Any],
) -> str:

    candidates = []

    ckpt_test_csv = first_ckpt_cfg.get(
        "test_csv",
        None,
    )

    if ckpt_test_csv:
        candidates.append(
            ckpt_test_csv
        )

    candidates.append(
        TEST_META_FALLBACK
    )

    return first_existing_file(
        candidates,
        "test metadata CSV",
    )


def get_key_cols(
    df: pd.DataFrame,
) -> List[str]:

    wanted = [
        "Chrom",
        "Position",
        "Reference",
        "Alternate",
    ]

    low = {
        str(c).lower(): str(c)
        for c in df.columns
    }

    out = []

    for name in wanted:
        if name in df.columns:
            out.append(
                name
            )
        elif name.lower() in low:
            out.append(
                low[
                    name.lower()
                ]
            )
        else:
            return []

    return out


def normalize_variant_key_frame(
    df: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:

    out = pd.DataFrame(
        index=df.index
    )

    out["Chrom"] = (
        df[
            cols[0]
        ]
        .astype(str)
        .str.strip()
        .str.replace(
            r"^chr",
            "",
            regex=True,
            case=False,
        )
    )

    # Normalize positions so "123" and "123.0" match.
    pos = pd.to_numeric(
        df[
            cols[1]
        ],
        errors="coerce",
    )

    if pos.isna().any():
        raise ValueError(
            "Position contains non-numeric values."
        )

    out["Position"] = (
        pos
        .round()
        .astype(
            "Int64"
        )
        .astype(str)
    )

    out["Reference"] = (
        df[
            cols[2]
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out["Alternate"] = (
        df[
            cols[3]
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return out


def build_variant_keys(
    df: pd.DataFrame,
) -> pd.Series:

    cols = get_key_cols(
        df
    )

    if not cols:
        raise KeyError(
            "Required variant keys not found: "
            "Chrom, Position, Reference, Alternate."
        )

    norm = normalize_variant_key_frame(
        df,
        cols,
    )

    return norm.astype(
        str
    ).agg(
        "|".join,
        axis=1,
    )


FUNCTIONAL_COLUMNS_PREFERRED = [
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
]


def build_functional_annotation_matrix(
    test_meta_csv: str,
    tool_score_csv: str,
) -> Tuple[
    np.ndarray,
    List[str],
]:

    meta = read_csv_smart(
        test_meta_csv
    )

    tool = read_csv_smart(
        tool_score_csv
    )

    meta_key = build_variant_keys(
        meta
    )

    tool_key = build_variant_keys(
        tool
    )

    if tool_key.duplicated().any():
        keep = (
            ~tool_key.duplicated(
                keep="first"
            )
        )

        tool = tool.loc[
            keep
        ].copy()

        tool_key = tool_key.loc[
            keep
        ].copy()

    tool.index = tool_key.values

    selected = [
        c
        for c in FUNCTIONAL_COLUMNS_PREFERRED
        if c in tool.columns
    ]

    if not selected:
        # Fallback: numeric tool columns, but EXPLICITLY exclude model output,
        # label, AF, and variant keys.
        excluded = {
            "label",
            "AF",
            "GenProt-DSM",
            "Chrom",
            "Position",
            "Reference",
            "Alternate",
        }

        selected = [
            c
            for c in tool.columns
            if c not in excluded
        ]

    sub = tool.reindex(
        meta_key.values
    )[
        selected
    ].copy()

    for c in selected:
        sub[c] = pd.to_numeric(
            sub[c],
            errors="coerce",
        )

    sub = sub.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    med = (
        sub.median(
            axis=0,
            skipna=True,
        )
        .fillna(
            0.0
        )
    )

    sub = (
        sub.fillna(
            med
        )
        .fillna(
            0.0
        )
    )

    print(
        "[Functional Annotation] columns:"
    )

    print(
        "  "
        + ", ".join(
            selected
        )
    )

    return (
        sub.to_numpy(
            dtype=np.float32
        ),
        selected,
    )


def load_or_extract_oldlogic_representations(
    force_reextract: bool = False,
):
    ensure_dir(
        CACHE_DIR
    )

    paths = {
        "prot_input": os.path.join(
            CACHE_DIR,
            "prot_input_delta_mean.npy",
        ),
        "dna_input": os.path.join(
            CACHE_DIR,
            "dna_input_delta_mean.npy",
        ),
        "prot_vec": os.path.join(
            CACHE_DIR,
            "protein_encoder_output_mean.npy",
        ),
        "dna_vec": os.path.join(
            CACHE_DIR,
            "dna_encoder_output_mean.npy",
        ),
        "fused_vec": os.path.join(
            CACHE_DIR,
            "gate_fusion_output_mean.npy",
        ),
        "metadata": os.path.join(
            CACHE_DIR,
            "umap_metadata.csv",
        ),
    }

    cached = all(
        os.path.isfile(
            p
        )
        for p in paths.values()
    )

    if (
        cached
        and not force_reextract
    ):
        print(
            "[CACHE] Reusing old-layout representation cache."
        )

        return {
            "prot_input": np.load(
                paths["prot_input"]
            ),
            "dna_input": np.load(
                paths["dna_input"]
            ),
            "prot_vec": np.load(
                paths["prot_vec"]
            ),
            "dna_vec": np.load(
                paths["dna_vec"]
            ),
            "fused_vec": np.load(
                paths["fused_vec"]
            ),
            "metadata": pd.read_csv(
                paths["metadata"]
            ),
        }

    model_dir = resolve_full_model_dir()

    folds = autodetect_folds(
        model_dir
    )

    if not folds:
        raise RuntimeError(
            "No folds detected."
        )

    print(
        f"[FOLDS] {folds}"
    )

    pred_csv = resolve_prediction_csv(
        model_dir
    )

    pred = read_csv_smart(
        pred_csv
    )

    if "prob_ensemble" not in pred.columns:
        raise KeyError(
            "pred_test_ensemble.csv missing prob_ensemble."
        )

    if "sample_index" in pred.columns:
        pred = (
            pred.sort_values(
                "sample_index"
            )
            .reset_index(
                drop=True
            )
        )
    else:
        pred = pred.reset_index(
            drop=True
        )

        pred.insert(
            0,
            "sample_index",
            np.arange(
                len(pred)
            ),
        )

    prot_vec_folds = []
    dna_vec_folds = []
    prot_input_folds = []
    dna_input_folds = []
    fused_folds = []

    first_cfg_dict = None
    test_memmaps = None

    device = torch.device(
        "cpu"
    )

    for fold in folds:
        ckpt_path = os.path.join(
            model_dir,
            "fold_artifacts",
            f"fold{fold}",
            f"c_model_fold{fold}.pth",
        )

        if not os.path.isfile(
            ckpt_path
        ):
            raise FileNotFoundError(
                ckpt_path
            )

        print(
            f"[Fold {fold}] loading C checkpoint..."
        )

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        cfg_dict = dict(
            ckpt.get(
                "cfg",
                {},
            )
        )

        if first_cfg_dict is None:
            first_cfg_dict = cfg_dict

        memmap_dir = resolve_memmap_dir(
            cfg_dict
        )

        if test_memmaps is None:
            test_memmaps = load_test_memmaps(
                memmap_dir
            )

        (
            t5_wt,
            t5_mut,
            gpn_ref,
            gpn_alt,
        ) = test_memmaps

        n_test = int(
            t5_wt.shape[0]
        )

        if len(
            pred
        ) != n_test:
            raise ValueError(
                "Prediction N != model test N:\n"
                f"{len(pred)} != {n_test}"
            )

        c_model, _cfg = build_c_model(
            ckpt=ckpt,
            t5_dim=int(
                t5_wt.shape[2]
            ),
            gpn_dim=int(
                gpn_ref.shape[2]
            ),
            seq_len=int(
                t5_wt.shape[1]
            ),
        )

        c_model.to(
            device
        )

        pack = extract_one_fold(
            c_model=c_model,
            t5_wt=t5_wt,
            t5_mut=t5_mut,
            gpn_ref=gpn_ref,
            gpn_alt=gpn_alt,
            device=device,
        )

        prot_vec_folds.append(
            pack[
                "prot_vec"
            ]
        )

        dna_vec_folds.append(
            pack[
                "dna_vec"
            ]
        )

        prot_input_folds.append(
            pack[
                "prot_input_delta"
            ]
        )

        dna_input_folds.append(
            pack[
                "dna_input_delta"
            ]
        )

        fused_path = first_existing_file(
            [
                os.path.join(
                    model_dir,
                    "fused_memmap",
                    f"fold{fold}_Xte.npy",
                ),
                os.path.join(
                    FORMAL_MODEL_DIR,
                    "fused_memmap",
                    f"fold{fold}_Xte.npy",
                ),
            ],
            f"fold{fold} fused Xte",
        )

        fused_fold = np.asarray(
            np.load(
                fused_path,
                mmap_mode="r",
            ),
            dtype=np.float32,
        )

        if fused_fold.shape[0] != n_test:
            raise ValueError(
                f"Fold {fold} fused N mismatch."
            )

        fused_folds.append(
            fused_fold
        )

        del c_model

    # EXACT old-code logic: mean across folds.
    prot_vec = np.mean(
        np.stack(
            prot_vec_folds,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )

    dna_vec = np.mean(
        np.stack(
            dna_vec_folds,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )

    prot_input = np.mean(
        np.stack(
            prot_input_folds,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )

    dna_input = np.mean(
        np.stack(
            dna_input_folds,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )

    fused_vec = np.mean(
        np.stack(
            fused_folds,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )

    np.save(
        paths["prot_input"],
        prot_input,
    )

    np.save(
        paths["dna_input"],
        dna_input,
    )

    np.save(
        paths["prot_vec"],
        prot_vec,
    )

    np.save(
        paths["dna_vec"],
        dna_vec,
    )

    np.save(
        paths["fused_vec"],
        fused_vec,
    )

    pred.to_csv(
        paths["metadata"],
        index=False,
        encoding="utf-8-sig",
    )

    test_meta_csv = resolve_test_metadata_csv(
        first_cfg_dict
        if first_cfg_dict is not None
        else {}
    )

    with open(
        os.path.join(
            CACHE_DIR,
            "source_manifest.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "model_dir": model_dir,
                "folds": folds,
                "prediction_csv": pred_csv,
                "test_metadata_csv": test_meta_csv,
                "representation_rule": "mean_across_folds_to_match_previous_code",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "prot_input": prot_input,
        "dna_input": dna_input,
        "prot_vec": prot_vec,
        "dna_vec": dna_vec,
        "fused_vec": fused_vec,
        "metadata": pred,
        "first_cfg_dict": first_cfg_dict,
        "test_meta_csv": test_meta_csv,
    }


# =============================================================================
# 8. UMAP — MATCH OLD SIX-PANEL LOGIC
# =============================================================================

def reduce_umap(
    X: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:

    X = np.asarray(
        X,
        dtype=np.float32,
    ).copy()

    X[
        ~np.isfinite(
            X
        )
    ] = np.nan

    col_med = np.nanmedian(
        X,
        axis=0,
    )

    col_med = np.where(
        np.isfinite(
            col_med
        ),
        col_med,
        0.0,
    )

    inds = np.where(
        np.isnan(
            X
        )
    )

    if len(
        inds[0]
    ) > 0:
        X[
            inds
        ] = col_med[
            inds[1]
        ]

    Xs = StandardScaler().fit_transform(
        X
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )

    return reducer.fit_transform(
        Xs
    )


def make_prob_scatter(
    ax,
    z2: np.ndarray,
    prob: np.ndarray,
    title: str,
):
    sc = ax.scatter(
        z2[
            :,
            0,
        ],
        z2[
            :,
            1,
        ],
        s=10,
        alpha=0.85,
        c=prob,
        vmin=0.0,
        vmax=1.0,
        cmap="coolwarm",
        edgecolors="none",
        rasterized=True,
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "UMAP-1"
    )

    ax.set_ylabel(
        "UMAP-2"
    )

    return sc


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    ensure_dir(
        OUTPUT_DIR
    )

    ensure_dir(
        CACHE_DIR
    )

    print(
        "=" * 100
    )

    print(
        "FIG. 6 — OLD LAYOUT REPRODUCTION"
    )

    print(
        "=" * 100
    )

    # -------------------------------------------------------------------------
    # (a) Ablation
    # -------------------------------------------------------------------------

    ablation_df = load_ablation_table()

    print(
        "\n[Ablation — root final_metrics]"
    )

    print(
        ablation_df[
            [
                "display",
                "AUROC",
                "AUPR",
                "metric_source",
            ]
        ].to_string(
            index=False
        )
    )

    ablation_csv = os.path.join(
        OUTPUT_DIR,
        "Fig6a_ablation_metrics.csv",
    )

    ablation_df.to_csv(
        ablation_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # (b) Representations / inputs
    # -------------------------------------------------------------------------

    rep = load_or_extract_oldlogic_representations(
        force_reextract=False
    )

    meta = rep[
        "metadata"
    ]

    prob = pd.to_numeric(
        meta[
            "prob_ensemble"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    # Resolve test metadata.
    if "test_meta_csv" in rep:
        test_meta_csv = rep[
            "test_meta_csv"
        ]
    else:
        manifest_path = os.path.join(
            CACHE_DIR,
            "source_manifest.json",
        )

        if os.path.isfile(
            manifest_path
        ):
            with open(
                manifest_path,
                "r",
                encoding="utf-8",
            ) as f:
                manifest = json.load(
                    f
                )

            test_meta_csv = manifest[
                "test_metadata_csv"
            ]
        else:
            test_meta_csv = TEST_META_FALLBACK

    func_X, func_cols = build_functional_annotation_matrix(
        test_meta_csv=test_meta_csv,
        tool_score_csv=TOOL_SCORE_CSV,
    )

    n = len(
        prob
    )

    matrices = [
        (
            "ProtT5 Input",
            rep[
                "prot_input"
            ],
        ),
        (
            "GPN-MSA Input",
            rep[
                "dna_input"
            ],
        ),
        (
            "Functional Annotation",
            func_X,
        ),
        (
            "Protein Encoder Output",
            rep[
                "prot_vec"
            ],
        ),
        (
            "DNA Encoder Output",
            rep[
                "dna_vec"
            ],
        ),
        (
            "Gate Fusion Output",
            rep[
                "fused_vec"
            ],
        ),
    ]

    for title, X in matrices:
        if X.shape[0] != n:
            raise ValueError(
                f"N mismatch for {title}: {X.shape[0]} vs {n}"
            )

        print(
            f"{title:24s} shape={X.shape}"
        )

    # Compute six UMAPs independently, exactly like old code.
    umap_results = []

    for title, X in matrices:
        print(
            f"[UMAP] {title}"
        )

        z2 = reduce_umap(
            X,
            random_state=RANDOM_STATE,
        )

        umap_results.append(
            (
                title,
                z2,
            )
        )

    # Save UMAP coordinates.
    coords = pd.DataFrame({
        "sample_index": (
            meta[
                "sample_index"
            ].to_numpy()
            if "sample_index" in meta.columns
            else np.arange(
                n
            )
        ),
        "prob_ensemble": prob,
    })

    for title, z2 in umap_results:
        prefix = (
            title
            .replace(
                " ",
                "_",
            )
            .replace(
                "-",
                "_",
            )
        )

        coords[
            f"{prefix}_UMAP1"
        ] = z2[
            :,
            0,
        ]

        coords[
            f"{prefix}_UMAP2"
        ] = z2[
            :,
            1,
        ]

    coords_csv = os.path.join(
        OUTPUT_DIR,
        "Fig6b_umap_coordinates.csv",
    )

    coords.to_csv(
        coords_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # COMBINED FIGURE — visually match the old example:
    # left = ablation scatter
    # right = 2x3 UMAP + one shared colorbar
    # -------------------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            18.6,
            7.0,
        ),
        dpi=FIG_DPI,
    )

    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[
            0.80,
            1.32,
        ],
        wspace=0.10,
    )

    # Panel a
    ax_a = fig.add_subplot(
        outer[
            0,
            0,
        ]
    )

    plot_ablation_scatter(
        ax_a,
        ablation_df,
    )

    panel_label(
        ax_a,
        "a",
        x=-0.11,
        y=1.07,
    )

    # Panel b container + 2x3 internal grid
    b_container = fig.add_subplot(
        outer[
            0,
            1,
        ]
    )

    b_container.axis(
        "off"
    )

    panel_label(
        b_container,
        "b",
        x=-0.035,
        y=1.07,
    )

    gs_b = outer[
        0,
        1,
    ].subgridspec(
        2,
        4,
        width_ratios=[
            1.0,
            1.0,
            1.0,
            0.055,
        ],
        wspace=0.45,
        hspace=0.28,
    )

    axes = [
        fig.add_subplot(
            gs_b[
                0,
                0,
            ]
        ),
        fig.add_subplot(
            gs_b[
                0,
                1,
            ]
        ),
        fig.add_subplot(
            gs_b[
                0,
                2,
            ]
        ),
        fig.add_subplot(
            gs_b[
                1,
                0,
            ]
        ),
        fig.add_subplot(
            gs_b[
                1,
                1,
            ]
        ),
        fig.add_subplot(
            gs_b[
                1,
                2,
            ]
        ),
    ]

    last_sc = None

    for ax, (
        title,
        z2,
    ) in zip(
        axes,
        umap_results,
    ):
        last_sc = make_prob_scatter(
            ax,
            z2,
            prob,
            title,
        )

    cax = fig.add_subplot(
        gs_b[
            :,
            3,
        ]
    )

    cbar = fig.colorbar(
        last_sc,
        cax=cax,
    )

    cbar.set_label(
        "Predicted pathogenicity probability"
    )

    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        top=0.955,
        bottom=0.095,
    )

    output_base = os.path.join(
        OUTPUT_DIR,
        OUTPUT_BASENAME,
    )

    png_path = (
        output_base
        + ".png"
    )

    pdf_path = (
        output_base
        + ".pdf"
    )

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
        f"PNG: {png_path}"
    )

    print(
        f"PDF: {pdf_path}"
    )

    print(
        f"Ablation audit: {ablation_csv}"
    )

    print(
        f"UMAP coordinates: {coords_csv}"
    )


if __name__ == "__main__":
    main()
