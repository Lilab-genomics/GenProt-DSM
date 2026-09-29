#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig. 7 — SHAP-based feature-level interpretation of GenProt-DSM

Panels
------
(a) Global SHAP importance
(b) SHAP beeswarm

This script follows the PREVIOUS manuscript code logic:

1. Load the five full-model C-stage checkpoints.
2. Extract, for the locked test set:
      - DNA encoder representation
      - Protein encoder representation
      - Gate alpha
3. Read the saved fused representation for each fold.
4. Average DNA / Protein / Fused / Gate representations across the five folds.
5. Construct the proxy feature matrix:
      [DNA | PROT | FUSED | GATE]
6. Fit an XGBoost proxy classifier to y_true.
7. Compute TreeSHAP values on the proxy model.
8. Keep GATE in the proxy model, but EXCLUDE GATE from the displayed SHAP plots,
   exactly as in the previous code.
9. Rename latent dimensions for the figure:
      DNA_i   -> GPN-MSA i
      PROT_i  -> ProtT5-XL i
      FUSED_i -> Fused i

IMPORTANT
---------
This is the same "proxy-SHAP on learned GenProt-DSM representations" analysis
used by the previous code. It is not direct SHAP on the original neural
network classifier.

Required inputs
---------------
A) Five C-stage checkpoints:
F:\\GenProt-DSM_Resubmit\\result\\pred\\ablation\\GenProt-DSM
  \\fold_artifacts\\fold1\\c_model_fold1.pth
  ...
  \\fold_artifacts\\fold5\\c_model_fold5.pth

B) Five saved fused test representations:
F:\\GenProt-DSM_Resubmit\\result\\pred\\ablation\\GenProt-DSM
  \\fused_memmap\\fold1_Xte.npy
  ...
  \\fused_memmap\\fold5_Xte.npy

C) Final ensemble prediction file:
F:\\GenProt-DSM_Resubmit\\result\\pred\\ablation\\GenProt-DSM
  \\pred\\pred_test_ensemble.csv
Required columns:
  sample_index
  y_true
  prob_ensemble

D) Test raw feature memmaps:
F:\\GenProt-DSM_Resubmit\\dataset\\data\\feature\\merged_memmap_raw
  test_t5_wt.npy
  test_t5_mut.npy
  test_gpn_ref.npy
  test_gpn_alt.npy

Outputs
-------
F:\\GenProt-DSM_Resubmit\\fig_Results\\Fig7_SHAP
  Fig7_SHAP.png
  Fig7_SHAP.pdf
  Fig7a_global_shap_importance.csv
  Fig7_proxy_audit.csv
  shap_values_display.npy
  shap_feature_matrix_display.npy
  shap_feature_names_display.txt
  representation_cache\\...

Requirements
------------
pip install numpy pandas torch matplotlib scikit-learn shap xgboost
"""

from __future__ import annotations

import json
import os
import re
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("once")


# =============================================================================
# 1. PATHS
# =============================================================================

MODEL_DIR = (
    r"F:\GenProt-DSM_Resubmit\result\pred"
    r"\ablation\GenProt-DSM"
)

# Per your current revised-project directory structure:
MEMMAP_DIR = (
    r"F:\GenProt-DSM_Resubmit\dataset\data"
    r"\feature\merged_memmap_raw"
)
LABEL_COL = "label"

PRED_ENSEMBLE_CSV = os.path.join(
    MODEL_DIR,
    "pred",
    "pred_test_ensemble.csv",
)

OUTPUT_DIR = (
    r"F:\GenProt-DSM_Resubmit\fig_Results"
    r"\Fig7_SHAP"
)

CACHE_DIR = os.path.join(
    OUTPUT_DIR,
    "representation_cache",
)

OUTPUT_BASENAME = "Fig7_SHAP"

FOLDS = [1, 2, 3, 4, 5]

BATCH_SIZE = 64
DEVICE = torch.device("cpu")

FIG_DPI = 600
PLOT_FONT_SIZE = 9
MAX_DISPLAY = 40
RANDOM_STATE = 42


# =============================================================================
# 2. GLOBAL PLOT STYLE
# =============================================================================

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
# 3. GENERAL UTILITIES
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def require_file(path: str, label: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def require_dir(path: str, label: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def panel_label(
    ax,
    label: str,
    x: float = -0.06,
    y: float = 1.01,
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
# 4. MINIMAL MODEL DEFINITIONS NEEDED TO LOAD THE C CHECKPOINTS
# =============================================================================

def pseudo_pos(
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
) -> torch.Tensor:
    delta = b_raw - a_raw
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
    ):
        if (
            not self.training
            or self.p <= 0
        ):
            return gpn, t5

        B = gpn.size(0)

        u = torch.rand(
            (B, 1),
            device=gpn.device,
            dtype=gpn.dtype,
        )

        drop_gpn = (
            u < (self.p / 2)
        ).float()

        drop_t5 = (
            (u >= (self.p / 2))
            & (u < self.p)
        ).float()

        return (
            gpn * (1.0 - drop_gpn),
            t5 * (1.0 - drop_t5),
        )


class BiLSTMEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d = int(cfg.d_model)
        h = d // 2

        if 2 * h != d:
            raise ValueError(
                "d_model must be even for BiLSTM."
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
                float(cfg.dropout)
                if n_layers > 1
                else 0.0
            ),
        )

        self.ln = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.ln(y)


class GRUEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d = int(cfg.d_model)

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
                float(cfg.dropout)
                if n_layers > 1
                else 0.0
            ),
        )

        self.ln = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.ln(y)


class HeadProj(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d_in = int(cfg.d_model)

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
                float(cfg.dropout)
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
            self.mlp(x)
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

        self.seq_len = int(seq_len)
        self.cfg = cfg

        self.proj = nn.Linear(
            int(d_in),
            int(cfg.d_model),
        )

        self.seg_emb = nn.Embedding(
            2,
            int(cfg.d_model),
        )

        self.abs_pos = nn.Embedding(
            2 * self.seq_len,
            int(cfg.d_model),
        )

        self.rel_max = int(
            cfg.relpos_max
        )

        self.rel_emb = nn.Embedding(
            2 * self.rel_max + 1,
            int(cfg.d_model),
        )

        self.site_emb = nn.Parameter(
            torch.zeros(
                1,
                1,
                int(cfg.d_model),
            )
        )

        b = str(backbone).lower()

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
                "Fig. 7 expects the final model configuration:\n"
                "ProtT5 = BiLSTM\n"
                "GPN-MSA = GRU\n"
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
            - pos_idx.unsqueeze(1)
        )

        rel = (
            torch.clamp(
                rel,
                -self.rel_max,
                self.rel_max,
            )
            + self.rel_max
        )

        rel_emb = self.rel_emb(
            rel
        )

        mask = torch.zeros(
            (B, L, 1),
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
            + mask * self.site_emb
        )

        b = (
            b
            + rel_emb
            + mask * self.site_emb
        )

        x = torch.cat(
            [a, b],
            dim=1,
        )

        seg = torch.cat(
            [
                torch.zeros(
                    (B, L),
                    dtype=torch.long,
                    device=x.device,
                ),
                torch.ones(
                    (B, L),
                    dtype=torch.long,
                    device=x.device,
                ),
            ],
            dim=1,
        )

        x = (
            x
            + self.seg_emb(seg)
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
            + self.abs_pos(pos)
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
                    getattr(
                        cfg,
                        "moddrop_p",
                        0.0,
                    )
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
                2 * branch_d,
                int(cfg.gate_hidden),
            ),
            nn.GELU(),
            nn.Dropout(
                float(cfg.dropout)
            ),
            nn.Linear(
                int(cfg.gate_hidden),
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
                    float(cfg.dropout)
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
            [g_gate, t_gate],
            dim=1,
        )

        alpha = torch.sigmoid(
            self.gate_mlp(xcat)
            / tau
        )

        return (
            alpha * g_in
            + (1.0 - alpha) * t_in
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
            self.head_t5(t5_vec)
            if self.head_t5 is not None
            else None
        )

        lgpn = (
            self.head_gpn(gpn_vec)
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
# 5. LOAD RAW TEST MEMMAPS
# =============================================================================

def load_test_memmaps():
    require_dir(
        MEMMAP_DIR,
        "MEMMAP_DIR",
    )

    paths = {
        "t5_wt": os.path.join(
            MEMMAP_DIR,
            "test_t5_wt.npy",
        ),
        "t5_mut": os.path.join(
            MEMMAP_DIR,
            "test_t5_mut.npy",
        ),
        "gpn_ref": os.path.join(
            MEMMAP_DIR,
            "test_gpn_ref.npy",
        ),
        "gpn_alt": os.path.join(
            MEMMAP_DIR,
            "test_gpn_alt.npy",
        ),
    }

    for name, path in paths.items():
        require_file(
            path,
            name,
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


# =============================================================================
# 6. BUILD C MODEL FROM CHECKPOINT
# =============================================================================

def build_c_model_from_ckpt(
    ckpt_path: str,
    t5_dim: int,
    gpn_dim: int,
    seq_len: int,
):
    require_file(
        ckpt_path,
        "C-stage checkpoint",
    )

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if "cfg" not in ckpt:
        raise KeyError(
            f"Checkpoint missing cfg:\n{ckpt_path}"
        )

    cfg = SimpleNamespace(
        **dict(
            ckpt["cfg"]
        )
    )

    t5_backbone = str(
        cfg.t5_backbone
    ).lower()

    gpn_backbone = str(
        cfg.gpn_backbone
    ).lower()

    if t5_backbone != "bilstm":
        raise ValueError(
            f"Expected ProtT5 backbone=bilstm, got {t5_backbone}"
        )

    if gpn_backbone != "gru":
        raise ValueError(
            f"Expected GPN-MSA backbone=gru, got {gpn_backbone}"
        )

    t5_enc = Scheme2PairEncoder(
        d_in=t5_dim,
        seq_len=seq_len,
        cfg=cfg,
        backbone=t5_backbone,
    )

    gpn_enc = Scheme2PairEncoder(
        d_in=gpn_dim,
        seq_len=seq_len,
        cfg=cfg,
        backbone=gpn_backbone,
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

    # Gate tau is not stored in state_dict.
    model.set_gate_tau(
        float(
            getattr(
                cfg,
                "gate_tau_end",
                2.0,
            )
        )
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model, cfg


# =============================================================================
# 7. EXTRACT DNA / PROTEIN / GATE FOR ONE FOLD
# =============================================================================

@torch.no_grad()
def extract_one_fold(
    model: DualBranchFusionModel,
    t5_wt,
    t5_mut,
    gpn_ref,
    gpn_alt,
):
    n = int(
        t5_wt.shape[0]
    )

    dna_all = []
    prot_all = []
    gate_all = []

    for start in range(
        0,
        n,
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            n,
        )

        bt5w = torch.from_numpy(
            np.asarray(
                t5_wt[start:end]
            ).copy()
        ).float().to(
            DEVICE
        )

        bt5m = torch.from_numpy(
            np.asarray(
                t5_mut[start:end]
            ).copy()
        ).float().to(
            DEVICE
        )

        bgref = torch.from_numpy(
            np.asarray(
                gpn_ref[start:end]
            ).copy()
        ).float().to(
            DEVICE
        )

        bgalt = torch.from_numpy(
            np.asarray(
                gpn_alt[start:end]
            ).copy()
        ).float().to(
            DEVICE
        )

        prot_vec = model.t5_enc(
            bt5w,
            bt5m,
        )

        dna_vec = model.gpn_enc(
            bgref,
            bgalt,
        )

        g_gate = model.gate_ln_g(
            dna_vec
        )

        t_gate = model.gate_ln_t(
            prot_vec
        )

        xcat = torch.cat(
            [g_gate, t_gate],
            dim=1,
        )

        gate_logit = model.gate_mlp(
            xcat
        )

        tau = max(
            float(
                getattr(
                    model,
                    "_gate_tau",
                    1.0,
                )
            ),
            1e-6,
        )

        alpha = torch.sigmoid(
            gate_logit / tau
        )

        dna_all.append(
            dna_vec
            .detach()
            .cpu()
            .numpy()
        )

        prot_all.append(
            prot_vec
            .detach()
            .cpu()
            .numpy()
        )

        gate_all.append(
            alpha
            .detach()
            .cpu()
            .numpy()
        )

    return {
        "dna_vec": np.concatenate(
            dna_all,
            axis=0,
        ),
        "prot_vec": np.concatenate(
            prot_all,
            axis=0,
        ),
        "gate_alpha": np.concatenate(
            gate_all,
            axis=0,
        ),
    }


# =============================================================================
# 8. FIVE-FOLD MEAN REPRESENTATIONS — OLD CODE LOGIC
# =============================================================================

def load_or_extract_mean_representations(
    force_reextract: bool = False,
):
    ensure_dir(
        CACHE_DIR
    )

    cache_paths = {
        "dna": os.path.join(
            CACHE_DIR,
            "dna_vec_mean.npy",
        ),
        "prot": os.path.join(
            CACHE_DIR,
            "prot_vec_mean.npy",
        ),
        "fused": os.path.join(
            CACHE_DIR,
            "fused_vec_mean.npy",
        ),
        "gate": os.path.join(
            CACHE_DIR,
            "gate_alpha_mean.npy",
        ),
        "metadata": os.path.join(
            CACHE_DIR,
            "metadata.csv",
        ),
    }

    if (
        not force_reextract
        and all(
            os.path.isfile(p)
            for p in cache_paths.values()
        )
    ):
        print(
            "[CACHE] Reusing Fig. 7 representation cache."
        )

        return {
            "dna": np.load(
                cache_paths["dna"]
            ),
            "prot": np.load(
                cache_paths["prot"]
            ),
            "fused": np.load(
                cache_paths["fused"]
            ),
            "gate": np.load(
                cache_paths["gate"]
            ),
            "metadata": pd.read_csv(
                cache_paths["metadata"]
            ),
        }

    require_dir(
        MODEL_DIR,
        "MODEL_DIR",
    )

    require_file(
        PRED_ENSEMBLE_CSV,
        "pred_test_ensemble.csv",
    )

    meta = pd.read_csv(
        PRED_ENSEMBLE_CSV
    )

    required_cols = [
        "y_true",
        "prob_ensemble",
    ]

    missing = [
        c
        for c in required_cols
        if c not in meta.columns
    ]

    if missing:
        raise KeyError(
            f"{PRED_ENSEMBLE_CSV} missing columns: {missing}"
        )

    if "sample_index" in meta.columns:
        meta = (
            meta.sort_values(
                "sample_index"
            )
            .reset_index(
                drop=True
            )
        )
    else:
        meta = meta.reset_index(
            drop=True
        )

        meta.insert(
            0,
            "sample_index",
            np.arange(
                len(meta)
            ),
        )

    (
        t5_wt,
        t5_mut,
        gpn_ref,
        gpn_alt,
    ) = load_test_memmaps()

    n_test = int(
        t5_wt.shape[0]
    )

    if len(meta) != n_test:
        raise ValueError(
            "Prediction row count does not match test memmaps:\n"
            f"prediction N={len(meta)}\n"
            f"memmap N={n_test}"
        )

    dna_all = []
    prot_all = []
    gate_all = []
    fused_all = []

    for fold in FOLDS:
        print(
            f"[Fold {fold}] extracting representations..."
        )

        c_ckpt = os.path.join(
            MODEL_DIR,
            "fold_artifacts",
            f"fold{fold}",
            f"c_model_fold{fold}.pth",
        )

        model, _cfg = build_c_model_from_ckpt(
            ckpt_path=c_ckpt,
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

        pack = extract_one_fold(
            model=model,
            t5_wt=t5_wt,
            t5_mut=t5_mut,
            gpn_ref=gpn_ref,
            gpn_alt=gpn_alt,
        )

        dna_all.append(
            pack["dna_vec"]
        )

        prot_all.append(
            pack["prot_vec"]
        )

        gate_all.append(
            pack["gate_alpha"]
        )

        fused_path = os.path.join(
            MODEL_DIR,
            "fused_memmap",
            f"fold{fold}_Xte.npy",
        )

        require_file(
            fused_path,
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
                f"Fold {fold} fused Xte N mismatch."
            )

        fused_all.append(
            fused_fold
        )

        del model

    # EXACT previous-code logic:
    # mean across the five independently trained folds.
    dna_mean = np.mean(
        np.stack(
            dna_all,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )

    prot_mean = np.mean(
        np.stack(
            prot_all,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )

    gate_mean = np.mean(
        np.stack(
            gate_all,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )

    fused_mean = np.mean(
        np.stack(
            fused_all,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )

    np.save(
        cache_paths["dna"],
        dna_mean,
    )

    np.save(
        cache_paths["prot"],
        prot_mean,
    )

    np.save(
        cache_paths["gate"],
        gate_mean,
    )

    np.save(
        cache_paths["fused"],
        fused_mean,
    )

    meta.to_csv(
        cache_paths["metadata"],
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[SAVED] {cache_paths['dna']}"
    )
    print(
        f"[SAVED] {cache_paths['prot']}"
    )
    print(
        f"[SAVED] {cache_paths['fused']}"
    )
    print(
        f"[SAVED] {cache_paths['gate']}"
    )

    return {
        "dna": dna_mean,
        "prot": prot_mean,
        "fused": fused_mean,
        "gate": gate_mean,
        "metadata": meta,
    }


# =============================================================================
# 9. PROXY FEATURES — EXACT OLD LOGIC
# =============================================================================

def build_proxy_X(
    dna_vec: np.ndarray,
    prot_vec: np.ndarray,
    fused_vec: np.ndarray,
    gate_alpha: np.ndarray,
):
    parts = []
    feat_names = []

    def add(
        arr: np.ndarray,
        prefix: str,
    ):
        parts.append(
            arr
        )

        feat_names.extend(
            [
                f"{prefix}_{i}"
                for i in range(
                    arr.shape[1]
                )
            ]
        )

    add(
        dna_vec,
        "DNA",
    )

    add(
        prot_vec,
        "PROT",
    )

    add(
        fused_vec,
        "FUSED",
    )

    add(
        gate_alpha.reshape(
            -1,
            1,
        ),
        "GATE",
    )

    X = np.concatenate(
        parts,
        axis=1,
    ).astype(
        np.float32
    )

    return (
        X,
        feat_names,
    )


def train_proxy_xgb(
    X: np.ndarray,
    y: np.ndarray,
):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "Fig. 7 follows your previous XGBoost-proxy SHAP logic.\n"
            "Please install xgboost first:\n"
            "    pip install xgboost"
        ) from e

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )

    model.fit(
        X,
        y,
    )

    return model


# =============================================================================
# 10. SHAP
# =============================================================================

def pretty_feature_name(
    name: str,
) -> str:

    s = str(name).strip()

    m = re.match(
        r"^(DNA|PROT|FUSED)_(\d+)$",
        s,
        flags=re.IGNORECASE,
    )

    if m:
        prefix = m.group(1).upper()
        idx = m.group(2)

        if prefix == "DNA":
            return f"GPN-MSA {idx}"

        if prefix == "PROT":
            return f"ProtT5-XL {idx}"

        if prefix == "FUSED":
            return f"Fused {idx}"

    return s.replace(
        "_",
        " ",
    )


def get_shap_matrix(
    explainer,
    X: np.ndarray,
):
    sv = explainer.shap_values(
        X
    )

    if isinstance(
        sv,
        list,
    ):
        if len(sv) >= 2:
            sv = sv[1]
        else:
            sv = sv[0]

    sv = np.asarray(
        sv
    )

    # Newer SHAP versions can return [N, F, 2].
    if sv.ndim == 3:
        if sv.shape[-1] >= 2:
            sv = sv[
                ...,
                1,
            ]
        else:
            sv = sv[
                ...,
                0,
            ]

    if sv.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP value shape: {sv.shape}"
        )

    return sv


def make_shap_figure(
    proxy_model,
    X: np.ndarray,
    feat_names: List[str],
):
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "SHAP is required:\n"
            "    pip install shap"
        ) from e

    # -------------------------------------------------------------------------
    # EXACT old logic:
    # keep GATE in proxy training, but remove GATE from displayed SHAP panels.
    # -------------------------------------------------------------------------

    keep_mask = np.array(
        [
            not str(n)
            .upper()
            .startswith(
                "GATE_"
            )
            for n in feat_names
        ],
        dtype=bool,
    )

    explainer = shap.TreeExplainer(
        proxy_model
    )

    shap_full = get_shap_matrix(
        explainer,
        X,
    )

    if (
        keep_mask.size
        != shap_full.shape[1]
    ):
        raise ValueError(
            "Feature-name / SHAP dimension mismatch:\n"
            f"feature names={keep_mask.size}, SHAP F={shap_full.shape[1]}"
        )

    X_plot = X[
        :,
        keep_mask,
    ]

    shap_plot = shap_full[
        :,
        keep_mask,
    ]

    feat_plot = [
        n
        for n, keep in zip(
            feat_names,
            keep_mask,
        )
        if keep
    ]

    feat_plot_pretty = [
        pretty_feature_name(
            n
        )
        for n in feat_plot
    ]

    # -------------------------------------------------------------------------
    # Preserve the previous display scaling logic.
    # -------------------------------------------------------------------------

    target_beeswarm_halfspan = 0.16

    max_abs_shap = (
        float(
            np.nanmax(
                np.abs(
                    shap_plot
                )
            )
        )
        if shap_plot.size
        else 1.0
    )

    display_scale = max(
        1.0,
        float(
            np.ceil(
                max_abs_shap
                / target_beeswarm_halfspan
            )
        ),
    )

    shap_disp = (
        shap_plot
        / display_scale
    )

    # -------------------------------------------------------------------------
    # Save SHAP audit arrays.
    # -------------------------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "shap_values_display.npy",
        ),
        shap_disp,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "shap_feature_matrix_display.npy",
        ),
        X_plot,
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "shap_feature_names_display.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        for n in feat_plot_pretty:
            f.write(
                n + "\n"
            )

    mean_abs = np.mean(
        np.abs(
            shap_disp
        ),
        axis=0,
    )

    importance = pd.DataFrame({
        "feature": feat_plot_pretty,
        "mean_abs_SHAP_display": mean_abs,
        "raw_feature_name": feat_plot,
    }).sort_values(
        "mean_abs_SHAP_display",
        ascending=False,
    )

    importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Fig7a_global_shap_importance.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Combined 1 × 2 Fig. 7
    # -------------------------------------------------------------------------

    fig, (
        ax_a,
        ax_b,
    ) = plt.subplots(
        1,
        2,
        figsize=(
            14.8,
            8.8,
        ),
        dpi=FIG_DPI,
        gridspec_kw={
            "wspace": 0.28,
        },
    )

    # ----- (a) Global SHAP importance -----

    plt.sca(
        ax_a
    )

    shap.summary_plot(
        shap_disp,
        X_plot,
        feature_names=feat_plot_pretty,
        plot_type="bar",
        max_display=MAX_DISPLAY,
        show=False,
        plot_size=None,
    )

    # Same previous-code display range.
    ax_a.set_xlim(
        0.0,
        0.10,
    )

    ax_a.set_xticks(
        [
            0.0,
            0.05,
            0.10,
        ]
    )

    ax_a.set_xlabel(
        "mean(|SHAP value|)"
    )

    ax_a.set_title(
        "Global SHAP importance"
    )

    panel_label(
        ax_a,
        "a",
        x=-0.06,
        y=1.01,
    )

    # ----- (b) SHAP beeswarm -----

    plt.sca(
        ax_b
    )

    shap.summary_plot(
        shap_disp,
        X_plot,
        feature_names=feat_plot_pretty,
        max_display=MAX_DISPLAY,
        show=False,
        plot_size=None,
    )

    beeswarm_lim = max(
        0.12,
        float(
            np.nanmax(
                np.abs(
                    shap_disp
                )
            )
        )
        * 1.05,
    )

    ax_b.set_xlim(
        -beeswarm_lim,
        beeswarm_lim,
    )

    tick_step = (
        0.05
        if beeswarm_lim <= 0.20
        else 0.10
    )

    tick_max = (
        np.ceil(
            beeswarm_lim
            / tick_step
        )
        * tick_step
    )

    ticks = np.arange(
        -tick_max,
        tick_max
        + tick_step * 0.5,
        tick_step,
    )

    ax_b.set_xticks(
        ticks
    )

    ax_b.tick_params(
        axis="x",
        labelsize=9,
    )

    ax_b.set_xlabel(
        "SHAP value (impact on model output)"
    )

    ax_b.set_title(
        "SHAP beeswarm"
    )

    panel_label(
        ax_b,
        "b",
        x=-0.06,
        y=1.01,
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        top=0.95,
        bottom=0.12,
        wspace=0.28,
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
        pad_inches=0.28,
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.28,
        facecolor="white",
    )

    plt.close(
        fig
    )

    return {
        "shap_full": shap_full,
        "shap_display": shap_disp,
        "importance": importance,
        "display_scale": display_scale,
        "png": png_path,
        "pdf": pdf_path,
    }


# =============================================================================
# 11. RIGOROUS SURROGATE-SHAP WORKFLOW
# =============================================================================

TRAIN_CSV_FALLBACK = (
    r"F:\GenProt-DSM_Resubmit\dataset\data"
    r"\VarGeneDisjointTrain.csv"
)

SUMMARY_JSON = os.path.join(
    MODEL_DIR,
    "ckpt",
    "test_ensemble_summary.json",
)


class Classifier2L(nn.Module):
    """The original D-stage classifier used by GenProt-DSM."""

    def __init__(self, hidden: int, hidden2: int, dropout: float, input_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.l2 = nn.Linear(hidden, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.l3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.drop(self.bn1(self.l1(x))))
        x = self.relu(self.drop(self.bn2(self.l2(x))))
        return self.l3(x)


def _select_representative_fold() -> Tuple[int, pd.DataFrame]:
    """
    Select one representative fold using VALIDATION AUROC only.
    For five folds, choose the median validation-AUROC fold.
    Test performance is never used for fold selection.
    """
    require_file(SUMMARY_JSON, "test_ensemble_summary.json")
    js = json.load(open(SUMMARY_JSON, "r", encoding="utf-8"))
    rows = []
    for rec in js.get("fold_summaries", []):
        vm = rec.get("d_best_val_metrics", {})
        if "auc_roc" not in vm:
            raise KeyError(f"Fold {rec.get('fold')} missing d_best_val_metrics.auc_roc")
        v = float(vm["auc_roc"])
        if v > 1.0:
            v /= 100.0
        rows.append({"fold": int(rec["fold"]), "validation_AUROC": v})
    if not rows:
        raise RuntimeError("No fold validation results found in summary JSON.")
    df = pd.DataFrame(rows).sort_values("validation_AUROC").reset_index(drop=True)
    fold = int(df.iloc[len(df)//2]["fold"])
    return fold, df


def _load_all_memmaps() -> Dict[str, np.ndarray]:
    """Load train/test raw feature memmaps lazily."""
    require_dir(MEMMAP_DIR, "MEMMAP_DIR")
    files = [
        "train_t5_wt.npy", "train_t5_mut.npy",
        "train_gpn_ref.npy", "train_gpn_alt.npy",
        "test_t5_wt.npy", "test_t5_mut.npy",
        "test_gpn_ref.npy", "test_gpn_alt.npy",
    ]
    out = {}
    for fn in files:
        p = os.path.join(MEMMAP_DIR, fn)
        require_file(p, fn)
        out[fn[:-4]] = np.load(p, mmap_mode="r")
    return out


def _resolve_train_csv(cfg_obj) -> str:
    candidates = []
    p = getattr(cfg_obj, "train_csv", None)
    if p:
        candidates.append(str(p))
    candidates.append(TRAIN_CSV_FALLBACK)
    for p in candidates:
        if os.path.isfile(p):
            print(f"[FOUND] train CSV: {p}")
            return p
    raise FileNotFoundError(
        "Could not locate training CSV. Checked:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )


def _reconstruct_fold_indices(y_train: np.ndarray, cfg_obj, selected_fold: int):
    """Reconstruct the exact StratifiedKFold split used by the current training code."""
    skf = StratifiedKFold(
        n_splits=int(cfg_obj.n_folds),
        shuffle=True,
        random_state=int(cfg_obj.seed),
    )
    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(y_train)), y_train), start=1
    ):
        if fold == selected_fold:
            return tr_idx.astype(int), va_idx.astype(int)
    raise RuntimeError(f"Could not reconstruct fold {selected_fold}.")


def _load_d_model(selected_fold: int, cfg_obj):
    candidates = [
        os.path.join(MODEL_DIR, "ckpt", f"d_best_model_fold{selected_fold}.pth"),
        os.path.join(
            MODEL_DIR, "fold_artifacts", f"fold{selected_fold}",
            f"d_best_model_fold{selected_fold}.pth"
        ),
    ]
    ckpt_path = next((p for p in candidates if os.path.isfile(p)), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            "D-stage checkpoint not found. Checked:\n" +
            "\n".join(f"  - {p}" for p in candidates)
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = Classifier2L(
        hidden=int(cfg_obj.d_hidden1),
        hidden2=int(cfg_obj.d_hidden2),
        dropout=float(cfg_obj.d_dropout),
        input_dim=int(cfg_obj.fused_d),
    )
    net.load_state_dict(ckpt["model_state_dict"], strict=True)
    net.to(DEVICE).eval()
    return net, ckpt_path


@torch.no_grad()
def _extract_split_representations(
    c_model: DualBranchFusionModel,
    t5_wt, t5_mut, gpn_ref, gpn_alt,
    indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extract DNA / Protein / Fused / Gate representations for specified rows."""
    dna_all, prot_all, fused_all, gate_all = [], [], [], []
    indices = np.asarray(indices, dtype=int)

    for start in range(0, len(indices), BATCH_SIZE):
        idx = indices[start:start+BATCH_SIZE]
        bt5w = torch.from_numpy(np.asarray(t5_wt[idx]).copy()).float().to(DEVICE)
        bt5m = torch.from_numpy(np.asarray(t5_mut[idx]).copy()).float().to(DEVICE)
        bgref = torch.from_numpy(np.asarray(gpn_ref[idx]).copy()).float().to(DEVICE)
        bgalt = torch.from_numpy(np.asarray(gpn_alt[idx]).copy()).float().to(DEVICE)

        prot_vec = c_model.t5_enc(bt5w, bt5m)
        dna_vec = c_model.gpn_enc(bgref, bgalt)

        g_gate = c_model.gate_ln_g(dna_vec)
        t_gate = c_model.gate_ln_t(prot_vec)
        gate_logit = c_model.gate_mlp(torch.cat([g_gate, t_gate], dim=1))
        tau = max(float(getattr(c_model, "_gate_tau", 1.0)), 1e-6)
        alpha = torch.sigmoid(gate_logit / tau)

        fused = c_model._gate_fuse(dna_vec, prot_vec)
        fused = c_model.fuse_proj(fused)
        fused = c_model.fuse_ln(fused)

        dna_all.append(dna_vec.cpu().numpy())
        prot_all.append(prot_vec.cpu().numpy())
        fused_all.append(fused.cpu().numpy())
        gate_all.append(alpha.cpu().numpy())

    return {
        "dna": np.concatenate(dna_all, axis=0).astype(np.float32),
        "prot": np.concatenate(prot_all, axis=0).astype(np.float32),
        "fused": np.concatenate(fused_all, axis=0).astype(np.float32),
        "gate": np.concatenate(gate_all, axis=0).astype(np.float32),
    }


@torch.no_grad()
def _predict_original_d(d_model: nn.Module, fused: np.ndarray) -> np.ndarray:
    probs = []
    for start in range(0, len(fused), 256):
        bx = torch.from_numpy(np.asarray(fused[start:start+256]).copy()).float().to(DEVICE)
        probs.append(torch.sigmoid(d_model(bx)).cpu().numpy().reshape(-1))
    return np.concatenate(probs).astype(float)


def _train_surrogate_regressor(X_train: np.ndarray, target_prob: np.ndarray):
    """
    Fit a surrogate to the ORIGINAL MODEL PROBABILITY, not to y_true.
    Thus the surrogate learns to mimic GenProt-DSM rather than learn a second classifier.
    """
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("Please install xgboost: pip install xgboost") from e

    model = xgb.XGBRegressor(
        n_estimators=900,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )
    model.fit(X_train, target_prob)
    return model


def _fidelity_row(target: np.ndarray, pred: np.ndarray, split: str) -> Dict[str, Any]:
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "split": split,
        "N": len(target),
        "MAE": mean_absolute_error(target, pred),
        "RMSE": mean_squared_error(target, pred) ** 0.5,
        "R2": r2_score(target, pred),
        "Spearman_rho": float(spearmanr(target, pred).statistic),
        "Pearson_r": float(pearsonr(target, pred).statistic),
    }


def _make_rigorous_shap_figure(proxy_model, X_test: np.ndarray, feat_names: List[str]):
    """Compute TreeSHAP on the locked test set using RAW SHAP values."""
    try:
        import shap
    except ImportError as e:
        raise ImportError("Please install shap: pip install shap") from e

    explainer = shap.TreeExplainer(proxy_model)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list):
        sv = sv[0] if len(sv) == 1 else sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    if sv.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

    # Keep the previous visual focus: DNA / protein / fused dimensions.
    # Gate is available to the surrogate but is not displayed in the main panels.
    keep = np.array([not n.upper().startswith("GATE_") for n in feat_names], dtype=bool)
    X_plot = X_test[:, keep]
    sv_plot = sv[:, keep]
    names = [pretty_feature_name(n) for n, k in zip(feat_names, keep) if k]

    # IMPORTANT: no arbitrary rescaling of SHAP values.
    mean_abs = np.mean(np.abs(sv_plot), axis=0)
    imp = pd.DataFrame({
        "feature": names,
        "mean_abs_SHAP": mean_abs,
    }).sort_values("mean_abs_SHAP", ascending=False)

    imp.to_csv(
        os.path.join(OUTPUT_DIR, "Fig7_global_importance.csv"),
        index=False, encoding="utf-8-sig"
    )
    np.save(os.path.join(OUTPUT_DIR, "shap_values_test.npy"), sv_plot)
    np.save(os.path.join(OUTPUT_DIR, "shap_feature_matrix_test.npy"), X_plot)
    with open(os.path.join(OUTPUT_DIR, "shap_feature_names.txt"), "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(14.8, 8.8), dpi=FIG_DPI,
        gridspec_kw={"wspace": 0.28}
    )

    plt.sca(ax_a)
    shap.summary_plot(
        sv_plot, X_plot,
        feature_names=names,
        plot_type="bar",
        max_display=MAX_DISPLAY,
        show=False,
        plot_size=None,
    )
    ax_a.set_title("Global SHAP importance")
    ax_a.set_xlabel("mean(|SHAP value|)")
    panel_label(ax_a, "a")

    plt.sca(ax_b)
    shap.summary_plot(
        sv_plot, X_plot,
        feature_names=names,
        max_display=MAX_DISPLAY,
        show=False,
        plot_size=None,
    )
    ax_b.set_title("SHAP beeswarm")
    ax_b.set_xlabel("SHAP value (impact on surrogate output)")
    panel_label(ax_b, "b")

    fig.subplots_adjust(left=0.14, right=0.985, top=0.95, bottom=0.12, wspace=0.28)
    base = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME)
    png = base + ".png"
    pdf = base + ".pdf"
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.28, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.28, facecolor="white")
    plt.close(fig)
    return imp, png, pdf


def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("FIG. 7 — RIGOROUS TRAINING-DERIVED SURROGATE SHAP")
    print("=" * 100)

    # 1) Select representative fold using validation AUROC only.
    selected_fold, fold_table = _select_representative_fold()
    print("\n[Fold selection: validation AUROC only]")
    print(fold_table.to_string(index=False))
    print(f"\n[Selected representative fold] {selected_fold}")

    # 2) Load train/test raw features.
    arrays = _load_all_memmaps()

    # 3) Load the representative C-stage model.
    c_ckpt = os.path.join(
        MODEL_DIR, "fold_artifacts", f"fold{selected_fold}",
        f"c_model_fold{selected_fold}.pth"
    )
    c_model, cfg_obj = build_c_model_from_ckpt(
        c_ckpt,
        t5_dim=int(arrays["train_t5_wt"].shape[2]),
        gpn_dim=int(arrays["train_gpn_ref"].shape[2]),
        seq_len=int(arrays["train_t5_wt"].shape[1]),
    )

    # 4) Reconstruct the exact fold train/validation split.
    train_csv = _resolve_train_csv(cfg_obj)
    train_df = pd.read_csv(train_csv)
    if LABEL_COL not in train_df.columns:
        raise KeyError(f"{train_csv} missing label column '{LABEL_COL}'")
    y_train = train_df[LABEL_COL].to_numpy(dtype=int)
    if len(y_train) != arrays["train_t5_wt"].shape[0]:
        raise ValueError(
            f"Train CSV N={len(y_train)} != train memmap N={arrays['train_t5_wt'].shape[0]}"
        )
    tr_idx, va_idx = _reconstruct_fold_indices(y_train, cfg_obj, selected_fold)
    te_idx = np.arange(arrays["test_t5_wt"].shape[0], dtype=int)
    print(f"[Split sizes] train={len(tr_idx)} validation={len(va_idx)} test={len(te_idx)}")

    # 5) Same-fold representations; NO cross-fold averaging.
    rep_tr = _extract_split_representations(
        c_model,
        arrays["train_t5_wt"], arrays["train_t5_mut"],
        arrays["train_gpn_ref"], arrays["train_gpn_alt"], tr_idx
    )
    rep_va = _extract_split_representations(
        c_model,
        arrays["train_t5_wt"], arrays["train_t5_mut"],
        arrays["train_gpn_ref"], arrays["train_gpn_alt"], va_idx
    )
    rep_te = _extract_split_representations(
        c_model,
        arrays["test_t5_wt"], arrays["test_t5_mut"],
        arrays["test_gpn_ref"], arrays["test_gpn_alt"], te_idx
    )

    X_tr, feat_names = build_proxy_X(
        rep_tr["dna"], rep_tr["prot"], rep_tr["fused"], rep_tr["gate"]
    )
    X_va, names_va = build_proxy_X(
        rep_va["dna"], rep_va["prot"], rep_va["fused"], rep_va["gate"]
    )
    X_te, names_te = build_proxy_X(
        rep_te["dna"], rep_te["prot"], rep_te["fused"], rep_te["gate"]
    )
    if feat_names != names_va or feat_names != names_te:
        raise RuntimeError("Feature definitions differ across train/validation/test.")

    # 6) Original D-stage model provides surrogate targets.
    d_model, d_ckpt_path = _load_d_model(selected_fold, cfg_obj)
    target_tr = _predict_original_d(d_model, rep_tr["fused"])
    target_va = _predict_original_d(d_model, rep_va["fused"])
    target_te = _predict_original_d(d_model, rep_te["fused"])

    # 7) Train surrogate ONLY on training samples and ORIGINAL MODEL probability.
    proxy = _train_surrogate_regressor(X_tr, target_tr)
    pred_tr = proxy.predict(X_tr)
    pred_va = proxy.predict(X_va)
    pred_te = proxy.predict(X_te)

    fidelity = pd.DataFrame([
        _fidelity_row(target_tr, pred_tr, "train"),
        _fidelity_row(target_va, pred_va, "validation"),
        _fidelity_row(target_te, pred_te, "test"),
    ])
    fidelity_path = os.path.join(OUTPUT_DIR, "Fig7_surrogate_fidelity.csv")
    fidelity.to_csv(fidelity_path, index=False, encoding="utf-8-sig")
    print("\n[Surrogate fidelity to original GenProt-DSM fold model]")
    print(fidelity.to_string(index=False))

    test_row = fidelity.loc[fidelity["split"] == "test"].iloc[0]
    if float(test_row["Spearman_rho"]) < 0.90 or float(test_row["R2"]) < 0.80:
        warnings.warn(
            "Surrogate fidelity is weaker than preferred on the locked test set. "
            "Interpret SHAP cautiously before manuscript use."
        )

    # 8) Freeze surrogate, then compute SHAP on locked TEST representations.
    importance, png, pdf = _make_rigorous_shap_figure(proxy, X_te, feat_names)

    # 9) Save transparent method audit.
    audit = {
        "analysis": "training-derived surrogate SHAP on learned representations",
        "representative_fold_selection": "median validation AUROC only; no test performance used",
        "selected_fold": int(selected_fold),
        "surrogate_model": "XGBoost regressor",
        "surrogate_target": "original representative-fold GenProt-DSM probability",
        "ground_truth_labels_used_to_train_surrogate": False,
        "test_labels_used_to_train_surrogate": False,
        "surrogate_training_split": "representative-fold training subset only",
        "surrogate_fidelity_splits": ["validation", "test"],
        "SHAP_split": "locked test set",
        "cross_fold_latent_dimension_averaging": False,
        "raw_SHAP_values_plotted": True,
        "displayed_feature_groups": ["GPN-MSA", "ProtT5-XL", "Fused"],
        "gate_used_in_surrogate_but_hidden_from_main_panels": True,
        "C_checkpoint": c_ckpt,
        "D_checkpoint": d_ckpt_path,
        "train_csv": train_csv,
        "memmap_dir": MEMMAP_DIR,
    }
    audit_path = os.path.join(OUTPUT_DIR, "Fig7_method_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("\n[Top 20 SHAP features]")
    print(importance.head(20).to_string(index=False))
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")
    print(f"Fidelity audit: {fidelity_path}")
    print(f"Method audit: {audit_path}")


if __name__ == "__main__":
    main()
