"""
Paper 2 release identifiers (distinct from Paper 1's v18 / vNN env lineage).

Paper 1 convention: incremental simulator tags (v14, v18) + ``results/v18_certified``.
Paper 2 convention: product codename + semantic release tag + campaign folders under
``results/oacs/``.
"""

from __future__ import annotations

from pathlib import Path

from configs.paths import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR

# ---------------------------------------------------------------------------
# Identity (use in papers, logs, and result paths — not "v20")
# ---------------------------------------------------------------------------
PAPER2_CODENAME = "OACS"  # Online ACI Conformal Shield
PAPER2_RELEASE = "oacs-r1"
PAPER2_SHORT = "paper2"

# ---------------------------------------------------------------------------
# Evaluation arms (Paper 2 scripts — avoid reusing v18 arm string names)
# ---------------------------------------------------------------------------
ARM_RAW = "raw"
ARM_P1_STATIC = "static_cps"  # Paper 1 CertifiedPerceptualShieldWrapper baseline
ARM_ACI_FEASIBILITY = "aci_feasibility"  # online ACI, banking off
ARM_ACI_FULL = "aci_oacs"  # online ACI + per-chunk VMAF knee + masks

# Legacy aliases for cross-paper tables only (do not use in new Paper 2 filenames)
ARM_ALIAS_CPS_V18 = "cps_v18"
ARM_ALIAS_ACI_CERTIFIED = "aci_certified"
ARM_ALIAS_ACI_SAFETY = "aci_safety"

# ---------------------------------------------------------------------------
# Results layout (server-friendly: one campaign dir per sweep)
# ---------------------------------------------------------------------------
OACS_RESULTS_ROOT = RESULTS_DIR / "oacs"
OACS_CAMPAIGNS_DIR = OACS_RESULTS_ROOT / "campaigns"
OACS_MODELS_ROOT = MODELS_DIR / "oacs"
OACS_PAPER_ASSETS = PROJECT_ROOT / "paper" / "oacs"

DEFAULT_CAMPAIGN = f"{PAPER2_RELEASE}_main"


def campaign_dir(name: str | None = None) -> Path:
    d = OACS_CAMPAIGNS_DIR / (name or DEFAULT_CAMPAIGN)
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir(model_key: str, seed: int) -> Path:
    d = OACS_MODELS_ROOT / model_key / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def release_banner() -> str:
    return f"{PAPER2_CODENAME} release {PAPER2_RELEASE}"
