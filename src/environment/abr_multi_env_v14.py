"""
Multi-Video ABR Environment V14 (reviewer-response version).

This version inherits all dynamics from the V12 chain (V12 -> V11 -> ... -> V5)
and changes ONLY the rebuffering reward scale, which is the single most
consequential fix requested in the expert review.

------------------------------------------------------------------------------
Why the reward scale changed (P0.1 in the review)
------------------------------------------------------------------------------
The V5-V13 reward used a fidelity term on the VMAF scale (0-100) together with
a rebuffering penalty ``REBUF_PENALTY_BASE = 6.0`` (training) and an evaluation
penalty ``beta = 4.3``. That ``4.3`` is the Pensieve rebuffering weight, but in
Pensieve the *fidelity* term is bitrate in Mbps (roughly a 0-5 scale), so a
1-second stall costs ``4.3`` == the maximum single-chunk quality reward.

When the fidelity axis is VMAF (0-100) but ``beta`` is left at ``4.3``, one
second of stall costs only ~4.3 out of a ~100-point fidelity range, i.e. the
relative cost of a stall is under-weighted by ~20-23x. Under such an objective a
QoE-optimal controller (and even the offline Genie oracle) will *choose* to
stall, which is exactly the pathology the review flagged (RobustMPC ~30% and an
oracle ~8% rebuffering).

Fix: keep fidelity on the VMAF scale and set the rebuffering penalty to the
maximum per-chunk fidelity, so that

    one second of rebuffering  ==  losing one chunk at maximum perceptual quality.

Our ladder's maximum VMAF is ~97-98, so we use ``REBUF_PENALTY_BASE = 100.0``
(training env, used by the non-Lagrangian arms) and ``beta = 100.0`` in the
evaluation surrogate (see ``evaluate_all_models_v14.py``). The Lagrangian arms
do not use ``REBUF_PENALTY_BASE`` directly; their rebuffering weight is the
adaptive dual variable ``lambda_r`` whose range is widened accordingly in
``constrained_abr_v14.py`` so the rebuffering constraint can actually bind.

The smoothness weight stays at 1.0: smoothness is |Delta VMAF| (VMAF points),
already on the fidelity unit, matching Pensieve.

Everything else (observation layout, buffer dynamics, VMAF lookup, VBR
profiles, episode length, and the V6 stabiliser tuning) is inherited unchanged,
so V14 results are directly comparable to the V12/V13 pipeline apart from the
corrected objective.
"""

from __future__ import annotations

from .abr_multi_env_v12 import ABREnv as _ABREnvV12


class ABREnv(_ABREnvV12):
    """V14 environment: identical dynamics to V12 with a corrected reward scale.

    Only ``REBUF_PENALTY_BASE`` is overridden; all other constants (smoothness
    weight, buffer-deviation weight, Lyapunov parameters) are inherited from the
    V12 chain so no unrelated tuning is silently changed.
    """

    # One second of stall ~ losing one chunk at maximum perceptual quality.
    # (max VMAF on this ladder is ~97-98; 100.0 is the clean, defensible value.)
    REBUF_PENALTY_BASE = 100.0


__all__ = ["ABREnv"]
