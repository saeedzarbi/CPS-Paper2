# Two-day pilot — go / no-go

## Direction A: buffer-conditioned \(\alpha_k(B_k)\)

1. Fixed grid: \(\varepsilon \in \{0.5,1,2,4\} \times \alpha \in \{0.05,0.10,0.20\}\) on greedy and BBA (12 cells, paired episodes).
2. Two or three schedules: step and linear \(\alpha_k(B_k)\).
3. At **matched rebuffering**, compare bitrate and VMAF to the best fixed pair.
4. **Go** only if a schedule beats the best fixed pair on at least one of {bitrate, rebuffer, VMAF} without losing the other two — separately on **both** hosts.
5. Do **not** motivate from Certified–Safety bitrate-saved % under \(\alpha\) sweeps; use intervention rate and coverage.
6. Report **empirical** coverage (marginal and buffer-binned). No conditional conformal claim.

If A fails: fixed-grid Pareto table is useful but not enough for a strong Q1 paper alone.

## Direction B: cold-start / calibration (fallback)

Use stall-second decomposition across hosts; fair \(\rho_{\mathrm{fb}}\) sweep; warm-start / weighted / ACI as targeted methods; regime recipe table.
Keep ACI fair (carry \(\alpha_t\) across sessions or long \(K_e\)).
