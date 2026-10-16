# Tube-v2: finite-sample coverage sketch

## Proposition (finite-sample tube coverage)

Fix a horizon step \(h \in \{1,\ldots,H\}\) and let the nonconformity scores
\(s^{(h)}_1,\ldots,s^{(h)}_{n_h}, s^{(h)}_{n_h+1}\) be exchangeable (e.g. i.i.d.).
Let \(\hat c^{(t)}_{t+h}\) be the point predictor issued at decision time \(t\), and
\[
s^{(h)} = \max\!\left(0,\frac{\hat c^{(t)}_{t+h} - c_{t+h}}{\hat c^{(t)}_{t+h}}\right)
\]
(or the log-ratio variant). With \(q_h\) the \(\lceil (n_h+1)(1-\alpha_h)\rceil\)-th
order statistic of the calibration scores in \(S_h\), set
\(L_h = \max(\text{min\_kbps}, \hat c^{(t)}_{t+h}(1-q_h))\). Then
\(\Pr[c_{t+h} \ge L_h] \ge 1-\alpha_h\).

Consequently:

- **Bonferroni mode** (\(\alpha_h = \alpha/H\)): union bound gives
  \(\Pr[\forall h\le H : c_{t+h} \ge L_h] \ge 1-\alpha\) for the *planned* tube at time \(t\).
- **Per-step mode** (\(\alpha_h = \alpha\)): each step has marginal coverage \(\ge 1-\alpha\).

If MPC marks a sequence safe when buffer stays above margin under download times
\(\mathrm{size}/L_h\), then with probability \(\ge 1-\alpha\) (bonferroni) no stall occurs
within the planned horizon **for that fixed sequence**, given the tube holds.

## Limitations (report honestly)

- Exchangeability fails under drift; ACI only targets long-run average miscoverage.
- Scores from overlapping horizons are dependent.
- Receding-horizon re-planning reuses data; guarantees apply to the tube issued at one decision, not to the closed-loop trajectory.
- Application-layer Irish traces are not i.i.d. physics; empirical coverage tables are mandatory.
