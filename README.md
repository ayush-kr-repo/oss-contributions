# Open Source Contributions

**16 merged PRs across 4 repositories** — GSSoC 2026, May–August 2026

| Repository | Merged | Area |
|---|---|---|
| [AegisGraph Sentinel 2.0](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0) | 12 | Explainability, calibration, training integrity |
| [AegisAI](https://github.com/SdSarthak/AegisAI) | 2 | API endpoints, integration testing |
| [CodeGraphContext](https://github.com/CodeGraphContext/CodeGraphContext) | 1 | CLI error propagation |
| [HELPDESK.AI](https://github.com/riteshbonthalakoti/HELPDESK.AI) | 1 | Unit testing |

7 rated `level:advanced` by maintainers · 3 rated `quality:exceptional` or `quality:clean`

[→ All merged PRs](https://github.com/pulls?q=is%3Apr+author%3Aayush-kr-repo+is%3Amerged+-user%3Aayush-kr-repo)

---

# AegisGraph Sentinel 2.0 — 12 merged PRs

AegisGraph Sentinel is a graph-based fraud detection platform. A heterogeneous
graph attention network (HTGAT) scores accounts for fraud risk, and an
"Oracle" explainability layer produces the analyst- and regulator-facing
justification for each decision.

I worked on the layer between the model and the analyst.

## The recurring problem

Several components in that layer produced output that *looked* model-derived
but wasn't. Counterfactual explanations were random perturbations scored with
`random.uniform(0.7, 0.95)` and never checked against the model. The drift
monitor compared production traffic against baselines drawn from unseeded
`np.random` calls, so it was measuring drift against noise. The Oracle accepted
`attention_weights` and never read them. The graph explainer reported
GNNExplainer edge-mask weights to analysts as "fraud_contribution" with no
validation that removing those edges changed anything.

Individually these look like bugs. Together they meant a fraud system was
showing confident, specific reasons for decisions that were partly decorative —
in a domain where those explanations are shown to blocked customers and auditors.

Most of my PRs replaced one of these with a real implementation plus a way to
verify it was real.

---

## 1. Making explanations faithful

**[#1180](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0/pull/1180)** — expose HTGAT attention weights through the explainability pipeline
**[#1368](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0/pull/1368)** — generate ranked investigation reports from attention weights
**[#1426](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0/pull/1426)** — add confidence scoring to Oracle explanations
**[#1930](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0/pull/1930)** — ground Oracle causal factors in HTGAT attention weights
**[#1954](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0/pull/1954)** — add fidelity/sparsity faithfulness metrics to explanation subgraphs
**[#2271](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0/pull/2271)** — replace mock counterfactuals with model-probing search

These six build on each other: expose the attention weights, use them to
generate reports, actually read them in the causal-factor extractor, then
measure whether the resulting explanations are faithful at all.

**#1930** — `generate_explanation()` accepted an `attention_weights` argument
that `_extract_causal_factors()` never read, so every GRAPH-type causal factor
fell back to a static evidence template regardless of the subgraph. I parsed
attention payloads into a ranked edge list, accepting three shapes already in
use across the codebase (the explainability pipeline's edge list,
`ProductionRiskScorer`'s `top_relationships`, and flat `"SRC->TGT"` mappings),
replaced the canned evidence with the highest-attention transfer paths, and
lifted the GRAPH factor's weight to the strongest attention score so factor
ranking reflects what the model attended to. Multi-head weights are averaged,
non-finite values skipped, and malformed payloads fall back to the old template
rather than raising.

**#1954** — `extract_critical_topology()` selected the top-10 edges above a
0.5 mask weight and reported them as `fraud_contribution` / "AI Impact Weight".
Mask importance is not evidence of faithfulness. I added
`explain_with_faithfulness()`, which measures the explanation by perturbing
the graph and re-running the model:

- **fidelity+** (necessity) — prediction drop when the explanation edges are removed
- **fidelity−** (sufficiency) — prediction change when *only* the explanation edges are kept
- **sparsity** — fraction of the graph not needed for the explanation

An explanation is `FAITHFUL` when fidelity+ ≥ 0.1 and |fidelity−| ≤ 0.1,
`PARTIALLY_FAITHFUL` when one holds, `UNFAITHFUL` when neither. The verdict
surfaces in the Oracle's analyst output, so an unfaithful explanation is now
labelled as such instead of being presented as fact.

**#2271** — counterfactuals were fabricated: random perturbations of the first
three features, with `proximity_score = random.uniform(0.7, 0.95)`, never
validated against the model. I replaced this with a search that probes the
real scoring function:

- greedy feature selection — try to flip the decision with one feature; if none
  suffices, lock in the most effective single move and repeat
- per-feature bisection — binary search (25 steps) for the smallest change
  magnitude that still flips the decision
- a sparsity budget (`max_features_changed`), an `immutable_features` set for
  fields that must not be altered (account age, regulatory attributes), and a
  margin so the counterfactual doesn't land exactly on the boundary
- every returned counterfactual is verified against the model before being
  stored; when no flip is reachable within bounds, it returns `None` rather
  than inventing an answer

That last point was the design decision I cared most about. A fraud system
telling a blocked customer "reduce transaction velocity by 12% and you'd be
approved" is worthless unless that statement is true of the actual model.

> **Hardest part:** TODO — one specific technical obstacle. Candidates:
> defining fidelity for a heterogeneous graph where removing edges changes
> node typing; getting bisection to terminate when the model is non-monotonic
> in a feature; deciding the `FIDELITY_PLUS_MIN` / `FIDELITY_MINUS_TOLERANCE`
> thresholds. Pick the one that actually cost you time. 2–3 sentences.

---

## 2. Calibrated risk scores

**[#2242](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0/pull/2242)** — temperature scaling and ECE calibration in risk training
**[#3636](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0/pull/3636)** — MC-Dropout epistemic uncertainty, routing uncertain scores to review

Risk scores were consumed as probabilities throughout the system — decision
thresholds, REVIEW bands, regulator-facing narratives like "Risk Score: 56%" —
but the model was trained with focal loss and never calibrated, so those
percentages didn't correspond to observed fraud rates. #2242 added a
`calibration` module (temperature scaling fit on a held-out split, ECE as the
reported metric) and wired it into the trainer.

#3636 addressed the complementary gap: scores were point estimates fed
straight into hard thresholds, with no measure of how reliable any individual
score was. MC-Dropout gives an epistemic uncertainty estimate, and
high-uncertainty decisions are routed to human review rather than
auto-actioned.

> **Hardest part:** TODO — 2–3 sentences. If nothing here was hard, delete
> this line rather than inventing one.

---

## 3. Training and monitoring integrity

**[#2525](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0/pull/2525)** — replace random train/val split with temporal split
**[#2998](https://github.com/Puneet04-tech/AegisGraph-Sentinel-2.0/pull/2998)** — add PSI and real training baselines to the drift monitor

**#2525** — `AegisGraphLoader` documented itself as preventing data leakage
through future-peeking, then split train/validation with
`torch.rand(num_accounts) < 0.8`. `NeighborLoader`'s `time_attr` only
constrains neighbour sampling; it does nothing about which accounts land in
which split. Fraud labels are strongly time-correlated, so a random split lets
the model validate against periods it trained on. Replaced with a temporal
split.

**#2998** — the drift monitor's K-S test and webhook alerting were real, but
`_load_training_baselines()` returned unseeded `np.random.normal` /
`np.random.exponential` samples. Every instantiation compared production
traffic against a different
