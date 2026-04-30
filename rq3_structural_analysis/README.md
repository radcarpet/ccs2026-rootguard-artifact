# RQ3 — Structural analysis

**Paper section:** §5.3 (`\label{sec:experiments-rq3}` in `PAPER/sections/5_experiments_worst_case_adversary.tex`).

## Question

What structural properties of a template (sensitivity profile, domain
widths) and budget allocation determine when dependency-aware noising
provides the largest M-Opt advantage?

## What's analyzed

- **Allocation power-law**: εᵢ ∝ (\|hᵢ\|·Dᵢ)^α, with α ≈ 0.5 fitted across mechanisms.
- **Per-template breakdown**: 8 templates split into:
  - *Compressing* (large M-Opt vs M-All gaps): AIP, HOMA, FIB4, NLR.
  - *Amplifying* (small absolute gaps but 2–3× multiplicative): ANEMIA, CONICITY, TYG, VASCULAR.
- **Template metadata table**: roots / target / formula / domain widths per template.

This RQ does **not** run a new sweep — it analyzes the M-Opt allocations
computed for RQ1 and the per-cell wMAPE values from RQ1's results.

## Reproduce

**Prerequisite:** the body figures and the `tab:per_template_summary` table
read RQ1's root-space sweep results
(`rq1_target_utility/results_rq1_adversarial_2kplus1_wmape_root_based/`),
so RQ1 must have been run first (`python rq1_target_utility/run_root_space.py …`
for all 27 cells; the root-space sweep does not need a `recompute_wmape.py`
pass — it emits the `_wmape_*` form directly). The frozen tables shipped in
`tables/` cover the RQ3 body without rerunning anything.

Then from this folder:

```bash
# Body §5: allocation power-law fit.
python analysis/gen_allocation_plot.py
python analysis/gen_allocation_plot_2kplus1.py

# Body §5 RQ3 table: tab:per_template_summary — per-template wMAPE at the
# focal cell (ε=0.1, B=(2k+1)ε, Exponential), grouped compressing /
# amplifying. The default --results-dir resolves to
# ../rq1_target_utility/results_rq1_adversarial_2kplus1_wmape_root_based
# (override with --results-dir if your data lives elsewhere).
python analysis/gen_per_template_summary.py

# Template-properties metadata table (roots, targets, formulas, domains).
python analysis/gen_template_metadata_table.py
```

### Appendix-only scripts (not reproducible from this artifact alone)

`gen_per_template_plot.py` and `gen_fair_utility.py` are appendix-only
analyses that consume `results_rq2_implied/s_dom_info.json` — a
domain-statistics file that the public artifact does not ship and no
included script produces. Per AUDIT.md, the body of RQ3 reproduces
without these. Both scripts now exit cleanly with an explanatory message
when invoked in this artifact:

```bash
python analysis/gen_per_template_plot.py    # exits with "appendix-only, requires s_dom_info.json"
python analysis/gen_fair_utility.py          # same exit
```

## Pre-built outputs

```
plots/
  rq2_allocation_powerlaw_*.pdf       allocation vs sensitivity, log-log
  rq1_allocation_vs_sensitivity.pdf
  rq1_per_template_grid.pdf           per-template wMAPE (also in RQ1)
  rq1_cross_budget*.pdf               same numbers cross-cutting B settings
  rq1_improvement_decomp.pdf          decomposition of M-Opt advantage
tables/
  latex_template_metadata.tex         template metadata
  latex_rq2_fair_utility.tex          fair-utility analysis
  per_template_summary.tex            tab:per_template_summary (main text §5 RQ3)
```
