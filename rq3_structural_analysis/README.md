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

Make sure RQ1 has been run first (or its frozen tables / aggregates are
available). Then from this folder:

```bash
# Allocation power-law fit (across mechanisms, ε, B settings).
python analysis/gen_allocation_plot.py
python analysis/gen_allocation_plot_2kplus1.py

# Per-template wMAPE breakdown (compressing vs amplifying).
python analysis/gen_per_template_plot.py

# §5 RQ3 body table tab:per_template_summary — per-template wMAPE at the
# focal cell (ε=0.1, B=(2k+1)ε, Exponential), grouped into compressing
# vs amplifying. Reads root-space data from
# results_rq1_adversarial_2kplus1_wmape_root_based/ at the repo root.
python analysis/gen_per_template_summary.py

# Template-properties metadata table (roots, targets, formulas, domains).
python analysis/gen_template_metadata_table.py

# Fair-utility analysis (paper appendix).
python analysis/gen_fair_utility.py
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
