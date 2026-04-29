# Paper-to-Artifact Audit

This file maps every numbered table and figure in the paper
(`PAPER/sections/{1_Introduction,5_experiments_worst_case_adversary,Appendix}.tex`)
to the artifact code that reproduces it. After running each `Reproduce`
section in the per-RQ READMEs, every label and figure path below should
resolve from artifact-generated outputs.

## §1 Introduction

| Paper artifact | Artifact source | Reproducibility |
|---|---|---|
| `figures/rootguard_overview.pdf` (`fig:rootguard_overview`) | hand-drawn system diagram, **not regenerated from data** | static asset; ship as-is |

## §5 Body — RQ1 / RQ2 / RQ3 / RQ4

| Paper label | Artifact generator | Output path |
|---|---|---|
| `tab:double_asymmetry` (RQ1, body) | `rq1_target_utility/analysis/gen_rce_tables.py` (root-space, paper-canonical) — also `gen_double_asymmetry_table.py` (target-space, legacy) | `rq1_target_utility/tables/root_space/rq1_double_asymmetry.tex` |
| `tab:rce_per_template_rq1` (RQ1, body) | `rq1_target_utility/analysis/gen_rce_tables.py` | `rq1_target_utility/tables/root_space/rq1_rce_per_template.tex` |
| `tab:recon_main` (RQ2, body) | `rq2_reconstruction/analysis/gen_paper_tables.py` | `rq2_reconstruction/tables/recon_main.tex` |
| `fig:recon_uniform`, `fig:recon_informed`, `fig:recon_main` (`figures/fig_mae_vs_q_eps0.1_{uniform,informed}.pdf`) | `rq2_reconstruction/analysis/gen_main_figure.py` | `rq2_reconstruction/plots/main/fig_mae_vs_q_eps0.1_{uniform,informed}.pdf` |
| `tab:per_template_summary` (RQ3, body) | `rq3_structural_analysis/analysis/gen_per_template_summary.py` | `rq3_structural_analysis/tables/per_template_summary.tex` |
| `\input{figures/rq4_aggregate_wmape}` → `tab:agent_aggregate_wmape` (RQ4, body) | `rq4_agent_eval/analysis/gen_tables.py` | `rq4_agent_eval/tables/aggregate_wmape.tex` (paper does `\input{figures/rq4_aggregate_wmape}` so the file should be copied/symlinked into the paper's `figures/` directory at build time) |
| `tab:rce_per_template` (RQ4, body) | `rq4_agent_eval/analysis/gen_tables.py` (`gen_rce_body`) | `rq4_agent_eval/tables/rce_eps0.1_body.tex` |

## Appendix tables

| Paper label | Artifact generator | Output path |
|---|---|---|
| `tab:rq1_kplus1_wmape`, `tab:rq1_2kplus1_wmape`, `tab:rq1_3kplus1_wmape` (RQ1 appendix) | `rq1_target_utility/analysis/gen_rce_tables.py` (root-space, canonical) — also `gen_appendix_summary.py` (target-space, legacy) | `rq1_target_utility/tables/root_space/rq1_{k,2k,3k}plus1_wmape.tex` |
| `tab:rq1_kplus1_rce`, `tab:rq1_2kplus1_rce`, `tab:rq1_3kplus1_rce` | `rq1_target_utility/analysis/gen_rce_tables.py` | `rq1_target_utility/tables/root_space/rq1_{k,2k,3k}plus1_rce.tex` |
| `tab:rq1_kplus1_per_tmpl_wmape`, `tab:rq1_2kplus1_per_tmpl_wmape`, `tab:rq1_3kplus1_per_tmpl_wmape` | `rq1_target_utility/analysis/gen_rce_tables.py` | `rq1_target_utility/tables/root_space/rq1_{k,2k,3k}plus1_per_tmpl_wmape.tex` |
| `tab:rq1_win_rate` (RQ1 appendix) | `rq1_target_utility/analysis/gen_appendix_summary.py` | inside `rq1_target_utility/tables/target_space/latex_rq1_adversarial_summary.tex` |
| `tab:rq2_per_template` | `rq2_reconstruction/analysis/gen_paper_tables.py` | `rq2_reconstruction/tables/rq2_per_template.tex` |
| `tab:rq2_exp_A_uniform`, `tab:rq2_exp_A_informed`, `tab:rq2_exp_B_uniform`, `tab:rq2_exp_B_informed` | `rq2_reconstruction/analysis/gen_paper_tables.py` | `rq2_reconstruction/tables/rq2_exp_{A,B}_{uniform,informed}.tex` |
| `tab:rq2_blap_agg`, `tab:rq2_stair_agg` | `rq2_reconstruction/analysis/gen_paper_tables.py` | `rq2_reconstruction/tables/rq2_{blap,stair}_agg.tex` |
| `tab:agent_rce_eps0.1_app` (RQ4 appendix) | `rq4_agent_eval/analysis/gen_tables.py` (`gen_rce_appendix`) | `rq4_agent_eval/tables/rce_eps0.1_appendix.tex` |

## Appendix figures

| Paper file | Artifact generator | Output path |
|---|---|---|
| `figures/rq1_per_template_grid.pdf` (`fig:per_template_app`) | `rq1_target_utility/analysis/gen_plots_root_based.py` | `rq1_target_utility/plots/root_space/rq1_per_template_grid.pdf` |
| `figures/rq2_allocation_powerlaw_eps1p0.pdf` (`fig:allocation_powerlaw_10`) | `rq3_structural_analysis/analysis/gen_allocation_plot.py` | `rq3_structural_analysis/plots/rq2_allocation_powerlaw_eps1p0.pdf` |
| `figures/rq2_allocation_powerlaw_eps0p1.pdf` (`fig:allocation_powerlaw_01`) | `rq3_structural_analysis/analysis/gen_allocation_plot_2kplus1.py` | `rq3_structural_analysis/plots/rq2_allocation_powerlaw_eps0p1.pdf` (alias of `..._2kplus1_eps0p1.pdf`) |
| `figures/rq2_allocation_by_mechanism.pdf` (`fig:allocation_bars`) | `rq3_structural_analysis/analysis/gen_allocation_plot.py` | `rq3_structural_analysis/plots/rq2_allocation_by_mechanism.pdf` |

## Notes on reproducibility

* **Root-space vs. target-space.** The paper's RQ1 numbers come from the
  *root-space* threat model (M-All sanitizes each root release independently
  with budget ε; total leakage scales with k). The artifact ships both
  threat-model variants — generators in `rq1_target_utility/analysis/` write
  root-space outputs to `tables/root_space/` and target-space outputs to
  `tables/target_space/`. **Always import the root-space tables for the
  paper.** The target-space `latex_rq1_adversarial_summary.tex` is retained
  for the legacy comparison reported in the introduction.
* **RQ4 RCE bootstrap snapshot.** The agent-eval RCE numbers in the paper
  come from a slightly older `aggregates.json` snapshot. Re-running
  `analysis/aggregate.py` and `analysis/gen_tables.py` reproduces the same
  data within bootstrap noise (~1–2 SE) but not character-for-character.
* **Static figures.** `figures/rootguard_overview.pdf` is hand-drawn and
  not regenerated from data.
* **Paper-side path conventions.** The paper's `\input{figures/X}` and
  `\includegraphics{figures/X.pdf}` resolve relative to the paper's own
  `figures/` directory. To wire artifact outputs in, copy or symlink each
  file from its location in this audit into `PAPER/figures/`.
