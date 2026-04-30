# RootGuard: Reproducibility artifact

Companion code, derived artifacts, and per-RQ reproduction folders for the
paper. Per-RQ READMEs link each folder back to the relevant section in the
paper.

## Layout

```
rootguard-artifact/
├── README.md                     this file
├── INSTALL.md                    setup instructions
├── .env.example                  template for environment variables (RQ4 only)
├── pyproject.toml                pinned Python dependencies
├── data/                         NHANES benchmark + population stats (input data)
│   ├── nhanes_benchmark_200.json
│   └── holdout_population_means.json
├── preempt/                      core sanitizer module (legacy package name; RootGuard wraps this)
├── utils/                        shared utilities (sensitivity, allocation, topo order)
├── baselines/                    Bounded-Laplace and Staircase mechanism implementations (used in RQ1, RQ2)
├── rq1_target_utility/           RQ1 (paper §5.1)
├── rq2_reconstruction/           RQ2 (paper §5.2)
├── rq3_structural_analysis/      RQ3 (paper §5.3)
└── rq4_agent_eval/               RQ4 (paper §5.4)
```

## Reproducing the paper's results

Each `rq*/` folder is **self-contained**: it has a `README.md`, a `run.py`
(main entry point), an `analysis/` directory of table/plot generators, and
pre-built `tables/` and `plots/` so reviewers can verify outputs without
rerunning. To reproduce from scratch:

| RQ | Section | Command (run from the rq folder) | Wall-clock | Notes |
|---|---|---|---|---|
| RQ1 — Target utility | §5.1 | `python run_target_space.py ...` (target-space sweep, all 27 cells) → `python run_root_space.py ...` (root-space sweep, paper-canonical, all 27 cells) → `python analysis/gen_rce_tables.py` (root-space wMAPE + RCE for body and appendix) → `python analysis/gen_double_asymmetry_table.py` → `python analysis/gen_appendix_summary.py` → `python analysis/gen_plots.py` | ~10 min | Pure CPU |
| RQ2 — Reconstruction attacks | §5.2 | `python precompute_allocations.py` → `python run.py` → `python analysis/gen_paper_tables.py` (paper labels: tab:recon_main, tab:rq2_*) → `python analysis/gen_main_figure.py` | ~30 min | Pure CPU |
| RQ3 — Structural analysis | §5.3 | `python analysis/gen_allocation_plot.py` → `python analysis/gen_allocation_plot_2kplus1.py` → `python analysis/gen_per_template_summary.py` (tab:per_template_summary, body §5) → `python analysis/gen_template_metadata_table.py` | <5 min | Reads RQ1's allocations + root-space results |
| RQ4 — End-to-end LLM agent eval | §5.4 | `python precompute_allocations.py` → `python experiments/run_sweep.py --n-patients 100 --workers 20 --eps-list 0.01,0.05,0.1` → `python analysis/aggregate.py` → `python analysis/gen_tables.py` (emits four RCE tables incl. tab:rce_per_template, tab:agent_rce_eps0.1_app, tab:agent_aggregate_rce) → `python analysis/gen_plots.py` | ~3 hours | Requires `OPENAI_API_KEY` (see `.env.example`) |

See each RQ's `README.md` for full options, expected outputs, and which
paper tables / figures each artifact corresponds to. **`AUDIT.md` (top
level)** maps every `\label{}` and `\includegraphics{figures/...}` in
`PAPER/sections/*.tex` to the artifact source that reproduces it.

## Inspect pre-built outputs

Pre-generated tables and figures are committed under each `rq*/tables/`
and `rq*/plots/` so reviewers can browse the paper-reported numbers
without rerunning any experiment:

```bash
ls rq1_target_utility/tables/ rq1_target_utility/plots/
ls rq2_reconstruction/tables/ rq2_reconstruction/plots/main/
ls rq3_structural_analysis/plots/
ls rq4_agent_eval/tables/ rq4_agent_eval/plots/
cat rq2_reconstruction/SUMMARY.md
cat rq4_agent_eval/SUMMARY.md
```

## System requirements

- Python ≥ 3.11
- ~2 GB free disk for results (RQ1–3) plus ~200 MB for RQ4 raw JSONs
- 8+ CPU cores recommended for the parallel cells
- For RQ4 only: a working OpenAI API key (any tier) and ~$2–5 of API budget

See [`INSTALL.md`](INSTALL.md) for full install instructions.

