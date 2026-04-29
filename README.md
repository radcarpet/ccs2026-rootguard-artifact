# PREEMPT++: Reproducibility artifact

Companion code, derived artifacts, and per-RQ reproduction folders for the
paper. The paper source is in [`PAPER/`](PAPER/); see `PAPER/main.tex` for
the full text.

## Layout

```
preempt-artifact/
├── README.md                     this file
├── INSTALL.md                    setup instructions
├── .env.example                  template for environment variables (RQ4 only)
├── pyproject.toml                pinned Python dependencies
├── data/                         NHANES benchmark + population stats (input data)
│   ├── nhanes_benchmark_200.json
│   └── holdout_population_means.json
├── preempt/                      core PREEMPT++ algorithm
├── utils/                        shared utilities (sensitivity, allocation, topo order)
├── baselines/                    baseline DP-text mechanisms (CAPE, SBYW-DPMLM, CluSanT, …)
├── PAPER/                        paper source (LaTeX + figures)
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
| RQ1 — Target utility | §5.1 | `python run_target_space.py --method exp_all --budget_mode kplus1` (and 26 sibling cells) → `python analysis/gen_double_asymmetry_table.py` → `python analysis/gen_plots.py` | ~10 min | Pure CPU |
| RQ2 — Reconstruction attacks | §5.2 | `python precompute_allocations.py` → `python run.py` → `python analysis/gen_main_table.py` → `python analysis/gen_main_figure.py` | ~30 min | Pure CPU |
| RQ3 — Structural analysis | §5.3 | `python analysis/gen_allocation_plot.py` → `python analysis/gen_per_template_plot.py` → `python analysis/gen_template_metadata_table.py` | <5 min | Reads RQ1's allocations |
| RQ4 — End-to-end LLM agent eval | §5.4 | `python precompute_allocations.py` → `python experiments/run_sweep.py --n-patients 100 --workers 20 --eps-list 0.01,0.05,0.1` → `python analysis/aggregate.py` → `python analysis/gen_tables.py` → `python analysis/gen_plots.py` | ~3 hours | Requires `OPENAI_API_KEY` (see `.env.example`) |

See each RQ's `README.md` for full options, expected outputs, and which
paper tables / figures each artifact corresponds to.

## Quick smoke check (no rerun)

Frozen pre-built tables and plots are committed under each `rq*/tables/`
and `rq*/plots/`. To inspect what the paper reports without running
anything:

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

## Citing

```bibtex
@inproceedings{preempt-plus-plus,
  title  = {…},
  author = {…},
  year   = {…},
  booktitle = {…}
}
```
