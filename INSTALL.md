# Install

Tested with Python 3.11 on Linux. Other versions ≥ 3.11 should work.

## Option A — pip (any virtualenv tool)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

`pyproject.toml` pins the dependencies (numpy, scipy, matplotlib, openai,
python-dotenv). Total install size is < 200 MB.

## Option B — uv (faster, recommended if you have it)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Verifying the install

From the artifact root:

```bash
python -c "import numpy, scipy, matplotlib, openai; print('ok')"
```

Each RQ folder also has a tiny smoke test:

```bash
for d in rq1_target_utility rq2_reconstruction rq3_structural_analysis rq4_agent_eval; do
  (cd "$d" && python tests/test_smoke.py)
done
```

## Environment variables (RQ4 only)

RQ4 requires an OpenAI API key. Copy the template and fill in your key:

```bash
cp .env.example .env
# edit .env
```

The runner in `rq4_agent_eval/experiments/run_sweep.py` reads
`OPENAI_API_KEY` from the process environment (load with
`set -a; source .env; set +a` or via `python-dotenv` if you import it).

RQ1, RQ2, RQ3 do **not** need any API access — they are pure-CPU
numerical / mechanism simulations.

## What is *not* in this artifact

- The 28,800 raw per-session JSON files for RQ4 (~150 MB). Reviewers
  regenerate them by running `experiments/run_sweep.py`. Frozen
  `aggregates.json`, `tables/`, `plots/`, and `SUMMARY.md` are shipped so
  outputs can be verified without an API key.

## Troubleshooting

- **`scipy.optimize` complains about constraints in `optimal_budget_allocation_*`**: ensure scipy ≥ 1.10. The pyproject pins this.
- **RQ4 hits OpenAI rate limits**: lower `--workers`. We tested 20-way
  concurrency on a Tier-4 account; lower tiers may need 5–10.
- **Bootstrap SE columns are empty in regenerated tables**: this happens
  if a cell had < 30 successful samples. Increase `--n-patients` or check
  the cell-level `parse_failures` counter in `aggregates.json`.
