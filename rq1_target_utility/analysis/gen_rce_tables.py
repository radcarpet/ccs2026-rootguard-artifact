"""Generate RQ1 Risk Class Error (RCE) tables in root-space.

Emits four LaTeX table files into ``tables/root_space/``:

* ``rq1_kplus1_rce.tex``    — appendix aggregate RCE, B = (k+1)\\varepsilon
* ``rq1_2kplus1_rce.tex``   — appendix aggregate RCE, B = (2k+1)\\varepsilon
* ``rq1_3kplus1_rce.tex``   — appendix aggregate RCE, B = (3k+1)\\varepsilon
* ``rq1_rce_per_template.tex`` — body per-template RCE at \\varepsilon = 0.1, all
  three budgets, M-All vs. M-Opt (Exponential mechanism)

Reads from ``results_rq1_adversarial_{kplus1,2kplus1,3kplus1}_wmape_root_based/``,
the root-space sweep used by the latest paper. Each per-template summary JSON is
expected to contain ``risk_mae``, ``risk_std`` and ``n_samples``; bootstrap-style
SE is computed as ``risk_std / sqrt(n_samples)`` per template, propagated across
templates as ``sqrt(sum(SE_i^2)) / k``.

Run from the ``rq1_target_utility`` folder:

    python analysis/gen_rce_tables.py --out-dir tables/root_space

By default the script reads from ``./results_rq1_adversarial_*kplus1_wmape_root_based``
and writes to ``./tables/root_space/``.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TEMPLATES = ['ANEMIA', 'FIB4', 'AIP', 'CONICITY', 'VASCULAR', 'TYG', 'HOMA', 'NLR']
PAPER_TMPL_ORDER = ['HOMA', 'AIP', 'FIB4', 'NLR', 'TYG', 'ANEMIA', 'CONICITY', 'VASCULAR']
K = {'ANEMIA': 3, 'FIB4': 4, 'AIP': 3, 'CONICITY': 3,
     'VASCULAR': 2, 'TYG': 2, 'HOMA': 2, 'NLR': 2}
EPS_VALS = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
METHODS = [
    'exp_all', 'exp_roots', 'exp_opt',
    'blap_all', 'blap_roots', 'blap_opt',
    'stair_all', 'stair_roots', 'stair_opt',
]
EPS_FOCAL = 0.1
PMSTD = '\\providecommand{\\pmstd}[1]{{\\scriptsize $\\pm$#1}}'

RUNS = [
    ('kplus1',  'k+1',  'results_rq1_adversarial_kplus1_wmape_root_based',  lambda k: k + 1),
    ('2kplus1', '2k+1', 'results_rq1_adversarial_2kplus1_wmape_root_based', lambda k: 2 * k + 1),
    ('3kplus1', '3k+1', 'results_rq1_adversarial_3kplus1_wmape_root_based', lambda k: 3 * k + 1),
]


def load(base, eps, tmpl, method):
    fname = Path(base) / f'epsilon_{eps}' / tmpl / f'{method}_results.json'
    with open(fname) as f:
        return json.load(f)['summary']


def fmt(v):
    if v >= 100:
        return f'{v:.0f}'
    return f'{v:.1f}'


def fmt_se(v):
    if v >= 10:
        return f'{v:.0f}'
    return f'{v:.1f}'


def risk_se_per_template(s):
    """Per-template SE for risk_mae: sample std / sqrt(n)."""
    n = s.get('n_samples', 1) or 1
    return s['risk_std'] / np.sqrt(n)


def aggregate_risk_with_se(data, eps, method):
    summaries = [data[eps][t][method] for t in TEMPLATES]
    mean = float(np.mean([s['risk_mae'] for s in summaries]))
    se = float(np.sqrt(np.sum(np.array([risk_se_per_template(s) for s in summaries]) ** 2)) / len(TEMPLATES))
    return mean, se


def preload(base):
    data = {}
    for eps in EPS_VALS:
        data[eps] = {}
        for tmpl in TEMPLATES:
            data[eps][tmpl] = {}
            for m in METHODS:
                data[eps][tmpl][m] = load(base, eps, tmpl, m)
    return data


def make_aggregate_rce(data, run_key, run_label, budget_fn):
    out = [PMSTD, '']
    out.append('\\begin{table*}[t]')
    out.append('\\centering')
    out.append('\\setlength{\\tabcolsep}{2.8pt}')
    out.append('\\small')
    out.append(
        f'\\caption{{Aggregate Risk Class Error (\\%) averaged across 8 templates with bootstrap SE. '
        f'Budget $B = ({run_label}) \\cdot \\varepsilon$. Best mean per row in \\textbf{{bold}}.}}'
    )
    out.append(f'\\label{{tab:rq1_{run_key}_rce}}')
    out.append('\\begin{tabular}{cc|ccc|ccc|ccc}')
    out.append('\\toprule')
    out.append('& & \\multicolumn{3}{c|}{\\textbf{Exponential}} & '
               '\\multicolumn{3}{c|}{\\textbf{Bounded Laplace}} & '
               '\\multicolumn{3}{c}{\\textbf{Staircase}} \\\\')
    out.append('$\\varepsilon$ & $\\bar{B}$ & All & Roots & Opt & All & Roots & Opt & All & Roots & Opt \\\\')
    out.append('\\midrule')
    for eps in EPS_VALS:
        row = [aggregate_risk_with_se(data, eps, m) for m in METHODS]
        means = [d[0] for d in row]
        best_idx = min(range(9), key=lambda i: means[i])
        cells = []
        for i, (m, se) in enumerate(row):
            val = f'{fmt(m)}\\pmstd{{{fmt_se(se)}}}'
            cells.append(f'\\textbf{{{val}}}' if i == best_idx else val)
        avg_b = float(np.mean([budget_fn(K[t]) * eps for t in TEMPLATES]))
        out.append(f'{eps} & {avg_b:.2f} & ' + ' & '.join(cells) + ' \\\\')
    out.append('\\bottomrule')
    out.append('\\end{tabular}')
    out.append('\\end{table*}')
    return '\n'.join(out) + '\n'


def make_rce_per_template(all_data):
    out = [PMSTD, '']
    out.append('\\begin{table}[t]')
    out.append('\\centering')
    out.append('\\setlength{\\tabcolsep}{2pt}')
    out.append('\\small')
    out.append(
        '\\caption{Per-template Risk Class Error (\\%) at $\\varepsilon = 0.1$ for the discrete '
        'Exponential mechanism across three adversarial budgets, under the controlled harness (RQ1). '
        'Per-template SE = sample std / $\\sqrt{n}$ over patients ($n = 200$). '
        'Aggregate cross-mechanism results are in App.~\\ref{app:rq1-tables}, '
        'Tables~\\ref{tab:rq1_kplus1_rce}--\\ref{tab:rq1_3kplus1_rce}. Lower is better.}'
    )
    out.append('\\label{tab:rce_per_template_rq1}')
    out.append('\\begin{tabular}{l|ccc|ccc}')
    out.append('\\toprule')
    out.append('& \\multicolumn{3}{c|}{\\textbf{$\\mathcal{M}$-All}} & '
               '\\multicolumn{3}{c}{\\textbf{$\\mathcal{M}$-Opt}} \\\\')
    out.append('Template & $k{+}1$ & $2k{+}1$ & $3k{+}1$ & $k{+}1$ & $2k{+}1$ & $3k{+}1$ \\\\')
    out.append('\\midrule')
    for tmpl in PAPER_TMPL_ORDER:
        row_vals = []
        for cfg in ['all', 'opt']:
            for run_key, _, _, _ in RUNS:
                s = all_data[run_key][EPS_FOCAL][tmpl][f'exp_{cfg}']
                row_vals.append((s['risk_mae'], risk_se_per_template(s)))
        means = [v[0] for v in row_vals]
        best_idx = min(range(6), key=lambda i: means[i])
        cells = []
        for i, (m, se) in enumerate(row_vals):
            val = f'{fmt(m)}\\pmstd{{{fmt_se(se)}}}'
            cells.append(f'\\textbf{{{val}}}' if i == best_idx else val)
        out.append(f'{tmpl} & ' + ' & '.join(cells) + ' \\\\')
    out.append('\\bottomrule')
    out.append('\\end{tabular}')
    out.append('\\end{table}')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    ap.add_argument('--out-dir', default='tables/root_space',
                    help='Directory to write .tex files to (default: tables/root_space)')
    ap.add_argument('--data-prefix', default='',
                    help='Optional path prefix for the results_* directories '
                         '(useful if data lives outside this folder)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for run_key, run_label, base, _ in RUNS:
        full_base = Path(args.data_prefix) / base if args.data_prefix else base
        all_data[run_key] = preload(str(full_base))

    files = {}
    for run_key, run_label, _, budget_fn in RUNS:
        files[f'rq1_{run_key}_rce.tex'] = make_aggregate_rce(
            all_data[run_key], run_key, run_label, budget_fn
        )
    files['rq1_rce_per_template.tex'] = make_rce_per_template(all_data)

    for name, content in files.items():
        path = out_dir / name
        path.write_text(content)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
