#!/usr/bin/env python3
"""
RQ3 v2 main table: reconstruction wMAPE at eps_r=0.1, Strategy B, informed prior.

Rows: 8 templates.
Columns: q=1 | q=4 | q=8, each with M-All | M-Roots | M-Opt wMAPE.
Mechanism: Exponential (main text). Other mechanisms go to appendix.

Output: latex_rq3_v2_main_table.tex
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import (
    TEMPLATES, Q_VALUES, METHOD_LABELS,
    load_result, compute_wmape, fmt,
)

EPS = 0.1
PRIORS = ["uniform", "informed"]
STRATEGY = "B"
METHOD_TRIPLE = ["vanilla", "vanilla_roots", "popabs"]  # M-All, M-Roots, M-Opt
MECH_LABEL = "Exponential"


def row_cells(tmpl, prior):
    cells = [tmpl]
    for q in Q_VALUES:
        for method in METHOD_TRIPLE:
            r = load_result(tmpl, method, prior, STRATEGY, q, EPS)
            m, _ = compute_wmape(r, tmpl, STRATEGY)
            cells.append(fmt(m))
    return cells


def render(prior):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\footnotesize")
    lines.append(r"\caption{RQ3 v2 main: reconstruction wMAPE (\%) at $\varepsilon_r=" + f"{EPS}" +
                 r"$, " + MECH_LABEL + r" mechanism, Strategy B (adversary makes $q$ queries " +
                 r"per root), " + prior + r" prior. M-All decreases with $q$ as expected; " +
                 r"M-Roots and M-Opt are flat (cached draw carries no new info).}")
    lines.append(r"\label{tab:rq3_v2_main_" + prior + r"}")
    lines.append(r"\begin{tabular}{l" + "ccc" * len(Q_VALUES) + "}")
    lines.append(r"\toprule")
    top = [""] + [r"\multicolumn{3}{c}{$q=" + str(q) + r"$}" for q in Q_VALUES]
    lines.append(" & ".join(top) + r" \\")
    cmid = " ".join(
        r"\cmidrule(lr){" + f"{2+3*i}-{4+3*i}" + "}" for i in range(len(Q_VALUES)))
    lines.append(cmid)
    sub = ["Template"]
    for _ in Q_VALUES:
        sub += [METHOD_LABELS[m] for m in METHOD_TRIPLE]
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")
    for t in TEMPLATES:
        cells = row_cells(t, prior)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tables",
    )
    os.makedirs(out_dir, exist_ok=True)
    for prior in PRIORS:
        out = os.path.join(out_dir, f"main_table_{prior}.tex")
        tex = render(prior)
        with open(out, "w") as f:
            f.write(tex)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
