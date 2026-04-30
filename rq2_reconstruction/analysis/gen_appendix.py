#!/usr/bin/env python3
"""
RQ3 v2 appendix LaTeX: full tables across all (eps_r, q, strategy, prior, mechanism).

Sections:
  1. Aggregate tables (mean across templates) for each mechanism family.
  2. Per-template Strategy A tables at informed prior (one per mechanism).
  3. Per-template Strategy B tables at informed prior (one per mechanism).
  4. Uniform-prior variants (aggregate).
  5. Strategy A metadata: r_star per template + its eps_{r*}^Opt.

Output: latex_rq3_v2_appendix.tex
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import (
    TEMPLATES, EPSILONS, Q_VALUES, PRIORS, MECHANISMS, METHOD_LABELS,
    TEMPLATE_PAIRS,
    load_result, load_allocation, compute_wmape, fmt,
)

# Write to the standard tables/ directory (sibling of analysis/) regardless
# of the caller's CWD. The legacy path was rq3_v3_deliverables/appendix.tex,
# left over from the project's pre-rename layout.
OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tables",
    "appendix.tex",
)


def aggregate(method, prior, strategy, eps, q):
    vals = []
    for t in TEMPLATES:
        r = load_result(t, method, prior, strategy, q, eps)
        m, _ = compute_wmape(r, t, strategy)
        if m is not None:
            vals.append(m)
    return float(np.mean(vals)) if vals else None


def per_template(tmpl, method, prior, strategy, eps, q):
    r = load_result(tmpl, method, prior, strategy, q, eps)
    m, _ = compute_wmape(r, tmpl, strategy)
    return m


# ───────── Section 1: Aggregate tables ──────────────────────────────────

def aggregate_table(mech_label, m_all, m_roots, m_opt, prior, strategy):
    """One table: rows = epsilons, cols = q × (M-All, M-Roots, M-Opt)."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\footnotesize")
    lines.append(r"\caption{" + mech_label +
                 r": aggregate reconstruction wMAPE (\%) across all templates. " +
                 r"Strategy " + strategy + r", " + prior + r" prior.}")
    lines.append(r"\label{tab:rq3_v2_agg_" +
                 f"{mech_label.replace(' ', '').lower()}_{strategy}_{prior}" + r"}")
    lines.append(r"\begin{tabular}{c" + "ccc" * len(Q_VALUES) + "}")
    lines.append(r"\toprule")
    top = [r"$\varepsilon_r$"] + [
        r"\multicolumn{3}{c}{$q=" + str(q) + r"$}" for q in Q_VALUES]
    lines.append(" & ".join(top) + r" \\")
    cmid = " ".join(
        r"\cmidrule(lr){" + f"{2+3*i}-{4+3*i}" + "}" for i in range(len(Q_VALUES)))
    lines.append(cmid)
    sub = [""]
    for _ in Q_VALUES:
        sub += ["M-All", "M-Roots", "M-Opt"]
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")
    for eps in EPSILONS:
        cells = [f"{eps}"]
        for q in Q_VALUES:
            for method in [m_all, m_roots, m_opt]:
                v = aggregate(method, prior, strategy, eps, q)
                cells.append(fmt(v))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ───────── Section 2/3: Per-template tables ─────────────────────────────

def per_template_table(mech_label, m_all, m_roots, m_opt, strategy, prior):
    """Side-by-side pair tables. Rows = epsilons, cols = q × 3 methods.
    Eight templates rendered as four pairs, each pair in its own table."""
    out = []
    for t1, t2 in TEMPLATE_PAIRS:
        lines = [r"\begin{table}[h]",
                 r"\centering",
                 r"\setlength{\tabcolsep}{2pt}",
                 r"\scriptsize",
                 r"\caption{" + mech_label + r", Strategy " + strategy +
                 r", " + prior + r" prior. Templates: " + t1 + r" / " + t2 +
                 r". wMAPE (\%).}",
                 r"\label{tab:rq3_v2_pt_" +
                 f"{mech_label.replace(' ', '').lower()}_{strategy}_{prior}_{t1}_{t2}" +
                 r"}",
                 r"\resizebox{\textwidth}{!}{",
                 r"\begin{tabular}{c|" + "ccc" * len(Q_VALUES) +
                 "|" + "ccc" * len(Q_VALUES) + "}",
                 r"\toprule"]
        top = [r"$\varepsilon_r$"]
        for tmpl in (t1, t2):
            top += [r"\multicolumn{" + str(3 * len(Q_VALUES)) +
                    r"}{c}{" + tmpl + r"}"]
        lines.append(" & ".join(top) + r" \\")
        q_top = [""]
        for _ in (t1, t2):
            for q in Q_VALUES:
                q_top.append(r"\multicolumn{3}{c}{$q=" + str(q) + r"$}")
        lines.append(" & ".join(q_top) + r" \\")
        sub = [""]
        for _ in (t1, t2):
            for _ in Q_VALUES:
                sub += ["All", "Rts", "Opt"]
        lines.append(" & ".join(sub) + r" \\")
        lines.append(r"\midrule")
        for eps in EPSILONS:
            cells = [f"{eps}"]
            for tmpl in (t1, t2):
                for q in Q_VALUES:
                    for method in [m_all, m_roots, m_opt]:
                        v = per_template(tmpl, method, prior, strategy, eps, q)
                        cells.append(fmt(v))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")
        lines.append(r"\end{table}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


# ───────── Section 5: Strategy A metadata ───────────────────────────────

def strategy_a_metadata():
    lines = [r"\begin{table}[h]",
             r"\centering",
             r"\footnotesize",
             r"\caption{Strategy A target root ($r^*$) and allocated "
             r"$\varepsilon_{r^*}^{\text{Opt}}$ per template and mechanism. "
             r"$r^* = \arg\max_r \varepsilon_r^{\text{Opt}}$ at "
             r"$\varepsilon_r = 0.1$.}",
             r"\label{tab:rq3_v2_rstar}",
             r"\begin{tabular}{lccc}",
             r"\toprule",
             r"Template & Exp $r^*$ ($\varepsilon^*$) & BLap $r^*$ ($\varepsilon^*$) & Stair $r^*$ ($\varepsilon^*$) \\",
             r"\midrule"]
    for t in TEMPLATES:
        cells = [t]
        for mech_label, _, _, _, _ in MECHANISMS:
            fam = {"Exponential": "exponential",
                   "Bounded Laplace": "bounded_laplace",
                   "Staircase": "staircase"}[mech_label]
            alloc_rec = load_allocation(fam, t, 0.1)
            if alloc_rec is None:
                cells.append("--")
                continue
            alloc = alloc_rec["allocation"]
            r_star = max(alloc, key=alloc.get)
            eps_star = alloc[r_star]
            cells.append(f"{r_star} ({eps_star:.3f})")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ───────── Compose document ─────────────────────────────────────────────

def main():
    parts = []
    parts.append(r"% RQ3 v2 appendix — auto-generated. Include in the main document.")
    parts.append(r"\section{RQ3 v2: Full Results (Appendix)}")

    parts.append(r"\subsection{Aggregate wMAPE (mean across templates)}")
    for mech_label, _, m_all, m_roots, m_opt in MECHANISMS:
        for strategy in ["A", "B"]:
            for prior in PRIORS:
                parts.append(aggregate_table(
                    mech_label, m_all, m_roots, m_opt, prior, strategy))

    parts.append(r"\clearpage")
    parts.append(r"\subsection{Per-template wMAPE — Strategy A, informed prior}")
    for mech_label, _, m_all, m_roots, m_opt in MECHANISMS:
        parts.append(per_template_table(
            mech_label, m_all, m_roots, m_opt, "A", "informed"))

    parts.append(r"\clearpage")
    parts.append(r"\subsection{Per-template wMAPE — Strategy B, informed prior}")
    for mech_label, _, m_all, m_roots, m_opt in MECHANISMS:
        parts.append(per_template_table(
            mech_label, m_all, m_roots, m_opt, "B", "informed"))

    parts.append(r"\clearpage")
    parts.append(r"\subsection{Strategy A target roots}")
    parts.append(strategy_a_metadata())

    tex = "\n\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(tex)
    print(f"Wrote {OUT_PATH} ({len(tex)} chars)")


if __name__ == "__main__":
    main()
