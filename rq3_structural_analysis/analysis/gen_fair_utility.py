#!/usr/bin/env python3
"""
RQ2 Fair Utility: Generate tables comparing M-All (nominal) vs M-Opt (nominal)
vs M-Opt (fair/true budget) for each template.

Uses wMAPE (weighted Mean Absolute Percentage Error) for the raw metric
and mean risk class error for the risk metric.

Outputs:
  - Main-text summary table at a representative epsilon
  - Per-template appendix tables (all epsilon x 3 mechanisms)
"""
# _REPO_ROOT_BOOTSTRAP: ensure repo root is on sys.path so that
# 'from utils.*' / 'from preempt.*' imports resolve when this script
# is run from its own subfolder.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import json, os, math
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

TEMPLATES = ["ANEMIA", "AIP", "CONICITY", "VASCULAR",
             "FIB4", "TYG", "HOMA", "NLR"]
MECHANISMS = ["Exp", "BLap", "Stair"]
EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

V9_BASE = "results_v9_holdout"
RQ2_BASE = "results_rq2_matched"
N = 200

TARGET_KEYS = {
    "ANEMIA": "mchc", "AIP": "aip", "CONICITY": "conicity",
    "VASCULAR": "ppi", "FIB4": "fib4", "TYG": "tyg",
    "HOMA": "homa", "NLR": "nlr",
}

TARGET_NAMES = {
    "ANEMIA": "MCHC", "AIP": "AIP", "CONICITY": "Conicity",
    "VASCULAR": "PPI", "FIB4": "FIB-4", "TYG": "TyG",
    "HOMA": "HOMA-IR", "NLR": "NLR",
}

# Result file names in v9_holdout
V9_FILES = {
    "Exp":   {"all": "vanilla_results.json",
              "opt": "popabs_results.json"},
    "BLap":  {"all": "blap_all_results.json",
              "opt": "blap_roots_opt_results.json"},
    "Stair": {"all": "staircase_all_results.json",
              "opt": "staircase_roots_opt_results.json"},
}

# M-Root: roots-only with uniform budget allocation
V9_ROOT_FILES = {
    "Exp":   "vp_roots_results.json",
    "BLap":  "blap_roots_results.json",
    "Stair": "staircase_roots_results.json",
}

# Result file names in rq2_implied
RQ2_FILES = {
    "Exp":   "Exp_opt_matched.json",
    "BLap":  "BLap_opt_matched.json",
    "Stair": "Stair_opt_matched.json",
}

RQ2_ROOT_FILES = {
    "Exp":   "Exp_roots_matched.json",
    "BLap":  "BLap_roots_matched.json",
    "Stair": "Stair_roots_matched.json",
}

REPR_EPS = 0.1

THREE_CLASS = {"ANEMIA", "AIP", "FIB4", "HOMA"}


# ── Risk classification (hardcoded from plots/plotting.py) ───────────────────

def get_risk_class(tmpl, val):
    try:
        val = float(val)
    except Exception:
        return 0
    if tmpl == "ANEMIA":
        return 0 if val < 32 else (2 if val > 36 else 1)
    elif tmpl == "AIP":
        return 0 if val < 0.11 else (1 if val <= 0.21 else 2)
    elif tmpl == "CONICITY":
        return 1 if val > 1.25 else 0
    elif tmpl == "VASCULAR":
        return 1 if val > 0.60 else 0
    elif tmpl == "FIB4":
        return 0 if val < 1.30 else (2 if val > 2.67 else 1)
    elif tmpl == "TYG":
        return 1 if val > 8.5 else 0
    elif tmpl == "HOMA":
        return 0 if val < 1.0 else (1 if val < 2.5 else 2)
    elif tmpl == "NLR":
        return 1 if val >= 3.0 else 0
    return 0


def max_risk_dist(tmpl):
    return 2 if tmpl in THREE_CLASS else 1


# ── Metric computation ──────────────────────────────────────────────────────

RNG = np.random.RandomState(42)
N_BOOTSTRAP = 1000


def compute_wmape(gt_vals, sanitized_list, tmpl):
    """wMAPE = sum(|san - gt|) / sum(|gt|) * 100."""
    tgt = TARGET_KEYS[tmpl]
    n = min(len(gt_vals), len(sanitized_list))
    gt = np.array([float(gt_vals[i]) for i in range(n)])
    san = np.array([float(sanitized_list[i][tgt]) for i in range(n)])
    denom = np.sum(np.abs(gt))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(san - gt)) / denom * 100)


def bootstrap_wmape_se(gt_vals, sanitized_list, tmpl):
    """Bootstrap standard error for wMAPE."""
    tgt = TARGET_KEYS[tmpl]
    n = min(len(gt_vals), len(sanitized_list))
    gt = np.array([float(gt_vals[i]) for i in range(n)])
    san = np.array([float(sanitized_list[i][tgt]) for i in range(n)])
    abs_err = np.abs(san - gt)
    abs_gt = np.abs(gt)
    boots = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = RNG.randint(0, n, size=n)
        denom = np.sum(abs_gt[idx])
        boots[b] = np.sum(abs_err[idx]) / denom * 100 if denom > 0 else 0.0
    return float(np.std(boots))


def compute_risk_error(gt_vals, sanitized_list, tmpl):
    """Mean risk class error (%)."""
    tgt = TARGET_KEYS[tmpl]
    md = max_risk_dist(tmpl)
    n = min(len(gt_vals), len(sanitized_list))
    errs = []
    for i in range(n):
        gt_v = gt_vals[i]
        pred_v = float(sanitized_list[i][tgt])
        gc = get_risk_class(tmpl, gt_v)
        pc = get_risk_class(tmpl, pred_v)
        errs.append(abs(pc - gc) / md * 100.0)
    return sum(errs) / len(errs) if errs else 0.0


# ── Load data ────────────────────────────────────────────────────────────────

def load_ground_truth():
    with open("../../data/nhanes_benchmark_200.json") as f:
        raw = json.load(f)
    gt = {}
    for tmpl in TEMPLATES:
        tgt = TARGET_KEYS[tmpl]
        gt[tmpl] = []
        for sample in raw[tmpl][:N]:
            vals = {}
            for turn in sample["turns"]:
                vals.update(turn["private_value"])
            gt[tmpl].append(float(vals[tgt]))
    return gt


def load_all_data(gt):
    """Load all data: M-All, M-Root, M-Opt (nominal), M-Opt (fair)."""
    rows = []
    missing = []
    for eps in EPSILONS:
        for tmpl in TEMPLATES:
            for mech in MECHANISMS:
                # M-All
                all_path = os.path.join(V9_BASE, f"epsilon_{eps}", tmpl,
                                        V9_FILES[mech]["all"])
                # M-Root (uniform budget on roots)
                root_path = os.path.join(V9_BASE, f"epsilon_{eps}", tmpl,
                                         V9_ROOT_FILES[mech])
                # M-Opt (nominal)
                nom_path = os.path.join(V9_BASE, f"epsilon_{eps}", tmpl,
                                        V9_FILES[mech]["opt"])
                # M-Opt (fair)
                fair_path = os.path.join(RQ2_BASE, f"epsilon_{eps}", tmpl,
                                         RQ2_FILES[mech])
                # M-Roots (fair)
                roots_fair_path = os.path.join(RQ2_BASE, f"epsilon_{eps}", tmpl,
                                               RQ2_ROOT_FILES[mech])

                paths = [all_path, root_path, nom_path, fair_path,
                         roots_fair_path]
                for p in paths:
                    if not os.path.exists(p):
                        missing.append(p)

                if any(not os.path.exists(p) for p in paths):
                    continue

                with open(all_path) as f:
                    all_data = json.load(f)
                with open(root_path) as f:
                    root_data = json.load(f)
                with open(nom_path) as f:
                    nom_data = json.load(f)
                with open(fair_path) as f:
                    fair_data = json.load(f)
                with open(roots_fair_path) as f:
                    roots_fair_data = json.load(f)

                rows.append({
                    "eps": eps, "tmpl": tmpl, "mech": mech,
                    "all_wmape": compute_wmape(
                        gt[tmpl], all_data["sanitized_values"], tmpl),
                    "all_wmape_se": bootstrap_wmape_se(
                        gt[tmpl], all_data["sanitized_values"], tmpl),
                    "all_risk": compute_risk_error(
                        gt[tmpl], all_data["sanitized_values"], tmpl),
                    "root_wmape": compute_wmape(
                        gt[tmpl], root_data["sanitized_values"], tmpl),
                    "root_wmape_se": bootstrap_wmape_se(
                        gt[tmpl], root_data["sanitized_values"], tmpl),
                    "root_risk": compute_risk_error(
                        gt[tmpl], root_data["sanitized_values"], tmpl),
                    "opt_nom_wmape": compute_wmape(
                        gt[tmpl], nom_data["sanitized_values"], tmpl),
                    "opt_nom_wmape_se": bootstrap_wmape_se(
                        gt[tmpl], nom_data["sanitized_values"], tmpl),
                    "opt_nom_risk": compute_risk_error(
                        gt[tmpl], nom_data["sanitized_values"], tmpl),
                    "opt_fair_wmape": compute_wmape(
                        gt[tmpl], fair_data["sanitized_values"], tmpl),
                    "opt_fair_wmape_se": bootstrap_wmape_se(
                        gt[tmpl], fair_data["sanitized_values"], tmpl),
                    "opt_fair_risk": compute_risk_error(
                        gt[tmpl], fair_data["sanitized_values"], tmpl),
                    "roots_fair_wmape": compute_wmape(
                        gt[tmpl], roots_fair_data["sanitized_values"], tmpl),
                    "roots_fair_wmape_se": bootstrap_wmape_se(
                        gt[tmpl], roots_fair_data["sanitized_values"], tmpl),
                    "roots_fair_risk": compute_risk_error(
                        gt[tmpl], roots_fair_data["sanitized_values"], tmpl),
                })

    if missing:
        print(f"WARNING: {len(missing)} missing files")
        for p in missing[:5]:
            print(f"  {p}")
    return rows


def load_budget_info():
    """Load domain-space S_dom and compute per-template info.

    Returns:
        ratios: {tmpl: S_dom / sum(D_r)} — domain-space amplification factor
        n_nodes: {tmpl: n} — number of non-classification nodes
        s_dom_vals: {tmpl: S_dom} — domain-space max separation
    """
    with open(os.path.join("results_rq2_implied", "s_dom_info.json")) as f:
        sdom = json.load(f)
    from preempt.sanitizer import set_template_domains, MEDICAL_DOMAINS

    TMPL_ROOTS = {
        "ANEMIA": ["hb", "hct", "rbc"], "AIP": ["tc", "hdl", "tg"],
        "CONICITY": ["wt", "ht", "waist"], "VASCULAR": ["sbp", "dbp"],
        "FIB4": ["age", "ast", "alt", "plt"], "TYG": ["tg", "glu"],
        "HOMA": ["glu", "ins"], "NLR": ["neu", "lym"],
    }

    ratios = {}
    n_nodes = {}
    s_dom_vals = {}
    for tmpl in TEMPLATES:
        s_dom = sdom[tmpl]["S_dom"]
        n = len(sdom[tmpl]["per_node_contributions"])
        set_template_domains(tmpl)
        sum_D = sum(
            MEDICAL_DOMAINS[r][1] - MEDICAL_DOMAINS[r][0]
            for r in TMPL_ROOTS[tmpl]
        )
        ratios[tmpl] = s_dom / sum_D
        n_nodes[tmpl] = n
        s_dom_vals[tmpl] = s_dom
    return ratios, n_nodes, s_dom_vals


# ── Formatting ───────────────────────────────────────────────────────────────

def _fmt(v):
    if v >= 100:
        return f"{v:.0f}"
    elif v >= 10:
        return f"{v:.1f}"
    elif v >= 1:
        return f"{v:.2f}"
    else:
        return f"{v:.2f}"


def _fmt_se(v):
    """Format SE value (typically smaller, use fewer digits)."""
    if v >= 10:
        return f"{v:.1f}"
    elif v >= 1:
        return f"{v:.1f}"
    else:
        return f"{v:.2f}"


def _fmt_pm(v, se):
    """Format value ± SE for LaTeX."""
    return _fmt(v) + r"{\scriptsize$\pm$" + _fmt_se(se) + "}"


def _bold_winner(all_v, fair_v):
    """Return (all_str, fair_str) with the winner bolded."""
    all_s = _fmt(all_v)
    fair_s = _fmt(fair_v)
    if fair_v < all_v - 0.005:
        return all_s, r"\textbf{" + fair_s + "}"
    elif all_v < fair_v - 0.005:
        return r"\textbf{" + all_s + "}", fair_s
    else:
        return all_s, fair_s


def _bold_winner_pm(all_v, all_se, fair_v, fair_se):
    """Return (all_str, fair_str) with ±SE and winner bolded."""
    all_s = _fmt_pm(all_v, all_se)
    fair_s = _fmt_pm(fair_v, fair_se)
    if fair_v < all_v - 0.005:
        return all_s, r"\textbf{" + fair_s + "}"
    elif all_v < fair_v - 0.005:
        return r"\textbf{" + all_s + "}", fair_s
    else:
        return all_s, fair_s


# ── Text summary ─────────────────────────────────────────────────────────────

def print_text_tables(rows, ratios, n_nodes, s_dom_vals):
    print("=" * 150)
    print(f"MAIN TABLE: Fair utility comparison at eps={REPR_EPS} (wMAPE %)")
    print(f"  Domain-space weighted constraint: sum(eps_r * D_r) = eps * S_dom")
    print("=" * 150)
    print(f"{'Template':<12} {'S_dom':>12} {'Ratio':>8} "
          f"{'M-All':>8} {'M-Root':>8} {'M-Opt':>8} {'RootM':>8} {'OptM':>8}  "
          f"{'M-All':>8} {'M-Root':>8} {'M-Opt':>8} {'RootM':>8} {'OptM':>8}  "
          f"{'M-All':>8} {'M-Root':>8} {'M-Opt':>8} {'RootM':>8} {'OptM':>8}")
    print(f"{'':12} {'':12} {'':8} "
          f"{'--- Exp ---':^44}  {'--- BLap ---':^44}  {'--- Stair ---':^44}")
    print("-" * 150)
    for tmpl in TEMPLATES:
        ratio = ratios[tmpl]
        s_dom = s_dom_vals[tmpl]
        cells = f"{tmpl:<12} {s_dom:>12.1f} {ratio:>7.2f}x "
        for mech in MECHANISMS:
            r = [x for x in rows
                 if x["tmpl"] == tmpl and x["eps"] == REPR_EPS
                 and x["mech"] == mech][0]
            cells += (f"{r['all_wmape']:>8.2f} {r['root_wmape']:>8.2f} "
                      f"{r['opt_nom_wmape']:>8.2f} "
                      f"{r['roots_fair_wmape']:>8.2f} "
                      f"{r['opt_fair_wmape']:>8.2f}  ")
        print(cells)


# ── LaTeX generation ─────────────────────────────────────────────────────────

def generate_latex(rows, ratios, n_nodes, s_dom_vals):
    out = []
    out.append(r"% ==========================================================================")
    out.append(r"% RQ2: Fair utility comparison — M-All vs M-Opt (nominal) vs M-Opt (fair)")
    out.append(r"% Generated by gen_rq2_fair_utility.py")
    out.append(r"% ==========================================================================")
    out.append("")

    # ── Main-text summary table ──────────────────────────────────────────────

    out.append(r"% Main-text table at eps=" + str(REPR_EPS))
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\small")
    out.append(r"\caption{wMAPE (\%) at matched domain-space distinguishability ($\varepsilon = "
               + str(REPR_EPS) + r"$).  "
               r"\textbf{Columns:} "
               r"$R = S_{\mathrm{dom}} / \sum_r D_r$ is the amplification ratio (Table~\ref{tab:amplification}); "
               r"\emph{All} is $\mathcal{M}$-All at nominal per-node budget $\varepsilon$; "
               r"\emph{OptM} is \sysname{} at the matched budget $B^*$ satisfying "
               r"$\Delta_{\mathrm{PP}}^{\mathrm{dom}}(B^*) = \Delta_{\mathrm{All}}^{\mathrm{dom}}(\varepsilon)$ "
               r"(Sec.~\ref{sec:experiments-matching}).  "
               r"Both methods are compared at the same domain-space distinguishability level.  "
               r"Winner in \textbf{bold}; $\pm$ denotes bootstrap SE ($N{=}200$, 1000 resamples).}")
    out.append(r"\label{tab:fair-utility}")
    out.append(r"\begin{tabular}{lc|cc|cc|cc}")
    out.append(r"\toprule")
    out.append(r"& & \multicolumn{2}{c|}{\textbf{Exp}} "
               r"& \multicolumn{2}{c|}{\textbf{BLap}} "
               r"& \multicolumn{2}{c}{\textbf{Stair}} \\")
    out.append(r"\textbf{Template} & $R$ & All & OptM "
               r"& All & OptM "
               r"& All & OptM \\")
    out.append(r"\midrule")

    for tmpl in TEMPLATES:
        ratio = ratios[tmpl]
        cells = [f"{tmpl}", f"${ratio:.2f}\\times$"]
        for mech in MECHANISMS:
            r = [x for x in rows
                 if x["tmpl"] == tmpl and x["eps"] == REPR_EPS
                 and x["mech"] == mech][0]
            all_s, opt_fair_s = _bold_winner_pm(
                r["all_wmape"], r["all_wmape_se"],
                r["opt_fair_wmape"], r["opt_fair_wmape_se"])
            cells.extend([all_s, opt_fair_s])
        out.append(" & ".join(cells) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    out.append("")

    # ── Nominal appendix tables (M-All, M-Root, M-Opt) ───────────────────────

    METRIC_DEFS = [
        {"key": "wmape", "label": "wMAPE", "has_se": True,
         "all": "all_wmape", "all_se": "all_wmape_se",
         "root": "root_wmape", "root_se": "root_wmape_se",
         "nom": "opt_nom_wmape", "nom_se": "opt_nom_wmape_se"},
        {"key": "risk", "label": "Risk Class Error", "has_se": False,
         "all": "all_risk", "root": "root_risk",
         "nom": "opt_nom_risk"},
    ]

    app_nom = []
    app_nom.append(r"\subsection{Per-Template Utility Results (Nominal Budget)}")
    app_nom.append(r"\label{app:nominal-utility-details}")
    app_nom.append("")
    app_nom.append(r"Tables~\ref{tab:rq2-nom-anemia-wmape}--\ref{tab:rq2-nom-nlr-risk} "
                   r"report wMAPE (\%) and Risk Classification Error (\%) "
                   r"for all eight templates across ten privacy budgets "
                   r"$\varepsilon \in \{0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0\}$, "
                   r"all at the nominal budget $B_{\mathrm{nom}} = n\varepsilon$.  "
                   r"These tables compare three methods: "
                   r"$\mathcal{M}$-All (independent noising of all nodes), "
                   r"$\mathcal{M}$-Root (uniform root-only noising), and "
                   r"$\mathcal{M}$-Opt (sensitivity-weighted root-only noising).  "
                   r"All use the same total budget; the comparison isolates the effect "
                   r"of post-processing and optimal allocation without any privacy-cost adjustment.")
    app_nom.append("")

    for mdef in METRIC_DEFS:
        for tmpl in TEMPLATES:
            tgt = TARGET_NAMES[tmpl]
            n = n_nodes[tmpl]
            label = f"rq2-nom-{tmpl.lower()}-{mdef['key']}"

            app_nom.append(r"\begin{table}[t]")
            app_nom.append(r"\centering")
            app_nom.append(r"\setlength{\tabcolsep}{3pt}")
            app_nom.append(r"\footnotesize")
            cap = (f"{mdef['label']} (\\%) for {tmpl} (target: {tgt}). "
                   r"\textbf{Columns:} "
                   r"$\varepsilon$ is the per-node privacy parameter; "
                   r"$B_{\mathrm{nom}} = n\varepsilon$ is the total nominal budget; "
                   r"\emph{M-All} noises all $n$ nodes independently at $\varepsilon$; "
                   r"\emph{M-Root} noises only the $k$ roots with uniform allocation $B_{\mathrm{nom}}/k$; "
                   r"\emph{M-Opt} noises only the $k$ roots with sensitivity-weighted allocation "
                   r"(Eq.~\ref{eq:opt-budget}).  "
                   r"Winner between M-All and M-Opt in \textbf{bold}.")
            if mdef["has_se"]:
                cap += r"  $\pm$ denotes bootstrap SE ($N{=}200$, 1000 resamples)."
            app_nom.append(r"\caption{" + cap + "}")
            app_nom.append(r"\label{tab:" + label + "}")

            app_nom.append(r"\begin{tabular}{cc|ccc|ccc|ccc}")
            app_nom.append(r"\toprule")
            app_nom.append(r"& & \multicolumn{3}{c|}{\textbf{Exp}} "
                           r"& \multicolumn{3}{c|}{\textbf{BLap}} "
                           r"& \multicolumn{3}{c}{\textbf{Stair}} \\")
            app_nom.append(r"$\varepsilon$ & $B_{\mathrm{nom}}$ "
                           r"& M-All & M-Root & M-Opt "
                           r"& M-All & M-Root & M-Opt "
                           r"& M-All & M-Root & M-Opt \\")
            app_nom.append(r"\midrule")

            for eps in EPSILONS:
                b_nom = n * eps
                cells = [str(eps), f"{b_nom:.3f}"]
                for mech in MECHANISMS:
                    r = [x for x in rows
                         if x["tmpl"] == tmpl and x["eps"] == eps
                         and x["mech"] == mech][0]
                    if mdef["has_se"]:
                        all_s, nom_s = _bold_winner_pm(
                            r[mdef["all"]], r[mdef["all_se"]],
                            r[mdef["nom"]], r[mdef["nom_se"]])
                        root_s = _fmt_pm(r[mdef["root"]], r[mdef["root_se"]])
                    else:
                        all_s, nom_s = _bold_winner(
                            r[mdef["all"]], r[mdef["nom"]])
                        root_s = _fmt(r[mdef["root"]])
                    cells.extend([all_s, root_s, nom_s])
                app_nom.append(" & ".join(cells) + r" \\")

            app_nom.append(r"\bottomrule")
            app_nom.append(r"\end{tabular}")
            app_nom.append(r"\end{table}")
            app_nom.append("")

    # ── Matched appendix tables (M-All, RootM, OptM) ───────────────────────

    METRIC_DEFS_M = [
        {"key": "wmape", "label": "wMAPE", "has_se": True,
         "all": "all_wmape", "all_se": "all_wmape_se",
         "roots_m": "roots_fair_wmape", "roots_m_se": "roots_fair_wmape_se",
         "opt_m": "opt_fair_wmape", "opt_m_se": "opt_fair_wmape_se"},
        {"key": "risk", "label": "Risk Class Error", "has_se": False,
         "all": "all_risk",
         "roots_m": "roots_fair_risk",
         "opt_m": "opt_fair_risk"},
    ]

    app_matched = []
    app_matched.append(r"\subsection{Per-Template Utility at Matched Distinguishability}")
    app_matched.append(r"\label{app:matched-utility-details}")
    app_matched.append("")
    app_matched.append(r"Tables~\ref{tab:rq2-matched-anemia-wmape}--\ref{tab:rq2-matched-nlr-risk} "
                       r"compare $\mathcal{M}$-All against $\mathcal{M}$-Root and $\mathcal{M}$-Opt "
                       r"at matched domain-space distinguishability "
                       r"($\Delta_{\mathrm{PP}}^{\mathrm{dom}} = \Delta_{\mathrm{All}}^{\mathrm{dom}}$, "
                       r"Sec.~\ref{sec:experiments-matching}) "
                       r"across ten $\varepsilon$ values.  "
                       r"For each $\varepsilon$, $\mathcal{M}$-All runs at its nominal parameter "
                       r"while the root-only methods run at the budget $B^*$ that produces "
                       r"the same domain-space distinguishability as $\mathcal{M}$-All.  "
                       r"This isolates the utility difference attributable to the "
                       r"privacy amplification from derived releases.")
    app_matched.append("")

    for mdef in METRIC_DEFS_M:
        for tmpl in TEMPLATES:
            tgt = TARGET_NAMES[tmpl]
            ratio = ratios[tmpl]
            n = n_nodes[tmpl]
            s_dom = s_dom_vals[tmpl]
            label = f"rq2-matched-{tmpl.lower()}-{mdef['key']}"

            app_matched.append(r"\begin{table}[t]")
            app_matched.append(r"\centering")
            app_matched.append(r"\setlength{\tabcolsep}{3pt}")
            app_matched.append(r"\footnotesize")
            cap = (f"{mdef['label']} (\\%) for {tmpl} (target: {tgt}, "
                   f"$R = {ratio:.2f}\\times$). "
                   r"All methods compared at equal domain-space distinguishability.  "
                   r"\textbf{Columns:} "
                   r"$\varepsilon$ is $\mathcal{M}$-All's per-node parameter; "
                   r"$\Delta_{\mathrm{All}}^{\mathrm{dom}} = \varepsilon \cdot S_{\mathrm{dom}}$ "
                   r"is the shared distinguishability level; "
                   r"\emph{M-All} noises all $n$ nodes independently at $\varepsilon$; "
                   r"\emph{RootM} noises only the $k$ roots with uniform allocation at the matched budget "
                   r"$B^*$ where $\Delta_{\mathrm{PP}}^{\mathrm{dom}}(B^*) = \Delta_{\mathrm{All}}^{\mathrm{dom}}$; "
                   r"\emph{OptM} uses sensitivity-weighted allocation at the same matched budget.  "
                   r"Winner between M-All and OptM in \textbf{bold}.")
            if mdef["has_se"]:
                cap += r"  $\pm$ denotes bootstrap SE ($N{=}200$, 1000 resamples)."
            app_matched.append(r"\caption{" + cap + "}")
            app_matched.append(r"\label{tab:" + label + "}")

            app_matched.append(r"\begin{tabular}{cc|ccc|ccc|ccc}")
            app_matched.append(r"\toprule")
            app_matched.append(r"& & \multicolumn{3}{c|}{\textbf{Exp}} "
                               r"& \multicolumn{3}{c|}{\textbf{BLap}} "
                               r"& \multicolumn{3}{c}{\textbf{Stair}} \\")
            app_matched.append(r"$\varepsilon$ & $\Delta_{\mathrm{All}}^{\mathrm{dom}}$ "
                               r"& M-All & RootM & OptM "
                               r"& M-All & RootM & OptM "
                               r"& M-All & RootM & OptM \\")
            app_matched.append(r"\midrule")

            for eps in EPSILONS:
                delta_all = s_dom * eps
                cells = [str(eps), f"{delta_all:.1f}"]
                for mech in MECHANISMS:
                    r = [x for x in rows
                         if x["tmpl"] == tmpl and x["eps"] == eps
                         and x["mech"] == mech][0]
                    if mdef["has_se"]:
                        all_s, opt_m_s = _bold_winner_pm(
                            r[mdef["all"]], r[mdef["all_se"]],
                            r[mdef["opt_m"]], r[mdef["opt_m_se"]])
                        roots_m_s = _fmt_pm(
                            r[mdef["roots_m"]], r[mdef["roots_m_se"]])
                    else:
                        all_s, opt_m_s = _bold_winner(
                            r[mdef["all"]], r[mdef["opt_m"]])
                        roots_m_s = _fmt(r[mdef["roots_m"]])
                    cells.extend([all_s, roots_m_s, opt_m_s])
                app_matched.append(" & ".join(cells) + r" \\")

            app_matched.append(r"\bottomrule")
            app_matched.append(r"\end{tabular}")
            app_matched.append(r"\end{table}")
            app_matched.append("")

    # ── Write output files ─────────────────────────────────────────────────

    main_path = "latex_rq2_fair_utility_main.tex"
    with open(main_path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nMain-text table written to {main_path} ({len(out)} lines)")

    app_nom_path = "latex_rq2_fair_utility_appendix.tex"
    with open(app_nom_path, "w") as f:
        f.write("\n".join(app_nom) + "\n")
    print(f"Nominal appendix written to {app_nom_path} ({len(app_nom)} lines)")

    app_matched_path = "latex_rq2_matched_appendix.tex"
    with open(app_matched_path, "w") as f:
        f.write("\n".join(app_matched) + "\n")
    print(f"Matched appendix written to {app_matched_path} ({len(app_matched)} lines)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    gt = load_ground_truth()
    rows = load_all_data(gt)
    ratios, n_nodes, s_dom_vals = load_budget_info()

    print_text_tables(rows, ratios, n_nodes, s_dom_vals)
    generate_latex(rows, ratios, n_nodes, s_dom_vals)


if __name__ == "__main__":
    main()
