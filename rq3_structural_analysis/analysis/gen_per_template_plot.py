#!/usr/bin/env python3
"""
RQ2 per-template grid: wMAPE curves for M-All, M-Root, M-Opt (nominal),
and M-Opt (fair) across all epsilon values and 3 mechanisms.

Data sources:
  - M-All, M-Root, M-Opt: results_v9_holdout/
  - M-Opt (fair):          results_rq2_implied/
  - Ground truth:          data/nhanes_benchmark_200.json
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Configuration ────────────────────────────────────────────────────────────

EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
TEMPLATES = ['ANEMIA', 'FIB4', 'AIP', 'CONICITY',
             'VASCULAR', 'TYG', 'HOMA', 'NLR']
N = 200

V9_BASE = "results_v9_holdout"
RQ2_BASE = "results_rq2_matched"
OUTPUT_DIR = "plots_rq1"

TARGET_KEYS = {
    "ANEMIA": "mchc", "AIP": "aip", "CONICITY": "conicity",
    "VASCULAR": "ppi", "FIB4": "fib4", "TYG": "tyg",
    "HOMA": "homa", "NLR": "nlr",
}

# Result file names per mechanism
V9_FILES = {
    "Exp":   {"all": "vanilla_results.json",
              "root": "vp_roots_results.json",
              "opt": "popabs_results.json"},
    "BLap":  {"all": "blap_all_results.json",
              "root": "blap_roots_results.json",
              "opt": "blap_roots_opt_results.json"},
    "Stair": {"all": "staircase_all_results.json",
              "root": "staircase_roots_results.json",
              "opt": "staircase_roots_opt_results.json"},
}

RQ2_OPT_FILES = {
    "Exp":   "Exp_opt_matched.json",
    "BLap":  "BLap_opt_matched.json",
    "Stair": "Stair_opt_matched.json",
}
RQ2_ROOT_FILES = {
    "Exp":   "Exp_roots_matched.json",
    "BLap":  "BLap_roots_matched.json",
    "Stair": "Stair_roots_matched.json",
}

MECHANISMS = ["Exp", "BLap", "Stair"]

# 3 variants per mechanism (all at matched distinguishability)
VARIANTS = ["All", "RootM", "OptM"]

FAMILY_COLORS = {
    "Exp":   "#0072B2",
    "BLap":  "#E69F00",
    "Stair": "#009E73",
}

VARIANT_STYLES = {
    "All":   {"linestyle": "-",  "linewidth": 1.4, "alpha": 0.45},
    "Root":  {"linestyle": "--", "linewidth": 1.8, "alpha": 0.6},
    "Opt":   {"linestyle": "-.", "linewidth": 2.0, "alpha": 0.75},
    "RootM": {"linestyle": "--", "linewidth": 1.8, "alpha": 1.0},
    "OptM":  {"linestyle": "-",  "linewidth": 1.8, "alpha": 1.0},
}

VARIANT_MARKERS = {"All": "o", "Root": "s", "Opt": "D", "RootM": "^", "OptM": "*"}
MARKER_SIZES = {"All": 5, "Root": 6, "Opt": 6, "RootM": 7, "OptM": 9}


# ── wMAPE computation ───────────────────────────────────────────────────────

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


# ── Data loading ────────────────────────────────────────────────────────────

def load_ground_truth():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "nhanes_benchmark_200.json",
    )
    with open(data_path) as f:
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
    """Load wMAPE for all (eps, tmpl, mech, variant) combos.

    Returns: {(eps, tmpl, mech, variant): wmape_value}
    """
    data = {}
    missing = 0
    for eps in EPSILONS:
        for tmpl in TEMPLATES:
            for mech in MECHANISMS:
                # M-All from v9_holdout
                path = os.path.join(V9_BASE, f"epsilon_{eps}", tmpl,
                                    V9_FILES[mech]["all"])
                if not os.path.exists(path):
                    missing += 1
                    continue
                with open(path) as f:
                    d = json.load(f)
                w = compute_wmape(gt[tmpl], d["sanitized_values"], tmpl)
                data[(eps, tmpl, mech, "All")] = w

                # M-Opt (matched) from rq2_matched
                opt_m_path = os.path.join(RQ2_BASE, f"epsilon_{eps}", tmpl,
                                          RQ2_OPT_FILES[mech])
                if not os.path.exists(opt_m_path):
                    missing += 1
                else:
                    with open(opt_m_path) as f:
                        d = json.load(f)
                    w = compute_wmape(gt[tmpl], d["sanitized_values"], tmpl)
                    data[(eps, tmpl, mech, "OptM")] = w

                # M-Roots (matched) from rq2_matched
                root_m_path = os.path.join(RQ2_BASE, f"epsilon_{eps}", tmpl,
                                           RQ2_ROOT_FILES[mech])
                if not os.path.exists(root_m_path):
                    missing += 1
                else:
                    with open(root_m_path) as f:
                        d = json.load(f)
                    w = compute_wmape(gt[tmpl], d["sanitized_values"], tmpl)
                    data[(eps, tmpl, mech, "RootM")] = w

    if missing:
        print(f"WARNING: {missing} missing files")
    return data


# ── Plotting ────────────────────────────────────────────────────────────────

def load_s_dom():
    """Load S_dom per template for x-axis computation."""
    s_dom_path = os.path.join("results_rq2_implied", "s_dom_info.json")
    if not os.path.exists(s_dom_path):
        sys.stderr.write(
            "gen_per_template_plot.py: this appendix-only plot requires\n"
            "  results_rq2_implied/s_dom_info.json,\n"
            "which the public artifact does not ship (no producer script is\n"
            "included). The body of RQ3 reproduces without it; see AUDIT.md\n"
            "for the canonical scripts. Skipping.\n"
        )
        sys.exit(2)
    with open(s_dom_path) as f:
        sdom = json.load(f)
    return {tmpl: sdom[tmpl]["S_dom"] for tmpl in TEMPLATES}


def plot_per_template_grid(data, output_path):
    import warnings

    S_DOM = load_s_dom()

    TMPL_ORDER = ['FIB4', 'HOMA', 'VASCULAR', 'NLR',
                  'ANEMIA', 'AIP', 'CONICITY', 'TYG']

    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharey=False)

    for idx, tmpl in enumerate(TMPL_ORDER):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]
        target = TARGET_KEYS[tmpl]
        s_dom = S_DOM[tmpl]

        for mech in MECHANISMS:
            color = FAMILY_COLORS[mech]
            for variant in VARIANTS:
                style = VARIANT_STYLES[variant]
                marker = VARIANT_MARKERS[variant]
                ms = MARKER_SIZES[variant]

                vals, deltas = [], []
                for eps in EPSILONS:
                    key = (eps, tmpl, mech, variant)
                    if key in data:
                        vals.append(data[key])
                        deltas.append(eps * s_dom)

                if not deltas:
                    continue

                label = f"{mech}-{variant}"
                ax.plot(deltas, vals, marker=marker,
                        linestyle=style["linestyle"],
                        color=color, linewidth=style["linewidth"],
                        markersize=ms, label=label,
                        alpha=style["alpha"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{tmpl} (target: {target})",
                     fontsize=16, fontweight='bold')
        ax.set_xlabel(r"$\Delta_{\mathrm{All}}^{\mathrm{dom}}$", fontsize=14)
        if col == 0:
            ax.set_ylabel("wMAPE (%)", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.25)

        if idx == 0:
            ax.legend(fontsize=10, loc='upper right', ncol=3,
                      columnspacing=1.0, handletextpad=0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    png_path = output_path.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {png_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ground truth...")
    gt = load_ground_truth()

    print("Loading all data and computing wMAPE...")
    data = load_all_data(gt)
    print(f"Loaded {len(data)} data points")

    output_path = os.path.join(OUTPUT_DIR, "rq2_per_template_grid.pdf")
    plot_per_template_grid(data, output_path)


if __name__ == "__main__":
    main()
