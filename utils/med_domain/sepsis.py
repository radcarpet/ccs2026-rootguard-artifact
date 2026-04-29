import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def estimate_bi_lipschitz(func_name, domain, samples=10000):
    """
    Estimates m and L by sampling the gradient norm over a physiological domain.
    """
    # Define clinical ranges from your generator
    # Stable: hr(70-90), sbp(115-135), dbp(70-85)
    # Shock:  hr(110-145), sbp(75-95), dbp(35-50)
    hr = np.random.uniform(70, 145, samples)
    sbp = np.random.uniform(75, 135, samples)
    dbp = np.random.uniform(35, 85, samples)
    
    grads = []

    if func_name == "shock_index":
        # f(hr, sbp) = hr / sbp
        # Grad = [1/sbp, -hr/sbp^2]
        df_dhr = 1 / sbp
        df_dsbp = -hr / (sbp**2)
        grads = np.sqrt(df_dhr**2 + df_dsbp**2)

    elif func_name == "map_val":
        # f(sbp, dbp) = sbp + 2 * dbp
        # Grad = [1, 2]
        # This is a linear function, so grad is constant
        val = np.sqrt(1**2 + 2**2)
        grads = np.full(samples, val)

    elif func_name == "lactate":
        # f(si) = 1.0 + 1.5 * si
        # Grad = [1.5]
        grads = np.full(samples, 1.5)

    m = np.min(grads)
    L = np.max(grads)
    
    return m, L

def analysis():
    # --- Analysis ---
    functions = ["shock_index", "map_val", "lactate"]

    print(f"{'Function':<15} | {'Lower Bound (m)':<15} | {'Upper Bound (L)':<15}")
    print("-" * 50)

    for func in functions:
        m, L = estimate_bi_lipschitz(func, None)
        print(f"{func:<15} | {m:<15.4f} | {L:<15.4f}")

def get_cv_sofa(map_val, ne_dose):
    """
    Calculates the Cardiovascular SOFA score based on 
    Sepsis-3 clinical criteria.
    """
    if ne_dose > 0.1:
        return 4
    elif ne_dose > 0:
        return 3
    elif map_val < 70:
        return 1
    else:
        return 0

def sepsis_eval_batch(gt_list, vanilla_list, dep_aware_list):
    """
    Evaluate lists of dictionaries to get Mean Absolute Error (MAE) and variability.
    """
    metrics_data = {
        "vanilla": {"sofa_ae": [], "lactate_ae": []},
        "dep_aware": {"sofa_ae": [], "lactate_ae": []}
    }

    def get_metrics(data):
        # Assumes get_cv_sofa(map, dose) is defined in your environment
        return {
            "sofa": get_cv_sofa(float(data.get("map_val", 0)), float(data.get("ne_dose", 0))),
            "lactate": float(data.get("lactate", 0))
        }

    for gt, van, dep in zip(gt_list, vanilla_list, dep_aware_list):
        gt_m = get_metrics(gt)
        for name, vals in [("vanilla", van), ("dep_aware", dep)]:
            curr_m = get_metrics(vals)
            
            # Calculate Absolute Errors (MAE)
            s_ae = abs(curr_m["sofa"] - gt_m["sofa"])
            l_ae = abs(curr_m["lactate"] - gt_m["lactate"])
            
            metrics_data[name]["sofa_ae"].append(s_ae)
            metrics_data[name]["lactate_ae"].append(l_ae)

    summary = {}
    for name in ["vanilla", "dep_aware"]:
        summary[name] = {
            "sofa": (np.mean(metrics_data[name]["sofa_ae"]), np.std(metrics_data[name]["sofa_ae"])),
            "lactate": (np.mean(metrics_data[name]["lactate_ae"]), np.std(metrics_data[name]["lactate_ae"]))
        }
    return summary

def process_privacy_budgets(vanilla_eps, dep_aware_eps, epsilon):
    """
    Excludes the last element of each sublist, replaces items < epsilon with 0,
    sums components per sample, and returns mean/std.
    """
    # Slice [:-1] removes the last element, then filter and sum
    van_totals = [
        sum(val if val >= epsilon else 0 for val in sub_list) 
        for sub_list in vanilla_eps
    ]
    dep_totals = [
        sum(val if val >= epsilon else 0 for val in sub_list[:-1]) 
        for sub_list in dep_aware_eps
    ]
    
    return {
        "vanilla": (np.mean(van_totals), np.std(van_totals)),
        "dep_aware": (np.mean(dep_totals), np.std(dep_totals))
    }

def generate_comparison_plots(eval_summary, privacy_summary):
    """
    Generates three subplots to ensure appropriate y-axis scales for each metric.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    width = 0.35
    x = np.array([0]) # Single group per plot

    # --- Plot 1: SOFA MAE (Scale: Points) ---
    v_mean, v_std = eval_summary['vanilla']['sofa']
    d_mean, d_std = eval_summary['dep_aware']['sofa']
    ax1.bar(x - width/2, [v_mean], width, yerr=[v_std], label='Vanilla', capsize=6, color='#ff9999', edgecolor='black')
    ax1.bar(x + width/2, [d_mean], width, yerr=[d_std], label='Dependency-Aware', capsize=6, color='#66b3ff', edgecolor='black')
    ax1.set_title('SOFA Mean Absolute Error', fontweight='bold')
    ax1.set_ylabel('Absolute Error (Points)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['SOFA'])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # --- Plot 2: Lactate MAE (Scale: mmol/L) ---
    v_mean, v_std = eval_summary['vanilla']['lactate']
    d_mean, d_std = eval_summary['dep_aware']['lactate']
    ax2.bar(x - width/2, [v_mean], width, yerr=[v_std], label='Vanilla', capsize=6, color='#ff9999', edgecolor='black')
    ax2.bar(x + width/2, [d_mean], width, yerr=[d_std], label='Dependency-Aware', capsize=6, color='#66b3ff', edgecolor='black')
    ax2.set_title('Lactate Mean Absolute Error', fontweight='bold')
    ax2.set_ylabel('Absolute Error (mmol/L)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Lactate'])
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # --- Plot 3: Privacy Budget (Scale: Epsilon) ---
    v_p_mean, v_p_std = privacy_summary['vanilla']
    d_p_mean, d_p_std = privacy_summary['dep_aware']
    ax3.bar(x - width/2, [v_p_mean], width, yerr=[v_p_std], label='Vanilla', capsize=6, color='#ff9999', edgecolor='black')
    ax3.bar(x + width/2, [d_p_mean], width, yerr=[d_p_std], label='Dependency-Aware', capsize=6, color='#66b3ff', edgecolor='black')
    ax3.set_title('Total Privacy Budget ($\epsilon$)', fontweight='bold')
    ax3.set_ylabel('Average Cumulative $\epsilon$')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Privacy'])
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("sepsis_mae_results.png")
    plt.show()

# --- Example Usage ---
# eval_results = sepsis_eval_batch(gt_values, vanilla_values, dep_aware_values)
# priv_results = process_privacy_budgets(vanilla_epsilon_budgets, dep_aware_epsilon_budgets)
# generate_comparison_plots(eval_results, priv_results)

# To use:
# results = sepsis_eval_batch(gt_values, vanilla_values, dep_aware_values)
# plot_sepsis_errors(results)

sepsis_edges = [
    ("hr", "shock_index"), 
    ("sbp", "shock_index"),
    ("sbp", "map_val"), 
    ("implied_dbp", "map_val"),
    ("shock_index", "lactate"),
    ("map_val", "sofa_score"), 
    ("ne_dose", "sofa_score")
]

sepsis_nodes = {
    "hr": None,
    "sbp": None,
    "implied_dbp": None,
    "ne_dose": None,
    "map_val": ["sbp", "implied_dbp"],
    "shock_index": ["hr", "sbp"],
    "lactate": ["shock_index"],
    "sofa_score": ["map_val", "ne_dose"],
}

# Define Levels (Layer 0 = Roots, Layer 2 = Final Leaves)
sepsis_layers = {
    "hr": 0, "sbp": 0, "implied_dbp": 0, "ne_dose": 0,
    "shock_index": 1, "map_val": 1,
    "lactate": 2, "sofa_score": 2
}

sepsis_node_expressions = {
    # If a node is not present, it requires epsilon budget by default.
    "shock_index": "round(hr / sbp, 2)", 
    "map_val": "round((sbp + 2 * implied_dbp) / 3, 1)", 
    "lactate": "round(1.0 + (shock_index * 1.5) + random.uniform(-0.5, 1.0), 1)",
} 

sepsis_node_bi_lipschitz_constants = {
    "shock_index": (0.0084, 0.0288), 
    "map_val": (2.2361, 2.2361), 
    "lactate": (1.5, 1.5),
    "sofa_score": (1,1)
}

