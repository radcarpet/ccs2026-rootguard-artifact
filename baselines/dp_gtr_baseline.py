"""
DP-GTR Baseline Comparison Script

Runs the DP-GTR (Group Text Rewriting) text-level sanitization approach
on the NHANES medical benchmark and evaluates using the same metrics
(Relative MAE, Risk Class Error) as Preempt's numerical sanitization.

Pipeline per sample:
  1. Concatenate user_response fields into a paragraph (including target value)
  2. DP-GTR: generate N paraphrases via LLM API → extract private keywords → produce sanitized text
  3. Extract the target numerical value from the sanitized text via LLM API
  4. Compute risk class from extracted value
  5. Evaluate against ground truth

Supports two backends:
  - azure:  Azure OpenAI (default)
  - vertex: Google Vertex AI

Usage (Azure OpenAI):
  python baselines/dp_gtr_baseline.py \
      --backend azure \
      --azure_endpoint https://YOUR_RESOURCE.openai.azure.com/ \
      --azure_api_key YOUR_KEY \
      --azure_deployment gpt-35-turbo \
      --epsilons 2.0 \
      --templates HOMA \
      --max_samples 5

  # Or load credentials from a JSON file:
  python baselines/dp_gtr_baseline.py \
      --backend azure \
      --azure_config path/to/OpenAI.json \
      --templates HOMA \
      --max_samples 5

Usage (Vertex AI):
  python baselines/dp_gtr_baseline.py \
      --backend vertex \
      --project YOUR_GCP_PROJECT \
      --epsilons 2.0 \
      --templates HOMA \
      --max_samples 5
"""

import os
import sys
import json
import time
import re
import argparse
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root and DP-GTR subdir
# IMPORTANT: project root must come BEFORE DP-GTR dir on sys.path because
# baselines/DP-GTR/utils.py would shadow the project's utils/ package.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Project-root imports first (before DP-GTR dir pollutes the namespace)
from plots.plotting import multi_template_eval_batch, get_risk_class
from utils.med_domain.all_templates import template_target_keys, template_edges
from utils.utils import get_topological_order

# Now add DP-GTR dir for GTR/dpps imports (appended so it won't shadow utils/)
sys.path.append(os.path.join(PROJECT_ROOT, "baselines", "DP-GTR"))
from GTR import GTR
from dpps.jointEM import joint

# =============================================================================
# Configuration
# =============================================================================

# --- Default models per backend (change these to swap models) ---
DEFAULT_MODELS = {
    "azure": "gpt-35-turbo",
    "vertex": "gemini-2.0-flash-lite",
}

# --- Readable names for target keys (used in extraction prompts) ---
TARGET_READABLE_NAMES = {
    "mchc":        "MCHC (Mean Corpuscular Hemoglobin Concentration)",
    "kfre":        "KFRE (Kidney Failure Risk Equation score)",
    "aip":         "AIP (Atherogenic Index of Plasma)",
    "conicity":    "Conicity Index",
    "ppi":         "PPI (Pulse Pressure Index)",
    "fib4":        "FIB-4 (Fibrosis-4 Index)",
    "tyg":         "TyG (Triglyceride-Glucose Index)",
    "homa":        "HOMA-IR (Homeostatic Model Assessment for Insulin Resistance)",
    "liver_ratio": "Liver Stiffness-to-CAP Ratio",
    "nlr":         "NLR (Neutrophil-to-Lymphocyte Ratio)",
}

# --- Rate limiting ---
API_CALL_DELAY = 0.1  # seconds between API calls
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds, doubles on each retry

# --- Token budget ---
# Reasoning models (o-series, GPT-5-Nano, etc.) use most of the budget for
# internal chain-of-thought. 2048 leaves enough room for the visible response.
DEFAULT_MAX_TOKENS = 4096


# =============================================================================
# LLM Backends
# =============================================================================

class _BaseRewriter:
    """Common interface for all LLM backends.

    Subclasses implement _generate(prompt, temperature, max_tokens) -> str.
    """

    def __init__(self):
        self._call_count = 0
        self.model_id = "unknown"

    def _generate(self, prompt, temperature=1.0, max_tokens=256):
        raise NotImplementedError

    # --- rewrite_function interface for GTR ---
    def paraphrase(self, text, **kwargs):
        """Paraphrase text. Conforms to GTR's rewrite_function(text, **kwargs) -> str."""
        temperature = kwargs.get("temperature", 1.0)
        prompt = f"Document: {text}\nParaphrase of the document:"
        return self._generate(prompt, temperature=temperature, max_tokens=DEFAULT_MAX_TOKENS)

    # --- Raw generation (no paraphrase wrapper) ---
    def generate_raw(self, prompt, **kwargs):
        """Send prompt directly to the LLM without wrapping in a paraphrase template."""
        temperature = kwargs.get("temperature", 1.0)
        return self._generate(prompt, temperature=temperature, max_tokens=DEFAULT_MAX_TOKENS)

    # --- Value extraction ---
    def extract_value(self, sanitized_text, target_key):
        """Extract a numerical value from sanitized text.

        Returns (parsed_float_or_None, extraction_prompt_str).
        """
        readable = TARGET_READABLE_NAMES.get(target_key, target_key)
        prompt = (
            f"The following text describes a patient's medical data. "
            f"Extract the value of {readable} from it.\n\n"
            f"Text: {sanitized_text}\n\n"
            f"Return ONLY the numerical value, nothing else. "
            f"If the value is not present, return 'N/A'."
        )
        response = self._generate(prompt, temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS)
        return _parse_float(response), prompt


class AzureOpenAIRewriter(_BaseRewriter):
    """Azure OpenAI backend.

    Credentials can be passed directly or loaded from a JSON config file
    (same format as baselines/DP-GTR/OpenAI.json):
        {"api_key": "...", "api_base": "...", "api_version": "...",
         "deployment_name": {"GPT3.5": "..."}}
    """

    def __init__(self, model_id=None, endpoint=None, api_key=None,
                 api_version="2024-02-01", config_path=None):
        super().__init__()
        from openai import AzureOpenAI

        # Load from config file if provided
        if config_path:
            with open(config_path) as f:
                cfg = json.load(f)
            endpoint = endpoint or cfg.get("api_base")
            api_key = api_key or cfg.get("api_key")
            api_version = cfg.get("api_version", api_version)
            if model_id is None:
                # Fall back to first deployment name in config
                names = cfg.get("deployment_name", {})
                model_id = next(iter(names.values()), "gpt-35-turbo")

        if not endpoint or not api_key:
            raise ValueError(
                "Azure OpenAI requires --azure_endpoint and --azure_api_key, "
                "or --azure_config pointing to a JSON file."
            )

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.model_id = model_id or DEFAULT_MODELS["azure"]

    def _generate(self, prompt, temperature=1.0, max_tokens=256):
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(API_CALL_DELAY)
                kwargs = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": max_tokens,
                }
                # Some models (e.g. GPT-5-Nano) only support temperature=1;
                # omit the parameter in that case to use the model default.
                if temperature != 1.0:
                    kwargs["temperature"] = temperature
                response = self.client.chat.completions.create(**kwargs)
                self._call_count += 1
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e)
                # If the model rejects our temperature, retry without it
                if "temperature" in err_str:
                    try:
                        time.sleep(API_CALL_DELAY)
                        response = self.client.chat.completions.create(
                            model=self.model_id,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=max_tokens,
                        )
                        self._call_count += 1
                        return response.choices[0].message.content or ""
                    except Exception as e2:
                        e = e2  # fall through to retry logic
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"  [retry {attempt+1}/{MAX_RETRIES}] {e} — waiting {delay}s")
                    time.sleep(delay)
                else:
                    print(f"  [FAILED] {e}")
                    return ""


class VertexRewriter(_BaseRewriter):
    """Google Vertex AI backend."""

    def __init__(self, model_id=None, project=None, location="us-central1"):
        super().__init__()
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=project, location=location)
        self.model_id = model_id or DEFAULT_MODELS["vertex"]
        self.model = GenerativeModel(self.model_id)

    def _generate(self, prompt, temperature=1.0, max_tokens=256):
        from vertexai.generative_models import GenerationConfig

        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(API_CALL_DELAY)
                response = self.model.generate_content(prompt, generation_config=config)
                self._call_count += 1
                return response.text
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"  [retry {attempt+1}/{MAX_RETRIES}] {e} — waiting {delay}s")
                    time.sleep(delay)
                else:
                    print(f"  [FAILED] {e}")
                    return ""


# =============================================================================
# Epsilon-Aware GTR Wrapper
# =============================================================================

class EpsilonGTR(GTR):
    """GTR subclass that injects epsilon into the Joint-EM token selection."""

    # Local GPT-2 fallback when HuggingFace Hub downloads are broken
    GPT2_LOCAL_PATH = "/tmp/gpt2_manual"

    def __init__(self, epsilon=2.0, num_rewrites=10, remian_tokens=10):
        super().__init__(
            num_rewrites=num_rewrites,
            releasing_strategy="jem",
            remian_tokens=remian_tokens,
        )
        self.epsilon = epsilon
        # Use local GPT-2 if available, else fall back to HF hub
        self.gpt2_path = self.GPT2_LOCAL_PATH if os.path.isdir(self.GPT2_LOCAL_PATH) else None

    def icl(self, rewrites, rewrite_function=None, supplied_keywords=None,
            raw_generate=None, **kwargs):
        """Generate sanitized text from paraphrases.

        If supplied_keywords is given, skip the Joint-EM keyword extraction
        and use those keywords directly. Otherwise fall back to DP token selection.

        If raw_generate is provided, use it for the final ICL generation instead
        of rewrite_function (avoids wrapping the ICL prompt in a paraphrase template).
        """
        import pandas as pd

        if supplied_keywords is not None:
            released_tokens = list(supplied_keywords)
        else:
            import nltk
            import string
            from nltk.util import ngrams

            # ---- Privacy: token frequency ----
            all_tokens = {}
            for rewrite in rewrites:
                tokens = nltk.word_tokenize(rewrite)
                onegrams = set(ngrams(tokens, 1))
                for token in onegrams:
                    if token in all_tokens:
                        all_tokens[token] += 1
                    else:
                        all_tokens[token] = 1

            all_tokens_sorted = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)

            filtered_tokens = {}
            for token, count in all_tokens_sorted:
                if (
                    not all(word in string.punctuation for word in token)
                    and token[0] not in self.stopword_set
                ):
                    filtered_tokens[token] = count
            filtered_tokens_sorted = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)

            # Joint-EM with parameterized epsilon
            item_counts = np.array([count for _, count in filtered_tokens_sorted])
            k = min(self.remian_tokens, len(item_counts))
            if k == 0:
                released_tokens = []
            else:
                joint_out = joint(item_counts, k=k, epsilon=self.epsilon, neighbor_type=1)
                filtered_tokens_sorted_jem = np.array(filtered_tokens_sorted, dtype=object)[joint_out]
                released_tokens = [token_tuple[0][0] for token_tuple in filtered_tokens_sorted_jem]

            import random as _rand
            _rand.shuffle(released_tokens)

        # ---- Utility: perplexity-based reference selection ----
        paraphrase_sentences = [r if len(r) > 0 else " " for r in rewrites]
        gpt2_path = self.gpt2_path or "gpt2"
        perplexity_res = self.perplexity_metric.compute(
            predictions=paraphrase_sentences, model_id=gpt2_path
        )
        tmp_df = pd.DataFrame({
            "Predictions": paraphrase_sentences,
            "Perplexity": perplexity_res["perplexities"],
        })
        reference_question = tmp_df.loc[tmp_df["Perplexity"].idxmin(), "Predictions"]

        # ---- Final prompt ----
        suggest_tokens = ", ".join(released_tokens) if released_tokens else ""
        icl_prompt = (
            "Refer the following question to generate a new question:\n"
            + reference_question
            + "\nAvoid using following tokens:\n"
            + suggest_tokens
            + "\nGenerated question:\n"
        )
        gen_fn = raw_generate or rewrite_function
        sanitized = gen_fn(icl_prompt, **kwargs)
        return sanitized, released_tokens


# =============================================================================
# Data Helpers
# =============================================================================

def sample_to_paragraph(sample, template_name):
    """Build a natural-language paragraph from the sample's private values."""
    # Collect all values across turns
    vals = {}
    for turn in sample["turns"]:
        vals.update(turn.get("private_value", {}))

    formatter = _PARAGRAPH_TEMPLATES.get(template_name)
    if formatter:
        return formatter(vals)

    # Fallback: join user_response fields
    parts = []
    for turn in sample["turns"]:
        resp = turn.get("user_response", "")
        if resp:
            parts.append(resp)
    return ". ".join(parts) + "."


def _fmt(val, decimals=2):
    """Format a number to a fixed number of decimal places."""
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _para_anemia(v):
    return (
        f"My hemoglobin level is {_fmt(v['hb'], 1)} g/dL and my hematocrit is {_fmt(v['hct'], 1)}%. "
        f"My red blood cell count is {_fmt(v['rbc'], 2)} million/µL. "
        f"My mean corpuscular volume (MCV) is {_fmt(v['mcv'])}. "
        f"My mean corpuscular hemoglobin (MCH) is {_fmt(v['mch'])}. "
        f"My mean corpuscular hemoglobin concentration (MCHC) is {_fmt(v['mchc'])}. "
        f"What is my anemia classification?"
    )

def _para_kfre(v):
    gender_str = "male" if v.get("gender", 1) == 1 else "female"
    return (
        f"I am a {int(v['age'])}-year-old {gender_str} with a serum creatinine of {_fmt(v['scr'])} mg/dL. "
        f"My estimated GFR is {_fmt(v['egfr'])}. "
        f"My urine albumin is {_fmt(v['u_alb'], 1)} mg/L and urine creatinine is {_fmt(v['u_cr'], 1)} mg/dL. "
        f"My albumin-to-creatinine ratio (ACR) is {_fmt(v['acr'])}. "
        f"My kidney failure risk equation (KFRE) score is {_fmt(v['kfre'])}. "
        f"What is my kidney failure risk?"
    )

def _para_aip(v):
    return (
        f"My total cholesterol is {_fmt(v['tc'], 1)} mg/dL and my HDL is {_fmt(v['hdl'], 1)} mg/dL. "
        f"My non-HDL cholesterol is {_fmt(v['non_hdl'], 1)} mg/dL. "
        f"My triglycerides are {_fmt(v['tg'], 1)} mg/dL. "
        f"My LDL cholesterol is {_fmt(v['ldl'], 1)} mg/dL. "
        f"My atherogenic index of plasma (AIP) is {_fmt(v['aip'], 4)}. "
        f"What is my cardiovascular risk?"
    )

def _para_conicity(v):
    return (
        f"My weight is {_fmt(v['wt'], 1)} kg and my height is {_fmt(v['ht'], 2)} m. "
        f"My BMI is {_fmt(v['bmi'])}. "
        f"My waist circumference is {_fmt(v['waist'], 2)} m. "
        f"My waist-to-height ratio is {_fmt(v['wthr'])}. "
        f"My conicity index is {_fmt(v['conicity'])}. "
        f"What is my body shape classification?"
    )

def _para_vascular(v):
    return (
        f"My systolic blood pressure is {_fmt(v['sbp'], 1)} mmHg and my diastolic blood pressure is {_fmt(v['dbp'], 1)} mmHg. "
        f"My pulse pressure is {_fmt(v['pp'], 1)} mmHg. "
        f"My mean arterial pressure (MAP) is {_fmt(v['map'])}. "
        f"My mid blood pressure (MBP) is {_fmt(v['mbp'])}. "
        f"My pulse pressure index (PPI) is {_fmt(v['ppi'], 4)}. "
        f"What is my vascular risk status?"
    )

def _para_fib4(v):
    return (
        f"I am {int(v['age'])} years old with an AST of {_fmt(v['ast'], 1)} U/L. "
        f"My ALT is {_fmt(v['alt'], 1)} U/L and my platelet count is {_fmt(v['plt'], 1)} × 10⁹/L. "
        f"The age-AST product is {_fmt(v['fib4_prod'])}. "
        f"The platelet-ALT denominator is {_fmt(v['fib4_denom'])}. "
        f"My FIB-4 index is {_fmt(v['fib4'])}. "
        f"What is my fibrosis risk?"
    )

def _para_tyg(v):
    return (
        f"My triglycerides are {_fmt(v['tg'], 1)} mg/dL. "
        f"My fasting glucose is {_fmt(v['glu'], 1)} mg/dL. "
        f"The triglyceride-glucose product is {_fmt(v['tyg_prod'])}. "
        f"The half-product is {_fmt(v['tyg_half'])}. "
        f"My TyG index is {_fmt(v['tyg'], 4)}. "
        f"What is my metabolic risk status?"
    )

def _para_homa(v):
    return (
        f"My fasting glucose level is {_fmt(v['glu'], 1)} mg/dL. "
        f"My fasting insulin level is {_fmt(v['ins'])} µU/mL. "
        f"The glucose-insulin product is {_fmt(v['homa_prod'])}. "
        f"My HOMA-IR score is {_fmt(v['homa'], 4)}. "
        f"What is my insulin resistance status?"
    )

def _para_liver_us(v):
    return (
        f"My liver stiffness measurement is {_fmt(v['stiff'], 1)} kPa. "
        f"My controlled attenuation parameter (CAP) is {_fmt(v['cap'], 1)} dB/m. "
        f"The stiffness in pascals is {_fmt(v['stiff_pa'], 1)}. "
        f"My liver stiffness-to-CAP ratio is {_fmt(v['liver_ratio'], 4)}. "
        f"What is my liver condition?"
    )

def _para_nlr(v):
    return (
        f"My neutrophil count is {_fmt(v['neu'], 1)} × 10⁹/L. "
        f"My lymphocyte count is {_fmt(v['lym'], 1)} × 10⁹/L. "
        f"The neutrophil-lymphocyte sum is {_fmt(v['nlr_sum'])}. "
        f"The neutrophil-lymphocyte difference is {_fmt(v['nlr_diff'])}. "
        f"My neutrophil-to-lymphocyte ratio (NLR) is {_fmt(v['nlr'], 4)}. "
        f"What is my inflammatory risk status?"
    )


_PARAGRAPH_TEMPLATES = {
    "ANEMIA": _para_anemia,
    "KFRE": _para_kfre,
    "AIP": _para_aip,
    "CONICITY": _para_conicity,
    "VASCULAR": _para_vascular,
    "FIB4": _para_fib4,
    "TYG": _para_tyg,
    "HOMA": _para_homa,
    "LIVER_US": _para_liver_us,
    "NLR": _para_nlr,
}


def _parse_float(text):
    """Best-effort parse a float from LLM output. Returns None on failure."""
    if not text:
        return None
    text = text.strip()
    if text.upper() == "N/A":
        return None
    # Try to find a number (possibly negative, possibly with decimal)
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    rewriter,
    epsilons,
    templates,
    max_samples,
    num_rewrites,
    data_path,
    results_base,
):
    print(f"Backend: {rewriter.__class__.__name__}")
    print(f"Model: {rewriter.model_id}")
    print(f"Epsilons: {epsilons}")
    print(f"Templates: {templates}")
    print(f"Max samples per template: {max_samples}")

    # --- Load data ---
    print("\nLoading data...")
    with open(data_path, "r") as fp:
        all_samples = json.load(fp)

    fields = templates if templates else list(all_samples.keys())
    for f in fields:
        if f not in all_samples:
            print(f"  WARNING: template '{f}' not found in data, skipping")
    fields = [f for f in fields if f in all_samples]

    # Pre-compute ground truth
    gt_values = defaultdict(list)
    for key in fields:
        samples = all_samples[key][:max_samples] if max_samples else all_samples[key]
        for s in samples:
            _, vals = get_topological_order(s, template_edges)
            gt_values[key].append(vals)

    # --- Run experiments ---
    for epsilon in epsilons:
        print(f"\n{'='*60}")
        print(f"Epsilon = {epsilon}")
        print(f"{'='*60}")

        results_dir = f"{results_base}/epsilon_{epsilon}"
        os.makedirs(results_dir, exist_ok=True)

        gtr = EpsilonGTR(epsilon=epsilon, num_rewrites=num_rewrites)
        all_results = {}

        for key in fields:
            target_key = template_target_keys.get(key)
            if not target_key:
                print(f"  Skipping {key}: no target key defined")
                continue

            samples = all_samples[key][:max_samples] if max_samples else all_samples[key]
            n = len(samples)
            print(f"\n  Processing {key} ({n} samples, target={target_key})...")

            dp_gtr_values = []
            sanitized_texts = []
            input_paragraphs = []
            extraction_prompts = []
            all_keywords = []

            for i, sample in enumerate(samples):
                paragraph = sample_to_paragraph(sample, key)
                input_paragraphs.append(paragraph)
                print(f"    [{i+1}/{n}] Sanitizing ({len(paragraph)} chars)...")

                # Collect node values to use as keywords to suppress
                node_keywords = []
                for turn in sample["turns"]:
                    for v in turn.get("private_value", {}).values():
                        node_keywords.append(str(v))

                # Step 1: Generate paraphrases
                rewrites = gtr.gtr(paragraph, rewriter.paraphrase, temperature=1.0)

                # Step 2: ICL → sanitized text (keywords = node values)
                sanitized, keywords = gtr.icl(
                    rewrites, rewriter.paraphrase,
                    supplied_keywords=node_keywords,
                    raw_generate=rewriter.generate_raw,
                    temperature=1.0,
                )
                sanitized_texts.append(sanitized)
                all_keywords.append(keywords)
                print(f"           Keywords ({len(keywords)}): {keywords}")

                # Step 3: Extract target value
                extracted, extraction_prompt = rewriter.extract_value(sanitized, target_key)
                extraction_prompts.append(extraction_prompt)
                print(f"           Extracted {target_key} = {extracted}")

                # Build result dict matching Preempt's format
                val_dict = {}
                _, node_vals = get_topological_order(sample, template_edges)
                for node_name in node_vals:
                    val_dict[node_name] = node_vals[node_name]
                # Override the target with the extracted value
                if extracted is not None:
                    val_dict[target_key] = extracted
                else:
                    # Extraction failed — use 0.0 so evaluation counts it as error
                    val_dict[target_key] = 0.0

                dp_gtr_values.append(val_dict)

            # --- Evaluate ---
            utility_results = multi_template_eval_batch(
                gt_values[key],
                {"dp_gtr": dp_gtr_values},
                template_name=key,
            )

            all_results[key] = {
                "utility": utility_results,
                "sanitized_texts": sanitized_texts,
            }

            # Print summary
            if utility_results and "summary" in utility_results:
                s = utility_results["summary"]["dp_gtr"]
                print(f"    Results — Raw MAE: {s['raw'][0]:.2f}% ± {s['raw'][1]:.2f}")
                print(f"              Risk Err: {s['risk'][0]:.2f}% ± {s['risk'][1]:.2f}")

            # --- Save results ---
            template_dir = f"{results_dir}/{key}"
            os.makedirs(template_dir, exist_ok=True)

            with open(f"{template_dir}/dp_gtr_results.json", "w") as f:
                json.dump({
                    "method": "dp_gtr",
                    "model": rewriter.model_id,
                    "epsilon": epsilon,
                    "num_rewrites": num_rewrites,
                    "input_paragraphs": input_paragraphs,
                    "sanitized_texts": sanitized_texts,
                    "extraction_prompts": extraction_prompts,
                    "extracted_keywords": all_keywords,
                    "sanitized_values": dp_gtr_values,
                    "evaluation": utility_results,
                }, f, indent=2, default=str)

        # --- Epsilon-level summary ---
        summary = {"epsilon": epsilon, "model": rewriter.model_id, "templates": {}}
        for key in fields:
            if key in all_results and all_results[key]["utility"]:
                s = all_results[key]["utility"]["summary"]["dp_gtr"]
                summary["templates"][key] = {
                    "dp_gtr_raw_mae": s["raw"][0],
                    "dp_gtr_risk_mae": s["risk"][0],
                }
        with open(f"{results_dir}/dp_gtr_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Results saved to {results_dir}/")

    print(f"\nTotal API calls: {rewriter._call_count}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DP-GTR baseline on NHANES benchmark")

    # --- Backend selection ---
    parser.add_argument(
        "--backend", choices=["azure", "vertex"], default="azure",
        help="LLM backend to use (default: azure)",
    )
    parser.add_argument("--model", default=None, help="Model/deployment name (default: per-backend)")

    # --- Azure OpenAI options ---
    azure = parser.add_argument_group("Azure OpenAI")
    azure.add_argument("--azure_config", default=None,
                       help="Path to OpenAI.json config file (same format as DP-GTR)")
    azure.add_argument("--azure_endpoint", default=None, help="Azure OpenAI endpoint URL")
    azure.add_argument("--azure_api_key", default=None,
                       help="Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var)")
    azure.add_argument("--azure_deployment", default=None, help="Azure deployment name")
    azure.add_argument("--azure_api_version", default="2024-02-01", help="Azure API version")

    # --- Vertex AI options ---
    vertex = parser.add_argument_group("Vertex AI")
    vertex.add_argument("--project", default=None, help="GCP project ID")
    vertex.add_argument("--location", default="us-central1", help="Vertex AI location")

    # --- Experiment options ---
    parser.add_argument(
        "--epsilons", nargs="+", type=float, default=[2.0],
        help="Epsilon values to test (default: 2.0)",
    )
    parser.add_argument(
        "--templates", nargs="+", default=None,
        help="Template names to run (default: all). E.g., HOMA NLR ANEMIA",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples per template (default: all)",
    )
    parser.add_argument(
        "--num_rewrites", type=int, default=10,
        help="Number of DP-GTR paraphrases per sample (default: 10)",
    )
    parser.add_argument(
        "--data_path", default=os.path.join(PROJECT_ROOT, "data", "nhanes_benchmark_final.json"),
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--results_dir", default=os.path.join(PROJECT_ROOT, "results"),
        help="Base directory for results output",
    )
    args = parser.parse_args()

    # --- Construct backend ---
    if args.backend == "azure":
        api_key = args.azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        rewriter = AzureOpenAIRewriter(
            model_id=args.model or args.azure_deployment,
            endpoint=args.azure_endpoint,
            api_key=api_key,
            api_version=args.azure_api_version,
            config_path=args.azure_config,
        )
    elif args.backend == "vertex":
        rewriter = VertexRewriter(
            model_id=args.model,
            project=args.project,
            location=args.location,
        )

    run_experiment(
        rewriter=rewriter,
        epsilons=args.epsilons,
        templates=args.templates,
        max_samples=args.max_samples,
        num_rewrites=args.num_rewrites,
        data_path=args.data_path,
        results_base=args.results_dir,
    )


if __name__ == "__main__":
    main()
