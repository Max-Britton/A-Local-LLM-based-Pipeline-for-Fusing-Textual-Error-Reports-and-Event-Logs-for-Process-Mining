import pandas as pd
import json
import ollama
from tqdm import tqdm
import re
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import glob
import os

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "gpt-oss:20b"

# Input/Output naming (GemmaN must match)
IN_PREFIX = "Extracted_Error_Texts_Paper_GPT"
IN_SUFFIX = "_final.csv"

OUT_PREFIX = "Klassisch_Classified_Errors_Paper_GPT"
OUT_SUFFIX = "_final.csv"

MAX_FILES = 8

# Results column name
RESULTS_COLUMN = "classification_eval_results"

# Tracking columns (per entry)
CLS_LATENCY_COL = "cls_latency_sec"
CLS_PROMPT_TOK_COL = "cls_prompt_tokens"
CLS_COMPLETION_TOK_COL = "cls_completion_tokens"
CLS_TOTAL_TOK_COL = "cls_total_tokens"

# Where to save confusion matrices
CM_DIR = "confusion_matrices"


# ----------------------------
# HELPERS
# ----------------------------
def save_cm_heatmap(cm, labels, out_path_png, title):
    """Save confusion matrix heatmap as PNG (no UI popups)."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300)
    plt.close()


def extract_gemma_num(filename, prefix, suffix):
    """Extract N from prefix{N}suffix; returns int or None."""
    base = os.path.basename(filename)
    m = re.match(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}$", base)
    return int(m.group(1)) if m else None


# This list is used only to print categories "created during classification"
classified_examples = []


def classify_error_with_llm(error_text, latency_list, prompt_tokens_list, completion_tokens_list, total_tokens_list):
    """Uses Ollama to classify an error into one of the fixed categories only (timing + real token tracking)."""
    fixed_categories = [
        "Air Pressure Error",
        "Lubrication Deficiency",
        "Power Outage",
        "Part Misalignment",
        "Clamping Error",
        "Software Glitch",
        "Cooling System Failure",
        "Feed Path Error",
        "Other"
    ]

    known_categories_text = ", ".join(fixed_categories)
    category_guidance = (
        f"Choose exactly one category from the following list: [{known_categories_text}]. "
        f"Do NOT invent new categories.\n"
    )

    prompt = f"""
    You are an expert in classifying machine errors. Your task is to assign the correct category to the following error based only on the given list.

    {category_guidance}

    Example:
    Error information: "During the Lapping process on Machine 1, an error occurred at 2012/02/14 09:38. My worker ID is ID4882. The machine ran normally at first, but as the process continued, the lapping pattern on the surface became noticeably uneven, with one side worn more than the other.  There were no alerts from the system, but the final part had a clear angular distortion. I stopped the process and informed the supervisor for correction."

    Expected Category:
    {{
      "Category": "Part Misalignment"
    }}

    Now, classify the following error:

    Error information: "{error_text}"

    Provide your output in JSON format:
    {{
      "Category": "..."
    }}
    """

    start = time.perf_counter()
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    end = time.perf_counter()

    latency_sec = end - start

    prompt_tokens = response.get("prompt_eval_count", None)
    completion_tokens = response.get("eval_count", None)
    if prompt_tokens is None or completion_tokens is None:
        raise RuntimeError(
            "Ollama hat keine Token-Zahlen zurückgegeben (prompt_eval_count/eval_count fehlen). "
            "Ohne echte Tokens keine faire Effizienzbewertung möglich."
        )

    total_tokens = int(prompt_tokens) + int(completion_tokens)

    latency_list.append(float(latency_sec))
    prompt_tokens_list.append(int(prompt_tokens))
    completion_tokens_list.append(int(completion_tokens))
    total_tokens_list.append(int(total_tokens))

    raw_output = response.get("message", {}).get("content", "").strip()
    clean_output = raw_output.replace("```json", "").replace("```", "").strip()

    # Parse JSON (fallback regex)
    try:
        output = json.loads(clean_output)
    except json.JSONDecodeError:
        match = re.search(r'{\s*"Category":\s*".+?"\s*}', raw_output, re.DOTALL)
        if match:
            try:
                output = json.loads(match.group(0))
            except Exception:
                output = {"Category": "Unknown"}
        else:
            output = {"Category": "Unknown"}

    if output.get("Category") in fixed_categories:
        classified_examples.append({"text": error_text, "category": output["Category"]})
    else:
        output["Category"] = "Unknown"

    return output


def build_classification_results_json(
    component_name,
    df_component,
    component_total_sec,
    report_overall=None,
    report_per_level=None,
    used_categories=None,
    cm_overall=None,
    cm_per_level=None
):
    timing = {
        "avg_latency_sec": float(df_component[CLS_LATENCY_COL].mean()) if CLS_LATENCY_COL in df_component.columns else None,
        "sum_latency_sec": float(df_component[CLS_LATENCY_COL].sum()) if CLS_LATENCY_COL in df_component.columns else None,
        "component_total_sec": float(component_total_sec)
    }

    tokens = {
        "avg_total_tokens": float(df_component[CLS_TOTAL_TOK_COL].mean()) if CLS_TOTAL_TOK_COL in df_component.columns else None,
        "total_tokens": int(df_component[CLS_TOTAL_TOK_COL].sum()) if CLS_TOTAL_TOK_COL in df_component.columns else None,
        "entries": int(len(df_component))
    }

    out = {
        "component": component_name,
        "used_categories": used_categories if used_categories is not None else [],
        "classification_report_overall": report_overall,
        "classification_report_per_level": report_per_level if report_per_level is not None else {},
        "confusion_matrix_overall": cm_overall,
        "confusion_matrix_per_level": cm_per_level if cm_per_level is not None else {},
        "timing": timing,
        "tokens": tokens
    }
    return json.dumps(out, ensure_ascii=False)


def run_one_classification(input_file, output_file):
    """Run exactly one classification job (one input -> one output), save confusion matrices only."""
    global classified_examples
    classified_examples = []  # reset per run (prevents mixing used_categories across runs)

    df = pd.read_csv(input_file)

    # per-run tracking lists
    latency_list = []
    prompt_tokens_list = []
    completion_tokens_list = []
    total_tokens_list = []

    component_start = time.perf_counter()

    df["Classification"] = [
        classify_error_with_llm(
            error_text,
            latency_list,
            prompt_tokens_list,
            completion_tokens_list,
            total_tokens_list
        )
        for error_text in tqdm(df["Error Text"], total=len(df), desc=f"Classifying ({os.path.basename(input_file)})", unit="entry")
    ]

    component_end = time.perf_counter()
    component_total_sec = component_end - component_start

    # sanity check
    if not (len(latency_list) == len(df) == len(prompt_tokens_list) == len(completion_tokens_list) == len(total_tokens_list)):
        raise RuntimeError("Tracking lists length mismatch. Something went wrong in classification loop.")

    # add tracking columns
    df[CLS_LATENCY_COL] = latency_list
    df[CLS_PROMPT_TOK_COL] = prompt_tokens_list
    df[CLS_COMPLETION_TOK_COL] = completion_tokens_list
    df[CLS_TOTAL_TOK_COL] = total_tokens_list

    # Extract the category from the JSON result
    df["Category"] = df["Classification"].apply(
        lambda x: x["Category"] if isinstance(x, dict) and "Category" in x else "Unknown"
    )
    df.drop(columns=["Classification"], inplace=True)

    print(f"\n➡️ Classified errors will be saved to: {output_file}")

    final_categories = sorted(set(ex["category"] for ex in classified_examples))
    print("\n📚 Categories created during classification:")
    for cat in final_categories:
        print(f"- {cat}")

    # Timing + Token summary
    print(f"\n⏱️ Timing Summary:")
    print(f"  Avg latency per entry: {df[CLS_LATENCY_COL].mean():.3f} sec")
    print(f"  Sum latency (entries): {df[CLS_LATENCY_COL].sum():.3f} sec")
    print(f"  Component total time (start-end): {component_total_sec:.3f} sec")

    print(f"\n🔢 Token Summary (ONLY real Ollama counts):")
    print(f"  Avg total tokens per entry: {df[CLS_TOTAL_TOK_COL].mean():.1f}")
    print(f"  Total tokens (all entries): {int(df[CLS_TOTAL_TOK_COL].sum())}")

    # Metrics: overall + per Prompt_Level
    report_overall = None
    report_per_level = {}
    cm_overall = None
    cm_per_level = {}

    if "True_Category" in df.columns:
        print("\n📊 Classification Report (Overall):")
        print(classification_report(df["True_Category"], df["Category"], zero_division=0))
        report_overall = classification_report(df["True_Category"], df["Category"], zero_division=0, output_dict=True)

        if "Prompt_Level" in df.columns:
            for lvl in [1, 2, 3]:
                sub = df[df["Prompt_Level"] == lvl]
                print(f"\n📊 Classification Report (Level {lvl}):")
                if len(sub) == 0:
                    print("  (no rows)")
                    report_per_level[str(lvl)] = {"note": "no rows"}
                    continue

                print(classification_report(sub["True_Category"], sub["Category"], zero_division=0))
                report_per_level[str(lvl)] = classification_report(
                    sub["True_Category"],
                    sub["Category"],
                    zero_division=0,
                    output_dict=True
                )

    # Confusion matrix: SAVE ONLY (no plt.show)
    if "True_Category" in df.columns:
        os.makedirs(CM_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(output_file))[0]

        labels = sorted(df["True_Category"].unique())
        cm = confusion_matrix(df["True_Category"], df["Category"], labels=labels)
        cm_overall = {"labels": labels, "matrix": cm.tolist()}

        save_cm_heatmap(
            cm=cm,
            labels=labels,
            out_path_png=os.path.join(CM_DIR, f"{base}_cm_overall.png"),
            title="Confusion Matrix (Overall)"
        )

        if "Prompt_Level" in df.columns:
            for lvl in [1, 2, 3]:
                sub = df[df["Prompt_Level"] == lvl]
                if len(sub) == 0:
                    cm_per_level[str(lvl)] = {"note": "no rows"}
                    continue

                labels_lvl = sorted(sub["True_Category"].unique())
                cm_lvl = confusion_matrix(sub["True_Category"], sub["Category"], labels=labels_lvl)
                cm_per_level[str(lvl)] = {"labels": labels_lvl, "matrix": cm_lvl.tolist()}

                save_cm_heatmap(
                    cm=cm_lvl,
                    labels=labels_lvl,
                    out_path_png=os.path.join(CM_DIR, f"{base}_cm_level{lvl}.png"),
                    title=f"Confusion Matrix (Level {lvl})"
                )

    # Store ALL printed results as JSON in one column (same value for all rows)
    results_json = build_classification_results_json(
        component_name="Classification",
        df_component=df,
        component_total_sec=component_total_sec,
        report_overall=report_overall,
        report_per_level=report_per_level if report_per_level else None,
        used_categories=final_categories,
        cm_overall=cm_overall,
        cm_per_level=cm_per_level if cm_per_level else None
    )
    df[RESULTS_COLUMN] = results_json

    # Save output CSV
    df.to_csv(output_file, index=False)
    print(f"\n✅ Classified errors saved to: {output_file}")


# ----------------------------
# MAIN: auto-match Extracted GemmaN -> Classified GemmaN, max 5 outputs
# ----------------------------
if __name__ == "__main__":
    # Existing outputs
    existing_out = glob.glob(f"{OUT_PREFIX}*{OUT_SUFFIX}")
    out_nums = set()
    for fp in existing_out:
        n = extract_gemma_num(fp, OUT_PREFIX, OUT_SUFFIX)
        if n is not None:
            out_nums.add(n)

    print(f"Found existing classification CSVs: {sorted(out_nums)} (count={len(out_nums)})")

    if len(out_nums) >= MAX_FILES:
        print(f"✅ Already have {MAX_FILES} classification CSVs. Nothing to do.")
        raise SystemExit(0)

    # Existing inputs
    existing_in = glob.glob(f"{IN_PREFIX}*{IN_SUFFIX}")
    in_nums = set()
    for fp in existing_in:
        n = extract_gemma_num(fp, IN_PREFIX, IN_SUFFIX)
        if n is not None:
            in_nums.add(n)

    print(f"Found existing extracted input CSVs: {sorted(in_nums)} (count={len(in_nums)})")

    # Candidates = inputs that exist but are not yet classified
    candidates = sorted([n for n in in_nums if n not in out_nums])
    if not candidates:
        print("✅ No new inputs to classify (all extracted LLM already have classification outputs).")
        raise SystemExit(0)

    # Run up to MAX_FILES total outputs
    remaining_slots = MAX_FILES - len(out_nums)
    to_run = candidates[:remaining_slots]

    print(f"➡️ Will run classification for LLM numbers: {to_run}")

    for n in to_run:
        input_file = f"{IN_PREFIX}{n}{IN_SUFFIX}"
        output_file = f"{OUT_PREFIX}{n}{OUT_SUFFIX}"

        print(f"\n=== RUN LLM{n} ===")
        print(f"Input : {input_file}")
        print(f"Output: {output_file}")

        run_one_classification(input_file, output_file)
