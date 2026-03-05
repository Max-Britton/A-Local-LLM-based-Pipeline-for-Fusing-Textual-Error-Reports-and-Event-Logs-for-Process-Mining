import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
import time
import glob
import os



def extract_json_block(text):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print("❌ Fehler beim Extrahieren des JSON-Blocks:", e)
    return ""


# Load Data (Unmatched Versions)
def load_data_with_errors(unmatched_csv):
    """
    Loads the dataset that includes generated error texts.
    """
    unmatched_df = pd.read_csv(unmatched_csv, dtype={"index": int})
    print(f"Loaded {len(unmatched_df)} rows of unmatched error texts.")
    return unmatched_df


# NEW: tracked variant (keeps your structure intact)
def extract_information_from_text_tracked(error_text):
    """
    Same logic as extract_information_from_text, but returns:
    (content, latency_sec, prompt_tokens, completion_tokens, total_tokens)

    IMPORTANT: NO approximation. Requires Ollama to return:
      - prompt_eval_count
      - eval_count
    """
    prompt = f"""
   You are an expert in extracting information. Extract essential details from the error description:

    "{error_text}"

    Provide output in JSON format with the following keys:
    - "Activity": Activity name, as string or use "Not specified"
    - "Machine": MUST be exactly "Machine <digits>" or use "Not specified"
    - "Error time and Error date": Format as "YYYY/MM/DD HH:MM" or use "Not specified"
    - "WorkerID": MUST be exactly "ID<digits>" or use "Not specified"


    Return ONLY a single JSON object wrapped in a markdown code fence that starts with exactly ```json (lowercase).
    No extra text before or after. Use double quotes only. No trailing commas.

    Example 1:
    Input:
    "During Turning & Milling on Machine 4 at 2012/03/15 09:30, I’m Worker ID ID4521. The finish looked uneven and dimensions were inconsistent."

    Output:
    ```json
    {{
        "Activity": "Turning & Milling",
        "Machine": "Machine 4",
        "Error time and Error date": "2012/03/15 09:30",
        "WorkerID": "ID4521"

    }}
    ```

    Example 2:
    Input:
    "On 14 Feb 2012 around 09:38, while running Lapping on machine one, I’m worker 4-8-8-2. The part seating looked off and the surface marks came out uneven."

    Output:
    ```json
    {{
        "Activity": "Lapping",
        "Machine": "Machine 1",
        "Error time and Error date": "2012/02/14 09:38",
        "WorkerID": "ID4882"

    }}
    ```
    """

    start = time.perf_counter()
    response = ollama.chat(model="gpt-oss:20b", messages=[
        {"role": "user", "content": prompt}
    ])
    end = time.perf_counter()

    content = response.get("message", {}).get("content", "").strip()
    latency_sec = end - start

    prompt_tokens = response.get("prompt_eval_count", None)
    completion_tokens = response.get("eval_count", None)

    if prompt_tokens is None or completion_tokens is None:
        raise RuntimeError(
            "Ollama hat keine Token-Zahlen zurückgegeben (prompt_eval_count/eval_count fehlen). "
            "Ohne echte Tokens keine faire Effizienzbewertung möglich."
        )

    total_tokens = int(prompt_tokens) + int(completion_tokens)

    return content, float(latency_sec), int(prompt_tokens), int(completion_tokens), int(total_tokens)


# Process Extracted Information (keeps name; adds timing/tokens; keeps Raw column behavior)
def process_extracted_data(df):
    """
    Applies information extraction using Ollama to extract structured data.
    Also extracts only the JSON part.

    NEW:
      - latency_sec
      - prompt_tokens
      - completion_tokens
      - total_tokens
    """
    raw_outputs = []
    latencies = []
    p_tokens = []
    c_tokens = []
    t_tokens = []

    component_start = time.perf_counter()

    for text in tqdm(df["Error Text"], total=len(df), desc="Extracting Information", unit="entry"):
        if pd.isna(text):
            raw_outputs.append("")
            latencies.append(0.0)
            p_tokens.append(0)
            c_tokens.append(0)
            t_tokens.append(0)
            continue

        raw, latency, pt, ct, tt = extract_information_from_text_tracked(text)
        raw_outputs.append(raw)
        latencies.append(latency)
        p_tokens.append(pt)
        c_tokens.append(ct)
        t_tokens.append(tt)

    component_end = time.perf_counter()
    component_total_sec = component_end - component_start

    df["Raw Ollama Output"] = raw_outputs

    # Nur der JSON-Teil
    df["Extracted Info"] = df["Raw Ollama Output"].apply(extract_json_block)

    # Tracking columns
    df["latency_sec"] = latencies
    df["prompt_tokens"] = p_tokens
    df["completion_tokens"] = c_tokens
    df["total_tokens"] = t_tokens

    del df["Raw Ollama Output"]

    return df, component_total_sec


# Evaluation der extrahierten Informationen (gegen Ground Truth) (same name; extended)
def evaluate_extracted_information(validation_df, ground_truth_column="Ground Truth",
                                   extracted_column="Extracted Info",
                                   level_column="Prompt_Level"):
    """
    Vergleicht extrahierte Informationen mit erstellter Ground Truth.
    Erwartet, dass beide Spalten JSON-Strings enthalten.

    NEW:
      - Druckt zusätzlich Metriken pro Level (1,2,3), wenn level_column existiert.
      - Gibt ein summary dict zurück (für CSV-Spalte "extraction_eval_results").
    """
    fields_to_check = ["Activity", "Machine", "Error time and Error date", "WorkerID"]

    def eval_subset(df_subset):
        results = {field: {"correct": 0, "total": 0} for field in fields_to_check}
        parsed_ok = 0
        parsed_total = 0

        for _, row in df_subset.iterrows():
            parsed_total += 1
            try:
                gt = json.loads(row[ground_truth_column])
                pred = json.loads(row[extracted_column])
                parsed_ok += 1
                for field in fields_to_check:
                    if field in gt and field in pred:
                        results[field]["total"] += 1
                        if str(gt[field]).strip() == str(pred[field]).strip():
                            results[field]["correct"] += 1
            except Exception:
                continue

        metrics = {}
        for field in fields_to_check:
            total = results[field]["total"]
            correct = results[field]["correct"]
            acc = correct / total if total > 0 else 0
            metrics[field] = {"acc": acc, "correct": correct, "total": total}

        metrics["_parsed_ok"] = parsed_ok
        metrics["_parsed_total"] = parsed_total
        return metrics

    overall = eval_subset(validation_df)

    print("🔍 Evaluation Results (Field-level Accuracy):")
    for field in fields_to_check:
        m = overall[field]
        print(f"  {field:30}: {m['acc']:.2%} ({m['correct']}/{m['total']})")
    print(f"  Parsed JSON rows: {overall['_parsed_ok']}/{overall['_parsed_total']}")

    per_level = {}
    if level_column in validation_df.columns:
        for lvl in [1, 2, 3]:
            subset = validation_df[validation_df[level_column] == lvl]
            lvl_metrics = eval_subset(subset)
            per_level[lvl] = lvl_metrics

            print(f"\n🔍 Evaluation Results (Level {lvl}):")
            for field in fields_to_check:
                m = lvl_metrics[field]
                print(f"  {field:30}: {m['acc']:.2%} ({m['correct']}/{m['total']})")
            print(f"  Parsed JSON rows: {lvl_metrics['_parsed_ok']}/{lvl_metrics['_parsed_total']}")

    return {"overall": overall, "per_level": per_level}


def build_extraction_results_json(component_name, eval_summary, df_component, component_total_sec):
    """
    Packs everything that is printed (metrics + timing + tokens) into one JSON string.
    Stored in one CSV column: extraction_eval_results (same value for all rows).
    """
    timing = {
        "avg_latency_sec": float(df_component["latency_sec"].mean()) if "latency_sec" in df_component.columns else None,
        "sum_latency_sec": float(df_component["latency_sec"].sum()) if "latency_sec" in df_component.columns else None,
        "component_total_sec": float(component_total_sec)
    }

    tokens = {
        "avg_total_tokens": float(
            df_component["total_tokens"].mean()) if "total_tokens" in df_component.columns else None,
        "total_tokens": int(df_component["total_tokens"].sum()) if "total_tokens" in df_component.columns else None,
        "entries": int(len(df_component))
    }

    out = {
        "component": component_name,
        "validation": eval_summary,
        "timing": timing,
        "tokens": tokens
    }
    return json.dumps(out, ensure_ascii=False)


def print_timing_token_summary(df_component, component_total_sec):
    avg_latency = df_component["latency_sec"].mean()
    total_latency_sum = df_component["latency_sec"].sum()

    avg_tokens = df_component["total_tokens"].mean()
    total_tokens_sum = df_component["total_tokens"].sum()

    print(f"\n⏱️ Timing Summary:")
    print(f"  Avg latency per entry: {avg_latency:.3f} sec")
    print(f"  Sum latency (entries): {total_latency_sum:.3f} sec")
    print(f"  Component total time (start-end): {component_total_sec:.3f} sec")

    print(f"\n🔢 Token Summary (ONLY real Ollama counts):")
    print(f"  Avg total tokens per entry: {avg_tokens:.1f}")
    print(f"  Total tokens (all entries): {int(total_tokens_sum)}")


# Save and Display Results
def save_results(df, validation_df, output_path, validation_output_path):
    """
    Saves the processed DataFrame to CSV and validation output.
    """
    df.to_csv(output_path, index=False)
    validation_df.to_csv(validation_output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")
    print(f"Validation output saved to: {validation_output_path}")


#  MAIN
if __name__ == "__main__":

    unmatched_file = "Generated_Error_Texts_Paper_Final.csv"  # INPUT
    validation_input_path = "Validation_Generated_Event_Log_With_Texts_Paper_final.csv"  # INPUT (bleibt gleich)

    RESULTS_COLUMN = "extraction_eval_results"
    MAX_FILES = 8

    # Wir erzeugen Dateien nach Schema: Extracted_Error_Texts_Paper_GPT{N}_new.csv
    extracted_prefix = "Extracted_Error_Texts_Paper_GPT"
    extracted_suffix = "_final.csv"

    validation_prefix = "Validation_Generated_Event_Log_With_Texts_Paper_GPT"
    validation_suffix = "_final.csv"

    # -------------------------
    # Finde existierende LLM Dateien
    # -------------------------
    pattern = f"{extracted_prefix}*{extracted_suffix}"
    existing_files = glob.glob(pattern)

    existing_nums = set()
    for fp in existing_files:
        base = os.path.basename(fp)
        m = re.match(rf"{re.escape(extracted_prefix)}(\d+){re.escape(extracted_suffix)}$", base)
        if m:
            existing_nums.add(int(m.group(1)))

    existing_count = len(existing_nums)
    print(f"Found existing extracted files: {sorted(existing_nums)} (count={existing_count})")

    if existing_count >= MAX_FILES:
        print(f"✅ Already have {MAX_FILES} extracted CSVs. Nothing to do.")
        raise SystemExit(0)

    # -------------------------
    n_to_create = MAX_FILES - existing_count

    nums_to_create = []
    candidate = 1
    while len(nums_to_create) < n_to_create and candidate <= MAX_FILES:
        if candidate not in existing_nums:
            nums_to_create.append(candidate)
        candidate += 1

    print(f"Will create new CSVs for numbers: {nums_to_create}")

    if not nums_to_create:
        print("⚠️ No free file numbers left in range 1..MAX_FILES.")
        raise SystemExit(0)

    # -------------------------
    # Run loop: jede Nummer = 1 neuer Durchlauf = 1 neue CSV
    # -------------------------
    for num in nums_to_create:
        print(f"\n==================== RUN -> {num} ====================\n")

        output_unmatched_file = f"{extracted_prefix}{num}{extracted_suffix}"
        validation_output_path = f"{validation_prefix}{num}{validation_suffix}"

        # fresh load every run
        validation_df = pd.read_csv(validation_input_path)
        unmatched_df = load_data_with_errors(unmatched_file)

        # Extract structured information
        unmatched_df, extraction_total_sec = process_extracted_data(unmatched_df)

        # Merge into validation_df (deine Logik)
        for _, row in unmatched_df.iterrows():
            orig_idx = row["index"]
            match = validation_df[validation_df["original_index"] == orig_idx]
            if len(match) == 1:
                match_idx = match.index[0]
                validation_df.at[match_idx, "Extracted Info"] = row["Extracted Info"]
                validation_df.at[match_idx, "latency_sec"] = row["latency_sec"]
                validation_df.at[match_idx, "prompt_tokens"] = row["prompt_tokens"]
                validation_df.at[match_idx, "completion_tokens"] = row["completion_tokens"]
                validation_df.at[match_idx, "total_tokens"] = row["total_tokens"]

        # Evaluation
        if "Ground Truth" in unmatched_df.columns:
            eval_summary = evaluate_extracted_information(unmatched_df)
        else:
            eval_summary = {"note": "No Ground Truth column available -> skipped validation metrics."}
            print("\n⚠️ Ground Truth not found -> skipped validation metrics.")

        print_timing_token_summary(unmatched_df, extraction_total_sec)

        # Store printed results JSON
        results_json = build_extraction_results_json(
            component_name=f"Information Extraction ({num})",
            eval_summary=eval_summary,
            df_component=unmatched_df,
            component_total_sec=extraction_total_sec
        )
        unmatched_df[RESULTS_COLUMN] = results_json
        validation_df[RESULTS_COLUMN] = results_json

        # Save
        save_results(unmatched_df, validation_df, output_unmatched_file, validation_output_path)

        print(f"✅ Saved: {output_unmatched_file}")
        print(f"✅ Saved: {validation_output_path}")
