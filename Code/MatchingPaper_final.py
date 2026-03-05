import pandas as pd
import json
from tqdm import tqdm
import re
from sklearn.metrics.pairwise import cosine_similarity
import ollama
import time
import glob
import os

# ============================
# CONFIG (RUN-ONLY SCHEME like your classification)
# ============================

LLM_PICK_MODEL = "deepseek-r1:14b"
EMBED_MODEL = "mxbai-embed-large"

LLM_PREFIX = "Deep"
MAX_RUNS = 8  # set to 5 if you want max 5 outputs

ERRORS_FILE_PATTERN = "Klassisch_Classified_Errors_Paper_{llm}{r}_final.csv"
FULL_EVENTLOG_OUT_PATTERN = "Matched_Errors_To_EventLog_Paper_{llm}{r}_final.csv"
MATCHED_ERRORLEVEL_OUT_PATTERN = "Matched_ErrorLevel_Paper_{llm}{r}_final.csv"
VALIDATION_RESULTS_OUT_PATTERN = "Validation_Results_Paper_{llm}{r}_final.csv"

EVENTS_FILE = "Filtered_Production_Data_Paper.csv"
VALIDATION_FILE = "Validation_Generated_Event_Log_With_Texts_Paper_final.csv"

LEVEL_COLUMN = "Prompt_Level"
ERROR_TEXT_MATCHED_COL = "Error Text"
ERROR_TEXT_VALID_COL = "Unstructured Text"

cols_to_compare = ["Activity", "Resource", "End Time", "Worker ID"]

FILTER_ALL_ATTRIBUTES = True

ERROR_ORIG_COL = "error_original_index"

MATCH_METHOD_COL = "match_method"
MATCH_LATENCY_COL = "match_latency_sec"
MATCH_PROMPT_TOK_COL = "match_prompt_tokens"
MATCH_COMPLETION_TOK_COL = "match_completion_tokens"
MATCH_TOTAL_TOK_COL = "match_total_tokens"
MATCH_RESULTS_COL = "matching_eval_results"

COSINE_THRESHOLD = 0.99
MIN_MARGIN = 0.0000

# ============================
# HELPERS: run discovery (like your classification)
# ============================

def extract_run_num(filename, prefix, file_prefix, file_suffix):
    base = os.path.basename(filename)
    m = re.match(
        rf"^{re.escape(file_prefix)}{re.escape(prefix)}(\d+){re.escape(file_suffix)}$",
        base
    )
    return int(m.group(1)) if m else None


def discover_available_runs_from_errors(prefix):
    paths = glob.glob(f"Klassisch_Classified_Errors_Paper_{prefix}*_final.csv")
    runs = set()
    for p in paths:
        r = extract_run_num(p, prefix, "Klassisch_Classified_Errors_Paper_", "_final.csv")
        if r is not None:
            runs.add(r)
    return sorted(runs)


def discover_existing_runs_from_outputs(prefix):
    # This is THE key: decide "already done" by OUTPUTS, not inputs
    paths = glob.glob(f"Validation_Results_Paper_{prefix}*_final.csv")
    runs = set()
    for p in paths:
        r = extract_run_num(p, prefix, "Validation_Results_Paper_", "_final.csv")
        if r is not None:
            runs.add(r)
    return sorted(runs)


# ============================
# LOAD EVENTS (shared across runs)
# ============================
events_df = pd.read_csv(EVENTS_FILE)
events_df.columns = events_df.columns.str.strip().str.replace("\n", "")

raw_events_df = pd.read_csv(EVENTS_FILE)
raw_events_df.columns = raw_events_df.columns.str.strip().str.replace("\n", "")

machine_rows = events_df[events_df["Resource"].str.contains("Machine", case=False, na=False)].copy()
machine_rows["original_event_index"] = machine_rows.index
machine_rows = machine_rows.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================
# MATCHING HELPERS (logic unchanged)
# ============================

def get_ollama_embedding(text):
    try:
        result = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return result["embedding"]
    except Exception as e:
        print("❌ Embedding error:", e)
        return None


def clean_extracted_info(text):
    return str(text).strip().removeprefix("```json").removesuffix("```").strip()


def ns_to_empty(v) -> str:
    s = "" if v is None else str(v).strip()
    return "" if s.lower() == "not specified" else s


def extract_embedding_text(extracted_info):
    if not str(extracted_info).strip():
        return ""
    try:
        info = json.loads(extracted_info)

        activity = ns_to_empty(info.get("Activity", ""))
        resource = ns_to_empty(info.get("Machine", ""))  # machine corresponds to Resource in event log
        end_time = ns_to_empty(info.get("Error time and Error date", ""))
        worker_id = ns_to_empty(info.get("WorkerID", ""))

        return (
            f"Activity: {activity} | "
            f"Resource: {resource} | "
            f"End Time: {end_time} | "
            f"Worker ID: {worker_id}"
        )
    except Exception as e:
        print("❌ Failed to extract embedding text:", e)
        return ""


def flatten_row(row):
    columns_of_interest = ["Activity", "Resource", "End Time", "Worker ID"]
    return " | ".join(
        [f"{col}: {str(row[col])}" for col in columns_of_interest if col in row and pd.notna(row[col])]
    )


def parse_json_info(info_text):
    try:
        info_clean = re.sub(r"^```json|```$", "", str(info_text).strip())
        return json.loads(info_clean)
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        return {}


def ask_llm_to_pick_best(error_text, candidates):
    """
    Returns:
      (choice_int_or_None, latency_sec, prompt_tokens, completion_tokens, total_tokens)
    """
    prompt = f"""
You are an expert at matching structured factory error descriptions to the most likely event-log entry.

Task:
Choose the best matching candidate among [0]...[9].
If NONE of the candidates is a sufficient match, reply with: none

Error:
{error_text}

Candidates:
{chr(10).join([f"[{i}] {c}" for i, c in enumerate(candidates)])}

Reply with ONLY: a single digit 0-9 OR the word none. No other text or explanation.
"""
    start = time.perf_counter()
    response = ollama.chat(
        model=LLM_PICK_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    end = time.perf_counter()
    latency_sec = float(end - start)

    prompt_tokens = response.get("prompt_eval_count", None)
    completion_tokens = response.get("eval_count", None)
    if prompt_tokens is None or completion_tokens is None:
        raise RuntimeError(
            "Ollama hat keine Token-Zahlen zurückgegeben (prompt_eval_count/eval_count fehlen). "
            "Ohne echte Tokens keine faire Effizienzbewertung möglich."
        )
    total_tokens = int(prompt_tokens) + int(completion_tokens)

    output = response["message"]["content"].strip()

    if output.strip().lower() == "none":
        return None, latency_sec, int(prompt_tokens), int(completion_tokens), int(total_tokens)

    m = re.search(r"\b([0-9])\b", output)
    choice = int(m.group(1)) if m else None
    return choice, latency_sec, int(prompt_tokens), int(completion_tokens), int(total_tokens)


def build_matching_results_json(df_component, component_total_sec, pass_counts=None, validation_summary=None):
    timing = {
        "avg_latency_sec": float(df_component[MATCH_LATENCY_COL].mean()) if MATCH_LATENCY_COL in df_component.columns else None,
        "sum_latency_sec": float(df_component[MATCH_LATENCY_COL].sum()) if MATCH_LATENCY_COL in df_component.columns else None,
        "component_total_sec": float(component_total_sec)
    }
    tokens = {
        "avg_total_tokens": float(df_component[MATCH_TOTAL_TOK_COL].mean()) if MATCH_TOTAL_TOK_COL in df_component.columns else None,
        "total_tokens": int(df_component[MATCH_TOTAL_TOK_COL].sum()) if MATCH_TOTAL_TOK_COL in df_component.columns else None,
        "entries": int(len(df_component))
    }
    out = {
        "component": "Matching",
        "timing": timing,
        "tokens": tokens,
        "pass_counts": pass_counts if pass_counts is not None else {},
        "validation": validation_summary if validation_summary is not None else {}
    }
    return json.dumps(out, ensure_ascii=False)


def smart_filter_candidates(error_row):
    parsed_info = parse_json_info(error_row["Extracted Info"])

    expected_time = ns_to_empty(parsed_info.get("Error time and Error date", ""))
    expected_worker = ns_to_empty(parsed_info.get("WorkerID", ""))

    expected_machine = ns_to_empty(parsed_info.get("Machine", ""))
    expected_activity = ns_to_empty(parsed_info.get("Activity", ""))

    def is_within_time_margin(candidate_time, reference_time, margin_minutes=1):
        try:
            from dateutil import parser
            ct = parser.parse(str(candidate_time))
            rt = parser.parse(str(reference_time))
            delta = abs((ct - rt).total_seconds()) / 60.0
            return delta <= margin_minutes
        except:
            return False

    candidate_events = machine_rows.copy()

    if FILTER_ALL_ATTRIBUTES:
        if expected_worker or expected_time or expected_machine or expected_activity:
            candidate_events = candidate_events[
                candidate_events.apply(
                    lambda r: (
                        (not expected_worker or expected_worker.lower() in str(r.get("Worker ID", "")).lower()) and
                        (not expected_time or is_within_time_margin(r.get("End Time", ""), expected_time)) and
                        (not expected_activity or expected_activity.lower() in str(r.get("Activity", "")).lower()) and
                        (
                            (not expected_machine) or
                            (expected_machine.lower() in str(r.get("Resource", "")).lower()) or
                            (expected_machine.lower() in str(r.get("Machine", "")).lower())
                        )
                    ),
                    axis=1
                )
            ]
            if candidate_events.empty:
                candidate_events = machine_rows.copy()
    else:
        if expected_worker or expected_time:
            candidate_events = candidate_events[
                candidate_events.apply(
                    lambda r: (
                        (not expected_worker or expected_worker.lower() in str(r.get("Worker ID", "")).lower()) and
                        (not expected_time or is_within_time_margin(r.get("End Time", ""), expected_time))
                    ),
                    axis=1
                )
            ]
            if candidate_events.empty:
                candidate_events = machine_rows.copy()

    return candidate_events.reset_index(drop=True)


def compute_topk(error_row, candidate_events):
    candidate_events = candidate_events.copy()
    candidate_events["embedding_text"] = candidate_events.apply(flatten_row, axis=1)
    candidate_events["embedding"] = candidate_events["embedding_text"].apply(
        lambda x: get_ollama_embedding(x) if x else None
    )

    top_10 = sorted(
        [
            (i, cosine_similarity([error_row["embedding"]], [candidate_events.iloc[i]["embedding"]])[0][0])
            for i in range(len(candidate_events))
            if candidate_events.iloc[i]["embedding"] is not None
        ],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return candidate_events, top_10


def open_ids(index_matched_list, n):
    return [i for i in range(n) if index_matched_list[i] is None]


# ============================
# RUN ONE MATCHING JOB (one RUN id)
# ============================
def run_one_matching_job(llm_prefix, run_id):
    ERRORS_FILE = ERRORS_FILE_PATTERN.format(llm=llm_prefix, r=run_id)
    FULL_EVENTLOG_OUT = FULL_EVENTLOG_OUT_PATTERN.format(llm=llm_prefix, r=run_id)
    MATCHED_ERRORLEVEL_OUT = MATCHED_ERRORLEVEL_OUT_PATTERN.format(llm=llm_prefix, r=run_id)
    VALIDATION_RESULTS_OUT = VALIDATION_RESULTS_OUT_PATTERN.format(llm=llm_prefix, r=run_id)

    if not os.path.exists(ERRORS_FILE):
        print(f"⚠️ Missing errors file, skipping: {ERRORS_FILE}")
        return

    print(f"\n=============================")
    print(f"🚀 MATCHING RUN: {llm_prefix}{run_id}")
    print(f"Errors file: {ERRORS_FILE}")
    print(f"Outputs: {MATCHED_ERRORLEVEL_OUT}, {FULL_EVENTLOG_OUT}, {VALIDATION_RESULTS_OUT}")
    print(f"=============================\n")

    # LOAD DATA (per-run errors)
    errors_df = pd.read_csv(ERRORS_FILE)

    # PREP ERROR EMBEDDINGS
    errors_df["Extracted Info"] = errors_df["Extracted Info"].fillna("").apply(clean_extracted_info)
    errors_df["embedding_text"] = errors_df["Extracted Info"].apply(extract_embedding_text)
    errors_df["embedding"] = errors_df["embedding_text"].apply(lambda x: get_ollama_embedding(x) if x else None)

    # Rename original_index to avoid collisions later
    if "original_index" not in errors_df.columns:
        errors_df["original_index"] = errors_df.index
    errors_df.rename(columns={"original_index": ERROR_ORIG_COL}, inplace=True)

    # Only keep rows with extracted info
    errors_df = errors_df[errors_df["Extracted Info"].notna() & (errors_df["Extracted Info"].str.strip() != "")].copy()
    errors_df = errors_df.reset_index(drop=True)
    n_rows = len(errors_df)

    # 3-PASS MATCHING (unchanged)
    index_matched_list = [None] * n_rows
    match_method_list = [""] * n_rows

    # per-entry tracking
    match_latency_list = [0.0] * n_rows
    match_prompt_tok_list = [0] * n_rows
    match_completion_tok_list = [0] * n_rows
    match_total_tok_list = [0] * n_rows

    used_event_indices = set()

    pass_counts = {
        "pass1_single_candidate": 0,
        "pass2_cosine_direct": 0,
        "pass3_llm_choice": 0,
        "unmatched": 0
    }

    component_start = time.perf_counter()

    # ----------------------------
    # PASS 1: single candidate after smart filter
    # ----------------------------
    for i in tqdm(open_ids(index_matched_list, n_rows), desc="Pass 1/3: single-candidate", unit="entry"):
        entry_start = time.perf_counter()
        row = errors_df.iloc[i]

        if row["embedding"] is None:
            match_latency_list[i] += float(time.perf_counter() - entry_start)
            continue

        candidate_events = smart_filter_candidates(row)

        if len(candidate_events) == 1:
            original_idx = int(candidate_events.iloc[0]["original_event_index"])
            if original_idx not in used_event_indices:
                index_matched_list[i] = original_idx
                used_event_indices.add(original_idx)
                match_method_list[i] = "single_candidate"
                pass_counts["pass1_single_candidate"] += 1

        match_latency_list[i] += float(time.perf_counter() - entry_start)

    # ----------------------------
    # PASS 2: cosine direct matches collected then assigned by confidence
    # ----------------------------
    candidates_for_assignment = []  # (score, i, original_event_index)

    for i in tqdm(open_ids(index_matched_list, n_rows), desc="Pass 2/3: cosine-direct", unit="entry"):
        entry_start = time.perf_counter()
        row = errors_df.iloc[i]

        if row["embedding"] is None:
            match_latency_list[i] += float(time.perf_counter() - entry_start)
            continue

        candidate_events = smart_filter_candidates(row)

        # Remove already-used events
        candidate_events = candidate_events[
            ~candidate_events["original_event_index"].isin(used_event_indices)
        ].reset_index(drop=True)

        # fallback to all unused machine events
        if candidate_events.empty:
            candidate_events = machine_rows[
                ~machine_rows["original_event_index"].isin(used_event_indices)
            ].copy().reset_index(drop=True)

        candidate_events, top_10 = compute_topk(row, candidate_events)

        if top_10:
            top1_idx, top1_score = top_10[0]
            top2_score = top_10[1][1] if len(top_10) > 1 else 0.0
            margin = float(top1_score - top2_score)

            if float(top1_score) >= COSINE_THRESHOLD and margin >= MIN_MARGIN:
                original_idx = int(candidate_events.loc[top1_idx, "original_event_index"])
                candidates_for_assignment.append((float(top1_score), i, original_idx))

        match_latency_list[i] += float(time.perf_counter() - entry_start)

    # Assign highest confidence first
    candidates_for_assignment.sort(key=lambda x: x[0], reverse=True)

    for score, i, original_idx in candidates_for_assignment:
        if index_matched_list[i] is not None:
            continue
        if original_idx in used_event_indices:
            continue
        index_matched_list[i] = original_idx
        used_event_indices.add(original_idx)
        match_method_list[i] = "cosine_direct"
        pass_counts["pass2_cosine_direct"] += 1

    # ----------------------------
    # PASS 3: LLM pick for remaining
    # ----------------------------
    for i in tqdm(open_ids(index_matched_list, n_rows), desc="Pass 3/3: llm-pick", unit="entry"):
        entry_start = time.perf_counter()
        row = errors_df.iloc[i]

        if row["embedding"] is None:
            match_method_list[i] = "no_match"
            match_latency_list[i] += float(time.perf_counter() - entry_start)
            continue

        candidate_events = smart_filter_candidates(row)

        # Remove already-used events
        candidate_events = candidate_events[
            ~candidate_events["original_event_index"].isin(used_event_indices)
        ].reset_index(drop=True)

        if candidate_events.empty:
            candidate_events = machine_rows[
                ~machine_rows["original_event_index"].isin(used_event_indices)
            ].copy().reset_index(drop=True)

        candidate_events, top_10 = compute_topk(row, candidate_events)

        if not top_10:
            match_method_list[i] = "no_match"
            match_latency_list[i] += float(time.perf_counter() - entry_start)
            continue

        candidates_text = [candidate_events.iloc[idx]["embedding_text"] for idx, _ in top_10]

        parsed_info = parse_json_info(row["Extracted Info"])
        error_struct = (
            f"Activity: {ns_to_empty(parsed_info.get('Activity', ''))}\n"
            f"Machine: {ns_to_empty(parsed_info.get('Machine', ''))}\n"
            f"Time: {ns_to_empty(parsed_info.get('Error time and Error date', ''))}\n"
            f"WorkerID: {ns_to_empty(parsed_info.get('WorkerID', ''))}"
        )

        llm_choice, llm_latency, pt, ct, tt = ask_llm_to_pick_best(error_struct, candidates_text)

        # If abstained or parse failed, do NOT assign
        if llm_choice is None:
            match_method_list[i] = "llm_none_or_parse_failed_no_assign"
            match_latency_list[i] += float(time.perf_counter() - entry_start)
            continue

        # store real tokens (no approximation)
        match_prompt_tok_list[i] = pt
        match_completion_tok_list[i] = ct
        match_total_tok_list[i] = tt

        chosen = max(0, min(llm_choice, len(top_10) - 1))
        chosen_candidate_index = top_10[chosen][0]
        original_idx = int(candidate_events.loc[chosen_candidate_index, "original_event_index"])

        if original_idx not in used_event_indices:
            index_matched_list[i] = original_idx
            used_event_indices.add(original_idx)
            match_method_list[i] = "llm_choice"
            pass_counts["pass3_llm_choice"] += 1
        else:
            # do NOT fall back to another candidate
            match_method_list[i] = "no_match"

        match_latency_list[i] += float(time.perf_counter() - entry_start)

    component_total_sec = float(time.perf_counter() - component_start)
    pass_counts["unmatched"] = sum(1 for x in index_matched_list if x is None)

    # ============================
    # BUILD matched_rows
    # ============================
    matched_rows = []
    for idx in index_matched_list:
        if idx is None or (isinstance(idx, float) and pd.isna(idx)):
            matched_rows.append(pd.Series(dtype="object"))
        else:
            matched_rows.append(raw_events_df.loc[int(idx)])

    # Prefix event columns to avoid collisions
    matched_events_df = pd.DataFrame(matched_rows).reset_index(drop=True)
    matched_events_df = matched_events_df.add_prefix("event_")

    matched_errorlevel_df = pd.concat([errors_df.reset_index(drop=True), matched_events_df], axis=1)

    matched_errorlevel_df["index_matched"] = index_matched_list
    matched_errorlevel_df[MATCH_METHOD_COL] = match_method_list

    matched_errorlevel_df[MATCH_LATENCY_COL] = match_latency_list
    matched_errorlevel_df[MATCH_PROMPT_TOK_COL] = match_prompt_tok_list
    matched_errorlevel_df[MATCH_COMPLETION_TOK_COL] = match_completion_tok_list
    matched_errorlevel_df[MATCH_TOTAL_TOK_COL] = match_total_tok_list

    # save error-level
    matched_errorlevel_df.to_csv(MATCHED_ERRORLEVEL_OUT, index=False)
    print(f"✅ Saved error-level matches to: {MATCHED_ERRORLEVEL_OUT}")

    # ============================
    # WRITE INTO FULL EVENTLOG
    # ============================
    full_df = raw_events_df.copy()
    full_df[ERROR_TEXT_MATCHED_COL] = ""
    full_df["Extracted Info"] = ""
    full_df["Category"] = ""

    for i, row in matched_errorlevel_df.iterrows():
        try:
            idx = row["index_matched"]
            if pd.isna(idx):
                continue
            idx = int(idx)
            full_df.at[idx, ERROR_TEXT_MATCHED_COL] = row.get(ERROR_TEXT_MATCHED_COL, "")
            full_df.at[idx, "Extracted Info"] = row.get("Extracted Info", "")
            full_df.at[idx, "Category"] = row.get("Category", "")
        except Exception as e:
            print(f"❌ Could not assign row {i}: {e}")

    full_df.to_csv(FULL_EVENTLOG_OUT, index=False)
    print(f"✅ Saved full event log to: {FULL_EVENTLOG_OUT}")

    # ============================
    # VALIDATION (overall + by method + by level×method)
    # ============================
    validation_summary = {}

    try:
        validation_df = pd.read_csv(VALIDATION_FILE)

        merged = pd.merge(
            matched_errorlevel_df,
            validation_df[["original_index"] + cols_to_compare].rename(
                columns={c: c + "_original" for c in cols_to_compare}
            ),
            left_on=ERROR_ORIG_COL,
            right_on="original_index",
            how="inner"
        )

        merged["Match"] = [
            all(r.get(f"event_{c}", None) == r.get(f"{c}_original", None) for c in cols_to_compare)
            for _, r in merged.iterrows()
        ]

        # --------------------------
        # PASS / METHOD BREAKDOWN (RESTORED)
        # --------------------------
        if MATCH_METHOD_COL in merged.columns:
            method_stats = (
                merged.groupby(MATCH_METHOD_COL)["Match"]
                .agg(total="count", correct="sum")
                .reset_index()
            )
            method_stats["wrong"] = method_stats["total"] - method_stats["correct"]
            method_stats["accuracy_pct"] = (method_stats["correct"] / method_stats["total"] * 100).round(2)
            method_stats = method_stats.sort_values("accuracy_pct", ascending=False)

            print("\n-----------------------------")
            print("🧪 Accuracy by match_method (pass source)")
            print("-----------------------------")
            for _, r in method_stats.iterrows():
                m = str(r[MATCH_METHOD_COL])
                t = int(r["total"])
                c = int(r["correct"])
                a = float(r["accuracy_pct"])
                print(f"  {m}: {c}/{t} correct ({a:.2f}%)")

            if LEVEL_COLUMN in merged.columns:
                lvl_method = (
                    merged.groupby([LEVEL_COLUMN, MATCH_METHOD_COL])["Match"]
                    .agg(total="count", correct="sum")
                    .reset_index()
                )
                lvl_method["wrong"] = lvl_method["total"] - lvl_method["correct"]
                lvl_method["accuracy_pct"] = (lvl_method["correct"] / lvl_method["total"] * 100).round(2)
                lvl_method = lvl_method.sort_values([LEVEL_COLUMN, "accuracy_pct"], ascending=[True, False])

                print("\n-----------------------------")
                print("🧩 Accuracy by Prompt_Level × match_method")
                print("-----------------------------")
                for lvl in sorted(merged[LEVEL_COLUMN].dropna().unique()):
                    sub = lvl_method[lvl_method[LEVEL_COLUMN] == lvl]
                    print(f"  Level {int(lvl) if float(lvl).is_integer() else lvl}:")
                    for _, r in sub.iterrows():
                        m = str(r[MATCH_METHOD_COL])
                        t = int(r["total"])
                        c = int(r["correct"])
                        a = float(r["accuracy_pct"])
                        print(f"    - {m}: {c}/{t} correct ({a:.2f}%)")

        total = len(merged)
        correct = int(merged["Match"].sum())
        acc = (correct / total * 100) if total > 0 else 0.0

        print("\n=============================")
        print("✅ MATCHING VALIDATION")
        print("=============================")
        print("📌 Columns compared for matching:", cols_to_compare)
        print(f"✅ Overall: {correct}/{total} correct ({acc:.2f}%)")

        validation_summary["overall"] = {"correct": correct, "total": total, "accuracy_pct": acc}

        if LEVEL_COLUMN in merged.columns:
            validation_summary["per_level"] = {}
            for lvl in [1, 2, 3]:
                sub = merged[merged[LEVEL_COLUMN] == lvl]
                t = len(sub)
                c = int(sub["Match"].sum()) if t > 0 else 0
                a = (c / t * 100) if t > 0 else 0.0
                print(f"✅ Level {lvl}: {c}/{t} correct ({a:.2f}%)")
                validation_summary["per_level"][str(lvl)] = {"correct": c, "total": t, "accuracy_pct": a}

        # --------------------------
        # Ensure match_method is visible early in the CSV (RESTORED)
        # --------------------------
        if MATCH_METHOD_COL in merged.columns:
            front_cols = [MATCH_METHOD_COL, "Match"]
            if LEVEL_COLUMN in merged.columns:
                front_cols.insert(0, LEVEL_COLUMN)
            front_cols = [c for c in front_cols if c in merged.columns]
            other_cols = [c for c in merged.columns if c not in front_cols]
            merged = merged[front_cols + other_cols]

        merged.to_csv(VALIDATION_RESULTS_OUT, index=False)
        print(f"📁 Detailed validation saved to: {VALIDATION_RESULTS_OUT}")

    except Exception as e:
        print("❌ Validation failed:", e)

    # ============================
    # PRINT MATCHING SUMMARY + STORE JSON COLUMN (RESTORED)
    # ============================
    print("\n=============================")
    print("🧩 MATCHING PASS SUMMARY")
    print("=============================")
    print(f"  Pass 1 (single candidate): {pass_counts['pass1_single_candidate']}")
    print(f"  Pass 2 (cosine direct):    {pass_counts['pass2_cosine_direct']}")
    print(f"  Pass 3 (LLM choice):       {pass_counts['pass3_llm_choice']}")
    print(f"  Unmatched:                 {pass_counts['unmatched']}")

    print("\n=============================")
    print("⏱️ MATCHING TIMING SUMMARY")
    print("=============================")
    print(f"  Avg latency per entry: {matched_errorlevel_df[MATCH_LATENCY_COL].mean():.3f} sec")
    print(f"  Sum latency (entries): {matched_errorlevel_df[MATCH_LATENCY_COL].sum():.3f} sec")
    print(f"  Component total time (start-end): {component_total_sec:.3f} sec")

    print("\n=============================")
    print("🔢 MATCHING TOKEN SUMMARY (LLM pick only; other passes = 0)")
    print("=============================")
    print(f"  Avg total tokens per entry: {matched_errorlevel_df[MATCH_TOTAL_TOK_COL].mean():.1f}")
    print(f"  Total tokens (all entries): {int(matched_errorlevel_df[MATCH_TOTAL_TOK_COL].sum())}")

    results_json = build_matching_results_json(
        df_component=matched_errorlevel_df,
        component_total_sec=component_total_sec,
        pass_counts=pass_counts,
        validation_summary=validation_summary
    )
    matched_errorlevel_df[MATCH_RESULTS_COL] = results_json

    # overwrite to include summary column (RESTORED)
    matched_errorlevel_df.to_csv(MATCHED_ERRORLEVEL_OUT, index=False)
    print(f"\n✅ Updated {MATCHED_ERRORLEVEL_OUT} with '{MATCH_RESULTS_COL}' column.")


# ============================
# MAIN: run missing outputs up to MAX_RUNS (based on outputs)
# ============================
if __name__ == "__main__":
    available_runs = discover_available_runs_from_errors(LLM_PREFIX)
    if not available_runs:
        print(f"❌ No error files found for prefix '{LLM_PREFIX}'. Expected files like:")
        print(f"   Klassisch_Classified_Errors_Paper_{LLM_PREFIX}1_final.csv")
        raise SystemExit(1)

    existing_out_runs = discover_existing_runs_from_outputs(LLM_PREFIX)

    print(f"✅ Found available INPUT runs for '{LLM_PREFIX}': {available_runs}")
    print(f"✅ Found existing OUTPUT runs (by Validation_Results) for '{LLM_PREFIX}': {existing_out_runs}")

    missing = [r for r in available_runs if r not in existing_out_runs]

    remaining_slots = max(0, MAX_RUNS - len(existing_out_runs))
    to_run = missing[:remaining_slots]

    if not to_run:
        print(f"✅ Nothing to do. Already have {len(existing_out_runs)} outputs (max {MAX_RUNS}).")
        raise SystemExit(0)

    print(f"➡️ Will run MATCHING for runs: {to_run}")

    for run_id in to_run:
        run_one_matching_job(LLM_PREFIX, run_id)
