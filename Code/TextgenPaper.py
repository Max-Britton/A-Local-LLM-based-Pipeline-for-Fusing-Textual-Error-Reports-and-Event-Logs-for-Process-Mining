import pandas as pd
import random
from tqdm import tqdm
import ollama
import json
from collections import defaultdict
import re

# Set random seed for reproducibility
random.seed(42)

# ======================
# Prompt-level config
# ======================
LEVELS = [1, 2, 3]
TARGET_PER_LEVEL = 80  # => 240 total


def loose_time_match(time_str, text):
    text = text.lower()
    time_str = time_str.lower()
    time_tokens = time_str.split()

    if all(token in text for token in time_tokens):
        return "yes"

    partial_keywords = ["january", "february", "march", "april", "may", "june",
                        "july", "august", "september", "october", "november", "december"]
    if any(token in text for token in time_tokens) or any(kw in text for kw in partial_keywords):
        return "partial"
    return "no"


# ======================
# Text generation (L1/L2/L3)
# ======================
def generate_error_text(activity, resource, end_time, worker_id, error_type=None, level=1):
    # LEVEL 1
    def prompt_L1():
        return f"""
        You are a factory worker writing an incident report about a problem that occurred.
        
        Use the input values below to describe the issue in a natural and realistic way. Do not mention the error category "{error_type}" directly. Focus on the symptoms and consequences.
        
        Input:
        - Activity: Lapping
        - Machine: Machine 1
        - Time: 2012/02/14 09:38
        - Worker ID: ID4882
        - Category (hidden): Part Misalignment
        
        Output:
        During the Lapping process on Machine 1, an error occurred at 2012/02/14 09:38. My worker ID is ID4882. The machine ran normally at first, but I noticed that the component was not sitting correctly in its holder. As the process continued, the lapping pattern on the surface became noticeably uneven, with one side worn more than the other. There were no alerts from the system, but the final part had a clear angular distortion. I stopped the process and informed the supervisor for correction.
        
        ---
        
        Now use the following input:
        
        - Activity: {activity}
        - Machine: {resource}
        - Time: {end_time}
        - Worker ID: {worker_id}
        - Category (hidden): {error_type}
        
        Write the incident report below as one paragraph.
        """

    # LEVEL 2
    def prompt_L2():
        return f"""
        You are a factory worker writing an incident report about a problem that occurred.
        
        Use the input values below to describe the issue in a natural and realistic way. Do not mention the error category "{error_type}" directly. Focus on the symptoms and consequences.
        
        
        Rules:
        1) Do NOT use the exact timestamp format "YYYY/MM/DD HH:MM". Write it naturally, but keep date and time unambiguous.
        2) Mention the worker ID in a natural way (e.g., "worker 4882" or "operator 4882"), but keep the digits together and correct.
        
        Example:
        Input:
        - Activity: Lapping
        - Machine: Machine 1
        - Time: 2012/02/14 09:38
        - Worker ID: ID4882
        - Category (hidden): Part Misalignment
        
        Output:
        During the Lapping process on Machine 1 on 14 Feb 2012 at 09:38. I am worker 4882. The machine ran normally at first, but I noticed that the component was not sitting correctly in its holder. As the process continued, the lapping pattern on the surface became noticeably uneven, with one side worn more than the other. There were no alerts from the system, but the final part had a clear angular distortion. I stopped the process and informed the supervisor for correction.
        
        ---
        
        Now use the following input:
        
        - Activity: {activity}
        - Machine: {resource}
        - Time: {end_time}
        - Worker ID: {worker_id}
        - Category (hidden): {error_type}
        
        Write the incident report below as one paragraph, following the rules above.
        """

    # LEVEL 3
    def prompt_L3():
        return f"""
        You are a factory worker writing an incident report about a problem that occurred.
        
        Use the input values below to describe the issue in a natural and realistic way. Do not mention the error category "{error_type}" directly. Focus on the symptoms and consequences.
        
        Rules:
        1) Do NOT use "YYYY/MM/DD HH:MM". Express date and time in words and you may split them across sentences, but keep them unambiguous.
        2) Do NOT write the machine as "Machine <digits>" in the paragraph. Refer to it using words (e.g., "machine one", "the first machine") so the machine number is still clear.
        3) Write the worker ID digits separated by dashes or spaces (e.g., "4-8-8-2" or "4 8 8 2"), but keep the same digits.
        4) Add 1–2 short distractor details (batch/temperature/checks), but do not change the incident symptoms.
        
        Example:
        Input:
        - Activity: Lapping
        - Machine: Machine 1
        - Time: 2012/02/14 09:38
        - Worker ID: ID4882
        - Category (hidden): Part Misalignment
        
        Output:
        This entry concerns the Lapping task on the morning of the 14th of February 2012. It was around nine thirty-eight when the issue showed up on the first machine. My operator code is 4-8-8-2. The coolant temperature was stable at about 22 degrees and the batch label on the tray read 500, so nothing unusual there. The machine ran normally at first, but I noticed that the component was not sitting correctly in its holder. As the process continued, the lapping pattern on the surface became noticeably uneven, with one side worn more than the other. There were no alerts from the system, but the final part had a clear angular distortion. I stopped the process and informed the supervisor for correction.
        
        ---
        
        Now use the following input:
        
        - Activity: {activity}
        - Machine: {resource}
        - Time: {end_time}
        - Worker ID: {worker_id}
        - Category (hidden): {error_type}
        
        Write the incident report below as one paragraph, following the rules above.
        """

    # Auswahl der Prompt-Funktion
    if level == 3:
        prompt = prompt_L3()
    elif level == 2:
        prompt = prompt_L2()
    else:
        prompt = prompt_L1()

    try:
        response = ollama.chat(
            model="mistral:7b",
            messages=[{"role": "user", "content": prompt}],
        )
        clean_text = response["message"]["content"].replace("Output:", "").strip()
        return clean_text
    except Exception as e:
        raise RuntimeError(f"Error during text generation: {e}")


# ======================
# Generation + saving
# ======================
def generate_and_save_error_texts(input_csv, output_texts_csv, output_eventlog_csv):
    """
    Generates multiple error texts for machine-related events, stores results in two output CSVs:
    - One with only the error texts and metadata
    - One containing the original event log enriched with the generated texts
    """

    df = pd.read_csv(input_csv)

    # Keep original index for later matching/validation (traceability)
    if "original_index" not in df.columns:
        df["original_index"] = df.index

    # Add columns for ground truth category + prompt level
    df["True_Category"] = None
    df["Prompt_Level"] = None

    # List for error texts without metadata
    error_texts_list = []

    # Add new column for unstructured error descriptions
    df["Unstructured Text"] = None
    text_count = 0
    stats = {"activity": 0, "resource": 0, "end_time": 0, "worker_id": 0}

    # --- Begin category grouping logic ---
    category_counters = defaultdict(int)
    max_per_category = 30
    error_categories = [
        "Air Pressure Error",
        "Lubrication Deficiency",
        "Power Outage",
        "Part Misalignment",
        "Clamping Error",
        "Control Software Glitch",
        "Cooling System Failure",
        "Feed Path Error"
    ]
    # --- End category grouping logic ---

    # --- Level planning: exactly 80 per level ---
    level_counters = {lvl: 0 for lvl in LEVELS}

    def next_level():
        for lvl in LEVELS:
            if level_counters[lvl] < TARGET_PER_LEVEL:
                return lvl
        return None  # done

    machine_rows = df[df["Resource"].str.contains("Machine", case=False, na=False)].copy()
    machine_rows = machine_rows.sample(frac=1, random_state=42)  # shuffle

    for _, row in tqdm(machine_rows.iterrows(), total=machine_rows.shape[0], desc="Generating Error Texts by Rows"):
        lvl = next_level()
        if lvl is None:
            break  # we have 80 per level

        # This is the real df index for writing back into df
        row_idx = row.name

        # This is the traceability index from the original (source) dataset
        original_idx = row["original_index"]

        available_categories = [cat for cat in error_categories if category_counters[cat] < max_per_category]
        if not available_categories:
            break

        error_type = random.choice(available_categories)

        activity = row["Activity"]
        resource = row["Resource"]
        end_time = row["End Time"]
        worker_id = row["Worker ID"]

        try:
            generated_text = generate_error_text(activity, resource, end_time, worker_id, error_type, level=lvl)

            # count only if generation succeeded
            category_counters[error_type] += 1
            level_counters[lvl] += 1

            # write back using the actual df index
            df.at[row_idx, "Unstructured Text"] = generated_text
            df.at[row_idx, "True_Category"] = error_type
            df.at[row_idx, "Prompt_Level"] = lvl

            # Statistical checks
            activity_core = str(activity).split("-")[0].strip().lower()
            resource_tokens = str(resource).split()

            activity_present = activity_core in generated_text.lower()
            resource_present = any(token.lower() in generated_text.lower() for token in resource_tokens)
            end_match_level = loose_time_match(str(end_time), generated_text)
            worker_id_present = str(worker_id) in generated_text

            if activity_present:
                stats["activity"] += 1
            if resource_present:
                stats["resource"] += 1
            if end_match_level == "yes":
                stats["end_time"] += 1
            if worker_id_present:
                stats["worker_id"] += 1

            stats_summary = [
                f"activity: {'yes' if activity_present else 'no'}",
                f"end_time: {end_match_level}",
                f"worker_id: {'yes' if worker_id_present else 'no'}"
            ]
            df.at[row_idx, "statics"] = "; ".join(stats_summary)

            ground_truth_json = {
                "Activity": activity,
                "Machine": resource,
                "Error time and Error date": end_time,
                "WorkerID": worker_id,
            }

            # store BOTH indices:
            # - index = df row index (used to apply texts back)
            # - original_index = traceability to source CSV row
            error_texts_list.append({
                "index": row_idx,
                "original_index": original_idx,
                "Error Text": generated_text,
                "True_Category": error_type,
                "Prompt_Level": lvl,
                "Ground Truth": json.dumps(ground_truth_json, ensure_ascii=False),
                "Match_Activity": activity_present,
                "Match_Resource": resource_present,
                "Match_EndTime": end_match_level,
                "Match_WorkerID": worker_id_present
            })

            text_count += 1

        except RuntimeError as e:
            print(
                f"❌ Error generating text for df_index {row_idx} (original_index {original_idx}), Activity {activity}: {e}")

    # Convert list to DataFrame and save only the error texts
    error_texts_df = pd.DataFrame(error_texts_list)
    error_texts_df.to_csv(output_texts_csv, index=False)
    print(f"✅ Unmatched error texts saved to: {output_texts_csv}")

    # Print summary of errors per category
    print("\n📊 Anzahl generierter Fehler pro Kategorie:")
    for cat, count in category_counters.items():
        print(f"- {cat}: {count}")

    # Print summary per level
    print("\n📊 Anzahl generierter Texte pro Prompt-Level:")
    for lvl in LEVELS:
        print(f"- L{lvl}: {level_counters[lvl]}")

    # Save enriched event log with error texts
    df.to_csv(output_eventlog_csv, index=False)
    print(f"✅ Matched event log saved to: {output_eventlog_csv}")

    print("\n📊 Field Presence Statistics in Generated Texts:")
    for key, value in stats.items():
        print(f"- {key}: {value}/{text_count} present")


def apply_error_texts_to_log(master_texts_csv, input_eventlog_csv, output_csv):
    df_log = pd.read_csv(input_eventlog_csv)
    df_texts = pd.read_csv(master_texts_csv)

    for _, row in df_texts.iterrows():
        idx = row["index"]
        if idx in df_log.index:
            df_log.at[idx, "Unstructured Text"] = row["Error Text"]
            df_log.at[idx, "True_Category"] = row["True_Category"]
            df_log.at[idx, "Prompt_Level"] = row.get("Prompt_Level", None)
            df_log.at[idx, "statics"] = row.get("statics", "")
    df_log.to_csv(output_csv, index=False)
    print(f"✅ Texts applied to: {output_csv}")


# Run the script
if __name__ == "__main__":
    input_file = "Filtered_Production_Data_Paper.csv"
    output_texts_file = "Generated_Error_Texts_Paper_new_test.csv"
    output_eventlog_file = "Validation_Generated_Event_Log_With_Texts_Paper_new_test.csv"

    generate_and_save_error_texts(input_file, output_texts_file, output_eventlog_file)
