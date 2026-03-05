import pandas as pd
import re

# ======================
# Flags
# ======================
filter_machine_resource = True
filter_machine_activity = True
filter_only_legacy_activities = True
remove_unwanted_columns = True
use_only_complete_timestamp = True

use_sample = True
sample_n = 500
sample_mode = "random"
sample_seed = 42  # nur für random

apply_activity_mapping = True  # mapping later (after filters)

# ======================
# 1) Load
# ======================
data = pd.read_csv("Production_Data.csv")

# Originale Zeilen-ID direkt sichern (für Validierung)
data["original_index"] = data.index

# Keep track of the "real/original" columns for final export
original_cols = list(data.columns)

# ======================
# 2) Timestamps
# ======================
data["End Time"] = pd.to_datetime(data["Complete Timestamp"], errors="coerce").dt.strftime("%Y/%m/%d %H:%M")

if use_only_complete_timestamp:
    data = data.drop(columns=["Start Timestamp", "Start Time"], errors="ignore")
    original_cols = [c for c in original_cols if c not in ["Start Timestamp", "Start Time"]]


# ======================
# 3) Normalize activity
# ======================
def normalize_activity(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^SETUP\s+", "", s, flags=re.IGNORECASE)
    return s


data["Activity_raw"] = data["Activity"].astype(str)
data["Activity_norm"] = data["Activity_raw"].map(normalize_activity)

# ======================
# 4) Normalize Resource to "Machine X" (overwrite Resource)
# ======================
if "Resource" in data.columns:
    data["Resource"] = (
        data["Resource"].astype(str)
        .str.extract(r"(Machine\s*\d+)", expand=False)
        .fillna(data["Resource"].astype(str))
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

# Helper only (won't be exported)
data["Resource_norm"] = data["Resource"] if "Resource" in data.columns else None

# ======================
# 5) Filters
# ======================
if filter_machine_resource and "Resource" in data.columns:
    data = data[data["Resource"].str.contains(r"\bMachine\b", case=False, na=False)]

if filter_machine_activity:
    data = data[data["Activity_norm"].str.contains(r"\bMachine\b", case=False, na=False)]

if filter_only_legacy_activities:
    legacy_keywords = [
        "Turning & Milling",
        "Round Grinding",
        "Laser Marking",
        "Lapping",
        "Wire Cut",
        "Flat Grinding",
    ]
    pat = "|".join(re.escape(k) for k in legacy_keywords)
    data = data[data["Activity_norm"].str.contains(pat, case=False, na=False)]

# ======================
# 6) Drop unwanted columns (and remove them from export list)
# ======================
columns_to_remove = [
    "Report Type", "Qty Completed", "Qty Rejected", "Qty for MRB",
    "Rework", "Work Order  Qty", "Span", "Part Desc."
]
if remove_unwanted_columns:
    drop_cols = [c for c in columns_to_remove if c in data.columns]
    data = data.drop(columns=drop_cols)
    original_cols = [c for c in original_cols if c not in drop_cols]


# ======================
# 7) Mapping
# ======================
def map_to_process_step(activity: str) -> str:
    a = str(activity)
    a = re.sub(r"\s*-\s*Machine\s*\d+\s*$", "", a, flags=re.IGNORECASE)
    a = re.sub(r"\s*-\s*Machine\s*$", "", a, flags=re.IGNORECASE)
    return a.strip()


if apply_activity_mapping:
    data["Activity_mapped"] = data["Activity_norm"].map(map_to_process_step)
else:
    data["Activity_mapped"] = data["Activity_norm"]

data["Activity"] = data["Activity_mapped"]

# ======================
# 8) Deduplicate
# ======================
data = data.drop_duplicates()

# ======================
# 9) Sample
# ======================
if use_sample:
    if len(data) >= sample_n:
        if sample_mode == "random":
            data = (
                data.sample(n=sample_n, random_state=sample_seed)
                .sort_values("original_index")
                .copy()
            )
        else:  # "first"
            data = data.sort_values("original_index").iloc[:sample_n].copy()
    else:
        print(f"WARNING: Only {len(data)} rows left after filters; cannot sample {sample_n}.")

add_end_time = True
export_cols = original_cols.copy()

# Ensure original_index is exported
if "original_index" in data.columns and "original_index" not in export_cols:
    export_cols.insert(0, "original_index")

if add_end_time and "End Time" in data.columns and "End Time" not in export_cols:
    export_cols.append("End Time")

# Drop helper columns (but NOT original_index!)
helper_cols = {"Activity_raw", "Activity_norm", "Resource_norm", "Activity_mapped"}
export_cols = [c for c in export_cols if c in data.columns and c not in helper_cols]

data[export_cols].to_csv("Filtered_Production_Data_Paper.csv", index=False)
print("Saved Filtered_Production_Data_Paper.csv with rows:", len(data))
