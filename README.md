# A Local LLM-based Pipeline for Fusing Textual Error Reports and Event Logs for Process Mining

This repository is the **artifact** accompanying the paper  
**“A Local LLM-based Pipeline for Fusing Textual Error Reports and Event Logs for Process Mining.”**

It provides an end-to-end **local** pipeline (via **Ollama**) to fuse unstructured, worker-style **textual error reports** with structured **event logs** for process mining by:

1) filtering the Production Data event log to the paper scenario  
2) generating synthetic error reports aligned to the log  
3) extracting structured attributes from the reports  
4) classifying each report into one fixed error class  
5) matching reports to event-log events and enriching the log  
   (**3-level matching:** attribute filtering → embedding similarity → LLM-based selection)

---

## Repository Layout

- `Code/` — pipeline scripts  
  - `FilterPaper.py` — filter/prepare the event log  
  - `TextgenPaper.py` — generate synthetic textual error reports  
  - `ExtractPaper.py` — extract structured attributes from reports  
  - `ClassificationPaper.py` — classify reports into fixed error classes  
  - `MatchingPaper_final.py` — 3-level matching + enrichment

- `InputData/` — input files (event log + supporting files)  
- `ExtractionData/` — outputs of extraction  
- `ClassificationData/` — outputs of classification  
- `MatchingData/` — outputs of matching + validation  
- `confusion_matrices/` — confusion matrices created during classification  
- `requirements.txt` — Python dependencies

---

## Requirements

### 1) Python
- Python **3.13** recommended
- requirements.txt

### 2) Ollama
This artifact requires a local installation of **Ollama**. Before running any script, please install Ollama and make sure the following models are available locally (pull them once if needed): `mistral:7b` (text generation), `gpt-oss:20b`, `deepseek-r1:14b`, `gemma3:12b`, `llama3.2:3b`, `llama3.2:1b` (pipeline runs), and `mxbai-embed-large` (embeddings for matching). You can verify installation with `ollama --version` and check installed models with `ollama list`.

### 3) Run

Run 1. `ExtractPaper.py`
Run 2. `ClassificationPaper.py`
Run 3. `MatchingData/`
