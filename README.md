# Domain-Specific Query Expansion with LLMs

## Members

- Ahmed Badis Lakrach
- Berktug Kaan Özkan
- Rami Baffoun

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:AhmedBadis/llm-query-expansion.git
cd llm-query-expansion
```

### 2. Create and activate a virtual environment

```bash
# Create
python3 -m venv .venv  # Linux/macOS
python -m venv .venv   # Windows (Git Bash)

# Activate
source .venv/bin/activate  # Linux/macOS
source .venv/Scripts/activate  # Windows (Git Bash)
```

### 3. Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

## 4. Run the pipeline

To run the pipeline using a certain method, run all cells at one of:

- [notebook/baseline.ipynb](notebook/baseline.ipynb)
- [notebook/append.ipynb](notebook/baseline.ipynb)
- [notebook/reformulate.ipynb](notebook/reformulate.ipynb)
- [notebook/agr.ipynb](notebook/agr.ipynb)

## Summary

For running the summary notebook, run all cells at [notebook/summary.ipynb](notebook/summary.ipynb).

## Testing

For testing commands and evaluation scripts, run all cells at [notebook/test.ipynb](notebook/test.ipynb).

### Deactivate virtual environment

```bash
deactivate  # Linux/macOS & Windows (Git Bash)
```

## Documentation & Patch Notes

- [doc/progress_report](doc/progress_report) - Progress reports

## Data & Ingestion

You can use the ingest CLI to manually troubleshoot the process of preparing the workspace, downloading the datasets, and materializing the ingested format:

```bash
python -m src.ingest prepare  # creates folders and downloads NLTK assets
python -m src.ingest download --dataset trec_covid
python -m src.ingest ingest --dataset trec_covid
```

Run `python -m src.ingest --help` to see all available commands.