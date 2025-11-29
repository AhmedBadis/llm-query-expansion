# Domain-Specific Query Expansion with LLMs

## Members

- Ahmed Badis Lakrach
- Berktug Kaan Özkan
- Rami Baffoun
- Rami Kallel

## Setup

### 1. Clone the repository

```bash
git clone git@gitlab.informatik.uni-bonn.de:lab-information-retrieval/domain-specific-query-expansion-with-llms.git
cd domain-specific-query-expansion-with-llms
```

### 2. Create and activate virtual environment

```bash
# Create
python3 -m venv .venv  # Linux/macOS
python -m venv .venv   # Windows

# Activate
source .venv/bin/activate  # Linux/macOS
source .venv/Scripts/activate  # Windows (Git Bash)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Data & Ingestion

Use the ingest CLI to prepare the workspace and download datasets:

```bash
python -m src.ingest prepare       # creates folders and downloads NLTK assets
python -m src.ingest download --dataset trec-covid
```

Run `python -m src.ingest --help` to see all available commands.

## How to run?

To run the pipeline using a certain method, run all cells at one of:

- [runner/eval/baseline.ipynb](runner/eval/baseline.ipynb)
- [runner/eval/append.ipynb](runner/eval/baseline.ipynb)
- [runner/eval/reformulate.ipynb](runner/eval/reformulate.ipynb)
- [runner/eval/agr.ipynb](runner/eval/agr.ipynb)

## Testing

For testing commands and evaluation scripts, run all cells at [runner/test.ipynb](runner/test.ipynb).

## Documentation & Patch Notes

- [doc/progress_report](doc/progress_report) - Progress reports
- #TODO: Generate official documentation for all methods/files.
