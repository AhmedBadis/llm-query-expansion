# Testing Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

## Run All Tests

```bash
# Run all unit tests and verify all commands work
pytest tests/test_eval.py -v && \
python -m src.eval.compute_metrics --run tests/data/sample_run.csv --qrels tests/data/sample_qrels.csv --metric all && \
python -m src.eval.stats_tests tests/data/sample_run.csv tests/data/sample_run.csv tests/data/sample_qrels.csv ndcg@10 10
```

**Note:** Robustness analysis test requires a vocabulary file and is not included in the automated test suite. Run it separately when vocab file is available.

## Unit Tests

```bash
pytest tests/test_eval.py -v
```

## Compute Metrics

```bash
python -m src.eval.compute_metrics \
    --run tests/data/sample_run.csv \
    --qrels tests/data/sample_qrels.csv \
    --metric all
```

## Statistical Tests

```bash
python -m src.eval.stats_tests \
    tests/data/sample_run.csv \
    tests/data/sample_run.csv \
    tests/data/sample_qrels.csv \
    ndcg@10 10
```

## Robustness Analysis

```bash
python -m src.eval.robustness_slices \
    --run tests/data/sample_run.csv \
    --vocab dataset/trec-covid/vocab_top50k.txt \
    --queries dataset/trec-covid/queries.json \
    --out results/slices.csv
```

**Note:** Replace `vocab_file` and `queries_file` with actual file paths. The `--queries` argument is optional if queries can be extracted from the run file. These files don't exist yet so it will throw an error.
