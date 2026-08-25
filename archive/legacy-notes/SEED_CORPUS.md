# Seed Corpus for Evaluation

The `scripts/seed_corpus.py` script is used to ingest the test documents located in `uploads_stress/` into the `default` tenant's vector collection.

This is necessary for the Evaluation Harness (`scripts/eval_run.py`) to function correctly, as the test cases depend on specific content (e.g., Wifi passwords, onboarding guides) being present in the Retrieve step.

## Usage

```powershell
# Run from repository root
python scripts/seed_corpus.py
```

## When to run
- Before running `python scripts/eval_run.py` or `scripts/eval.ps1` for the first time.
- If you modify files in `uploads_stress/`.
- If you clear the `vectorstore/` directory.
