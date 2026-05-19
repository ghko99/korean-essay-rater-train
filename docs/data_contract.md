# Training Data Contract

This project expects precomputed JSONL files for training, validation, and test splits. Keep the contract explicit so future experiments can compare runs without rediscovering data assumptions.

## Required Files

- `data_precomputed/train.jsonl`: training split used by `train.py`.
- `data_precomputed/valid.jsonl`: validation split for early stopping and checkpoint selection.
- `data_precomputed/test.jsonl`: held-out evaluation split.

## Expected Fields

Each row should preserve the essay text, rubric scores, rubric feedback targets, and any lightweight features consumed by the prompt template or feature sampler. Avoid changing field names between experiments unless the preprocessing script is updated at the same time.

## Pre-Run Checks

- Confirm Git LFS files are pulled before launching training.
- Sample a few rows from each split and verify UTF-8 decoding.
- Check that score labels stay in the documented 1-9 range.
- Keep raw source exports outside the repository unless they are intentionally tracked through Git LFS.
