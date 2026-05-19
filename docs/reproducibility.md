# Reproducibility Notes

Use this checklist when comparing Korean essay rater training runs.

## Run Metadata

Record the following values with every run:

- Base model path and revision.
- Dataset split version or Git LFS pointer revision.
- LoRA rank, alpha, dropout, learning rate, batch size, gradient accumulation, and epoch count.
- Random seed, CUDA version, GPU type, and effective batch size.
- Whether WandB logging, dry-run mode, or resume mode was enabled.

## Suggested Flow

1. Pull Git LFS data and verify split counts.
2. Run a dry load check before a full training job.
3. Save the exact command line and environment variables.
4. Archive the selected checkpoint, final metrics, and evaluation logs together.

## Comparison Notes

Compare QWK, MAE, RMSE, and per-rubric behavior from the same split. Avoid mixing results from different preprocessing revisions unless the report calls that out explicitly.
