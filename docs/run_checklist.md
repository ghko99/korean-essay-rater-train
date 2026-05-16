# Training Run Checklist

Use this checklist before starting a full QLoRA training run.

## Environment

- Confirm Python 3.10+ is active.
- Install dependencies with `pip install -r requirements.txt`.
- Set `BASE_MODEL` to the local EXAONE checkpoint path.
- Run `git lfs pull` so `data_precomputed/` contains the expected JSONL files.

## Smoke test

```bash
bash run_train.sh --dry_run --no_wandb
```

## Full run

```bash
bash run_train.sh
```

## Outputs

Keep generated runs, checkpoints, W&B files, and temporary local data out of Git. The `.gitignore` file covers the common output paths.
