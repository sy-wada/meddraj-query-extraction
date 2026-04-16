#!/usr/bin/env python3
"""
Evaluate a saved Distilled NER (ner1) checkpoint on dev/test and write metrics_* files.

This is evaluation-only (no training).
It mirrors the evaluation logic used in train_ner1_optuna.py (CRF decode -> query -> ranking -> IR metrics).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer, Trainer, TrainingArguments

# Ensure project root (parent of scripts/) is on sys.path so we can import train_ner1_optuna.
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import train_ner1_optuna as ner1


def _build_ds(dataset_path: Path, split: str, *, tokenizer, data_args: ner1.DataArguments, ranking_args: ner1.RankingArguments):
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(dataset)}")
    if split not in dataset:
        raise KeyError(f"split={split} not found. available={list(dataset.keys())}")

    # label mapping must match training (built from train labels)
    raw_train = dataset["train"].filter(lambda ex: ex.get(data_args.label_field) is not None)
    label2id = ner1._build_label2id_from_dataset(raw_train, data_args.label_field)
    id2label = {v: k for k, v in label2id.items()}

    def _inject_gold(ex):
        ex["gold"] = ner1._extract_gold_codes(ex, data_args.gold_field, ranking_args.code_level)
        return ex

    raw_split = dataset[split].map(_inject_gold, batched=False)
    fn_kwargs = dict(
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=data_args.max_seq_length,
        label_field=data_args.label_field,
    )
    ds = raw_split.map(lambda ex: ner1._preprocess_ner1(ex, **fn_kwargs), batched=False)
    return ds, tokenizer, id2label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint dir containing model.safetensors/config.json")
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["dev", "test"], help="Which splits to evaluate")

    # Match main script args
    ap.add_argument("--mdra_suggest_dir", type=str, default="./mdra_suggest")
    ap.add_argument("--config_path", type=str, default="./mdra_suggest/config.yml")
    ap.add_argument("--use_shionogi_db", type=str, default="false")
    ap.add_argument("--code_level", type=str, default="llt")
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--k_eval", type=int, default=20)
    ap.add_argument("--csv_list_sep", type=str, default=" ||| ")

    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=128)
    ap.add_argument("--bf16", type=str, default="True")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path not found: {dataset_path}")

    # Build args objects compatible with helper functions
    ranking_args = ner1.RankingArguments(
        mdra_suggest_dir=args.mdra_suggest_dir,
        config_path=args.config_path,
        use_shionogi_db=str(args.use_shionogi_db).lower() in ("1", "true", "yes", "y"),
        code_level=args.code_level,
        top_k=int(args.top_k),
        k_eval=int(args.k_eval),
        csv_list_sep=args.csv_list_sep,
    )
    # DataArguments in train_ner1_optuna.py expects fields; we fill minimally.
    data_args = ner1.DataArguments(
        dataset_path=str(dataset_path),
        max_seq_length=int(args.max_seq_length),
        label_field="labels_ner1",
        text_field="text",
        offset_field="offset_mapping",
        gold_field="mdra_labels",
        gold_code_level="pt",
        max_train_samples=None,
        max_eval_samples=None,
    )
    # Use checkpoint tokenizer for preprocessing (special tokens etc.)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdra_assets = ner1.load_mdra_assets(
        mdra_suggest_dir=ranking_args.mdra_suggest_dir,
        config_path=ranking_args.config_path,
        use_shionogi_db=ranking_args.use_shionogi_db,
        device=device,
    )

    # Load model
    config = ner1.AutoConfig.from_pretrained(str(ckpt_dir))
    model = ner1.ModernBertCRFForTokenClassification.from_pretrained(str(ckpt_dir), config=config).to(device)

    # Eval-only Trainer
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        dataloader_drop_last=False,
        report_to=[],
        bf16=str(args.bf16).lower() in ("1", "true", "yes", "y"),
    )

    for split in args.splits:
        ds, _tok, id2label = _build_ds(dataset_path, split, tokenizer=tokenizer, data_args=data_args, ranking_args=ranking_args)
        collator = ner1.TokenClassificationDataCollator(tokenizer=tokenizer, label_field=data_args.label_field, pad_to_multiple_of=8)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=ds,
            data_collator=collator,
            processing_class=tokenizer,
        )

        metrics_avg, df_list = ner1.evaluate_ranking_metrics(
            model=model,
            trainer=trainer,
            eval_ds=ds,
            id2label=id2label,
            text_field=data_args.text_field,
            offset_field=data_args.offset_field,
            mdra_assets=mdra_assets,
            top_k=ranking_args.top_k,
            k_eval=ranking_args.k_eval,
            code_level=ranking_args.code_level,
            device=device,
            csv_list_sep=ranking_args.csv_list_sep,
        )

        if split == "dev":
            out_metrics = ckpt_dir / "metrics_dev.json"
            out_list = ckpt_dir / "metrics_dev_list.csv"
        else:
            out_metrics = ckpt_dir / "metrics_test.json"
            out_list = ckpt_dir / "metrics_test_list.csv"

        with out_metrics.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "k_eval": int(ranking_args.k_eval),
                    "code_level": ranking_args.code_level,
                    "top_k": int(ranking_args.top_k),
                    "metrics": metrics_avg,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        if df_list is not None and len(df_list):
            df_list.to_csv(out_list, index=False)

        print(f"[{split}] wrote {out_metrics}")


if __name__ == "__main__":
    main()

