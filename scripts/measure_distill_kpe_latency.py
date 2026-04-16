#!/usr/bin/env python3
"""
Distill KPE（NER）モデルで test データを推論し、1文書あたりの latency を計測する。
意味的類似度（FAISS ランキング）は含めず、モデル入力から extracted_clean が得られるまでを
一事例ずつ計測し、median / mean を算出する。CPU と GPU でそれぞれ実行し、結果を test_latency.csv に追記する。

実行例:
  # GPU で計測（ckpt_dir は best_trial_98/seed_* のいずれかを指定）
  python scripts/measure_distill_kpe_latency.py \
    --ckpt_dir outputs/20251212_ner1_final_s10/unsloth_medgemma-4b-it-unsloth-bnb-4bit_checkpoint-10000_filtered/best_trial_98/seed_5 \
    --dataset_path data/unsloth_medgemma-4b-it-unsloth-bnb-4bit_checkpoint-10000 \
    --device cuda \
    --out_dir outputs/summary_test_metrics_for_simple_manuscript

  # CPU で計測
  python scripts/measure_distill_kpe_latency.py \
    --ckpt_dir ... --dataset_path ... --device cpu --out_dir ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import eval_ner1_checkpoint as eval_ner
import train_ner1_optuna as ner1
from tools.ner_decode import bio_tags_to_spans, spans_to_strings


@torch.no_grad()
def _run_one_doc(
    *,
    model: ner1.ModernBertCRFForTokenClassification,
    ex: dict,
    id2label: dict[int, str],
    text_field: str,
    offset_field: str,
    device: torch.device,
) -> tuple[float, list[str]]:
    """
    一事例について、モデル入力から extracted_clean が得られるまでの時間を計測する。
    train_ner1_optuna.evaluate_ranking_metrics の「logits取得→CRF decode→span→文字列」部分に相当。
    戻り値: (elapsed_sec, extracted_clean)
    """
    input_ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([ex["attention_mask"]], dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # (1, T, C)

    model_device = next(model.parameters()).device
    logits_t = logits.float()
    if logits_t.size(1) >= 2:
        emissions = logits_t[:, 1:-1, :]
        mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=model_device)
    else:
        emissions = logits_t
        mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=model_device)

    paths = model.crf.decode(emissions, mask=mask)
    tags = [id2label.get(int(x), "O") for x in paths[0]]
    spans = bio_tags_to_spans(tags)
    text = ex.get(text_field) or ""
    offset_mapping = ex.get(offset_field) or []
    extracted = spans_to_strings(spans, offset_mapping=offset_mapping, text=text)
    extracted = [s for s in extracted if s and str(s).strip()]
    extracted_clean = [str(s).replace("\r", " ").replace("\n", " ").strip() for s in extracted]

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, extracted_clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--device", type=str, required=True, choices=["cpu", "cuda"])
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/summary_test_metrics_for_simple_manuscript",
        help="test_latency.csv を更新するディレクトリ",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="Distill_NER_filtered",
        help="test_latency.csv の model 列に書く名前",
    )
    ap.add_argument("--warmup_fraction", type=float, default=0.0, help="計測前に warmup する割合（0〜1）。0 で無効")
    ap.add_argument("--mdra_suggest_dir", type=str, default="./mdra_suggest")
    ap.add_argument("--config_path", type=str, default="./mdra_suggest/config.yml")
    ap.add_argument("--use_shionogi_db", type=str, default="false")
    ap.add_argument("--code_level", type=str, default="llt")
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--k_eval", type=int, default=20)
    ap.add_argument("--csv_list_sep", type=str, default=" ||| ")
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--max_docs", type=int, default=None, help="計測する最大文書数（未指定で全件）")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    dataset_path = Path(args.dataset_path)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    model_name = args.model_name

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path not found: {dataset_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking_args = ner1.RankingArguments(
        mdra_suggest_dir=args.mdra_suggest_dir,
        config_path=args.config_path,
        use_shionogi_db=str(args.use_shionogi_db).lower() in ("1", "true", "yes", "y"),
        code_level=args.code_level,
        top_k=int(args.top_k),
        k_eval=int(args.k_eval),
        csv_list_sep=args.csv_list_sep,
    )
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

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    config = ner1.AutoConfig.from_pretrained(str(ckpt_dir))
    model = ner1.ModernBertCRFForTokenClassification.from_pretrained(str(ckpt_dir), config=config).to(device)
    model.eval()

    ds, _tok, id2label = eval_ner._build_ds(
        dataset_path, "test", tokenizer=tokenizer, data_args=data_args, ranking_args=ranking_args
    )
    n_total = len(ds)
    if args.max_docs is not None:
        ds = ds.select(range(min(args.max_docs, n_total)))
    n_docs = len(ds)

    # Warmup（任意）
    if args.warmup_fraction > 0:
        n_warmup = max(1, int(n_docs * args.warmup_fraction))
        for i in range(n_warmup):
            _run_one_doc(
                model=model,
                ex=ds[i],
                id2label=id2label,
                text_field=data_args.text_field,
                offset_field=data_args.offset_field,
                device=device,
            )
        print(f"Warmup: {n_warmup} docs")

    # 本計測: 一事例ずつ計測し、per-doc latency の median / mean を算出
    latencies: list[float] = []
    for i in tqdm(range(n_docs)):
        elapsed, _ = _run_one_doc(
            model=model,
            ex=ds[i],
            id2label=id2label,
            text_field=data_args.text_field,
            offset_field=data_args.offset_field,
            device=device,
        )
        latencies.append(elapsed)

    median_sec = float(np.median(latencies))
    mean_sec = float(np.mean(latencies))
    device_label = "GPU" if device.type == "cuda" else "CPU"
    print(f"[{model_name}] {device_label}: {n_docs} docs -> median={median_sec:.4f} s/doc, mean={mean_sec:.4f} s/doc")

    # test_latency.csv を更新
    latency_path = out_dir / "test_latency.csv"
    if latency_path.exists():
        latency_df = pd.read_csv(latency_path)
        if "mean_attempts" not in latency_df.columns:
            latency_df["mean_attempts"] = ""
        latency_df["mean_attempts"] = latency_df["mean_attempts"].astype(object)
    else:
        latency_df = pd.DataFrame(columns=[
            "model", "device", "latency_sec_per_doc_median", "latency_sec_per_doc_mean",
            "n_docs", "mean_attempts", "note",
        ])

    mask = (latency_df["model"] == model_name) & (latency_df["device"] == device_label)
    if mask.any():
        latency_df.loc[mask, "latency_sec_per_doc_median"] = round(median_sec, 6)
        latency_df.loc[mask, "latency_sec_per_doc_mean"] = round(mean_sec, 6)
        latency_df.loc[mask, "n_docs"] = n_docs
        latency_df.loc[mask, "mean_attempts"] = ""  # NER は attempts なし
        latency_df.loc[mask, "note"] = "計測"
    else:
        new_row = pd.DataFrame([{
            "model": model_name,
            "device": device_label,
            "latency_sec_per_doc_median": round(median_sec, 6),
            "latency_sec_per_doc_mean": round(mean_sec, 6),
            "n_docs": n_docs,
            "mean_attempts": "",
            "note": "計測",
        }])
        latency_df = pd.concat([latency_df, new_row], ignore_index=True)

    latency_df.to_csv(latency_path, index=False)
    print(f"Updated: {latency_path}")


if __name__ == "__main__":
    main()
