#!/usr/bin/env python3
"""
PEFT_RL（QLoRA adapter）を Transformers + PEFT でロードし、test データで推論 latency を計測する。
CPU / GPU 両対応。結果を test_latency.csv に追記する。

実行例:
  # GPU
  python scripts/measure_peftrl_latency.py \
    --base_model_name_or_path unsloth/medgemma-4b-it-unsloth-bnb-4bit \
    --path_adapter models/qlora_adapter_medgemma-4b-it/checkpoint-10000 \
    --dataset_path data/mdra_rl_dataset_v20251210 \
    --device cuda --out_dir outputs/summary_test_metrics_for_simple_manuscript

  # CPU（4bit は CPU 非対応のため fp32 でロードする場合あり）
  python scripts/measure_peftrl_latency.py ... --device cpu --out_dir ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from string import Template

# Unsloth + PEFT 推論時の Dynamo エラー回避（latency 計測のみで compile は不要）
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# プロンプトは run_inference_with_adapter と同一
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
list_start = "<KeyPhrase>"
list_end = "</KeyPhrase>"

SYSTEM_PROMPT = Template(
    """
これからあなたは、入力本文 `<DOCUMENT> ... </DOCUMENT>` に含まれる医薬品に関する問い合わせ文から、定義された3つのカテゴリーに該当する **Key Phrase** を漏れなく抽出する。

## 入力の解釈
- 対象本文は `<DOCUMENT>` と `</DOCUMENT>` の間の文字列のみ。
- `<DOCUMENT>` の外側にある文字列はすべて無視する。

## 出力形式（厳守）
- 出力は **必ず** 次の2ブロックのみを、この順序で返す。追加の文章は禁止。

${reasoning_start}
（3カテゴリの観点で、該当箇所の有無と根拠を簡潔に。）
${reasoning_end}
${list_start}
- （抽出したKey Phrase 1）
- （抽出したKey Phrase 2）
...
${list_end}

- `${list_start}` タグ内は、抽出結果を **箇条書き（行頭に `- `）** で列挙する。
- Key Phrase が **0件** の場合は、次の形で返す（箇条書き行は出さない）:
  - `${list_start}${list_end}`
- 各 Key Phrase は **必ず `DOCUMENT` からの連続した原文コピー** とする。

""".strip()
)

USER_PROMPT = Template(
    """
<DOCUMENT>
${document}
</DOCUMENT>
""".strip()
)


def build_messages(text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT.substitute(
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
            list_start=list_start,
            list_end=list_end,
        )},
        {"role": "user", "content": USER_PROMPT.substitute(document=text)},
    ]


def main() -> None:
    torch._dynamo.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--path_adapter", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--device", type=str, required=True, choices=["cpu", "cuda"])
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/summary_test_metrics_for_simple_manuscript",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="PEFT_RL",
        help="test_latency.csv の model 列に書く名前",
    )
    ap.add_argument("--max_docs", type=int, default=None, help="計測する最大文書数（未指定で全件）")
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--do_sample", action="store_true", help="サンプリング（未指定は greedy）")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    path_adapter = Path(args.path_adapter)
    dataset_path = Path(args.dataset_path)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    model_name = args.model_name
    device_label = "GPU" if device.type == "cuda" else "CPU"

    if not path_adapter.exists():
        raise FileNotFoundError(f"path_adapter not found: {path_adapter}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path not found: {dataset_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    # Unsloth ベースモデルは FastLanguageModel でロード（meta device エラー回避）
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth モデル用に unsloth が必要です: pip install unsloth")

    load_in_4bit = device.type == "cuda"
    print(f"Loading base model: {args.base_model_name_or_path} on {device_label} (4bit={load_in_4bit}) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_name_or_path,
        max_seq_length=4096,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        device_map="auto" if device.type == "cuda" else "cpu",
    )
    print(f"Loading adapter: {path_adapter} ...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(path_adapter), is_trainable=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from datasets import load_from_disk
    dataset = load_from_disk(str(dataset_path))
    if "test" not in dataset:
        raise KeyError(f"dataset has no 'test' split: {list(dataset.keys())}")
    test_ds = dataset["test"]
    n_total = len(test_ds)
    max_docs = int(args.max_docs) if args.max_docs is not None else n_total
    n_docs = min(max_docs, n_total)
    print(f"Test docs: {n_docs} / {n_total}")

    # 1 doc ずつ generate して時間計測
    times_sec = []
    for i in tqdm(range(n_docs)):
        ex = test_ds[i]
        text = ex.get("input_text") or ""
        messages = build_messages(text)
        # Gemma3/MedGemma は content を [{"type": "text", "text": "..."}] 形式で期待する
        messages_for_template = [
            {"role": "system", "content": [{"type": "text", "text": messages[0]["content"]}]},
            {"role": "user", "content": [{"type": "text", "text": messages[1]["content"]}]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages_for_template,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if device.type == "cuda":
            input_ids = input_ids.to(device)
        elif hasattr(model, "device"):
            input_ids = input_ids.to(next(model.parameters()).device)

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - t0
        times_sec.append(elapsed)

    import numpy as np
    times_arr = np.array(times_sec)
    mean_sec = float(times_arr.mean())
    median_sec = float(np.median(times_arr))
    print(f"[{model_name}] {device_label}: {n_docs} docs -> median={median_sec:.4f}s, mean={mean_sec:.4f}s per doc")

    # test_latency.csv 更新
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
        latency_df.loc[mask, "note"] = "Transformers計測"
    else:
        new_row = pd.DataFrame([{
            "model": model_name,
            "device": device_label,
            "latency_sec_per_doc_median": round(median_sec, 6),
            "latency_sec_per_doc_mean": round(mean_sec, 6),
            "n_docs": n_docs,
            "mean_attempts": "",
            "note": "Transformers計測",
        }])
        latency_df = pd.concat([latency_df, new_row], ignore_index=True)

    latency_df.to_csv(latency_path, index=False)
    print(f"Updated: {latency_path}")


if __name__ == "__main__":
    main()
