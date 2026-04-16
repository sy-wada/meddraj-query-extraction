#!/usr/bin/env python3
"""
LLM 推論結果の JSON から generate_time_seconds と attempts を集計し、
詳細 CSV と test_latency.csv の GPU 行を更新する。

- 入力: outputs/jsons/<run_name>/test/seed_*/<guid>.json（各JSONに generate_time_seconds, attempts 必須）
- 出力:
  - <out_dir>/llm_latency_detail.csv … model, seed, guid, generate_time_seconds, attempts
  - <out_dir>/test_latency.csv を更新（既存ファイルがあれば読み、LLM の device=GPU 行を上書き）

使い方:
  python aggregate_llm_latency_from_jsons.py --out_dir outputs/summary_test_metrics_for_simple_manuscript
  python aggregate_llm_latency_from_jsons.py --out_dir ... --config_json path/to/latency_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# モデル名 -> JSON ルート（outputs/jsons 直下のディレクトリ名）
# run_summarize_for_simple_manuscript で用いている評価と対応させる
DEFAULT_JSON_ROOTS = {
    "Base": "unsloth_medgemma-4b-it-unsloth-bnb-4bit_vanilla",
    "Base_Large": "unsloth_medgemma-27b-text-it-unsloth-bnb-4bit_vanilla",
    "PEFT_RL": "unsloth_medgemma-4b-it-unsloth-bnb-4bit_checkpoint-10000_peftrl_step1",
}


def _load_mapping_json(path: Path) -> dict[str, str]:
    """{"model_name": "json_dir_name", ...} 形式の JSON を読み込む。"""
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in obj.items()}


def _collect_from_json_dir(json_root: Path, model_name: str) -> list[dict]:
    """test/seed_*/*.json を走査し、(guid, seed, generate_time_seconds, attempts) を集める。"""
    test_dir = json_root / "test"
    if not test_dir.is_dir():
        return []
    rows = []
    for seed_dir in sorted(test_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        try:
            seed = int(seed_dir.name.replace("seed_", ""))
        except ValueError:
            continue
        for json_path in seed_dir.glob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            guid = data.get("guid") or json_path.stem
            t = data.get("generate_time_seconds")
            a = data.get("attempts")
            if t is None and a is None:
                continue
            rows.append({
                "model": model_name,
                "seed": seed,
                "guid": str(guid),
                "generate_time_seconds": float(t) if t is not None else np.nan,
                "attempts": int(a) if a is not None else None,
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="LLM JSON から latency/attempts を集計し CSV 出力・test_latency 更新")
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/summary_test_metrics_for_simple_manuscript",
        help="出力ディレクトリ（llm_latency_detail.csv と test_latency.csv の置き場）",
    )
    p.add_argument(
        "--jsons_base",
        type=str,
        default="outputs/jsons",
        help="JSON ルートの親ディレクトリ",
    )
    p.add_argument(
        "--config_json",
        type=str,
        default="",
        help="モデル名 -> JSON ディレクトリ名のマッピング JSON（未指定時は DEFAULT_JSON_ROOTS を使用）",
    )
    p.add_argument(
        "--no_update_latency_csv",
        action="store_true",
        help="test_latency.csv を更新しない（詳細 CSV のみ出力）",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsons_base = Path(args.jsons_base)

    if args.config_json:
        mapping = _load_mapping_json(Path(args.config_json))
    else:
        mapping = DEFAULT_JSON_ROOTS

    # 1) 詳細 CSV 作成
    all_rows = []
    for model_name, dir_name in mapping.items():
        json_root = jsons_base / dir_name
        if not json_root.is_dir():
            print(f"Skip (not found): {json_root}")
            continue
        rows = _collect_from_json_dir(json_root, model_name)
        all_rows.extend(rows)
        print(f"  {model_name}: {len(rows)} rows from {json_root}")

    if not all_rows:
        print("No JSON data collected. Exiting.")
        return

    detail_df = pd.DataFrame(all_rows)
    detail_path = out_dir / "llm_latency_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"Wrote: {detail_path}  (rows={len(detail_df)})")

    # 2) test_latency.csv の GPU 行を更新（LLM のみ）
    if args.no_update_latency_csv:
        return

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

    for model_name in detail_df["model"].unique():
        sub = detail_df[detail_df["model"] == model_name]
        n_docs = sub["guid"].nunique()
        median_sec = sub["generate_time_seconds"].median()
        mean_sec = sub["generate_time_seconds"].mean()
        mean_attempts = sub["attempts"].mean()
        if pd.isna(mean_attempts):
            mean_attempts_str = ""
        else:
            mean_attempts_str = f"{mean_attempts:.2f}"

        mask = (latency_df["model"] == model_name) & (latency_df["device"] == "GPU")
        if mask.any():
            latency_df.loc[mask, "latency_sec_per_doc_median"] = median_sec
            latency_df.loc[mask, "latency_sec_per_doc_mean"] = mean_sec
            latency_df.loc[mask, "n_docs"] = int(n_docs)
            latency_df.loc[mask, "mean_attempts"] = mean_attempts_str  # type: ignore[call-overload]
            latency_df.loc[mask, "note"] = "JSON集計"
        else:
            new_row = pd.DataFrame([{
                "model": model_name,
                "device": "GPU",
                "latency_sec_per_doc_median": median_sec,
                "latency_sec_per_doc_mean": mean_sec,
                "n_docs": int(n_docs),
                "mean_attempts": mean_attempts_str,
                "note": "JSON集計",
            }])
            latency_df = pd.concat([latency_df, new_row], ignore_index=True)

    latency_df.to_csv(latency_path, index=False)
    print(f"Updated: {latency_path}  (LLM GPU rows filled from JSON)")


if __name__ == "__main__":
    main()
