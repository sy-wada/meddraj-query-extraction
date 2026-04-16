#!/usr/bin/env python3
"""
AE/SS 層別の IR 指標を再計算するスクリプト。

- 入力:
  - test_scores_all.csv
      model, seed, guid, extracted, gold_llt_code, AP, RR, Recall, nDCG
      （既存評価結果。extracted は \" | \" や \" ||| \" を含む文字列。）
  - dataset_mdra_label_summary.csv
      AE, SS, num_AE, num_SS, n_tokens, guid, split
      （文書ごとの AE/SS ラベル情報。ここから AE/SS 用 gold LLT 集合を作る。）
  - mdra_suggest/config.yml および mdra_suggest/cache/*
      既存の MedDRA/J 検索ロジック（埋め込み＋Faiss＋社内DB拡張）を再利用。

- 出力:
  1. 文書単位の AE/SS 別 IR 指標:
       test_scores_ae_ss_recomputed.csv
       （test_scores_all.csv と同じ行数＋ AE/SS 用の AP/RR/Recall/nDCG 列および has_AE/has_SS フラグ）
  2. モデル別の AE/SS サブグループ要約:
       model_summary_ae_ss.csv
       model, metric, group, n_docs, n_seeds, mean
       （集約ルール: 文書ごとに seed 平均 → 文書平均）

注意:
- AE/SS を持たない文書は、そのカテゴリのサブ解析平均には含めない。
- 近傍探索は mdra_suggest のキャッシュを用いて、既存ロジックと同じ方法で実行する。
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import faiss  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# パス設定（プロジェクトルートと mdra_suggest を import 可能にする）
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "mdra_suggest") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "mdra_suggest"))

from eval_adapters import (  # type: ignore[import-untyped]
    load_config as load_mdra_config,
    load_se_model,
    create_ranking_for_predictions,
    modify_ranking,
)
from mdra_suggest.evaluate_ranking import (  # type: ignore[import-untyped]
    calculate_ir_metrics,
)


@dataclass(frozen=True)
class MdraResources:
    mdra_embed_dict: Dict[str, List[int]]
    faiss_index: faiss.IndexFlatIP
    faiss_tbl: pd.DataFrame
    cd_llt2pt: Dict[int, int]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="AE/SS 層別の IR 指標を既存の MedDRA/J 検索ロジックで再計算する。"
    )
    ap.add_argument(
        "--test_scores_csv",
        type=str,
        required=True,
        help="test_scores_all.csv へのパス",
    )
    ap.add_argument(
        "--label_summary_csv",
        type=str,
        required=True,
        help="dataset_mdra_label_summary.csv へのパス",
    )
    ap.add_argument(
        "--mdra_suggest_dir",
        type=str,
        default="mdra_suggest",
        help="mdra_suggest ディレクトリ（config.yml や cache/ を含む）",
    )
    ap.add_argument(
        "--config_path",
        type=str,
        default="mdra_suggest/config.yml",
        help="mdra_suggest/config.yml へのパス",
    )
    ap.add_argument(
        "--use_shionogi_db",
        type=str,
        default="false",
        help="Shionogi 追加辞書を使うかどうか（true/false）",
    )
    ap.add_argument(
        "--code_level",
        type=str,
        default="llt",
        choices=["llt", "pt"],
        help="評価粒度（既存結果と整合させるには llt 推奨）",
    )
    ap.add_argument(
        "--top_k_retrieval",
        type=int,
        default=200,
        help="Faiss から取得する候補数（top_k_retrieval）",
    )
    ap.add_argument(
        "--k_eval",
        type=int,
        default=20,
        help="IR 指標の @k（nDCG@k, Recall@k など）",
    )
    ap.add_argument(
        "--csv_list_sep",
        type=str,
        default=" ||| ",
        help="extracted/gold_llt_code に使われている外側の区切り記号（デフォルト: ' ||| '）",
    )
    ap.add_argument(
        "--device_embed",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="埋め込み計算に用いるデバイス（auto/cpu/cuda）。デフォルトは auto。",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/summary_test_metrics_for_simple_manuscript",
        help="再計算結果 CSV を保存するディレクトリ",
    )
    return ap.parse_args()


def _to_bool_flag(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y")


def _inspect_separators(series: pd.Series, name: str) -> None:
    """extracted / gold_llt_code に含まれる区切りパターンを簡易サマリ表示。"""
    s = series.fillna("").astype(str)
    n_total = len(s)
    n_has_pipe = (s.str.contains(r"\|")).sum()
    n_has_triple = (s.str.contains(r"\|\|\|")).sum()
    n_has_sep_space = (s.str.contains(r"\s\|\s")).sum()
    print(f"[separator-inspect] column={name}")
    print(f"  total={n_total}")
    print(f"  rows with any '|'          : {n_has_pipe}")
    print(f"  rows with '|||'            : {n_has_triple}")
    print(f"  rows with ' | ' (space bar): {n_has_sep_space}")


def split_list_field(raw: Any, csv_list_sep: str) -> List[str]:
    """
    extracted / gold_llt_code 用の分割関数。

    - まず '|||' / 指定 csv_list_sep で大きく分割
    - 各チャンクを '\\s*\\|\\s*' で再分割
    - 空文字は除去
    """
    if not isinstance(raw, str):
        return []
    text = raw.strip()
    if not text:
        return []

    # 改行や CR は空白扱いに寄せておく
    text = text.replace("\r", " ").replace("\n", " ")

    # '|||' 系を一旦統一
    text = re.sub(r"\s*\|\|\|\s*", csv_list_sep, text)

    items: List[str] = []
    # 外側区切りでチャンクに分解
    for chunk in re.split(re.escape(csv_list_sep), text):
        chunk = chunk.strip()
        if not chunk:
            continue
        # 内側の ' | ' / '|' をまとめて扱う
        subparts = re.split(r"\s*\|\s*", chunk)
        for sp in subparts:
            sp = sp.strip()
            if sp:
                items.append(sp)
    return items


def split_code_field(raw: Any) -> List[int]:
    """
    AE / SS 列（'10029410 | 10042670' 等）を LLT コードの整数リストに変換。
    ' | ' や '|||' が紛れても、'|' 周りの空白を正規表現で吸収して分割する。
    """
    if not isinstance(raw, str):
        return []
    text = raw.strip()
    if not text:
        return []

    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s*\|\|\|\s*", " | ", text)
    text = re.sub(r"\s*\|\s*", " | ", text)

    codes: List[int] = []
    for part in text.split(" | "):
        part = part.strip()
        if not part:
            continue
        # 念のためカンマや空白区切りにも対応
        for token in re.split(r"[,\s]+", part):
            token = token.strip()
            if not token:
                continue
            try:
                codes.append(int(token))
            except ValueError:
                continue
    return codes


def load_mdra_resources(
    *,
    mdra_suggest_dir: Path,
    config: Dict[str, Any],
    use_shionogi_db: bool,
) -> MdraResources:
    """mdra_suggest/cache から埋め込み・Faiss index・cd_llt2pt などを読み込む。"""
    mdra_path = Path(config["mdra"]["path"])
    mdra_ver = mdra_path.name.lower()
    cache_root = mdra_suggest_dir / "cache" / mdra_ver
    mdra_cache_path = cache_root / "mdra_dict.joblib"

    md_hier, code2name, name2code, cd_llt2pt, term2info, code2info = joblib.load(
        mdra_cache_path
    )

    embed_model_path = config["model"]["embeddings"]["embedding_model_path"]
    stem = embed_model_path.replace("/", "_")

    mdra_embed_dict = joblib.load(cache_root / f"{stem}.embed.dict.joblib")
    faiss_index = faiss.read_index(str(cache_root / f"{stem}.embed.faiss_index"))
    faiss_tbl = pd.DataFrame(
        {
            "index_string": mdra_embed_dict.keys(),
            "llt_codes": [[code] for code in mdra_embed_dict.values()],
        }
    )

    if use_shionogi_db:
        sheet_path = Path(config["mdra"]["shionogi_path"])
        extra_cache_dir = cache_root.parent / f"{mdra_ver}_{sheet_path.stem}"
        exception_list_path = config["mdra"]["exceptions_for_preventing_data_leak"]
        extra_stem = f"{stem}.exceptions_for_preventing_data_leak={int(bool(exception_list_path))}"
        additional_embeds = np.load(extra_cache_dir / f"{extra_stem}.embed.npy")
        additional_embeds_dict = joblib.load(
            extra_cache_dir / f"{extra_stem}.embed.dict.joblib"
        )
        faiss_index.add(additional_embeds)
        faiss_tbl = (
            pd.concat(
                [
                    faiss_tbl,
                    pd.DataFrame(
                        {
                            "index_string": additional_embeds_dict.keys(),
                            "llt_codes": additional_embeds_dict.values(),
                        }
                    ),
                ]
            )
            .reset_index(drop=True)
        )

    return MdraResources(
        mdra_embed_dict=dict(mdra_embed_dict),
        faiss_index=faiss_index,
        faiss_tbl=faiss_tbl,
        cd_llt2pt=cd_llt2pt,
    )


def _point_estimate_from_matrix(mat: np.ndarray) -> float:
    """summarize_test_metrics.py と同様: 文書ごとに seed 平均 → 文書平均。"""
    return float(mat.mean(axis=1).mean(axis=0))


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_scores_path = Path(args.test_scores_csv)
    label_summary_path = Path(args.label_summary_csv)
    mdra_suggest_dir = (PROJECT_ROOT / args.mdra_suggest_dir).resolve()
    config_path = (PROJECT_ROOT / args.config_path).resolve()

    if not test_scores_path.exists():
        raise FileNotFoundError(f"test_scores_csv not found: {test_scores_path}")
    if not label_summary_path.exists():
        raise FileNotFoundError(f"label_summary_csv not found: {label_summary_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config_path not found: {config_path}")

    use_shionogi_db = _to_bool_flag(args.use_shionogi_db)
    code_level = str(args.code_level)
    top_k_retrieval = int(args.top_k_retrieval)
    k_eval = int(args.k_eval)
    csv_list_sep = str(args.csv_list_sep)
    device_embed_arg = str(args.device_embed)

    # 1) 入力 CSV 読み込み
    df_scores = pd.read_csv(test_scores_path)
    df_labels = pd.read_csv(label_summary_path)

    # test split のみを利用
    df_labels = df_labels[df_labels["split"] == "test"].copy()

    # 区切りパターンの簡易チェック
    _inspect_separators(df_scores.get("extracted", pd.Series([], dtype=str)), "extracted")
    _inspect_separators(
        df_scores.get("gold_llt_code", pd.Series([], dtype=str)), "gold_llt_code"
    )

    # 2) AE/SS ラベル辞書の構築
    df_labels["AE_codes"] = df_labels["AE"].apply(split_code_field)
    df_labels["SS_codes"] = df_labels["SS"].apply(split_code_field)
    df_labels["has_AE"] = df_labels["num_AE"].astype(int) > 0
    df_labels["has_SS"] = df_labels["num_SS"].astype(int) > 0

    # num_AE / num_SS とパース結果の長さが極端に乖離していないか簡易チェック
    mismatch_ae = (
        (df_labels["has_AE"])
        & (df_labels["AE_codes"].apply(len) != df_labels["num_AE"].astype(int))
    ).sum()
    mismatch_ss = (
        (df_labels["has_SS"])
        & (df_labels["SS_codes"].apply(len) != df_labels["num_SS"].astype(int))
    ).sum()
    if mismatch_ae or mismatch_ss:
        print(
            f"[warning] AE/SS code count mismatch: mismatch_AE={mismatch_ae}, mismatch_SS={mismatch_ss}"
        )

    label_map = df_labels.set_index("guid")[
        ["AE_codes", "SS_codes", "has_AE", "has_SS"]
    ].to_dict(orient="index")

    # 3) mdra_suggest 関連リソースのロード（検索ロジックの再利用）
    config = load_mdra_config(str(config_path))
    embed_model_path = config["model"]["embeddings"]["embedding_model_path"]
    print(f"[info] embed_model_path={embed_model_path}")

    # 埋め込み計算に使うデバイスを引数から決定
    if device_embed_arg == "cpu":
        device = torch.device("cpu")
    elif device_embed_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device_embed=cuda が指定されましたが CUDA が利用できません。")
        device = torch.device("cuda")
    else:  # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] embedding device: {device}")

    se_model, se_tokenizer = load_se_model(embed_model_path)
    se_model.to(device)

    mdra_resources = load_mdra_resources(
        mdra_suggest_dir=mdra_suggest_dir,
        config=config,
        use_shionogi_db=use_shionogi_db,
    )

    # 4) 文書×モデル×seed ごとに AE/SS 別 IR 指標を計算
    rows: List[Dict[str, Any]] = []
    total_rows = len(df_scores)
    for idx, row in tqdm(
        df_scores.iterrows(),
        total=total_rows,
        desc="Recomputing AE/SS IR metrics",
    ):
        model = str(row["model"])
        seed = int(row["seed"])
        guid = str(row["guid"])
        extracted_raw = row.get("extracted", "")

        label_info = label_map.get(guid)
        if label_info is None:
            # ラベル情報がない guid はスキップ（通常は起きない想定）
            continue

        ae_codes: List[int] = label_info["AE_codes"]
        ss_codes: List[int] = label_info["SS_codes"]
        has_ae: bool = bool(label_info["has_AE"])
        has_ss: bool = bool(label_info["has_SS"])

        queries = split_list_field(extracted_raw, csv_list_sep=csv_list_sep)

        # 近傍探索（ranking）は文書ごとに1回だけ実行し、AE/SS 両方で共有する
        if queries:
            ranking_results = create_ranking_for_predictions(
                queries=queries,
                mdra_embed_dict=mdra_resources.mdra_embed_dict,
                faiss_index=mdra_resources.faiss_index,
                faiss_tbl=mdra_resources.faiss_tbl,
                n_tops=top_k_retrieval,
                model=se_model,
                tokenizer=se_tokenizer,
                config=config,
                cd_llt2pt=mdra_resources.cd_llt2pt,
                code_level=code_level,
            )
            df_list = [
                modify_ranking(
                    ranking_results.get(q, None),
                    top_k=top_k_retrieval,
                    code_level=code_level,
                )
                for q in queries
            ]
        else:
            df_list = [
                pd.DataFrame(columns=["index_string", f"{code_level}_code", "score"])
            ]

        mod_ranking = (
            pd.concat(df_list)
            .sort_values(by="score", ascending=False)
            .drop_duplicates(subset=f"{code_level}_code")
        )
        pred_codes = mod_ranking[f"{code_level}_code"].astype(int).tolist()

        # AE / SS ごとに IR 指標計算
        # gold が存在しない文書は、その group の平均から除外したいので NaN を入れる
        if has_ae and ae_codes:
            metrics_ae = calculate_ir_metrics(
                relevant_codes=ae_codes,
                predicted_codes=pred_codes,
                k_eval=k_eval,
            )
        else:
            metrics_ae = {"AP": math.nan, "RR": math.nan, "Recall": math.nan, "nDCG": math.nan}

        if has_ss and ss_codes:
            metrics_ss = calculate_ir_metrics(
                relevant_codes=ss_codes,
                predicted_codes=pred_codes,
                k_eval=k_eval,
            )
        else:
            metrics_ss = {"AP": math.nan, "RR": math.nan, "Recall": math.nan, "nDCG": math.nan}

        rows.append(
            {
                "model": model,
                "seed": seed,
                "guid": guid,
                "has_AE": has_ae,
                "has_SS": has_ss,
                "AP_AE": metrics_ae["AP"],
                "RR_AE": metrics_ae["RR"],
                "Recall_AE": metrics_ae["Recall"],
                "nDCG_AE": metrics_ae["nDCG"],
                "AP_SS": metrics_ss["AP"],
                "RR_SS": metrics_ss["RR"],
                "Recall_SS": metrics_ss["Recall"],
                "nDCG_SS": metrics_ss["nDCG"],
            }
        )

    df_ae_ss = pd.DataFrame(rows)

    # 5) 文書単位の再計算結果を保存（元の test_scores_all.csv と guid 粒度を揃える）
    merged = df_scores.merge(
        df_ae_ss,
        on=["model", "seed", "guid"],
        how="left",
        validate="one_to_one",
    )
    per_doc_out = out_dir / "test_scores_ae_ss_recomputed.csv"
    merged.to_csv(per_doc_out, index=False)
    print(f"[info] wrote per-document AE/SS metrics to: {per_doc_out}")

    # 6) モデル別 AE/SS サブグループ要約（文書→seed→文書平均）
    summary_rows: List[Dict[str, Any]] = []
    metrics = ["AP", "RR", "Recall", "nDCG"]
    groups = [("AE", "has_AE"), ("SS", "has_SS")]

    for model in sorted(df_ae_ss["model"].unique()):
        df_model = df_ae_ss[df_ae_ss["model"] == model]
        seeds = sorted(df_model["seed"].unique())

        for group_name, flag_col in groups:
            df_group = df_model[df_model[flag_col]]
            if df_group.empty:
                continue

            for metric_name in metrics:
                col = f"{metric_name}_{group_name}"
                if col not in df_group.columns:
                    continue

                sub = df_group[["guid", "seed", col]].dropna()
                if sub.empty:
                    continue

                guids = sorted(sub["guid"].unique())
                mat = (
                    sub.pivot(index="guid", columns="seed", values=col)
                    .reindex(index=guids, columns=seeds)
                )
                if mat.isna().any().any():
                    # 欠損がある場合はその行/列を落とす（通常は発生しない想定）
                    mat = mat.dropna(axis=0, how="any").dropna(axis=1, how="any")
                if mat.size == 0:
                    continue

                mean_val = _point_estimate_from_matrix(mat.to_numpy(dtype=float))
                summary_rows.append(
                    {
                        "model": model,
                        "metric": metric_name,
                        "group": group_name,
                        "n_docs": mat.shape[0],
                        "n_seeds": mat.shape[1],
                        "mean": mean_val,
                    }
                )

    df_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["metric", "group", "model"])
        .reset_index(drop=True)
    )
    summary_out = out_dir / "model_summary_ae_ss.csv"
    df_summary.to_csv(summary_out, index=False)
    print(f"[info] wrote AE/SS subgroup summary to: {summary_out}")


if __name__ == "__main__":
    # スレッド数を抑制（既存スクリプトに合わせた安全側設定）
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
