from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset


# -------------------------
# IR metrics (AP/RR/Recall/nDCG)
# -------------------------

def average_precision_at_k(relevant_labels: Iterable[int], predicted_labels: Iterable[int], k_eval: int) -> float:
    relevant_set = set(int(x) for x in relevant_labels)
    if len(relevant_set) == 0:
        return 0.0
    truncated = list(predicted_labels)[: int(k_eval)]
    hits = 0
    sum_precisions = 0.0
    for i, label in enumerate(truncated):
        if int(label) in relevant_set:
            hits += 1
            sum_precisions += hits / (i + 1)
    return 0.0 if hits == 0 else float(sum_precisions / len(relevant_set))


def reciprocal_rank_at_k(relevant_labels: Iterable[int], predicted_labels: Iterable[int], k_eval: int) -> float:
    relevant_set = set(int(x) for x in relevant_labels)
    truncated = list(predicted_labels)[: int(k_eval)]
    for i, label in enumerate(truncated):
        if int(label) in relevant_set:
            return float(1.0 / (i + 1))
    return 0.0


def recall_at_k(relevant_labels: Iterable[int], predicted_labels: Iterable[int], k_eval: int) -> float:
    relevant_set = set(int(x) for x in relevant_labels)
    if len(relevant_set) == 0:
        return 0.0
    truncated = set(int(x) for x in list(predicted_labels)[: int(k_eval)])
    return float(len(relevant_set.intersection(truncated)) / len(relevant_set))


def ndcg_at_k(relevant_labels: Iterable[int], predicted_labels: Iterable[int], k_eval: int) -> float:
    relevant_set = set(int(x) for x in relevant_labels)
    if len(relevant_set) == 0:
        return 0.0
    truncated = list(predicted_labels)[: int(k_eval)]
    dcg = 0.0
    for i, label in enumerate(truncated):
        if int(label) in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
    up_to = min(int(k_eval), len(relevant_set))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(up_to))
    return 0.0 if idcg == 0.0 else float(dcg / idcg)


def calculate_ir_metrics(relevant_codes: Iterable[int], predicted_codes: Iterable[int], k_eval: int) -> dict[str, float]:
    """
    mdra_suggest/evaluate_ranking.py の calculate_ir_metrics 相当（簡略版）
    - AP, RR, Recall, nDCG を返す
    - いずれも @k_eval で評価
    """
    relevant_codes = list(relevant_codes)
    predicted_codes = list(predicted_codes)
    return {
        "AP": average_precision_at_k(relevant_codes, predicted_codes, k_eval=k_eval),
        "RR": reciprocal_rank_at_k(relevant_codes, predicted_codes, k_eval=k_eval),
        "Recall": recall_at_k(relevant_codes, predicted_codes, k_eval=k_eval),
        "nDCG": ndcg_at_k(relevant_codes, predicted_codes, k_eval=k_eval),
    }


# -------------------------
# Embedding helpers (SE)
# -------------------------

def mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def get_embeddings_from_queries(
    *,
    queries: list[str],
    model_input_keys: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    device: torch.device,
    max_seq_length: int = 512,
    prefix: str = "",
) -> np.ndarray:
    def tokenize_function(example: dict[str, list[Any]], column_name: str):
        return tokenizer(
            [f"{prefix}{query}" for query in example[column_name]],
            max_length=max_seq_length,
            truncation=True,
        )

    dataset = Dataset.from_dict({"query": queries})
    tokenized = (
        dataset.map(lambda example: tokenize_function(example, column_name="query"), batched=True)
        .remove_columns([col for col in dataset.column_names if col not in model_input_keys])
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    model.to(device)
    model.eval()
    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            emb = mean_pooling(outputs.last_hidden_state, batch["attention_mask"])
            all_embeddings.append(emb.detach().cpu().numpy())

    embs = np.vstack(all_embeddings).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


# -------------------------
# MedDRA FAISS ranking
# -------------------------

@dataclass(frozen=True)
class MdraAssets:
    config: dict[str, Any]
    mdra_ver: str
    cd_llt2pt: dict[int, int]
    faiss_tbl: pd.DataFrame
    faiss_index: faiss.Index
    mdra_embed_dict: dict[str, list[int]]
    se_model: PreTrainedModel
    se_tokenizer: PreTrainedTokenizerBase


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mdra_assets(
    *,
    mdra_suggest_dir: str,
    config_path: str,
    use_shionogi_db: bool,
    device: torch.device,
) -> MdraAssets:
    mdra_suggest_dir_p = Path(mdra_suggest_dir)
    config = load_config(str(Path(config_path)))
    config = dict(config)  # shallow copy
    config.setdefault("evaluate", {})
    config["evaluate"]["use_shionogi_db"] = bool(use_shionogi_db)

    embed_model_path = config["model"]["embeddings"]["embedding_model_path"]
    q_prefix = config["model"]["embeddings"].get("embedding_model_q_prefix", "")
    _ = q_prefix  # kept for callers

    mdra_ver = Path(config["mdra"]["path"]).name.lower()
    mdra_cache_dir = mdra_suggest_dir_p / "cache" / mdra_ver
    mdra_cache_path = mdra_cache_dir / "mdra_dict.joblib"
    md_hier, code2name, name2code, cd_llt2pt, term2info, code2info = joblib.load(mdra_cache_path)
    _ = (md_hier, code2name, name2code, term2info, code2info)

    mdra_embed_dict_raw = joblib.load(mdra_cache_dir / f"{embed_model_path.replace('/', '_')}.embed.dict.joblib")
    mdra_faiss_index = faiss.read_index(str(mdra_cache_dir / f"{embed_model_path.replace('/', '_')}.embed.faiss_index"))
    faiss_index = mdra_faiss_index

    mdra_embed_dict: dict[str, list[int]] = {
        k: ([v] if isinstance(v, int) else list(v)) for k, v in mdra_embed_dict_raw.items()
    }
    faiss_tbl = pd.DataFrame({"index_string": list(mdra_embed_dict.keys()), "llt_codes": list(mdra_embed_dict.values())})

    if config["evaluate"]["use_shionogi_db"]:
        sheet_path = Path(config["mdra"]["shionogi_path"])
        cache_dir = mdra_cache_dir.parent / f"{mdra_ver}_{sheet_path.stem}"
        exception_list_path = config["mdra"].get("exceptions_for_preventing_data_leak")
        stem_name = f"{embed_model_path.replace('/', '_')}.exceptions_for_preventing_data_leak={int(bool(exception_list_path))}"

        additional_embeds = np.load(cache_dir / f"{stem_name}.embed.npy")
        additional_embeds_dict = joblib.load(cache_dir / f"{stem_name}.embed.dict.joblib")

        faiss_index.add(additional_embeds)
        faiss_tbl = pd.concat(
            [
                faiss_tbl,
                pd.DataFrame({"index_string": list(additional_embeds_dict.keys()), "llt_codes": list(additional_embeds_dict.values())}),
            ],
            ignore_index=True,
        )

    se_model = AutoModel.from_pretrained(embed_model_path)
    se_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
    se_model.to(device)

    return MdraAssets(
        config=config,
        mdra_ver=mdra_ver,
        cd_llt2pt={int(k): int(v) for k, v in cd_llt2pt.items()},
        faiss_tbl=faiss_tbl.reset_index(drop=True),
        faiss_index=faiss_index,
        mdra_embed_dict=mdra_embed_dict,
        se_model=se_model,
        se_tokenizer=se_tokenizer,
    )


def create_ranking_for_predictions(
    *,
    queries: list[str],
    assets: MdraAssets,
    n_tops: int,
    device: torch.device,
    code_level: str = "llt",
    apply_similarity_search_to_all_queries: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    notebook/solo.py 相当:
    - クエリを埋め込み→faiss search
    - `llt_codes` を explode して `llt_code` 列に
    - code_level='pt' の場合は cd_llt2pt で pt_code を付与し、ptのみ残す（llt==ptのもの）
    """
    mdra_embed_dict = assets.mdra_embed_dict
    faiss_index = assets.faiss_index
    faiss_tbl = assets.faiss_tbl
    config = assets.config

    unique_queries = list(dict.fromkeys(queries))
    result: dict[str, pd.DataFrame] = {}

    queries_for_faiss = unique_queries
    if not apply_similarity_search_to_all_queries:
        # 用語集に完全一致するものは score=1.0 を返す
        for q in unique_queries:
            if q in mdra_embed_dict:
                result[q] = pd.Series({"index_string": q, "llt_codes": mdra_embed_dict[q], "score": 1.0}).to_frame().T
        queries_for_faiss = [q for q in unique_queries if q not in result]

    if queries_for_faiss:
        q_prefix = config["model"]["embeddings"].get("embedding_model_q_prefix", "")
        q_embeddings = get_embeddings_from_queries(
            queries=queries_for_faiss,
            model_input_keys=config["model"]["embeddings"]["model_input_keys"],
            model=assets.se_model,
            tokenizer=assets.se_tokenizer,
            prefix=q_prefix,
            batch_size=int(config["model"]["embeddings"]["batch_size"]),
            device=device,
        )
        distances, indices = faiss_index.search(q_embeddings, int(n_tops))
        for i, q in enumerate(queries_for_faiss):
            result[q] = faiss_tbl.loc[indices[i]].assign(score=distances[i])

    out: dict[str, pd.DataFrame] = {}
    for q in queries:
        df = result[q].explode(column="llt_codes").drop_duplicates(subset="llt_codes", keep="first").copy()
        df.rename(columns={"llt_codes": "llt_code"}, inplace=True)
        df["llt_code"] = df["llt_code"].astype(int)
        if code_level == "pt":
            df["pt_code"] = df["llt_code"].map(assets.cd_llt2pt)
            df = df[df["pt_code"] == df["llt_code"]]  # PTのみ残す（ノートブック準拠）
            df["pt_code"] = df["pt_code"].astype(int)
        out[q] = df.reset_index(drop=True)
    return out


def merge_rankings(
    ranking_results: dict[str, pd.DataFrame],
    *,
    queries: list[str],
    top_k: int,
    code_level: str = "llt",
) -> pd.DataFrame:
    """
    複数クエリのランキングを結合し、score降順・重複コード除去で最終ランキングを作る。
    """
    if not queries:
        return pd.DataFrame(columns=["index_string", f"{code_level}_code", "score"])

    frames: list[pd.DataFrame] = []
    for q in queries:
        df = ranking_results.get(q)
        if df is None or len(df) == 0:
            continue
        key = "pt_code" if code_level == "pt" else "llt_code"
        mod = (
            df.sort_values(by="score", ascending=False)
            .drop_duplicates(subset=key)
            .head(int(top_k))
            .rename(columns={key: f"{code_level}_code"})
        )
        frames.append(mod[["index_string", f"{code_level}_code", "score"]])
    if not frames:
        return pd.DataFrame(columns=["index_string", f"{code_level}_code", "score"])
    merged = (
        pd.concat(frames, ignore_index=True)
        .sort_values(by="score", ascending=False)
        .drop_duplicates(subset=f"{code_level}_code")
        .reset_index(drop=True)
    )
    return merged


