


"""Common text processing helpers for Streamlit pages.
20251210
    RL用に移植
20250807
    `clean_text()`にExcelに書き込めない制御文字を半角スペースに置換する処理を追加。
"""
import sys
import re
from typing import Any, Dict, List, Tuple, Optional, Union

import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import Dataset
from tqdm.auto import tqdm

from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


def clean_text(text: str) -> str:
    """
    入力テキストの前後の空白と二重引用符を削除する。
    Excelに書き込めない文字（制御文字）を半角スペースに置換する。
    Args:
        text (str): クリーニング対象のテキスト。

    Returns:
        str: クリーニング後のテキスト。
    """
    if not isinstance(text, str):
        return ""
    
    cleaned_text = text.strip()
    if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
        cleaned_text = cleaned_text[1:-1].strip()

    cleaned_text = ILLEGAL_CHARACTERS_RE.sub(' ', cleaned_text)
    
    return cleaned_text


# mean_pooling関数の定義
def mean_pooling(
    last_hidden_state: Tensor,
    attention_mask: Tensor
) -> Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)

    return sum_embeddings / sum_mask

def get_embeddings_from_queries(
    queries: list[str],
    model_input_keys: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    device:torch.device,
    max_seq_length: int = 512,
    prefix: str = '',
    ) -> np.ndarray:

    def tokenize_function(
        example: dict[str, list[Any]],
        column_name: str
        ):
        return tokenizer([f'{prefix}{query}' for query in example[column_name]],
                         max_length=max_seq_length, truncation=True
                         )
    dataset = Dataset.from_dict({'query': queries})
    tokenized_dataset = dataset.map(lambda example: tokenize_function(example, column_name='query'),
                                    batched=True, desc='Tokenize all unique queries'
                                    ).remove_columns([col for col in dataset.column_names if col not in model_input_keys])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors='pt')
    dataloader = DataLoader(
                    dataset=tokenized_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=data_collator)
    model.to(device)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            embeddings = mean_pooling(
                            last_hidden_state=outputs.last_hidden_state,
                            attention_mask=batch['attention_mask']
            )
            embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)
    embeddings_normalized = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

    # メモリ解放
    del all_embeddings
    del embeddings
    del outputs
    del batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings_normalized

def get_similarities(
    string: str,
    _model: PreTrainedModel,
    _tokenizer: PreTrainedTokenizerBase,
    _index,  # faiss.Index
    llt_tbl,  # pd.DataFrame
    top_k: int,
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    与えられた文字列に対して、Faissインデックスを使用して類似度が高いLLTを検索し、
    その結果をDataFrameで返す。
    """
    # configからq_prefixを読み込む。存在しない場合は空文字をデフォルト値とする。
    q_prefix = config.get("model", {}).get("embeddings", {}).get("embedding_model_q_prefix", "")
    
    unique_queries = [string] # 単一の文字列をリストに格納
    result: Dict[str, pd.DataFrame] = {}

    q_embeddings = get_embeddings_from_queries(
        queries=unique_queries,
        model_input_keys=config["model"]["embeddings"]["model_input_keys"],
        model=_model,
        tokenizer=_tokenizer,
        prefix=q_prefix, # configから読み込んだprefixを渡す
        batch_size=config["model"]["embeddings"]["batch_size"],
        device=_model.device,
    )
    distances, indices = _index.search(q_embeddings, top_k * 5)
    result.update({q: llt_tbl.loc[indices[i]].assign(score=distances[i]) for i, q in enumerate(unique_queries)})

    # The return type is dict[str, pd.DataFrame], not pd.DataFrame
    return {
        s: result[s].explode(column="llt_codes").rename({"llt_codes": "llt_code"}, axis=1)
        for s in result
    }

def get_pt_ranking(
    df: pd.DataFrame,
    _pt_code2name: Dict[int, Dict[str, List[str]]],
    _cd_llt2pt: Dict[int, int],
    top_k: int,
    threshold: float,
) -> pd.DataFrame:
    """Aggregate LLT search results at the PT level."""
    pt_code2name = _pt_code2name
    cd_llt2pt = _cd_llt2pt
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    df = df.dropna()
    df["pt_code"] = df["llt_code"].apply(cd_llt2pt.get)
    df = df.dropna(subset=["pt_code"])
    df["pt_code"] = df["pt_code"].astype(int)
    df["pt_kanji"] = df["pt_code"].apply(lambda x: pt_code2name.get(x, {}).get("kanji", ["Unknown"])[0])
    df = df[df["score"] >= threshold]
    df["score"] = df["score"].apply(lambda x: f"{x:.2f}")
    return df


def get_pt_related_info(
    selected_code: int,
    targets: List[str],
    mdra_ver: str,
    md_hier: Optional[Dict[int, Dict[str, Any]]] = None,
    code2name: Optional[Dict[str, Dict[int, Dict[str, List[str]]]]] = None,
) -> Dict[str, Any]:
    """Return SOC/HLGT/HLT/PT information for a selected PT code."""
    if md_hier is None or code2name is None:
        return {}

    hier_info = md_hier.get(selected_code)
    if hier_info is None:
        return {}

    items: Dict[str, Any] = {"version": mdra_ver}
    for target in targets:
        key_prefix = target.lower()
        target_code_val: Any = None
        if key_prefix == "pt":
            target_code_val = selected_code
        else:
            target_code_val = hier_info.get(f"{key_prefix}_code")

        if target_code_val is not None:
            # コードを整数に変換
            try:
                numeric_code = int(target_code_val)
            except (ValueError, TypeError):
                continue # 変換できない場合はスキップ

            if key_prefix != "llt":
                code2name_level1 = code2name.get(f"{key_prefix}_code")
                if code2name_level1:
                    code2name_level2 = code2name_level1.get(numeric_code) # 整数キーで検索
                    if code2name_level2:
                        items[f"{key_prefix}_kanji"] = ", ".join(code2name_level2.get("kanji", []))
                        if key_prefix == "pt":
                            pt_name_list = code2name_level2.get("name", [])
                            if pt_name_list:
                                items[f"{key_prefix}_name"] = ", ".join(pt_name_list)
                items[f"{key_prefix}_code"] = numeric_code # 整数として格納
    return items


def get_llt_table(
    selected_row: Dict[str, int | str],
    _df_simPT: pd.DataFrame,
    _md_hier: Dict[int, Dict[str, Any]],
    _code2info: Dict[int, Dict[str, str | List[str]]],
) -> pd.DataFrame:
    """Return LLT table rows for a selected PT."""
    md_hier = _md_hier
    code2info = _code2info
    df_simPT = _df_simPT
    selected_pt_code = selected_row["pt_code"]
    if not isinstance(selected_pt_code, int):
        try:
            selected_pt_code = int(selected_pt_code)
        except ValueError:
            return pd.DataFrame()

    def get_primary_ja_term(code_info: Dict) -> str:
        """LLTコード情報から代表の日本語名を取得する（Y > N > S > $の優先度）"""
        jcurr_str = code_info.get("llt_jcurr", "")
        kanji_list = code_info.get("llt_kanji", [])
        if not kanji_list:
            return "N/A"
        
        for currency_pref in ['Y', 'N', 'S', '$']:
            if currency_pref in jcurr_str:
                try:
                    idx = jcurr_str.index(currency_pref)
                    if idx < len(kanji_list):
                        return kanji_list[idx]
                except (ValueError, IndexError):
                    continue
        return kanji_list[0] # フォールバック

    all_rows = []

    # --- 1. Process similarity search results ---
    df_simPT_filtered = df_simPT[df_simPT["pt_code"].notna()].copy()
    if not df_simPT_filtered.empty:
        df_simPT_filtered["pt_code"] = pd.to_numeric(df_simPT_filtered["pt_code"], errors='coerce').astype('Int64')
        df_simPT_filtered = df_simPT_filtered[df_simPT_filtered["pt_code"] == selected_pt_code]

        for _, row in df_simPT_filtered.iterrows():
            llt_code = int(row["llt_code"])
            similar_term = row["index_string"]
            code_info = _code2info.get(llt_code)
            if not code_info:
                continue

            primary_ja_term = get_primary_ja_term(code_info)
            kanji_list = code_info.get("llt_kanji", [])
            jcurr_str = code_info.get("llt_jcurr", "")
            jcurr_for_term = ""

            if row["source"] == "in-house":
                jcurr_for_term = "$"
            elif similar_term in kanji_list:
                try:
                    idx = kanji_list.index(similar_term)
                    if idx < len(jcurr_str):
                        jcurr_for_term = jcurr_str[idx]
                except (ValueError, IndexError):
                    pass
            
            all_rows.append({
                "similar_term": similar_term,
                "official_llt_ja": primary_ja_term,
                "official_llt_en": code_info.get("llt_name", "N/A"),
                "code": llt_code,
                "currency": code_info.get("llt_currency", "N/A"),
                "jcurr": jcurr_for_term,
                "jcurr_kanji": code_info.get("jcurrency_kanji", "N/A"),
                "score": row["score"],
                "source": row["source"],
            })

    # --- 2. Process all terms from PT hierarchy ---
    pt_info = _md_hier.get(selected_pt_code)
    if pt_info and "llt_codes" in pt_info:
        for llt_code in pt_info["llt_codes"]:
            code_info = _code2info.get(llt_code)
            if not code_info:
                continue
            
            primary_ja_term = get_primary_ja_term(code_info)
            kanji_list = code_info.get("llt_kanji", [])
            jcurr_str = code_info.get("llt_jcurr", "")

            if len(kanji_list) == len(jcurr_str):
                for i, kanji in enumerate(kanji_list):
                    all_rows.append({
                        "similar_term": kanji,
                        "official_llt_ja": primary_ja_term,
                        "official_llt_en": code_info.get("llt_name", "N/A"),
                        "code": llt_code,
                        "currency": code_info.get("llt_currency", "N/A"),
                        "jcurr": jcurr_str[i],
                        "jcurr_kanji": code_info.get("jcurrency_kanji", "N/A"),
                        "score": "-",
                        "source": "mdra",
                    })

    # --- 3. Finalize dataframe ---
    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.sort_values(by='score', ascending=False, na_position='last', inplace=True)
    df.drop_duplicates(subset=["similar_term", "code"], keep="first", inplace=True)
    
    return df