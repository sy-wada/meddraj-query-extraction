import re
import pandas as pd
import numpy as np
from fuzzysearch import find_near_matches
import yaml
from typing import Any, Tuple, Optional
from string import Template
from pathlib import Path
import joblib
import faiss
import torch
import sys
import argparse
import json

from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest

REPO_ROOT = Path(__file__).resolve().parent

add_syspath_list = [
    str(REPO_ROOT),
    str(REPO_ROOT / "mdra_suggest"),
]
for syspath in add_syspath_list:
    if syspath not in sys.path:
        sys.path.append(syspath)

from text_processing_utils import (
    get_embeddings_from_queries,
)

from evaluate_ranking import calculate_ir_metrics

reasoning_start = "<reasoning>"
reasoning_end   = "</reasoning>"
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
  - 1文字も変更しない（言い換え、要約、整形、句読点の補正、表記ゆれ統一、不要語の削除、引用符の追加などは不可）。
- 抽出範囲は、判定根拠として意味が成立する **最小限の連続スパン** とする（段落丸ごと等の過度に長い抽出は避ける）。
- 1つの `${list_start}` 箇条書きには、原則として1つの連続スパンのみを入れる。
- 同じ表現が本文中に複数回出る場合は、**出現箇所ごとに** 列挙してよい（重複出力は許容）。

## Key Phrase 判定ルール（共通）
- 次の3カテゴリのいずれかに該当する「事象が、対象者に実際に発生した（または曝露が実際に起きた）」と本文に記載されている場合のみ抽出する。
- 次は抽出しない:
  - 一般的な注意喚起・添付文書の一般論・文献/報告の一般的言及
  - 単なる質問（発生の記載がない）
  - 否定（例: 「症状はない」「発現していない」）
  - 仮定（例: 「もし〜なら」「〜した場合」）のみで終わっている記載
- ただし、不確実表現（例: 「〜かもしれない」「疑い」）でも、対象者に症状や曝露が起きた旨が本文から読み取ることが出来れば抽出してよい。

## 本タスクにおける Key Phrase の定義（3カテゴリ）
1. **有害事象 (Adverse Event)**
   - 医薬品の使用後に、対象者に症状・体調変化・検査値異常などが発生した記載（因果関係の確定は不要）。
   - 「副作用について知りたいだけ」「文献上の副作用の話だけ」は除外し、本文の対象者に発生したことが書かれている一節のみ抽出する。

2. **妊婦・授乳婦服薬 (Medication and Drug Exposure in Pregnancy and Breastfeeding)**
   - 妊娠中、妊娠の可能性がある、または授乳中の対象者が医薬品に曝露した記載（誤用を含む）。
   - 一般論のみは除外する。

3. **医薬品投与に関係する特殊な状況 (Special Situations)**
   - 通常の使用範囲外、または特別なリスク状況の記載。
   - 例: 適応外使用、誤用、自己判断による用法用量の変更・中断、乱用、相互作用、不適切な保存、依存、品質問題 など。
   - 一般論のみは除外する。

## オーバーラップの扱い
- 1つの記載が複数カテゴリに該当しうる場合でも、**Key Phrase として抽出対象**（カテゴリ分けは不要）。
- 有害事象と特殊状況が同時に記載される場合は、それぞれの根拠となる一節を **漏れなく** 抽出する。

""".strip())

USER_PROMPT = Template(
"""
<DOCUMENT>
${document}
</DOCUMENT>
""".strip())

def create_input_ids_for_vllm_generate(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    ) -> TokensPrompt:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.substitute(
            reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        list_start=list_start,
        list_end=list_end
    )},
    {"role": "user",   "content": USER_PROMPT.substitute(
            document=text
        )},
    ]
    # `<bos><start_of_turn>user\n` という始まり方。
    # vllmのTokensPromptを介して入力すると、<bos>の重複なく、入力を整えてくれる。
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = True,
    )
    return TokensPrompt(prompt_token_ids=input_ids)

def match_phrases_in_text(
    text: str,
    phrase_list: list[str],
    max_l_dist: int = 2,
    max_deletions: int = 0,
    max_insertions: int = 0
    ) -> list[dict]:

    results = []
    phrase_list = [phrase.strip() for phrase in phrase_list if phrase.strip() != '']

    # 1. 完全一致する事例と、fuzzysearchで近似する事例をそれぞれ分離する
    exact_matches = [phrase for phrase in phrase_list if phrase in text]
    fuzzy_matches = [phrase for phrase in phrase_list if phrase not in text]

    # 2-1. 各フレーズをテキスト内で検索
    for phrase in exact_matches:
        start = 0
        while True:
            # find() で norm_phrase を text 内で検索
            found_idx = text.find(phrase, start)
            if found_idx == -1:
                # 見つからなかったら break
                break
            # 見つかったら end_idx を計算
            end_idx = found_idx + len(phrase)

            # 結果を追加
            results.append({
                "matched_phrase": phrase, 
                "query": phrase,
                "span": (found_idx, end_idx),
                "l_dist": 0
            })

            # 次の検索は end_idx 以降から
            start = end_idx

    # 2-2. fuzzysearchで近似する事例を処理
    for phrase in fuzzy_matches:
        matches = find_near_matches(subsequence=phrase,
                                    sequence=text,
                                    max_l_dist=max_l_dist if len(phrase) > max_l_dist else len(phrase) - 1,
                                    max_deletions=max_deletions,
                                    max_insertions=max_insertions
                                    )
        for match in matches:
            results.append({
                "matched_phrase": match.matched,
                "query": phrase,
                "span": (match.start, match.end),
                "l_dist": match.dist
            })

    results.sort(key=lambda x: x['span'][0])
    return results

def merge_overlapping_spans(
    match_list: list[dict],
    text: str
    ) -> list[dict]:
    """
    match_list は以下のような形式を想定:
      [
        {"matched_phrase": "多汗症", "span": (11, 14)},
        {"matched_phrase": "寝汗",   "span": (15, 17)},
        ...
      ]
    """

    if not match_list:
        return []

    # 1. マージ結果を格納する配列を用意
    results = []

    # 2.1. 開始インデックスでソートし直す
    match_list.sort(key=lambda x: x["span"][0])

    # 2.2. 区間マージ
    merged_results = []
    current = {
        "merged_phrases": [match_list[0]["matched_phrase"]],
        "query": [match_list[0]["query"]],
        "span": match_list[0]["span"],
        "l_dist": [match_list[0]["l_dist"]]
    }

    for i in range(1, len(match_list)):
        next_item = match_list[i]
        next_start, next_end = next_item["span"]

        curr_start, curr_end = current["span"]

        if next_start <= curr_end:
            # 重複 or 内包 -> マージ
            merged_start = min(curr_start, next_start)
            merged_end   = max(curr_end,   next_end)
            current["span"] = (merged_start, merged_end)
            current["merged_phrases"].append(next_item["matched_phrase"])
            current['query'].append(next_item['query'])
            current['l_dist'].append(next_item['l_dist'])
        else:
            # 重複しない -> current を確定追加
            merged_results.append(current)
            current = {
                "merged_phrases": [next_item["matched_phrase"]],
                "query": [next_item['query']],
                "span": next_item["span"],
                "l_dist": [next_item['l_dist']]
            }

    # ループ終了後に最後の current を追加
    merged_results.append(current)
    results.extend(merged_results)
    final_results = []
    for item in results:
        item['matched_string'] = text[item['span'][0]:item['span'][1]]
        final_results.append(item)
    return final_results

def load_config(
    path: str
    ) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_se_model(
    pretrained_model_name_or_path: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]: # tuple を Tuple に変更
    model = AutoModel.from_pretrained(pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    return model, tokenizer

def create_ranking_for_predictions(
    queries: list[str],
    mdra_embed_dict: dict[str, list[int]],
    faiss_index: faiss.IndexFlatIP,
    faiss_tbl: pd.DataFrame,
    n_tops: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: dict,
    cd_llt2pt: dict[int, int],
    code_level: str = 'pt',
    ) -> dict[str, pd.DataFrame]:

    mdra_embed_dict = {key: [value] if isinstance(value, int) else value
                        for key, value in mdra_embed_dict.items()}

    device = model.device
    q_prefix = config['model']['embeddings']['embedding_model_q_prefix']

    unique_queries = list(set(queries))
    result = {}

    q_embeddings = get_embeddings_from_queries(
                        queries=unique_queries,
                        model_input_keys=config['model']['embeddings']['model_input_keys'],
                        model=model, tokenizer=tokenizer,
                        prefix=q_prefix, batch_size=config['model']['embeddings']['batch_size'], device=device
                        )
    distances, indices = faiss_index.search(q_embeddings, n_tops)
    
    result.update(
        {q: faiss_tbl.loc[indices[i]].assign(score=distances[i])
         for i, q in enumerate(unique_queries)}
    )
    return_object = {}
    for query in queries:
        object = (result[query].explode(column='llt_codes')
                                .drop_duplicates(subset='llt_codes', keep='first')
                    )
        object.rename(columns={'llt_codes': 'llt_code'}, inplace=True)
        object['pt_code'] = object['llt_code'].map(cd_llt2pt)
        if code_level == 'pt':
            object = object[object['pt_code'] == object['llt_code']]
        return_object[query] = object.reset_index(drop=True)
    return return_object
    

def modify_ranking(
    df_ranking: pd.DataFrame | None,
    top_k: int,
    code_level: str = 'pt',
) -> pd.DataFrame:
    if df_ranking is None or len(df_ranking) == 0:
        return pd.DataFrame(columns=['index_string', f'{code_level}_code', 'score'])
    else:
        return (df_ranking
                    .sort_values(by='score', ascending=False)
                    .drop_duplicates(subset=f'{code_level}_code')
                    .head(top_k))

def extract_llt_codes(
    mdra_info: list[dict[str, Any]]
    ) -> set[int]:
    """
    mdra_info の中から 正解ラベルを取り出す関数。
    「有害事象なし」は除外しておく。
    """
    result = []
    for obj in mdra_info:
        if obj['hlt_kanji'] == '副作用なし' or obj['hlt_code'] == 10001424:
            continue
        else:
            result.append({
                'pt_code': obj['pt_code'],
                'pt_kanji': obj['pt_kanji'],
                'llt_code': obj['llt_code'],
                'llt_kanji': obj['llt_kanji']
            })
    return result

def extract_list_items(text: str):
    """
    与えられた複数行テキストから、行頭に'- 'があってもなくてもよい
    リスト項目を抽出してリスト化する。
    """
    pattern = re.compile(r"^\s*-?\s*(.+)$", re.MULTILINE)
    items = pattern.findall(text)

    # 空行や空文字を除外（必要に応じて）
    cleaned = [item.strip() for item in items if item.strip()]

    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all LoRA adapters (checkpoint-* dirs) under an adapter_root using IR metrics on a dataset split."
    )

    # Paths
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="HuggingFace datasets disk path (load_from_disk). e.g. /path/to/data/mdra_rl_dataset_vYYYYMMDD",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split to evaluate. (default: dev)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to mdra_suggest/config.yml",
    )
    parser.add_argument(
        "--adapter_root",
        type=str,
        required=True,
        help="Directory that contains checkpoint-* adapter folders. e.g. outputs/qlora_adapter_...__resume...__llt",
    )
    parser.add_argument(
        "--checkpoint_glob",
        type=str,
        default="checkpoint-*",
        help="Glob pattern for adapter checkpoints under adapter_root. (default: checkpoint-*)",
    )

    # vLLM base model / decoding
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="unsloth/medgemma-4b-it-unsloth-bnb-4bit",
        help="Base model to load in vLLM (must match the adapter's base model).",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to use stochastic decoding. "
            "If disabled (--no-do_sample), greedy decoding is used by forcing "
            "temperature=0.0, top_p=1.0, top_k=-1 (vLLM equivalent setting)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="SamplingParams seed (default: 0). Used for stochastic decoding; logged for reproducibility.",
    )

    # Retrieval / evaluation
    parser.add_argument("--code_level", type=str, default="llt", choices=["llt", "pt"])
    parser.add_argument("--top_k_retrieval", type=int, default=200)
    parser.add_argument("--k_eval", type=int, default=20)
    parser.add_argument("--max_attempts", type=int, default=5)

    # Behavior flags
    parser.add_argument(
        "--use_shionogi_db",
        action="store_true",
        help="If set, extends Faiss index with Shionogi additional embeddings as configured in config.yml",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skips checkpoints that already have metrics_<split>.json (or legacy metrics_dev.json for dev).",
    )
    parser.add_argument(
        "--metrics_name",
        type=str,
        default="metrics",
        help="Base filename for metrics outputs inside each checkpoint dir. (default: metrics)",
    )

    return parser.parse_args()

def _resolve_path(path_str: str) -> Path:
    """
    - Absolute paths are returned as-is.
    - Relative paths are resolved from repo root (directory containing this script).
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def build_adapters(adapter_root: Path, checkpoint_glob: str) -> dict[str, LoRARequest]:
    checkpoints = sorted([p for p in adapter_root.glob(checkpoint_glob) if p.is_dir()])
    adapters: dict[str, LoRARequest] = {}
    for i, path in enumerate(checkpoints):
        # vLLM LoRARequest: (lora_name, lora_int_id, lora_local_path)
        adapters[path.name] = LoRARequest(path.name, i + 1, str(path))
    return adapters


def _metrics_paths(checkpoint_dir: Path, split: str, metrics_name: str) -> tuple[Path, Path]:
    """
    Returns:
      - list_path: per-example metrics list
      - summary_path: describe() summary
    """
    list_path = checkpoint_dir / f"{metrics_name}_{split}_list.json"
    summary_path = checkpoint_dir / f"{metrics_name}_{split}.json"
    return list_path, summary_path


def _should_skip(checkpoint_dir: Path, split: str, metrics_name: str) -> bool:
    # New naming
    _, summary_path = _metrics_paths(checkpoint_dir, split, metrics_name)
    if summary_path.exists():
        return True
    # Backward compat: the old code used metrics_dev.json
    if split == "dev" and (checkpoint_dir / "metrics_dev.json").exists():
        return True
    return False


def main() -> None:
    args = parse_args()

    dataset_path = _resolve_path(args.dataset_path)
    config_path = _resolve_path(args.config_path)
    adapter_root = _resolve_path(args.adapter_root)

    dataset = load_from_disk(str(dataset_path))
    if args.split not in dataset:
        raise ValueError(f"split '{args.split}' not found in dataset: {list(dataset.keys())}")
    eval_dataset = dataset[args.split]

    if not adapter_root.exists():
        raise FileNotFoundError(f"adapter_root does not exist: {adapter_root}")
    adapters = build_adapters(adapter_root=adapter_root, checkpoint_glob=args.checkpoint_glob)
    if not adapters:
        raise ValueError(f"No adapters found under {adapter_root} with glob '{args.checkpoint_glob}'")

    llm = LLM(
        model=args.base_model_name_or_path,
        enable_lora=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    base_sampling_kwargs: dict[str, Any] = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    if not args.do_sample:
        # Deterministic / greedy decoding equivalent in vLLM.
        base_sampling_kwargs.update(
            dict(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
            )
        )
    sampling_params = SamplingParams(**base_sampling_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    config = load_config(str(config_path))
    embed_model_path = config["model"]["embeddings"]["embedding_model_path"]
    print(f"{embed_model_path=}")
    se_model, se_tokenizer = load_se_model(embed_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    se_model.to(device)

    # MedDRA related cache
    mdra_ver = Path(config["mdra"]["path"]).name.lower()
    print(f"{mdra_ver=}")
    mdra_cache_dir = REPO_ROOT / "mdra_suggest" / "cache" / mdra_ver
    mdra_cache_path = mdra_cache_dir / "mdra_dict.joblib"
    md_hier, code2name, name2code, cd_llt2pt, term2info, code2info = joblib.load(mdra_cache_path)

    # Faiss / Embeddings
    mdra_embed_dict = joblib.load(mdra_cache_dir / f'{embed_model_path.replace("/", "_")}.embed.dict.joblib')
    mdra_faiss_index = faiss.read_index(str(mdra_cache_dir / f'{embed_model_path.replace("/", "_")}.embed.faiss_index'))
    faiss_index = mdra_faiss_index
    faiss_tbl = pd.DataFrame({
        "index_string": mdra_embed_dict.keys(),
        "llt_codes": [[code] for code in mdra_embed_dict.values()],
    })

    if args.use_shionogi_db:
        sheet_path = Path(config["mdra"]["shionogi_path"])
        cache_dir = mdra_cache_dir.parent / f"{mdra_ver}_{sheet_path.stem}"
        exception_list_path = config["mdra"]["exceptions_for_preventing_data_leak"]
        stem_name = f'{embed_model_path.replace("/", "_")}.exceptions_for_preventing_data_leak={int(bool(exception_list_path))}'
        additional_embeds = np.load(cache_dir / f"{stem_name}.embed.npy")
        additional_embeds_dict = joblib.load(cache_dir / f"{stem_name}.embed.dict.joblib")
        faiss_index.add(additional_embeds)
        faiss_tbl = pd.concat([
            faiss_tbl,
            pd.DataFrame({
                "index_string": additional_embeds_dict.keys(),
                "llt_codes": additional_embeds_dict.values(),
            }),
        ]).reset_index(drop=True)

    match_format = re.compile(
        rf"^[\s]{{0,}}"
        rf"{reasoning_start}.+?{reasoning_end}.*?"
        rf"{list_start}(.+?){list_end}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )

    progress_bar = tqdm(adapters.items(), desc="Evaluating adapters", total=len(adapters))
    for adapter_name, lora_request in progress_bar:
        progress_bar.desc = adapter_name

        if args.skip_existing and _should_skip(Path(lora_request.path), args.split, args.metrics_name):
            continue

        metrics_list: list[dict[str, Any]] = []
        progress_data_bar = tqdm(eval_dataset, desc=f"Evaluating data ({args.split})", total=len(eval_dataset))

        for data in progress_data_bar:
            text = data["input_text"]
            true_labels = [obj[f"{args.code_level}_code"] for obj in data["mdra_labels"]]

            kp_list: list[str] = []
            for i in range(args.max_attempts):
                progress_data_bar.desc = f"{adapter_name}: Attempt {i+1}/{args.max_attempts}"
                outputs = llm.generate(
                    prompts=create_input_ids_for_vllm_generate(text=text, tokenizer=tokenizer),
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                    use_tqdm=False,
                )
                m = match_format.search(outputs[0].outputs[0].text)
                if m:
                    kp_list = extract_list_items(m.group(1))
                    break
                kp_list = []

            if kp_list:
                matched_objects = match_phrases_in_text(text=text, phrase_list=kp_list, max_l_dist=2)
                matched_objects = merge_overlapping_spans(match_list=matched_objects, text=text)
                queries = [item["matched_string"] for item in matched_objects]
            else:
                queries = []

            if queries:
                ranking_results = create_ranking_for_predictions(
                    queries=queries,
                    mdra_embed_dict=mdra_embed_dict,
                    faiss_index=faiss_index,
                    faiss_tbl=faiss_tbl,
                    n_tops=args.top_k_retrieval,
                    config=config,
                    model=se_model,
                    tokenizer=se_tokenizer,
                    cd_llt2pt=cd_llt2pt,
                    code_level=args.code_level,
                )
                df_list = [modify_ranking(ranking_results.get(string, None), args.top_k_retrieval, code_level=args.code_level) for string in queries]
            else:
                df_list = [pd.DataFrame(columns=["index_string", f"{args.code_level}_code", "score"])]

            mod_ranking = (
                pd.concat(df_list)
                .sort_values(by="score", ascending=False)
                .drop_duplicates(subset=f"{args.code_level}_code")
            )
            metrics = calculate_ir_metrics(
                relevant_codes=true_labels,
                predicted_codes=mod_ranking[f"{args.code_level}_code"],
                k_eval=args.k_eval,
            )
            metrics_list.append(metrics)

        df_metrics = pd.DataFrame(metrics_list)
        df_metrics['guid'] = eval_dataset['guid']
        list_path, summary_path = _metrics_paths(Path(lora_request.path), args.split, args.metrics_name)
        df_metrics.to_csv(list_path, index=False)
        summary_obj: dict[str, Any] = {
            "split": args.split,
            "n_examples": len(eval_dataset),
            "adapter_root": str(adapter_root),
            "checkpoint": adapter_name,
            "checkpoint_path": str(lora_request.path),
            "base_model_name_or_path": args.base_model_name_or_path,
            "sampling_params": {
                "do_sample": bool(args.do_sample),
                **base_sampling_kwargs,
            },
            "evaluate_params": {
                "code_level": args.code_level,
                "top_k_retrieval": args.top_k_retrieval,
                "k_eval": args.k_eval,
                "max_attempts": args.max_attempts,
                "use_shionogi_db": bool(args.use_shionogi_db),
            },
            # pandas.describe() の出力（count/mean/std/min/25%/50%/75%/max）
            "metrics_describe": df_metrics.describe().to_dict(),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_obj, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()