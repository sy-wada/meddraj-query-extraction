import re
import argparse
import pandas as pd
import numpy as np
from fuzzysearch import find_near_matches
import yaml
from typing import Any, Tuple
from string import Template
from pathlib import Path
import joblib
import faiss
import torch
import sys
import json
import time

import torch
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

SCRIPT_DIR = Path(__file__).resolve().parent
add_syspath_list = [
    str(SCRIPT_DIR),
    str(SCRIPT_DIR / 'mdra_suggest'),
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
        description="Run vLLM inference with a LoRA adapter and save per-example JSON outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python run_inference_with_adapter.py \\\n"
            "    --path_adapter ./outputs/checkpoint-10000 \\\n"
            "    --dataset_path ./data/mdra_rl_dataset_v20251210 \\\n"
            "    --config_path ./mdra_suggest/config.yml \\\n"
            "    --output_dir ./outputs \\\n"
            "    --base_model_name_or_path unsloth/medgemma-4b-it-unsloth-bnb-4bit\n"
            "\n"
            "Deterministic generation for Step 6.3 (train/dev, one-pass):\n"
            "  python run_inference_with_adapter.py \\\n"
            "    --splits train dev \\\n"
            "    --no-do_sample \\\n"
            "    --temperature 0.0 \\\n"
            "    --max_attempts 1 \\\n"
            "    ...\n"
        ),
    )

    # path related (required)
    parser.add_argument(
        "--path_adapter",
        type=Path,
        default=None,
        help=(
            "Path to the LoRA adapter checkpoint directory. "
            "If omitted, runs vanilla (no adapter)."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to the dataset saved by datasets.load_from_disk (e.g. ./data/mdra_rl_dataset_v20251210).",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Path to the YAML config (e.g. ./mdra_suggest/config.yml).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Base output directory (e.g. ./outputs). JSONs will be written under <output_dir>/jsons/...",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        help="Base model name or local path (e.g. unsloth/medgemma-4b-it-unsloth-bnb-4bit).",
    )

    # numeric defaults: keep current values
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="vLLM gpu_memory_utilization (default: 0.6).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size (default: 1).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1536,
        help="SamplingParams max_tokens (default: 1536).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="SamplingParams temperature (default: 1.0).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="SamplingParams top_p (default: 0.95).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=64,
        help="SamplingParams top_k (default: 64).",
    )
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
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit seed list for repeated inference (e.g. --seeds 0 1 2 ... 29). "
            "If provided, overrides --seed/--k_iterations."
        ),
    )
    parser.add_argument(
        "--k_iterations",
        type=int,
        default=1,
        help=(
            "Number of repeated generations per example by varying seed. "
            "Used only when --seeds is not provided. "
            "Seeds will be generated as [seed + i for i in range(k_iterations)]."
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Dataset splits to run (e.g. --splits train dev). If omitted, all splits are processed.",
    )
    parser.add_argument(
        "--top_k_retrieval",
        type=int,
        default=200,
        help="Top-k for retrieval/ranking modification (default: 200).",
    )
    parser.add_argument(
        "--k_eval",
        type=int,
        default=20,
        help="k for evaluation metrics (default: 20).",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Max generation attempts per example (default: 5).",
    )

    # other
    parser.add_argument(
        "--code_level",
        type=str,
        default="llt",
        help="Which code level to use for labels (default: llt).",
    )
    parser.add_argument(
        "--use_shionogi_db",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use an optional local extension term cache for retrieval (default: False).",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    path_adapter = args.path_adapter
    use_adapter = path_adapter is not None
    lora_request = None
    checkpoint_name = "vanilla"
    if use_adapter:
        adapter_parts = path_adapter.name.split('-')
        lora_name = adapter_parts[1] if len(adapter_parts) > 1 else path_adapter.name
        lora_request = LoRARequest(lora_name, 1, str(path_adapter))
        checkpoint_name = path_adapter.name

    dataset = load_from_disk(str(args.dataset_path))
    dataset = dataset.remove_columns('datetime')

    base_model_name_or_path = args.base_model_name_or_path
    llm = LLM(model=base_model_name_or_path,
            enable_lora=use_adapter,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size    # 1枚に収まるなら1にしておく
            )
    # Record "requested" decoding params (before any deterministic/greedy normalization).
    sampling_params_requested: dict[str, Any] = dict(
        do_sample=bool(args.do_sample),
        seed=int(args.seed),
        seeds=list(args.seeds) if args.seeds is not None else None,
        k_iterations=int(args.k_iterations),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
    )

    base_sampling_kwargs: dict[str, Any] = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
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

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    config = load_config(str(args.config_path))
    embed_model_path = config['model']['embeddings']['embedding_model_path']
    print(f'{embed_model_path=}')
    se_model, se_tokenizer = load_se_model(embed_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config['evaluate']['use_shionogi_db'] = args.use_shionogi_db
    output_dir = args.output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # MedDRA 関連
    mdra_ver = Path(config['mdra']['path']).name.lower()
    print(f'{mdra_ver=}')
    mdra_cache_dir = (SCRIPT_DIR / 'mdra_suggest' / 'cache') / mdra_ver
    mdra_cache_path = mdra_cache_dir / 'mdra_dict.joblib'
    md_hier, code2name, name2code, cd_llt2pt, term2info, code2info = joblib.load(mdra_cache_path)

    # Faiss / Embeddings 準備
    mdra_embed_dict = joblib.load(mdra_cache_dir / f'{embed_model_path.replace("/", "_")}.embed.dict.joblib')
    mdra_faiss_index = faiss.read_index(str(mdra_cache_dir / f'{embed_model_path.replace("/", "_")}.embed.faiss_index'))
    faiss_index = mdra_faiss_index
    faiss_tbl = pd.DataFrame({
        'index_string': mdra_embed_dict.keys(),
        'llt_codes': [[code] for code in mdra_embed_dict.values()]
    })

    use_shionogi_db = config['evaluate']['use_shionogi_db']
    print(f'{use_shionogi_db=}')
    if use_shionogi_db:
        sheet_path = Path(config['mdra']['shionogi_path'])
        cache_dir = mdra_cache_dir.parent / f'{mdra_ver}_{sheet_path.stem}'
        exception_list_path = config['mdra']['exceptions_for_preventing_data_leak']
        stem_name = f'{embed_model_path.replace("/", "_")}.exceptions_for_preventing_data_leak={int(bool(exception_list_path))}'
        additional_embeds = np.load(cache_dir / f'{stem_name}.embed.npy')
        additional_embeds_dict = joblib.load(cache_dir / f'{stem_name}.embed.dict.joblib')
        faiss_index.add(additional_embeds)
        faiss_tbl = pd.concat([
            faiss_tbl, pd.DataFrame({
                'index_string': additional_embeds_dict.keys(),
                'llt_codes': additional_embeds_dict.values()
            })],
            ).reset_index(drop=True)

    match_format = re.compile(
        rf"^[\s]{{0,}}"\
        rf"{reasoning_start}.+?{reasoning_end}.*?"\
        rf"{list_start}(.+?){list_end}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )
    code_level = args.code_level
    top_k_retrieval = args.top_k_retrieval
    k_eval = args.k_eval

    max_attempts = args.max_attempts
    
    # JSON出力用のディレクトリを作成
    base_model_dir_name = base_model_name_or_path.replace("/", "_")
    json_output_base_dir = output_dir / 'jsons' / f'{base_model_dir_name}_{checkpoint_name}'
    json_output_base_dir.mkdir(parents=True, exist_ok=True)

    # Seed list for repeated inference (paired design is ensured externally by reusing this list across models)
    if args.seeds is not None and len(args.seeds) > 0:
        seed_list = list(args.seeds)
    else:
        k = max(int(args.k_iterations), 1)
        seed_list = [int(args.seed) + i for i in range(k)]
    
    selected_splits = set(args.splits) if args.splits else None
    for ds_name, ds in dataset.items():
        if selected_splits is not None and ds_name not in selected_splits:
            continue
        progress_data_bar = tqdm(ds, desc=f'{ds_name}', total=len(ds))
        for data in progress_data_bar:
            text = data['input_text']
            guid = data.get('guid', 'unknown')

            for seed in seed_list:
                # seedごとのディレクトリを作成
                seed_dir = json_output_base_dir / ds_name / f'seed_{seed}'
                seed_dir.mkdir(parents=True, exist_ok=True)

                sampling_kwargs: dict[str, Any] = dict(base_sampling_kwargs)
                sampling_kwargs["seed"] = int(seed)
                try:
                    sampling_params = SamplingParams(**sampling_kwargs)
                except TypeError:
                    # For compatibility if the installed vLLM SamplingParams doesn't accept some kwargs.
                    sampling_kwargs.pop("seed", None)
                    sampling_params = SamplingParams(**sampling_kwargs)

                kp_list: list[str] = []
                attempts = 0
                generate_time_seconds = 0.0
                matched_objects: list[dict] = []
                raw_text_last: str | None = None
                format_matched = False

                for i in range(max_attempts):
                    # llm.generate()の実行時間を計測
                    start_time = time.time()
                    outputs = llm.generate(
                        prompts=create_input_ids_for_vllm_generate(text=text),
                        sampling_params=sampling_params,
                        lora_request=lora_request,
                    )
                    end_time = time.time()
                    generate_time_seconds = end_time - start_time

                    raw_text_last = outputs[0].outputs[0].text
                    m = match_format.search(raw_text_last)
                    attempts = i + 1
                    if m:
                        format_matched = True
                        kp_list = extract_list_items(m.group(1))
                        break
                    kp_list = []

                if kp_list:
                    matched_objects = match_phrases_in_text(text=text, phrase_list=kp_list, max_l_dist=2)
                    matched_objects = merge_overlapping_spans(match_list=matched_objects, text=text)

                output_data = dict(data)  # 元のdataオブジェクトをコピー
                output_data['base_model'] = base_model_name_or_path
                output_data['checkpoint'] = checkpoint_name
                output_data['path_adapter'] = str(path_adapter) if use_adapter else None
                output_data['split'] = ds_name
                # Backward-compatible key (kept) + explicit requested/effective keys for reproducibility.
                output_data['sampling_params'] = {
                    "do_sample": bool(args.do_sample),
                    "seed": int(seed),
                    "max_tokens": sampling_kwargs.get("max_tokens"),
                    "temperature": sampling_kwargs.get("temperature"),
                    "top_p": sampling_kwargs.get("top_p"),
                    "top_k": sampling_kwargs.get("top_k"),
                }
                output_data['sampling_params_requested'] = dict(sampling_params_requested)
                output_data['sampling_params_effective'] = {
                    "do_sample": bool(args.do_sample),
                    "seed": int(seed),
                    "max_tokens": sampling_kwargs.get("max_tokens"),
                    "temperature": sampling_kwargs.get("temperature"),
                    "top_p": sampling_kwargs.get("top_p"),
                    "top_k": sampling_kwargs.get("top_k"),
                }
                output_data['k_iterations'] = len(seed_list)
                output_data['seed_list'] = seed_list
                output_data['format_matched'] = format_matched
                output_data['raw_text_last'] = raw_text_last
                output_data['kp_list'] = kp_list
                output_data['generate_time_seconds'] = generate_time_seconds
                output_data['attempts'] = attempts
                output_data['matched_objects'] = matched_objects

                json_file_path = seed_dir / f'{guid}.json'
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
