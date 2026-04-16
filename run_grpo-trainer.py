import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import yaml
from typing import Any, Tuple, Optional
from string import Template
from pathlib import Path
import joblib
import faiss
import torch
import sys

# vLLM + Unsloth Standby でメモリ節約
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # Efficient GRPO 用 Standby 有効化
# 上記モードでVRAMを使い切ってしまうため、一度offにしておく
os.environ.pop("UNSLOTH_VLLM_STANDBY", None)
os.environ["UNSLOTH_VLLM_STANDBY"] = "0"  # 明示的に 0 も念の為与えておく
# os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:False"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from tqdm.auto import tqdm
from unsloth import FastLanguageModel, FastModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, load_from_disk

add_syspath_list = [
    './',
    './mdra_suggest'
]
for syspath in add_syspath_list:
    if syspath not in sys.path:
        sys.path.append(syspath)

from text_processing_utils import (
    get_embeddings_from_queries,
)

from evaluate_ranking import calculate_ir_metrics
from prompts import PROMPT_SET

# -------------------------
# CLI / Resume utilities
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO training with optional checkpoint resume.")

    # -------------------------
    # Data / task settings
    # -------------------------
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/mdra_rl_dataset_v20251210",
        help="Path to the RL dataset (HF datasets saved with load_from_disk).",
    )
    parser.add_argument(
        "--code_level",
        type=str,
        default="pt",
        choices=["pt", "llt"],
        help="MedDRA code level for evaluation (pt or llt). Default: pt",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Top-K candidates retrieved per query for ranking. Default: 200",
    )
    parser.add_argument(
        "--k_eval",
        type=int,
        default=20,
        help="k for IR metrics evaluation. Default: 20",
    )

    # -------------------------
    # Prompts / sequence lengths
    # -------------------------
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Max prompt length (tokens). Default: 2048",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=3072,
        help="Max sequence length (prompt + completion). Default: 3072",
    )

    # -------------------------
    # Model / runtime
    # -------------------------
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
        help="Base model name/path for Unsloth FastModel.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string used for helper models (e.g., 'cuda:0', 'cuda:1', 'cpu').",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Transformers/Unsloth device_map for the base model. Default: auto",
    )

    # -------------------------
    # Config / evaluation toggles
    # -------------------------
    parser.add_argument(
        "--config_path",
        type=str,
        default="./mdra_suggest/config.yml",
        help="Path to mdra_suggest config.yml",
    )
    parser.add_argument(
        "--eval_output_root",
        type=str,
        default="../outputs",
        help="Directory to ensure exists for evaluation-related outputs. Default: ../outputs",
    )
    parser.add_argument(
        "--use_shionogi_db",
        default=False,
        action="store_true",
        help="Whether to use an optional local extension term cache for retrieval. Default: False",
    )
    parser.add_argument(
        "--no_shionogi_db",
        default=False,
        action="store_true",
        help="Disable the optional local extension term cache for retrieval.",
    )

    # -------------------------
    # Training hyperparameters (GRPOConfig)
    # -------------------------
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=3)
    # Generation / sampling (used during training-time rollouts)
    # TRL (0.24.0) defaults: temperature=1.0, top_p=1.0, top_k=None
    # Gemma3 recommended sampling: temperature=1.0, top_p=0.95, top_k=64
    parser.add_argument(
        "--gen_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used during GRPO rollouts. Default: 1.0",
    )
    parser.add_argument(
        "--gen_top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling p used during GRPO rollouts. Default: 0.95",
    )
    parser.add_argument(
        "--gen_top_k",
        type=int,
        default=64,
        help="Top-k sampling used during GRPO rollouts. Use -1 to disable (None). Default: 64",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Max training steps when not resuming with --additional_steps. Default: 5000",
    )
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--torch_empty_cache_steps", type=int, default=1)

    # -------------------------
    # LoRA settings (FastModel.get_peft_model)
    # -------------------------
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint directory to resume from (e.g., /.../outputs/checkpoint-5000).",
    )
    parser.add_argument(
        "--additional_steps",
        type=int,
        default=0,
        help="Additional steps to run after resuming. new_max_steps = current_step + additional_steps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints/logs. Default: outputs",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="extract_key_phrase_from_document",
        help="Key for prompt template in prompts.py PROMPT_SET. Default: extract_key_phrase_from_document",
    )
    parser.add_argument(
        "--llm_device",
        type=str,
        default="",
        help="Device for the LLM. Default: cuda:0",
    )
    parser.add_argument(
        "--reward_device",
        type=str,
        default="",
        help="Device for the reward function. Default: cuda:1",
    )
    return parser.parse_args()


def _read_global_step_from_trainer_state(checkpoint_dir: Path) -> Optional[int]:
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return None
    try:
        data = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        step = data.get("global_step", None)
        return int(step) if step is not None else None
    except Exception:
        return None


def _infer_global_step_from_checkpoint_name(checkpoint_dir: Path) -> Optional[int]:
    m = re.search(r"checkpoint-(\d+)$", str(checkpoint_dir).rstrip("/"))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def get_resume_global_step(checkpoint_dir: Path) -> Optional[int]:
    step = _read_global_step_from_trainer_state(checkpoint_dir)
    if step is not None:
        return step
    return _infer_global_step_from_checkpoint_name(checkpoint_dir)

args = parse_args()
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LLM_DEVICE = torch.device(args.llm_device if args.llm_device != "" else DEFAULT_DEVICE)
REWARD_DEVICE = torch.device(args.reward_device if args.reward_device != "" else DEFAULT_DEVICE)
print(f'{LLM_DEVICE=}')
print(f'{REWARD_DEVICE=}')

# 各種変数の設定
dataset_path = args.dataset_path
code_level = args.code_level
top_k = args.top_k
k_eval = args.k_eval

# max_prompt_length = 2560    # max input_text length: 1511を想定, system promptで約900 -> まだエラーが発生する
max_prompt_length = args.max_prompt_length    # max input_text length: 957を想定, system promptで約900

# MODEL_NAME = "unsloth/medgemma-4b-it"
# MODEL_NAME = "unsloth/medgemma-27b-text-it-bnb-4bit"
MODEL_NAME = args.model_name
# MAX_SEQ_LEN = 4096  # 入力プロンプト 2560, 出力プロンプト 1536想定 -> まだエラーが発生する
# MAX_SEQ_LEN = 3584  # 入力プロンプト 2560, 出力プロンプト 1024想定 -> まだエラーが発生する
MAX_SEQ_LEN = args.max_seq_len  # 入力プロンプト 2048, 出力プロンプト 1024想定

# For GRPO sampling: treat top_k < 0 as disabled (None)
grpo_gen_top_k: Optional[int] = None if int(args.gen_top_k) < 0 else int(args.gen_top_k)



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

    # se_model はロード直後CPUになりやすいため、埋め込み計算では CLI で指定された device を優先する
    # （get_embeddings_from_queries 内で model.to(device) が呼ばれる）。
    embed_device = REWARD_DEVICE
    q_prefix = config['model']['embeddings']['embedding_model_q_prefix']

    unique_queries = list(set(queries))
    result = {}

    q_embeddings = get_embeddings_from_queries(
                        queries=unique_queries,
                        model_input_keys=config['model']['embeddings']['model_input_keys'],
                        model=model, tokenizer=tokenizer,
                        prefix=q_prefix, batch_size=config['model']['embeddings']['batch_size'], device=embed_device
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

# reward functionの設計：フォーマットの正しさ
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 2.0
        scores.append(score)
    return scores

# 部分的に正解している場合の報酬設計
def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end)   == 1 else -0.5
        score += 0.5 if response.count(list_start)  == 1 else -0.5
        score += 0.5 if response.count(list_end)    == 1 else -0.5
        scores.append(score)
    return scores

def check_exact_phrases(prompts, input_text, completions, **kwargs):
    """
    フォーマットが正しい場合に、Key Phraseが正確に文書中から抽出されているかを評価する。
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]
    scores = []
    for guess, text in zip(extracted_responses, input_text):
        score = 0
        if guess is None:
            scores.append(0)
            continue

        # 正確な文字列の抽出には最大で2 pointsを与える
        kp_list = extract_list_items(guess)
        for kp in kp_list:
            if kp in text:
                score += 2.0 / len(kp_list)
        scores.append(score)
    return scores

def check_answer(prompts, completions, mdra_labels, **kwargs):
    """
    フォーマットが正しい場合に、MedDRAコード提案精度を評価点に加える。
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]
    true_labels = [[obj[f'{code_level}_code'] for obj in mdra_obj] for mdra_obj in mdra_labels]
    scores = []
    metrics_list = []
    for guess, labels in zip(extracted_responses, true_labels):
        score = 0
        if guess is None:
            scores.append(0)
            metrics_list.append(None)
            continue

        # metricsとして、AP, nDCGは2倍で計算
        kp_list = extract_list_items(guess)
        if len(kp_list) > 0:
            ranking_results = create_ranking_for_predictions(
                queries=kp_list,
                mdra_embed_dict=mdra_embed_dict, faiss_index=faiss_index, faiss_tbl=faiss_tbl, n_tops=top_k, config=config,
                model=se_model, tokenizer=se_tokenizer, code_level=code_level
            )
            df_list = [modify_ranking(ranking_results.get(string, None), top_k) for string in kp_list]
        else:
            df_list = [pd.DataFrame(columns=['index_string', f'{code_level}_code', 'score'])]
        mod_ranking = (pd.concat(df_list)
            .sort_values(by='score', ascending=False)
            .drop_duplicates(subset=f'{code_level}_code')
        )
        metrics = calculate_ir_metrics(
            relevant_codes=labels,
            predicted_codes=mod_ranking[f'{code_level}_code'],
            k_eval=k_eval
        )
        metrics_list.append(metrics)
        
        score += metrics['AP'] * 2.0
        score += metrics['Recall'] * 1.0
        score += metrics['nDCG'] * 2.0
        scores.append(score)
    
    # scoreが最高値の事例を表示
    if scores:
        max_idx = np.argmax(scores)
        print("*"*20)
        print(f"Score: {scores[max_idx]}")
        print(f'Suggest Metrics: {metrics_list[max_idx]}')
        print(f"Response: {responses[max_idx]}")
        print(f"Extracted KeyPhrase: {extracted_responses[max_idx]}")
        print(f"True Labels: {mdra_labels[max_idx]}")

    return scores

config = load_config(args.config_path)
embed_model_path = config['model']['embeddings']['embedding_model_path']
print(f'{embed_model_path=}')
se_model, se_tokenizer = load_se_model(embed_model_path)
se_model.to(REWARD_DEVICE)

use_shionogi_db_override = False if args.no_shionogi_db else bool(args.use_shionogi_db)
config['evaluate']['use_shionogi_db'] = use_shionogi_db_override
eval_output_root = Path(args.eval_output_root)
if not eval_output_root.exists():
    eval_output_root.mkdir(parents=True)

# MedDRA 関連
mdra_ver = Path(config['mdra']['path']).name.lower()
print(f'{mdra_ver=}')
mdra_cache_dir = Path('./mdra_suggest/cache') / mdra_ver
mdra_cache_path = mdra_cache_dir / 'mdra_dict.joblib'
md_hier, code2name, name2code, cd_llt2pt, term2info, code2info = joblib.load(mdra_cache_path)

# ss_listを作成するための準備
# pt_hierの作成（同一ptに紐づくlltの情報を整理）
pt_hier = {}
for pt_code, hier in md_hier.items():
    items = []
    for llt_code in hier['llt_codes']:
        obj = code2info[llt_code]
        if 'Y' in obj['llt_jcurr']:
            for jcurr, kanji in zip(obj['llt_jcurr'], obj['llt_kanji']):
                if jcurr == 'Y':
                    items.append({'code': llt_code, 'kanji': kanji})
    pt_hier[pt_code] = items

# hlt_hier, hlgt_hierの作成
hlt_hier = {}
hlgt_hier = {}
for pt_code, obj in md_hier.items():
    hlt_code = obj['hlt_code']
    hlgt_code = obj['hlgt_code']
    # hlt_hier の追加
    if hlt_code not in hlt_hier:
        hlt_hier[hlt_code] = []
    if pt_code not in hlt_hier[hlt_code]:
        hlt_hier[hlt_code].append(pt_code)
    # hlgt_hier の追加
    if hlgt_code not in hlgt_hier:
        hlgt_hier[hlgt_code] = []
    if hlt_code not in hlgt_hier[hlgt_code]:
        hlgt_hier[hlgt_code].append(hlt_code)
hlt_hier = {key: [{'code': code, 'kanji': code2name['pt_code'][code]['kanji'][0]}
                     for code in value] for key, value in hlt_hier.items()}
hlgt_hier = {key: [{'code': code, 'kanji': code2name['hlt_code'][code]['kanji'][0]}
                     for code in value] for key, value in hlgt_hier.items()}

# - soc	10077536    製品の問題
# - hlgt	10079145    投薬過誤、その他の製品使用過誤および問題
# - hlgt	10079156	適応外使用および製品の企図的誤用/企図的使用の問題
# - hlgt	10079159	過量投与および過少量投与NEC
ss_llt_list = []
ss_pt_list = []
for hlgt_code in [10079145, 10079156, 10079159]:
    for hlt_item in hlgt_hier[hlgt_code]:
        for pt_item in hlt_hier[hlt_item['code']]:
            ss_pt_list.append(pt_item['kanji'])
            for llt_item in pt_hier[pt_item['code']]:
                ss_llt_list.append(llt_item['kanji'])
print(f'{len(ss_llt_list)=}')
print(f'{len(ss_pt_list)=}')

if code_level == 'pt':
    term_list = ss_pt_list
elif code_level == 'llt':
    term_list = ss_llt_list
else:
    raise ValueError(f"Invalid code level: {code_level}")
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

# プロンプトテンプレートをprompts.pyから読み込む
if args.prompt_key not in PROMPT_SET:
    raise ValueError(
        f"Prompt key '{args.prompt_key}' not found in PROMPT_SET. "
        f"Available keys: {list(PROMPT_SET.keys())}"
    )
prompt_templates = PROMPT_SET[args.prompt_key]
SYSTEM_PROMPT_TEMPLATE = prompt_templates["system"]
USER_PROMPT_TEMPLATE = prompt_templates["user"]

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,        # 4bit QLoRA 用
    load_in_8bit=False,
    fast_inference=True,        # vLLM + Standby 等の最適化パス
    full_finetuning=False,
    device_map=args.device_map if args.llm_device == "" else {"": LLM_DEVICE},  # 通常は`auto`. 明示的な引数が与えられたらそれに従う
    gpu_memory_utilization=args.gpu_memory_utilization,
)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    # use_gradient_checkpointing=True,
    random_state=args.seed,
    max_seq_length=MAX_SEQ_LEN,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
)

# dataset読み込み
dataset = load_from_disk(dataset_path)['train']

# datasetをTrainerに渡すために加工


user_prompt_vars_base = {}
if args.prompt_key == "extract_key_phrase_from_document":
    # extract_key_phrase_from_document の場合、追加の変数はなし（documentのみ）
    reasoning_start = "<reasoning>"
    reasoning_end   = "</reasoning>"
    list_start = "<KeyPhrase>"
    list_end = "</KeyPhrase>"

    sys_prompt_vars = {
    "reasoning_start": reasoning_start,
    "reasoning_end": reasoning_end,
    "list_start": list_start,
    "list_end": list_end,
}

    # rewards定義
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_exact_phrases,
        check_answer,
    ]

elif args.prompt_key == "select_pt_from_list":
    # select_pt_from_list の場合、pt_listなどの変数を設定可能
    # 固定値の場合: user_prompt_vars_base["pt_list"] = some_value
    # データセットから取得する場合は、lambda内で設定（下記参照）
    pass
elif args.prompt_key in ["create_query_for_meddra",
                            "create_query_for_meddra_v2"]:
    # reasoning_start = "<evidence>"
    # reasoning_end   = "</evidence>"
    reasoning_start = "<reasoning>"
    reasoning_end   = "</reasoning>"
    list_start = "<queries>"
    list_end = "</queries>"

    sys_prompt_vars = {
    "reasoning_start": reasoning_start,
    "reasoning_end": reasoning_end,
    "list_start": list_start,
    "list_end": list_end,
    "ss_list": '\n'.join([f'- {term}' for term in term_list])
}

    user_prompt_vars_base = {
    }

    # rewards定義
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
    ]
else:
    raise ValueError(f"Invalid prompt key: {args.prompt_key}")

def build_user_prompt_vars(x):
    """
    データサンプル x から user プロンプト用の変数辞書を構築
    プロンプトキーごとに必要な変数を設定
    """
    vars_dict = {**user_prompt_vars_base, "document": x["input_text"]}
    
    if args.prompt_key == "create_query_for_meddra":
        pass

        # データセットから動的に取得する変数の例
        # vars_dict["pt_list"] = x.get("pt_list", "")
        # vars_dict["other_var"] = x.get("other_var", "")
    
    return vars_dict

dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.substitute(**sys_prompt_vars)},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE.substitute(**build_user_prompt_vars(x))},
    ],
})

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{list_start}(.+?){list_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

resume_from: Optional[Path] = Path(args.resume_from) if args.resume_from else None
if resume_from is not None:
    if not resume_from.exists():
        raise FileNotFoundError(f"--resume_from does not exist: {resume_from}")
    if not resume_from.is_dir():
        raise NotADirectoryError(f"--resume_from must be a directory: {resume_from}")

output_dir = args.output_dir

# default training steps
base_max_steps = args.max_steps
if resume_from is not None and args.additional_steps:
    current_step = get_resume_global_step(resume_from)
    if current_step is None:
        raise ValueError(
            f"Failed to infer current global_step from checkpoint: {resume_from}. "
            f"Expected trainer_state.json or checkpoint-XXXX naming."
        )
    new_max_steps = int(current_step) + int(args.additional_steps)
    if new_max_steps <= current_step:
        raise ValueError(
            f"Invalid max_steps computed: current_step={current_step}, "
            f"additional_steps={args.additional_steps}, new_max_steps={new_max_steps}"
        )
    base_max_steps = new_max_steps

training_args = GRPOConfig(
    learning_rate = args.learning_rate,
    adam_beta1 = args.adam_beta1,
    adam_beta2 = args.adam_beta2,
    weight_decay = args.weight_decay,
    warmup_ratio = args.warmup_ratio,
    lr_scheduler_type = args.lr_scheduler_type,
    optim = args.optim,
    per_device_train_batch_size = args.per_device_train_batch_size,
    gradient_accumulation_steps = args.gradient_accumulation_steps, # Increase to 4 for smoother training
    num_generations = args.num_generations, # 4 -> OOM # Decrease if out of memory
    temperature = args.gen_temperature,
    max_prompt_length = max_prompt_length,
    max_completion_length = MAX_SEQ_LEN - max_prompt_length,
    # max_completion_length = 512,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = base_max_steps,
    save_steps = args.save_steps,
    logging_steps = args.logging_steps,
    max_grad_norm = args.max_grad_norm,
    report_to = args.report_to, # Can use Weights & Biases
    output_dir = output_dir,
    torch_empty_cache_steps=args.torch_empty_cache_steps
)

# `top_p` / `top_k` are supported by TRL at runtime; set them explicitly to control rollout diversity.
setattr(training_args, "top_p", float(args.gen_top_p))
setattr(training_args, "top_k", grpo_gen_top_k)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = reward_funcs,
    args = training_args,
    train_dataset = dataset,
)
trainer.train(resume_from_checkpoint=str(resume_from) if resume_from is not None else None)
