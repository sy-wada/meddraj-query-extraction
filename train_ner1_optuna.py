"""
NER1 (ModernBERT + CRF) Optuna tuning script.

目的:
  - learning_rate / num_train_epochs をOptunaで探索
  - 各trialで trainer.train() 後に、devで
      BIO decode -> span文字列抽出 -> FAISSランキング -> nDCG@k_eval を計算し objective とする
  - 評価は document 単位に nDCG を計算し、平均値をOptuna objective とする

実行（単一GPU: 物理GPU id=1のみを可視化）:

  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python train_ner1_optuna.py \
    --model_name_or_path sbintuitions/modernbert-ja-130m \
    --dataset_path ./data/unsloth_medgemma-4b-it-unsloth-bnb-4bit_checkpoint-10000 \
    --output_dir ./outputs/ner1_optuna \
    --n_trials 20 \
    --lr_range \"[1,5]\" \
    --epoch_range \"[1,3]\" \
    --k_eval 20 \
    --top_k 200 \
    --use_shionogi_db false
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)

from tools.model_ner import ModernBertCRFForTokenClassification, TokenClassificationDataCollator
from tools.ner_decode import bio_tags_to_spans, spans_to_strings
from tools.metrics_ranking_ndcg import (
    MdraAssets,
    calculate_ir_metrics,
    create_ranking_for_predictions,
    load_mdra_assets,
    merge_rankings,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF model id or local path"})


@dataclass
class DataArguments:
    dataset_path: str = field(metadata={"help": "HF dataset saved by save_to_disk"})
    max_seq_length: int = field(default=1024)
    label_field: str = field(default="labels_ner1")
    text_field: str = field(default="text")
    offset_field: str = field(default="offset_mapping")
    gold_field: str = field(default="mdra_labels")
    gold_code_level: str = field(default="pt", metadata={"help": "gold field dict uses e.g. pt_code/llt_code"})
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


@dataclass
class RankingArguments:
    mdra_suggest_dir: str = field(default="./mdra_suggest")
    config_path: str = field(default="./mdra_suggest/config.yml")
    use_shionogi_db: bool = field(default=False)
    code_level: str = field(default="llt", metadata={"help": "llt or pt"})
    top_k: int = field(default=200, metadata={"help": "faiss search top candidates per query"})
    k_eval: int = field(default=20, metadata={"help": "nDCG@k"})
    csv_list_sep: str = field(
        default=" ||| ",
        metadata={"help": "metrics_test_list.csv で複数値を連結する区切り文字（改行は避けることを推奨）"},
    )


@dataclass
class OptunaArguments:
    n_trials: int = field(default=10)
    run_test_eval: bool = field(
        default=True,
        metadata={"help": "If True, evaluate best trial on test split after optimization (note: avoid for strict dev-only tuning)."},
    )
    # 互換: 旧指定（値は 1e-5 単位）
    lr_range: str = field(default="[1,5]", metadata={"help": "DEPRECATED: JSON list [low, high] in 1e-5 units"})
    # 推奨: 直接指定（絶対値）
    lr_min: Optional[float] = field(default=None, metadata={"help": "Recommended: min learning rate (absolute), e.g. 1e-6"})
    lr_max: Optional[float] = field(default=None, metadata={"help": "Recommended: max learning rate (absolute), e.g. 5e-5"})
    lr_log: bool = field(default=True, metadata={"help": "Sample learning_rate in log scale"})

    epoch_range: str = field(default="[1,3]", metadata={"help": "DEPRECATED: JSON list [low, high]"})
    epoch_min: Optional[int] = field(default=None, metadata={"help": "Recommended: min num_train_epochs"})
    epoch_max: Optional[int] = field(default=None, metadata={"help": "Recommended: max num_train_epochs"})

    optuna_metric: str = field(
        default="nDCG",
        metadata={"help": "optimization target metric key: one of [nDCG, AP, RR, Recall]"},
    )
    optuna_storage: Optional[str] = field(
        default=None,
        metadata={"help": "Optuna storage URL (e.g. sqlite:////abs/path/optuna.db). Default: output_dir/optuna.db"},
    )
    study_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optuna study name. Default: derived from output_dir"},
    )
    load_if_exists: bool = field(default=True, metadata={"help": "Resume existing study if storage/study_name exists"})

    # Non-optuna single run (fixed hyperparameters)
    run_single: bool = field(
        default=False,
        metadata={
            "help": "If True, skip Optuna and run a single training/eval with fixed TrainingArguments (learning_rate/num_train_epochs/seed/output_dir). "
            "Use run_test_eval=True only after hyperparameters are finalized (to avoid test leakage)."
        },
    )


def _parse_json_range(range_str: str) -> tuple[float, float]:
    """
    JSON list形式の範囲指定をパースする（例: [1,5]）。
    """
    vals = json.loads(range_str)
    if not isinstance(vals, list) or len(vals) != 2:
        raise ValueError(f"range must be JSON list of length 2, got: {range_str}")
    return float(vals[0]), float(vals[1])


def _build_label2id_from_dataset(ds: Dataset, label_field: str) -> dict[str, int]:
    """
    dataset内のlabel_fieldからユニークなラベル文字列を集めてlabel2idを作る。
    - Noneは無視
    - "O"を必ず含める
    """
    uniq: set[str] = {"O"}
    for ex in ds:
        labels = ex.get(label_field)
        if not labels:
            continue
        # labels は list[str] を想定
        for l in labels:
            if l is None:
                continue
            uniq.add(str(l))
    # "O"を0に寄せたい場合はここで並べ替え
    ordered = ["O"] + sorted([u for u in uniq if u != "O"])
    return {lab: i for i, lab in enumerate(ordered)}


def _preprocess_ner1(
    ex: dict[str, Any],
    *,
    tokenizer,
    label2id: dict[str, int],
    max_length: int,
    label_field: str,
) -> dict[str, Any]:
    """
    ex は少なくとも token_ids / label_field / text / offset_mapping を含む想定。
    token_ids は add_special_tokens=False で得たID列（special無し）。
    """
    token_ids = ex.get("token_ids")
    if token_ids is None:
        raise ValueError("dataset must contain 'token_ids'")
    token_ids = list(token_ids)

    # special token を付与（bos/eos or cls/sep）
    special_tokens_count = tokenizer.num_special_tokens_to_add(pair=False)
    truncated = token_ids[: max_length - special_tokens_count]
    input_ids = tokenizer.build_inputs_with_special_tokens(truncated)
    attention_mask = [1] * len(input_ids)

    # labels
    labels = ex.get(label_field)
    if labels:
        labels = list(labels)[: max_length - special_tokens_count]
        label_ids = [label2id[str(l)] for l in labels]
        # special token 分は -100
        if special_tokens_count == 2:
            label_ids = [-100] + label_ids + [-100]
        elif special_tokens_count == 1:
            # 片側のみspecialの場合（稀）
            label_ids = [-100] + label_ids
        else:
            # 0や>2は想定外だが、とりあえず全長を合わせる
            pad = [-100] * (len(input_ids) - len(label_ids))
            label_ids = label_ids + pad
    else:
        label_ids = None

    out = {
        "guid": ex.get("guid"),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        # 重要:
        # Trainer(remove_unused_columns=True) は model.forward の引数名に無い列を dataset から落とす。
        # model が受けるのは通常 "labels" なので、"labels_ner1" しか無いと学習時にラベル列が消え、
        # collator 側では labels=None 扱い→全 -100 パディング→loss=0 になりうる。
        # そのため、"labels" を必ず同梱して互換を確保する。
        label_field: label_ids,
        "labels": label_ids,
        # 評価用に残す（Trainerには不要だがdev側で使う）
        "text": ex.get("text"),
        "offset_mapping": ex.get("offset_mapping"),
        "gold": ex.get("gold"),
    }
    return out


def _extract_gold_codes(ex: dict[str, Any], gold_field: str, code_level: str) -> list[str]:
    """
    gold_field（例: mdra_labels）から正解コード集合を抽出。
    - mdra_labels は list[dict] を想定し、key f\"{code_level}_code\" を読む。
    """
    gold_obj = ex.get(gold_field)
    if not gold_obj:
        return []
    key = f"{code_level}_code"
    codes: set[str] = set()
    if isinstance(gold_obj, list):
        for item in gold_obj:
            if isinstance(item, dict) and key in item:
                codes.add(str(item[key]))
    return sorted(codes)


@torch.no_grad()
def evaluate_ranking_metrics(
    *,
    model: ModernBertCRFForTokenClassification,
    trainer: Trainer,
    eval_ds: Dataset,
    id2label: dict[int, str],
    text_field: str,
    offset_field: str,
    mdra_assets: MdraAssets,
    top_k: int,
    k_eval: int,
    code_level: str,
    device: torch.device,
    csv_list_sep: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Trainer.predictでlogits/labelsを取得し、BIO decode→文字列抽出→FAISSランキング→IR metrics@k_eval を計算する。
    - metrics_avg: 文書単位の各指標を平均したdict
    - metrics_list_df: 文書ごとの出力（guid, extracted, gold_llt_codes, 各指標）
    """
    pred = trainer.predict(eval_ds, metric_key_prefix="dev")
    logits = pred.predictions  # (B,T,C)
    labels = pred.label_ids    # (B,T) or None
    if logits is None or labels is None:
        return {"AP": 0.0, "RR": 0.0, "Recall": 0.0, "nDCG": 0.0}, pd.DataFrame()

    # CRF decode は model と同一deviceで行う必要がある
    model_device = next(model.parameters()).device
    logits_t = torch.from_numpy(np.asarray(logits)).to(model_device).float()
    labels_t = torch.from_numpy(np.asarray(labels)).to(model_device).long()

    # special token を除外して CRF decode を安定化
    if logits_t.size(1) >= 2:
        emissions = logits_t[:, 1:-1, :]
        # mask は全Trueの想定（token部のみ残す）
        mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=model_device)
    else:
        emissions = logits_t
        mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=model_device)

    paths = model.crf.decode(emissions, mask=mask)

    rows: list[dict[str, Any]] = []
    for i, path_ids in enumerate(paths):
        tags = [id2label.get(int(x), "O") for x in path_ids]
        spans = bio_tags_to_spans(tags)

        ex = eval_ds[i]
        text = ex.get(text_field) or ""
        offset_mapping = ex.get(offset_field) or []
        extracted = spans_to_strings(spans, offset_mapping=offset_mapping, text=text)
        extracted = [s for s in extracted if s and str(s).strip()]
        # CSV表示を崩す改行は潰す（元テキスト由来の改行がspanに含まれうる）
        extracted_clean = [str(s).replace("\r", " ").replace("\n", " ").strip() for s in extracted]

        gold_codes = ex.get("gold") or []
        if not isinstance(gold_codes, list):
            gold_codes = []
        # nDCG評価は llt_code (int) で一致判定
        gold_codes_int = [int(x) for x in gold_codes]

        metrics = {"AP": 0.0, "RR": 0.0, "Recall": 0.0, "nDCG": 0.0}
        if extracted_clean and gold_codes_int:
            ranking_results = create_ranking_for_predictions(
                queries=extracted_clean,
                assets=mdra_assets,
                n_tops=top_k,
                device=device,
                code_level=code_level,
            )
            merged = merge_rankings(ranking_results, queries=extracted_clean, top_k=top_k, code_level=code_level)
            pred_codes = merged[f"{code_level}_code"].tolist() if len(merged) else []
            pred_codes_int = [int(x) for x in pred_codes]
            metrics = calculate_ir_metrics(gold_codes_int, pred_codes_int, k_eval=int(k_eval))

        gold_col = f"gold_{code_level}_code"
        rows.append(
            {
                "guid": ex.get("guid"),
                "extracted": csv_list_sep.join(extracted_clean),
                gold_col: csv_list_sep.join(str(x) for x in gold_codes_int),
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return {"AP": 0.0, "RR": 0.0, "Recall": 0.0, "nDCG": 0.0}, df
    metrics_avg = {k: float(df[k].mean()) for k in ["AP", "RR", "Recall", "nDCG"]}
    return metrics_avg, df


def objective(
    trial: Optional[optuna.Trial],
    *,
    train_ds: Dataset,
    dev_ds: Dataset,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    ranking_args: RankingArguments,
    optuna_args: OptunaArguments,
    mdra_assets: MdraAssets,
    device: torch.device,
) -> float:
    # trial パラメータ
    if trial is not None:
        # learning_rate
        if optuna_args.lr_min is not None and optuna_args.lr_max is not None:
            lr_low, lr_high = float(optuna_args.lr_min), float(optuna_args.lr_max)
            lr_log = bool(optuna_args.lr_log)
        else:
            # 互換: lr_range は 1e-5 単位
            lr_low_u, lr_high_u = _parse_json_range(optuna_args.lr_range)
            lr_low, lr_high = lr_low_u * 1e-5, lr_high_u * 1e-5
            lr_log = True
        training_args.learning_rate = trial.suggest_float("learning_rate", low=lr_low, high=lr_high, log=lr_log)

        # num_train_epochs
        if optuna_args.epoch_min is not None and optuna_args.epoch_max is not None:
            ep_low, ep_high = int(optuna_args.epoch_min), int(optuna_args.epoch_max)
        else:
            ep_low_f, ep_high_f = _parse_json_range(optuna_args.epoch_range)
            ep_low, ep_high = int(ep_low_f), int(ep_high_f)
        training_args.num_train_epochs = trial.suggest_int("num_train_epochs", low=ep_low, high=ep_high)

    set_seed(training_args.seed)

    # model config
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = len(label2id)
    config.label2id = label2id
    config.id2label = id2label

    model = ModernBertCRFForTokenClassification.from_pretrained(model_args.model_name_or_path, config=config)

    data_collator = TokenClassificationDataCollator(tokenizer=tokenizer, label_field=data_args.label_field, pad_to_multiple_of=8)

    # Trainer（評価はtrain後に手動）
    training_args.evaluation_strategy = "no"
    training_args.save_strategy = "no"
    training_args.logging_strategy = "steps"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    # trial artifact: fine-tuned weights
    trial_out = Path(training_args.output_dir)
    trainer.save_model(str(trial_out))
    tokenizer.save_pretrained(str(trial_out))

    metrics_avg, _ = evaluate_ranking_metrics(
        model=model,
        trainer=trainer,
        eval_ds=dev_ds,
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

    # save metrics_dev.json (all metrics)
    metrics_payload = {
        "k_eval": int(ranking_args.k_eval),
        "code_level": ranking_args.code_level,
        "top_k": int(ranking_args.top_k),
        "optuna_metric": optuna_args.optuna_metric,
        "metrics": metrics_avg,
    }
    with open(trial_out / "metrics_dev.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    # objective value
    metric_key = optuna_args.optuna_metric
    if metric_key not in metrics_avg:
        raise ValueError(f"Unknown optuna_metric={metric_key}. Available keys: {list(metrics_avg.keys())}")
    return float(metrics_avg[metric_key])


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, RankingArguments, OptunaArguments, TrainingArguments))
    model_args, data_args, ranking_args, optuna_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    base_out_dir = Path(training_args.output_dir)
    out_dir = base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(data_args.dataset_path)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(dataset)}")

    # trainのみ: labels_ner1 == None を除外（dev/testは除外しない）
    raw_train = dataset["train"].filter(lambda ex: ex.get(data_args.label_field) is not None)
    raw_dev = dataset["dev"] if "dev" in dataset else dataset["validation"]
    if data_args.max_train_samples is not None:
        raw_train = raw_train.select(range(min(len(raw_train), data_args.max_train_samples)))
    if data_args.max_eval_samples is not None:
        raw_dev = raw_dev.select(range(min(len(raw_dev), data_args.max_eval_samples)))

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    label2id = _build_label2id_from_dataset(raw_train, data_args.label_field)
    id2label = {v: k for k, v in label2id.items()}

    def _inject_gold(ex):
        ex["gold"] = _extract_gold_codes(ex, data_args.gold_field, ranking_args.code_level)
        return ex
    raw_train = raw_train.map(_inject_gold, batched=False)
    raw_dev = raw_dev.map(_inject_gold, batched=False)

    fn_kwargs = dict(
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=data_args.max_seq_length,
        label_field=data_args.label_field,
    )
    train_ds = raw_train.map(lambda ex: _preprocess_ner1(ex, **fn_kwargs), batched=False)
    dev_ds = raw_dev.map(lambda ex: _preprocess_ner1(ex, **fn_kwargs), batched=False)

    # SE model (embedding) + FAISS assets は trial外でロードして使い回す
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdra_assets = load_mdra_assets(
        mdra_suggest_dir=ranking_args.mdra_suggest_dir,
        config_path=ranking_args.config_path,
        use_shionogi_db=ranking_args.use_shionogi_db,
        device=device,
    )

    # ------------------------------------------------------------------
    # Single run mode: fixed hyperparameters (no Optuna / no optuna.db)
    # ------------------------------------------------------------------
    if bool(optuna_args.run_single):
        out_dir_single = Path(training_args.output_dir)
        out_dir_single.mkdir(parents=True, exist_ok=True)

        # Train + dev eval (writes metrics_dev.json into output_dir)
        _ = objective(
            None,
            train_ds=train_ds,
            dev_ds=dev_ds,
            tokenizer=tokenizer,
            label2id=label2id,
            id2label=id2label,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            ranking_args=ranking_args,
            optuna_args=optuna_args,
            mdra_assets=mdra_assets,
            device=device,
        )

        # Optional test eval (only after hyperparameters are finalized)
        if optuna_args.run_test_eval and "test" in dataset:
            raw_test = dataset["test"].map(_inject_gold, batched=False)
            if data_args.max_eval_samples is not None:
                raw_test = raw_test.select(range(min(len(raw_test), data_args.max_eval_samples)))
            test_ds = raw_test.map(lambda ex: _preprocess_ner1(ex, **fn_kwargs), batched=False)

            # load weights from output_dir we just saved
            config = AutoConfig.from_pretrained(str(out_dir_single))
            best_model = ModernBertCRFForTokenClassification.from_pretrained(str(out_dir_single), config=config)
            test_trainer = Trainer(
                model=best_model,
                args=training_args,
                train_dataset=None,
                eval_dataset=test_ds,
                data_collator=TokenClassificationDataCollator(tokenizer=tokenizer, label_field=data_args.label_field, pad_to_multiple_of=8),
                processing_class=tokenizer,
            )

            metrics_avg, df_list = evaluate_ranking_metrics(
                model=best_model,
                trainer=test_trainer,
                eval_ds=test_ds,
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
            with open(out_dir_single / "metrics_test.json", "w", encoding="utf-8") as f:
                json.dump({"k_eval": int(ranking_args.k_eval), "metrics": metrics_avg}, f, ensure_ascii=False, indent=2)
            if df_list is not None and len(df_list):
                df_list.to_csv(out_dir_single / "metrics_test_list.csv", index=False)

        logger.info("Single run finished.")
        return

    def _obj(trial: optuna.Trial) -> float:
        # trialごとに出力dirを分ける（ログ混線回避）
        trial_out = base_out_dir / f"trial_{trial.number}"
        training_args.output_dir = str(trial_out)
        trial_out.mkdir(parents=True, exist_ok=True)
        return objective(
            trial,
            train_ds=train_ds,
            dev_ds=dev_ds,
            tokenizer=tokenizer,
            label2id=label2id,
            id2label=id2label,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            ranking_args=ranking_args,
            optuna_args=optuna_args,
            mdra_assets=mdra_assets,
            device=device,
        )

    # Optuna storage: default is output_dir/optuna.db (same level as trial_0)
    if optuna_args.optuna_storage is None:
        optuna_db_path = (base_out_dir / "optuna.db").resolve()
        storage_url = f"sqlite:///{optuna_db_path.as_posix()}"
    else:
        storage_url = optuna_args.optuna_storage

    study_name = optuna_args.study_name or f"ner1_optuna_{base_out_dir.name}"
    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=study_name,
        load_if_exists=bool(optuna_args.load_if_exists),
    )

    try:
        study.optimize(_obj, n_trials=optuna_args.n_trials)
    except KeyboardInterrupt:
        logger.warning("Interrupted. Writing current best_trial.json and exiting gracefully.")

    best = {
        "storage": storage_url,
        "study_name": study_name,
        "best_value": study.best_value if study.best_trial is not None else None,
        "best_params": study.best_params if study.best_trial is not None else None,
        "best_trial_number": study.best_trial.number if study.best_trial is not None else None,
    }
    with open(out_dir / "best_trial.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    # best trial model -> test evaluation
    if optuna_args.run_test_eval and "test" in dataset and best["best_trial_number"] is not None:
        best_trial_dir = base_out_dir / f"trial_{best['best_trial_number']}"

        # build test dataset (do not filter labels_ner1==None)
        raw_test = dataset["test"].map(_inject_gold, batched=False)
        if data_args.max_eval_samples is not None:
            raw_test = raw_test.select(range(min(len(raw_test), data_args.max_eval_samples)))
        test_ds = raw_test.map(lambda ex: _preprocess_ner1(ex, **fn_kwargs), batched=False)

        # load best model weights
        config = AutoConfig.from_pretrained(str(best_trial_dir))
        best_model = ModernBertCRFForTokenClassification.from_pretrained(str(best_trial_dir), config=config)
        test_trainer = Trainer(
            model=best_model,
            args=training_args,
            train_dataset=None,
            eval_dataset=test_ds,
            data_collator=TokenClassificationDataCollator(tokenizer=tokenizer, label_field=data_args.label_field, pad_to_multiple_of=8),
            processing_class=tokenizer,
        )

        metrics_avg, df_list = evaluate_ranking_metrics(
            model=best_model,
            trainer=test_trainer,
            eval_ds=test_ds,
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
        with open(best_trial_dir / "metrics_test.json", "w", encoding="utf-8") as f:
            json.dump({"k_eval": int(ranking_args.k_eval), "metrics": metrics_avg}, f, ensure_ascii=False, indent=2)
        if df_list is not None and len(df_list):
            df_list.to_csv(best_trial_dir / "metrics_test_list.csv", index=False)

    logger.info(f"Best: {best}")


if __name__ == "__main__":
    main()

