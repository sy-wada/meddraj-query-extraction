[English](./README.md)

# MedDRA/J Search Query Extraction for Pharmacovigilance

論文「MedDRA/J Search Query Extraction for Pharmacovigilance: Comparing Reinforcement Learning Optimized Large Language Models with Knowledge Distillation」で用いた、MedDRA/J 検索クエリ抽出のためのコード一式です。

本リポジトリには、以下のコードを含みます。

- PEFT-RL による LLM 学習
- LLM を用いたクエリ抽出推論
- Distill KPE モデル学習
- Distill KPE モデル評価
- latency 集計
- 非劣性判定を含む統計集計
- AE/SS 層別の再集計
- MedDRA/J キャッシュ生成

本リポジトリには以下を含みません。

- `data/` 配下のデータセット
- `outputs/` 配下のモデル出力 JSON や評価結果
- 学習済みモデル重み
- MedDRA/J 辞書本体
- 社内拡張辞書・社内データベース
- `mdra_suggest/cache/` に生成されるキャッシュ

## ディレクトリ構成

```text
.
├── aggregate_llm_latency_from_jsons.py
├── eval_adapters.py
├── external_assets/
├── mdra_suggest/
│   ├── cache/
│   ├── config.yml
│   ├── create_cache.py
│   ├── evaluate_ranking.py
│   └── text_processing_utils.py
├── prompts.py
├── run_grpo-trainer.py
├── run_inference_with_adapter.py
├── scripts/
├── summarize_test_metrics.py
├── tools/
└── train_ner1_optuna.py
```

## セットアップ

`uv` を使う前提です。

```bash
uv sync
```

GPU 環境では `torch` / `vllm` / `unsloth` が GPU 対応で入ることを前提にしています。CPU のみで使う場合、PEFT-RL 学習や vLLM 推論はそのままでは実行できません。

## 外部資産

このコードを動かすには、ユーザー自身で以下を用意してください。

- Hugging Face 形式で `load_from_disk` 可能なデータセット
- MedDRA/J 辞書本体
- `mdra_suggest/cache/` に生成したキャッシュ
- 必要に応じて追加語彙用のローカル拡張辞書

配置の詳細は以下を参照してください。

- `external_assets/README.md`
- `mdra_suggest/README.md`
- `mdra_suggest/cache/README.md`

## 実行例

以下の例は、拡張辞書を使わない最小構成です。公開版では `use_shionogi_db` 系フラグは未指定時に無効です。

### 1. MedDRA/J キャッシュ生成

```bash
uv run python mdra_suggest/create_cache.py \
  --config ./mdra_suggest/config.yml
```

### 2. PEFT-RL 学習

```bash
uv run python run_grpo-trainer.py \
  --dataset_path ./data/mdra_rl_dataset \
  --model_name unsloth/medgemma-4b-it-unsloth-bnb-4bit \
  --config_path ./mdra_suggest/config.yml \
  --code_level llt \
  --top_k 200 \
  --k_eval 20 \
  --output_dir ./outputs/qlora_adapter_medgemma-4b-it \
  --max_steps 5000 \
  --save_steps 1000 \
  --logging_steps 10 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1
```

### 3. LLM 推論

```bash
uv run python run_inference_with_adapter.py \
  --path_adapter ./models/qlora_adapter_medgemma-4b-it/checkpoint-10000 \
  --dataset_path ./data/mdra_rl_dataset \
  --config_path ./mdra_suggest/config.yml \
  --output_dir ./outputs \
  --base_model_name_or_path unsloth/medgemma-4b-it-unsloth-bnb-4bit \
  --splits test \
  --seeds 0 1 2 3 4 \
  --gpu_memory_utilization 0.8 \
  --tensor_parallel_size 1 \
  --max_tokens 1536 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k 64 \
  --top_k_retrieval 200 \
  --k_eval 20 \
  --max_attempts 5 \
  --code_level llt
```

### 4. Distill KPE 学習

```bash
uv run python train_ner1_optuna.py \
  --model_name_or_path sbintuitions/modernbert-ja-130m \
  --dataset_path ./data/distill_ner_dataset \
  --output_dir ./outputs/ner1_optuna \
  --study_name ner1_optuna_public \
  --n_trials 20 \
  --run_test_eval false \
  --optuna_metric nDCG \
  --code_level llt \
  --top_k 200 \
  --k_eval 20 \
  --lr_min 1e-6 \
  --lr_max 3e-5 \
  --epoch_min 15 \
  --epoch_max 20 \
  --max_seq_length 4096 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 4 \
  --warmup_ratio 0.05 \
  --logging_steps 10 \
  --save_strategy no \
  --report_to none \
  --bf16 true
```

### 5. Distill KPE 評価

```bash
uv run python scripts/eval_ner1_checkpoint.py \
  --ckpt_dir ./outputs/ner1_optuna/best_trial_xx/seed_0 \
  --dataset_path ./data/distill_ner_dataset \
  --splits test \
  --config_path ./mdra_suggest/config.yml \
  --code_level llt \
  --top_k 200 \
  --k_eval 20 \
  --use_shionogi_db false
```

### 6. LLM latency 集計

```bash
uv run python aggregate_llm_latency_from_jsons.py \
  --out_dir ./outputs/summary_test_metrics \
  --jsons_base ./outputs/jsons
```

### 7. Distill / PEFT-RL latency 計測

```bash
uv run python scripts/measure_peftrl_latency.py \
  --base_model_name_or_path unsloth/medgemma-4b-it-unsloth-bnb-4bit \
  --path_adapter ./models/qlora_adapter_medgemma-4b-it/checkpoint-10000 \
  --dataset_path ./data/mdra_rl_dataset \
  --device cuda \
  --out_dir ./outputs/summary_test_metrics
```

```bash
uv run python scripts/measure_distill_kpe_latency.py \
  --ckpt_dir ./outputs/ner1_optuna/best_trial_xx/seed_0 \
  --dataset_path ./data/distill_ner_dataset \
  --device cuda \
  --config_path ./mdra_suggest/config.yml \
  --use_shionogi_db false \
  --out_dir ./outputs/summary_test_metrics
```

### 8. 非劣性を含む統計集計

```bash
uv run python summarize_test_metrics.py \
  --out_dir ./outputs/summary_test_metrics/bootstrap10000 \
  --bootstrap 10000 \
  --rng_seed 1 \
  --metrics AP,RR,Recall,nDCG \
  --config_json ./example_model_specs.json \
  --ni_retain 0.9 \
  --ni_ref_a Base \
  --ni_ref_b PEFT_RL \
  --ni_contrast H2_NER_minus_PEFTRl \
  --ni_metric nDCG
```

### 9. AE/SS 層別の再集計

```bash
uv run python scripts/summarize_ae_ss_ir_metrics.py \
  --test_scores_csv ./outputs/summary_test_metrics/test_scores_all.csv \
  --label_summary_csv ./outputs/summary_test_metrics/dataset_mdra_label_summary.csv \
  --config_path ./mdra_suggest/config.yml \
  --use_shionogi_db false \
  --code_level llt \
  --top_k_retrieval 200 \
  --k_eval 20 \
  --out_dir ./outputs/summary_test_metrics
```

## 主要ファイル

- `run_grpo-trainer.py`: PEFT-RL 学習
- `run_inference_with_adapter.py`: LLM 推論
- `train_ner1_optuna.py`: Distill KPE 学習
- `scripts/eval_ner1_checkpoint.py`: Distill KPE 評価
- `aggregate_llm_latency_from_jsons.py`: LLM latency 集計
- `summarize_test_metrics.py`: bootstrap 集計と非劣性判定
- `scripts/measure_peftrl_latency.py`: PEFT-RL latency 計測
- `scripts/measure_distill_kpe_latency.py`: Distill KPE latency 計測
- `scripts/summarize_ae_ss_ir_metrics.py`: AE/SS 層別再集計

## 制約事項

- データセット形式は、元研究で用いた Hugging Face `load_from_disk` 形式を前提にしています。
- MedDRA/J 本体はライセンス上の理由から同梱していません。
- 社内拡張辞書は同梱していません。必要な場合はローカル拡張辞書として差し替えてください。
- `summarize_test_metrics.py` のデフォルト入出力パスは公開版向けに相対パス化していますが、実運用では `--config_json` で明示指定するのを推奨します。
