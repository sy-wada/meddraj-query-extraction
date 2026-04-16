[Japanese](#japanese) | [English](#english)

<a id="japanese"></a>

# `mdra_suggest` について

このディレクトリには、MedDRA/J を用いた埋め込み検索・評価・キャッシュ生成のための最小コードを置いています。

## 含まれるもの

- `create_cache.py`: MedDRA/J から検索キャッシュを作成
- `text_processing_utils.py`: 埋め込み生成と検索補助
- `evaluate_ranking.py`: IR 指標計算の補助
- `config.yml`: 公開用テンプレート設定

## 含まれないもの

- MedDRA/J 辞書本体
- 日本語シノニム辞書
- ローカル拡張辞書
- `cache/` に生成されるキャッシュ

## 設定ファイルの前提

`config.yml` はリポジトリルートを基準にした相対パスを想定しています。標準では以下を参照します。

- `./external_assets/meddra/`
- `./external_assets/local_extension/`
- `./mdra_suggest/cache/`

公開版では `shionogi_path` というキー名をそのまま残していますが、これは任意のローカル拡張辞書を指すプレースホルダです。社内データは含まれていません。

<a id="english"></a>

# About `mdra_suggest`

This directory contains the minimal code required for embedding-based retrieval, evaluation, and cache generation using MedDRA/J.

## Included

- `create_cache.py`: Creates a retrieval cache from MedDRA/J
- `text_processing_utils.py`: Helpers for embedding generation and retrieval
- `evaluate_ranking.py`: Helpers for IR metric calculation
- `config.yml`: Public template configuration

## Not Included

- The MedDRA/J dictionary itself
- Japanese synonym dictionary
- Local extension dictionary
- Caches generated under `cache/`

## Configuration Assumptions

`config.yml` assumes paths relative to the repository root. By default, it refers to:

- `./external_assets/meddra/`
- `./external_assets/local_extension/`
- `./mdra_suggest/cache/`

The public version keeps the key name `shionogi_path` as-is, but it is only a placeholder for an arbitrary local extension dictionary. No internal data is included.
