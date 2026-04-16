[Japanese](#japanese) | [English](#english)

<a id="japanese"></a>

# 外部資産の配置

このディレクトリは、公開リポジトリに含めない外部資産の配置場所を説明するためのものです。

## 想定ディレクトリ構成

```text
external_assets/
├── meddra/
│   ├── mdra_format.json
│   ├── MDRA_J/
│   │   ├── *.asc
│   │   └── ...
│   └── J_Synonym/
│       └── LLT_SYN.asc
└── local_extension/
    └── extension_terms.xlsx
```

## 各ファイルの役割

- `meddra/mdra_format.json`
  - `create_cache.py` が `.asc` ファイルのカラム構造を読むための定義
- `meddra/MDRA_J/`
  - MedDRA/J の `.asc` 一式
- `meddra/J_Synonym/LLT_SYN.asc`
  - 日本語シノニム辞書
- `local_extension/extension_terms.xlsx`
  - 任意のローカル拡張辞書
  - 公開版では社内拡張辞書を同梱していないため、必要なら同等形式のローカル辞書を自前で配置

## 注意

- MedDRA/J 辞書や拡張辞書はライセンス・機密性の都合で本リポジトリに含めていません。
- `config.yml` のキー名に `shionogi_path` が残っていますが、公開版では任意のローカル拡張辞書へのパスとして扱ってください。

<a id="english"></a>

# External Asset Placement

This directory explains where to place external assets that are not included in the public repository.

## Expected Directory Structure

```text
external_assets/
├── meddra/
│   ├── mdra_format.json
│   ├── MDRA_J/
│   │   ├── *.asc
│   │   └── ...
│   └── J_Synonym/
│       └── LLT_SYN.asc
└── local_extension/
    └── extension_terms.xlsx
```

## Role of Each File

- `meddra/mdra_format.json`
  - Definitions used by `create_cache.py` to read the column structure of `.asc` files
- `meddra/MDRA_J/`
  - The full set of MedDRA/J `.asc` files
- `meddra/J_Synonym/LLT_SYN.asc`
  - Japanese synonym dictionary
- `local_extension/extension_terms.xlsx`
  - Optional local extension dictionary
  - The public version does not bundle any internal extension dictionary, so place your own local dictionary in a compatible format if needed

## Notes

- The MedDRA/J dictionary and extension dictionaries are not included in this repository due to licensing and confidentiality constraints.
- The key name `shionogi_path` remains in `config.yml`, but in the public version it should be treated as the path to an arbitrary local extension dictionary.
