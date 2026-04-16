[Japanese](#japanese) | [English](#english)

<a id="japanese"></a>

# `mdra_suggest/cache` について

このディレクトリには、`mdra_suggest/create_cache.py` 実行後に生成されるキャッシュを配置します。公開リポジトリにはキャッシュ本体を含めません。

## 代表的な生成物

`config.yml` の `mdra.path` が `./external_assets/meddra/MDRA_J` の場合、以下のような構成を想定します。

```text
mdra_suggest/cache/
└── mdra_j/
    ├── mdra_dict.joblib
    ├── cl-nagoya_ruri-v3-310m.embed.dict.joblib
    ├── cl-nagoya_ruri-v3-310m.embed.faiss_index
    └── ...
```

拡張辞書を有効化した場合は、追加で以下のようなキャッシュが生成されます。

```text
mdra_suggest/cache/
└── mdra_j_extension_terms/
    ├── cl-nagoya_ruri-v3-310m.exceptions_for_preventing_data_leak=0.embed.npy
    ├── cl-nagoya_ruri-v3-310m.exceptions_for_preventing_data_leak=0.embed.dict.joblib
    └── ...
```

ファイル名は `config.yml` の埋め込みモデル名や拡張辞書ファイル名に依存します。

<a id="english"></a>

# About `mdra_suggest/cache`

This directory stores caches generated after running `mdra_suggest/create_cache.py`. The public repository does not include the cache contents themselves.

## Typical Generated Files

If `mdra.path` in `config.yml` is set to `./external_assets/meddra/MDRA_J`, the structure is expected to look like this:

```text
mdra_suggest/cache/
└── mdra_j/
    ├── mdra_dict.joblib
    ├── cl-nagoya_ruri-v3-310m.embed.dict.joblib
    ├── cl-nagoya_ruri-v3-310m.embed.faiss_index
    └── ...
```

If the extension dictionary is enabled, additional caches like the following are generated:

```text
mdra_suggest/cache/
└── mdra_j_extension_terms/
    ├── cl-nagoya_ruri-v3-310m.exceptions_for_preventing_data_leak=0.embed.npy
    ├── cl-nagoya_ruri-v3-310m.exceptions_for_preventing_data_leak=0.embed.dict.joblib
    └── ...
```

File names depend on the embedding model name in `config.yml` and the extension dictionary file name.
