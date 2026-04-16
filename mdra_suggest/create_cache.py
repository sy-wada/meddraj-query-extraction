"""
20251231
    全階層を対象としたキャッシュ構築をするオプションを追加
20251210
    RL用に移植
20250410
    日本語カレンシーが存在しない場合、代わりに'N/A'を設定するようにした
20241213
    文単位の入力によるsuggestを見据えたキャッシュ構築
    埋め込み表現の保存を、huggingface datasetsの保存(pyarrow)から、faissのディスク保存に変更
20240930
    `pkshatech/GLuCoSE-base-ja-v2`への対応: the prefix "query: "を追加
20240918
    キャッシュファイル作成について、アプリのための省メモリ動作をoffにした
    塩野義DBでMedDRAに収載されていない表現・llt_codeに関するデータセットをfeather形式で保存するようにした
"""
import json
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import Any

import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from tqdm.auto import tqdm

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from text_processing_utils import get_embeddings_from_queries
import faiss

# 各辞書から削除しないkey（アプリで使う情報）を指定
used_keys = {
    'md_hier': [
        'hlt_code',
        'hlgt_code',
        'soc_code',
        'pt_name',
        'llt_codes'
    ],
    'code2name': [
        'pt_code',
        'hlt_code',
        'hlgt_code',
        'soc_code'
    ],
    'name2code': [
        'llt_kanji'
    ]
}

def load_config(
    path: str
    ) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_mdra_dictionaries(
    path: Path
    ) -> tuple[dict[Any], dict[Any], dict[Any]]:
    md_hier, code2name, name2code, cd_llt2pt, term2info, code2info = joblib.load(path)
    return md_hier, code2name, name2code, cd_llt2pt, term2info, code2info

def check_soc_and_pt_for_llt(
    llt_code: int,
    ref_pt_code: int,
    ref_soc_code: int,
    md_hier: dict[int, dict[str, Any]],
    cd_llt2pt: dict[int, int]
    ) -> bool:
    pt_code = cd_llt2pt[llt_code]
    soc_code = md_hier[pt_code]['soc_code']
    res = False
    if pt_code == ref_pt_code and soc_code == ref_soc_code:
        res = True
    else:
        res = False
    return res

def create_mdra_dict(
        config: dict[str, Any],
        encoding: str,
        remove_unused_keys: bool
) -> tuple[dict]:
    """
    MedDRAのcsvデータ（拡張子は`.asc`で区切り文字は`$`）をpythonで扱いやすいように変換する。
    md_hier:
        階層構造。keyはpt_code、valueには辞書形式で、以下を格納している。
        - hlt_code
        - hlgt_code
        - soc_code　※複数のSOCが紐づく場合、pt_soc_codeに一致するものを選択した
        - pt_name
        - hlt_name
        - hlgt_name
        - soc_name
        - soc_abbrev
        - pt_soc_code
        - llt_codes: pt_codeに紐づく複数のllt_codeをリスト形式で格納
    code2name:
        コードを文字列に変換するための辞書
        - key: llt_code, pt_code, hlt_code, hlgt_code, soc_code
        - value: {code: {'name': [英語の表記], 'kanji': [日本語表記]}}
    name2code:
        文字列をコードに変換するための辞書
        - key: llt_kanji, llt_name, pt_kanji, pt_name, hlt_kanji, hlt_name, hlgt_kanji, hlgt_name, soc_kanji, soc_name
        - value: {文字列: code}
    """
    df = {}
    with open(config['mdra']['format'], 'r', encoding=encoding) as f:
        format = json.load(f)
    p = Path(config['mdra']['path'])
    for path in p.glob('*.asc'):
        if path.name not in format.keys():
            continue
        table, header = [format[path.name][k] for k in ['table', 'header']]
        if not header:
            continue
        df[table] = pd.read_csv(path, sep='$', header=None, encoding=encoding, usecols=range(len(header))).rename(
                            {i: name for i, name in enumerate(header)}, axis=1)

    # 日本語シノニム
    jsyn = pd.read_csv(Path(config['mdra']['syn_path']) / 'LLT_SYN.asc', encoding=encoding, header=None, sep='$', usecols=range(7))
    jsyn.rename({i: name for i, name in enumerate(format['LLT_SYN.asc']['header'])}, axis=1, inplace=True)

    # 1_md_hierarchyの加工：同テーブルを上位概念結合に利用する
    # プライマリーSOCのコードに紐づいたものを使う（primary_soc_fgが"Y"：soc_codeがpt_soc_sodeに一致しているもの）にする
    # Preferred Terms (PT)に集約して管理
    # md_hier: `pt_code`をkeyとした辞書
    print('md_hier: `pt_code`をkeyとした階層構造辞書の構築')
    md_hier = df['1_md_hierarchy'][df['1_md_hierarchy']['primary_soc_fg'] == 'Y']
    md_hier = md_hier[md_hier.columns.drop(['null_field', 'primary_soc_fg'])].set_index('pt_code')
    md_hier = {i: row.to_dict() for i, row in md_hier.iterrows()}

    # pt_codeに対応するllt_codeの整理
    for pt_code in tqdm(md_hier.keys(), desc='Organize llt_code corresponding to pt_code'):
        tbl = df['1_low_level_term']
        md_hier[pt_code]['llt_codes'] = tbl[tbl['pt_code'] == pt_code]['llt_code'].to_list()

    if remove_unused_keys:
        print('md_hier: 使用していない情報を削除')
        for pt_code, obj in md_hier.items():
            for key in list(obj.keys()):
                if key not in used_keys['md_hier']:
                    del obj[key]
    
    # コード-文字列変換辞書
    code2name = defaultdict(dict)
    name2code = defaultdict(dict)

    ## llt, pt, hlt, hlgt, soc
    for cd_name, term in zip(['llt', 'pt', 'hlt', 'hlgt', 'soc'],
                            ['low_level_term', 'pref_term', 'hlt_pref_term', 'hlgt_pref_term', 'soc_term']):
        tbl = df[f'1_{term}']
        tbl_j = df[f'1_{term}_j']
        if cd_name == 'llt':
            tbl_j = pd.concat([tbl_j[['llt_code', 'llt_kanji', 'llt_jcurr']],
                        jsyn.rename({'llt_s_kanji': 'llt_kanji', 'llt_s_jcurr': 'llt_jcurr'},
                            axis=1)[['llt_code', 'llt_kanji', 'llt_jcurr']]])
        tbl_j[f'{cd_name}_kanji'] = tbl_j[f'{cd_name}_kanji'].fillna('').apply(
                            lambda x: unicodedata.normalize('NFKC', x))

        # code2name
        for name, code in tqdm(zip(tbl[f'{cd_name}_name'], tbl[f'{cd_name}_code'], strict=True),
                                total=len(tbl), desc=f'code2name for {cd_name}: code->name dict.'):
            if code not in code2name[f'{cd_name}_code'].keys():
                code2name[f'{cd_name}_code'][code] = {'name': [], 'kanji': []}
            code2name[f'{cd_name}_code'][code]['name'].append(name)
        for name, code in tqdm(zip(tbl_j[f'{cd_name}_kanji'], tbl_j[f'{cd_name}_code'], strict=True),
                                total=len(tbl_j), desc=f'code2name for {cd_name}: code->kanji dict.'):
            code2name[f'{cd_name}_code'][code]['kanji'].append(name)

        # name2code
        for name, code in tqdm(zip(tbl_j[f'{cd_name}_kanji'], tbl_j[f'{cd_name}_code']),
                                total=len(tbl_j), desc=f'name2code for {cd_name}: kanji->code dict.'):
            if name not in name2code[f'{cd_name}_kanji'].keys():
                name2code[f'{cd_name}_kanji'][name] = []
            name2code[f'{cd_name}_kanji'][name].append(code)
        for name, code in tqdm(zip(tbl[f'{cd_name}_name'], tbl[f'{cd_name}_code']),
                            total=len(tbl), desc=f'name2code for {cd_name}: name->code dict.'):
            if name not in name2code[f'{cd_name}_name'].keys():
                name2code[f'{cd_name}_name'][name] = []
            name2code[f'{cd_name}_name'][name].append(code)

        # term2info: 日本語llt（シノニム含む）をkeyとして、カレンシーなどの情報を整理した辞書の作成
        # code2info: llt_codeをkeyとして、日本語lltやカレンシーなどの情報を整理した辞書の作成
        if cd_name == 'llt':
            code2info = {code: {'llt_name': name, 'llt_currency': c}
                            for code, name, c in zip(tbl['llt_code'],
                                                    tbl['llt_name'],
                                                    tbl['llt_currency'])}
            code2kanji_withJC = (lambda df:
                                dict(zip(df['llt_code'],
                                        df['llt_kanji'])))(tbl_j[tbl_j['llt_jcurr'] == 'Y'])

            term2info = {}
            for code, kanji, jcurr in tqdm(zip(tbl_j['llt_code'], tbl_j['llt_kanji'], tbl_j['llt_jcurr']),
                                           total=len(tbl_j),
                                           desc='term2info, code2info: llt_kanji/code->info with currency.'):
                if kanji not in term2info.keys():
                    term2info[kanji] = {
                        'llt_code': [],
                        'llt_name': [],
                        'llt_currency': '',
                        'llt_jcurr': '',
                        'llt_kanji': ''
                    }
                llt_name = code2info[code]['llt_name']
                llt_currency = code2info[code]['llt_currency']
                term2info[kanji]['llt_code'].append(code)
                term2info[kanji]['llt_name'].append(llt_name)
                term2info[kanji]['llt_currency'] += llt_currency
                term2info[kanji]['llt_jcurr'] += jcurr
                if jcurr == 'S':
                    if code in code2kanji_withJC.keys():
                        term2info[kanji]['llt_kanji'] = code2kanji_withJC[code]
                    else:
                        print(f'{kanji}（{llt_name=}, {code=}, {llt_currency=}, {jcurr=}）には対応する日本語カレンシーが存在しません。\n代わりに{dummy_text}を設定します。')
                        term2info[kanji]['llt_kanji'] = dummy_text
                
                # code2info
                if 'llt_kanji' not in code2info[code].keys():
                    code2info[code].update({
                        'llt_kanji': [],
                        'llt_jcurr': '',
                        'jcurrency_kanji': ''
                    })
                
                code2info[code]['llt_kanji'].append(kanji)
                code2info[code]['llt_jcurr'] += jcurr
                if jcurr == 'S':
                    if code in code2kanji_withJC.keys():
                        code2info[code]['jcurrency_kanji'] = code2kanji_withJC[code]
                    else:
                        code2info[code]['jcurrency_kanji'] = dummy_text
                elif jcurr == 'Y':
                    code2info[code]['jcurrency_kanji'] = code2kanji_withJC[code]
    if remove_unused_keys:
        print('使用していない情報を削除: code2name')
        for key in list(code2name.keys()):
            if key not in used_keys['code2name']:
                del code2name[key]
        print('使用していない情報を削除: name2code')
        for key in list(name2code.keys()):
            if key not in used_keys['name2code']:
                del name2code[key]    

    # lltコードをptコードに変換する辞書の準備
    tmp = pd.concat([pd.DataFrame({'pt_code': pt_code, 'llt_code': obj['llt_codes']}) for pt_code, obj in md_hier.items()])
    cd_llt2pt = {llt: pt for llt, pt in zip(tmp['llt_code'], tmp['pt_code'])}

    return md_hier, code2name, name2code, cd_llt2pt, term2info, code2info

def create_mdra_cache(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: dict,
        dict_path: dict
    ):
    remove_unused_keys = config['mdra']['remove_unused_keys']
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # config読み込み
    cache_dir = dict_path['cache_dir']
    mdra_ver = Path(config['mdra']['path']).name.lower()
    cache_path = dict_path['cache_path']
    embed_path = dict_path['mdra_embed_path']
    model_path = config['model']['embeddings']['embedding_model_path']
    embed_dict_path = cache_dir / f'{model_path.replace("/", "_")}.embed.dict.joblib'

    # MedDRA辞書の構築
    print('MedDRA辞書の構築:')
    try:
        mdra_dicts = create_mdra_dict(
            config=config,
            encoding='cp932',
            remove_unused_keys=remove_unused_keys
        )       

        joblib.dump(mdra_dicts, cache_path, compress=True)
        # llt_codeをpt_codeに変換する辞書の利用頻度は高いので、個別保存しておく
        _, _, _, cd_llt2pt, _, _ = mdra_dicts
        joblib.dump(cd_llt2pt, cache_path.parent.joinpath(cache_path.stem + '.cd_llt2pt.joblib'), compress=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise ValueError("A ValueError was raised after catching an exception") from e
    print(f'MedDRA辞書 {mdra_ver}保存完了: {cache_path.absolute()}')

    _, _, _, _, term2info, _ = mdra_dicts
    # MedDRA用語の埋め込み表現を取得
    # 20240930: MedDRAのj_curr = Y or Sに検索対象を絞り込んで取得する
    lst_term_code = []
    lst_llt_jcurr = []
    for term, obj in term2info.items():
        for code in obj['llt_code']:
            lst_term_code.append(f'{term}_+_{code}')
        lst_llt_jcurr.extend(obj['llt_jcurr'])
    df = pd.DataFrame({'key': lst_term_code, 'llt_jcurr': lst_llt_jcurr})
    df = df[df['llt_jcurr'] != 'N']
    df['llt_kanji'] = df['key'].apply(lambda x: x.split('_+_')[0])
    df['llt_code'] = df['key'].apply(lambda x: int(x.split('_+_')[1]))
    embed_dict = {k: v for k, v in zip(df['llt_kanji'], df['llt_code'])}
    # ver27.0では一対一対応のため、key重複による削除は発生しないことを確認済み
    try:
        # mdra_embed = create_embeddings(config=config, terms=list(term2info.keys()), device=device)
        q_prefix = config.get("model", {}).get("embeddings", {}).get("embedding_model_q_prefix", "")
        if q_prefix:
            print(f'prefix `{q_prefix}` を使用して埋め込み表現を計算します。')

        embeddings = get_embeddings_from_queries(
                        queries=list(embed_dict.keys()),
                        model_input_keys=config['model']['embeddings']['model_input_keys'],
                        model=model, tokenizer=tokenizer,
                        prefix=q_prefix, batch_size=config['model']['embeddings']['batch_size'], device=device
                        )
        print('SAVE TO DISK.')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)    # 内積
        index.add(embeddings)
        faiss.write_index(index, str(embed_path))
        joblib.dump(embed_dict, embed_dict_path, compress=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise ValueError("A ValueError was raised after catching an exception") from e
    
    print(f'MedDRAの辞書構築と埋込み表現獲得処理は正常に終了しました:  {embeddings.shape=:}\n{str(embed_path)=:}\n{str(embed_dict_path)=:}')

def _iter_code_kanji_pairs(
        tbl: pd.DataFrame,
        code_col: str,
        kanji_col: str
    ):
    """
    DataFrameから (normalized_kanji, code:int) を逐次生成する。
    - kanjiはNFKC正規化
    - codeは数値化できないものを除外
    - 空文字は除外
    """
    if code_col not in tbl.columns or kanji_col not in tbl.columns:
        return
    codes = pd.to_numeric(tbl[code_col], errors='coerce')
    for kanji, code in zip(tbl[kanji_col].fillna(''), codes):
        if pd.isna(code):
            continue
        s = unicodedata.normalize('NFKC', str(kanji))
        if not s:
            continue
        yield s, int(code)

def create_mdra_all_levels_cache(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: dict,
        dict_path: dict
    ):
    """
    MedDRA標準用語集に収録されている全階層（llt/pt/hlt/hlgt/soc）の *_kanji 表現を対象に、
    別ディレクトリ `cache/<mdra_ver>__all_levels` に埋め込みキャッシュを保存する。

    - 既存のLLTキャッシュ（cache/<mdra_ver>）は維持・不変更
    - FAISSの行順復元のため、`*.embed.surfaces.joblib` を必ず保存する
    """
    remove_unused_keys = config['mdra']['remove_unused_keys']
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    mdra_ver = Path(config['mdra']['path']).name.lower()
    base_cache_dir: Path = dict_path['cache_dir']
    cache_dir = base_cache_dir.parent / f'{mdra_ver}__all_levels'
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)

    model_path = config['model']['embeddings']['embedding_model_path']
    cache_path = cache_dir / 'mdra_dict.joblib'
    embed_path = cache_dir / f'{model_path.replace("/", "_")}.embed.faiss_index'
    embed_dict_path = cache_dir / f'{model_path.replace("/", "_")}.embed.dict.joblib'
    surfaces_path = cache_dir / f'{model_path.replace("/", "_")}.embed.surfaces.joblib'

    if cache_path.exists() and embed_path.exists() and embed_dict_path.exists() and surfaces_path.exists():
        print(f'全階層埋め込みキャッシュは作成済みです。スキップします: {str(cache_dir)=}')
        return

    # MedDRA辞書の構築（ディレクトリ単体で完結させるため保存しておく）
    print('MedDRA辞書（all_levels）の構築:')
    try:
        mdra_dicts = create_mdra_dict(
            config=config,
            encoding='cp932',
            remove_unused_keys=remove_unused_keys
        )
        joblib.dump(mdra_dicts, cache_path, compress=True)
        # 高頻度利用の辞書は個別保存（既存挙動に合わせる）
        _, _, _, cd_llt2pt, _, _ = mdra_dicts
        joblib.dump(cd_llt2pt, cache_path.parent.joinpath(cache_path.stem + '.cd_llt2pt.joblib'), compress=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise ValueError("A ValueError was raised after catching an exception") from e
    print(f'MedDRA辞書（all_levels）保存完了: {cache_path.absolute()}')

    # --- 全階層(kanji)表現の収集 ---
    print('MedDRA全階層(kanji)表現の収集:')
    with open(config['mdra']['format'], 'r', encoding='cp932') as f:
        format = json.load(f)
    p = Path(config['mdra']['path'])
    df = {}
    for path in p.glob('*.asc'):
        if path.name not in format.keys():
            continue
        table, header = [format[path.name][k] for k in ['table', 'header']]
        if not header:
            continue
        df[table] = pd.read_csv(path, sep='$', header=None, encoding='cp932', usecols=range(len(header))).rename(
                            {i: name for i, name in enumerate(header)}, axis=1)

    # 日本語シノニム（LLT）
    jsyn = pd.read_csv(Path(config['mdra']['syn_path']) / 'LLT_SYN.asc', encoding='cp932', header=None, sep='$', usecols=range(7))
    jsyn.rename({i: name for i, name in enumerate(format['LLT_SYN.asc']['header'])}, axis=1, inplace=True)

    from collections import defaultdict as _defaultdict
    meta: dict[str, dict[str, set[int]]] = _defaultdict(lambda: _defaultdict(set))

    # LLT（既存ロジックに合わせて llt_jcurr != 'N' のみ対象）
    if '1_low_level_term_j' in df:
        llt_j = df['1_low_level_term_j']
        llt_j = pd.concat([llt_j[['llt_code', 'llt_kanji', 'llt_jcurr']],
                    jsyn.rename({'llt_s_kanji': 'llt_kanji', 'llt_s_jcurr': 'llt_jcurr'},
                        axis=1)[['llt_code', 'llt_kanji', 'llt_jcurr']]])
        llt_j['llt_kanji'] = llt_j['llt_kanji'].fillna('').apply(lambda x: unicodedata.normalize('NFKC', x))
        if 'llt_jcurr' in llt_j.columns:
            llt_j = llt_j[llt_j['llt_jcurr'] != 'N']
        for surface, code in _iter_code_kanji_pairs(llt_j, 'llt_code', 'llt_kanji'):
            meta[surface]['llt'].add(code)

    # PT/HLT/HLGT/SOC（全表現）
    level_specs = [
        ('pt', '1_pref_term_j', 'pt_code', 'pt_kanji'),
        ('hlt', '1_hlt_pref_term_j', 'hlt_code', 'hlt_kanji'),
        ('hlgt', '1_hlgt_pref_term_j', 'hlgt_code', 'hlgt_kanji'),
        ('soc', '1_soc_term_j', 'soc_code', 'soc_kanji'),
    ]
    for level, table_name, code_col, kanji_col in level_specs:
        if table_name not in df:
            continue
        tbl_j = df[table_name].copy()
        if kanji_col in tbl_j.columns:
            tbl_j[kanji_col] = tbl_j[kanji_col].fillna('').apply(lambda x: unicodedata.normalize('NFKC', x))
        for surface, code in _iter_code_kanji_pairs(tbl_j, code_col, kanji_col):
            meta[surface][level].add(code)

    # 埋め込み対象は表記(surface)のユニーク集合
    surfaces = sorted(meta.keys())
    meta_dump = {
        surface: {level: sorted(list(codes)) for level, codes in levels.items()}
        for surface, levels in meta.items()
    }

    # --- 埋め込み計算＆保存 ---
    try:
        q_prefix = config.get("model", {}).get("embeddings", {}).get("embedding_model_q_prefix", "")
        if q_prefix:
            print(f'prefix `{q_prefix}` を使用して埋め込み表現を計算します。')

        embeddings = get_embeddings_from_queries(
                        queries=surfaces,
                        model_input_keys=config['model']['embeddings']['model_input_keys'],
                        model=model, tokenizer=tokenizer,
                        prefix=q_prefix, batch_size=config['model']['embeddings']['batch_size'], device=device
                        )
        print('SAVE TO DISK.')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(embed_path))
        joblib.dump(meta_dump, embed_dict_path, compress=True)
        joblib.dump(surfaces, surfaces_path, compress=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise ValueError("A ValueError was raised after catching an exception") from e

    print(
        'MedDRA全階層(kanji)の辞書構築と埋込み表現獲得処理は正常に終了しました:  '
        f'{embeddings.shape=:}\n{str(embed_path)=:}\n{str(embed_dict_path)=:}\n{str(surfaces_path)=:}'
    )

def create_shionogi_cache(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: dict,
        dict_path: dict
    ):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # config読み込み
    mdra_ver = Path(config['mdra']['path']).name.lower()
    mdra_cache_dir = dict_path['cache_dir']
    mdra_cache_path = dict_path['cache_path']
    model_path = config['model']['embeddings']['embedding_model_path']
    sheet_path = Path(config['mdra']['shionogi_path'])
    cache_dir = mdra_cache_dir.parent / f'{mdra_ver}_{sheet_path.stem}'
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    exception_list_path = config['mdra']['exceptions_for_preventing_data_leak']
    stem_name = f'{model_path.replace("/", "_")}.exceptions_for_preventing_data_leak={int(bool(exception_list_path))}'
    additional_embed_path = cache_dir / f'{stem_name}.embed.npy'
    additional_embed_dict_path = cache_dir / f'{stem_name}.embed.dict.joblib'
    if additional_embed_path.exists() and additional_embed_dict_path.exists():
        print(f'MedDRA/J「{mdra_ver}」 x データベースファイル「{sheet_path.name}」 x Embedding Model「{model_path}」の組み合わせは作成済みです。')
        return
    
    print('MedDRA/J辞書の読み込み：')
    md_hier, code2name, name2code, cd_llt2pt, term2info, code2info = load_mdra_dictionaries(path=mdra_cache_path)

    # 塩野義データベースから埋め込み対象の表現を取得する
    # 対象：MedDRA辞書にない表現かつデータでLLTコードが特定出来るもの

    print('データベースの読み込み：')
    sdf = pd.read_excel(sheet_path)
    sdf['読替有害事象'] = sdf['読替有害事象'].fillna('').apply(lambda x: unicodedata.normalize('NFKC', x))
    sdf.drop_duplicates(inplace=True)

    # code2nameを利用してpt, socのname2codeを拡張する
    # ! LLT codeが最初から「読替有害事象」に紐づいていれば、この処理は不要
    print('MedDRA/J辞書と照合：最新LLT, 最新PT, 最新SOCが収載されているかチェックする')
    for key in ['pt_code', 'soc_code']:
        res = {}
        for c, item in code2name[key].items():
            for k in item['kanji']:
                res[k] = c
        name2code[key.replace('_code', '_kanji')] = res

    for key in ['LLT', 'PT', 'SOC']:
        sdf[f'{key.lower()}_code'] = sdf[f'最新{key}'].fillna(''
                                        ).apply(lambda x: name2code[f'{key.lower()}_kanji'].get(
                                                                unicodedata.normalize('NFKC', x), '')
                                        ).replace('', None)

    # pt_code, soc_codeのマッチングが成立したllt_codeを取得する
    lst_res = []
    for _, row in sdf.iterrows():
        if not row['llt_code']:
            lst_res.append(None)
            continue
        res = []
        for code in row['llt_code']:
            if code not in code2info.keys():
                continue
            if check_soc_and_pt_for_llt(
                llt_code=code, ref_pt_code=row['pt_code'], ref_soc_code=row['soc_code'],
                md_hier=md_hier, cd_llt2pt=cd_llt2pt):
                res.append(code)
        lst_res.append(res)
    # 日本語カレンシー、英語カレンシー、どちらも存在しない表現の優先順位を与えてllt_codeを選択する
    result = []
    for codes in lst_res:
        if not codes:
            result.append(None)
            continue
        code_filtered = None
        # 日本語カレンシーに対応しているllt_codeを採用する
        for code in codes:
            if 'Y' in code2info[code]['llt_jcurr']:
                code_filtered = code
                break
        # 対応していない事例しかなくても、英語カレンシーである場合はそのコードを採用する
        if code_filtered is None:
            for code in codes:
                if code2info[code]['llt_currency'] == 'Y':
                    code_filtered = code
                    break
        # それでも対応するものがない場合、検索には有用な表現となってくれるので、リストに格納する
        if code_filtered is None:
            code_filtered = codes[0]
        result.append(code_filtered)

    sdf['llt_code'] = pd.Series(result, index=sdf.index, dtype='Int64')

    # 以下を出力しておくとエラー分析に使うことができる
    sdf[sdf['llt_code'].isna() | sdf['pt_code'].isna() | sdf['soc_code'].isna()].to_excel(cache_dir / f'{stem_name}.irregulars.xlsx')

    # 埋め込み表現取得対象文字列の選択
    # df = sdf[~sdf['llt_code'].isna()][['読替有害事象', 'llt_code']].drop_duplicates()
    # 検証のため、処理後のDataFrameを全て出力するように変更
    df = sdf.loc[sdf[~sdf['llt_code'].isna()][['読替有害事象', 'llt_code']].drop_duplicates().index]
    # 精度評価に使う予定の事例を、管理番号（トリアージ情報の`ADR_NO`に対応）で除外する
    if exception_list_path:
        with open(exception_list_path, 'r', encoding='utf-8') as file:
            exception_list = [line.strip() for line in file]
        # デバッグ目的で対象外になる事例を保存しておく
        df[df['管理番号'].isin(exception_list)].to_excel(cache_dir / f'{stem_name}.exceptions_for_validation.xlsx')
        df = df[~df['管理番号'].isin(exception_list)]
    # 既にMedDRA辞書にllt_codeも含めて収録されている場合は対象外にしておく
    df_not_in_mdra = df[~df['読替有害事象'].isin(term2info.keys())]
    df_in_mdra = df[df['読替有害事象'].isin(term2info.keys())]
    result = []
    for i, row in df_in_mdra.iterrows():
        result.append(row['llt_code'] not in term2info[row['読替有害事象']]['llt_code'])

    df = pd.concat([df_in_mdra[result], df_not_in_mdra]).sort_index()

    embed_dict = defaultdict(list)
    for string, code in zip(df['読替有害事象'], df['llt_code']):
        embed_dict[string].append(code)

    # MedDRA用語の埋め込み表現を取得 for シオノギDB
    try:
        q_prefix = config.get("model", {}).get("embeddings", {}).get("embedding_model_q_prefix", "")
        if q_prefix:
            print(f'prefix `{q_prefix}` を使用して埋め込み表現を計算します。')

        embeddings = get_embeddings_from_queries(
                        queries=list(embed_dict.keys()),
                        model_input_keys=config['model']['embeddings']['model_input_keys'],
                        model=model, tokenizer=tokenizer,
                        prefix=q_prefix, batch_size=config['model']['embeddings']['batch_size'], device=device
                        )
        print('SAVE TO DISK.')
        np.save(additional_embed_path, embeddings, allow_pickle=False)
        embed_dict = {key: [int(v) for v in values] for key, values in embed_dict.items()}
        joblib.dump(embed_dict, additional_embed_dict_path, compress=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise ValueError("A ValueError was raised after catching an exception") from e
    
    print(f'自社データベースに対応する埋込み表現獲得処理は正常に終了しました: {embeddings.shape=:}\n{str(additional_embed_path)=:}\n{str(additional_embed_dict_path)=:}')

def main(config: dict):
    # MedDRAのキャッシュを準備
    mdra_ver = Path(config['mdra']['path']).name.lower()
    cache_dir = Path('./cache') / mdra_ver
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    cache_path = cache_dir / 'mdra_dict.joblib'
    model_path = config['model']['embeddings']['embedding_model_path']
    mdra_embed_path = cache_dir / f'{model_path.replace("/", "_")}.embed.faiss_index'
    dict_path = {
        'cache_dir': cache_dir,
        'cache_path': cache_path,
        'mdra_embed_path': mdra_embed_path
    }
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if config.get('_runtime', {}).get('create_llt_cache', True):
        if not cache_path.exists() or not mdra_embed_path.exists():
            create_mdra_cache(model=model, tokenizer=tokenizer, config=config, dict_path=dict_path)
        else:
            print(f'既に`{mdra_ver}`のMedDRA辞書と埋込表現は作成済みです。\n作成をスキップします。')
    else:
        print('オプション指定により、LLTレベルのMedDRAキャッシュ作成をスキップします。')

    # MedDRA標準用語集の全階層(kanji)用キャッシュを作成（既存キャッシュとは別ディレクトリに保存）
    if config.get('_runtime', {}).get('create_all_levels_cache', True):
        create_mdra_all_levels_cache(model=model, tokenizer=tokenizer, config=config, dict_path=dict_path)
    else:
        print('オプション指定により、全階層(all_levels)キャッシュ作成をスキップします。')

    # 自社データベースのキャッシュを作成
    if config.get('_runtime', {}).get('create_shionogi_cache', True) and config['evaluate']['use_shionogi_db']:
        print('自社データベースに対応する埋め込み表現を獲得します。')
        create_shionogi_cache(model=model, tokenizer=tokenizer, config=config, dict_path=dict_path)
    elif config['evaluate']['use_shionogi_db']:
        print('オプション指定により、塩野義DBキャッシュ作成をスキップします。')

    print('すべての処理が正常に終了しました。')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='./config.yml',
        help='設定ファイル（YAML）のパス。既定は ./config.yml'
    )
    parser.add_argument(
        '--all-levels-only',
        action='store_true',
        help='全階層(all_levels)キャッシュのみ作成し、LLTキャッシュと塩野義DBキャッシュ作成をスキップする。'
    )
    parser.add_argument(
        '--skip-all-levels',
        action='store_true',
        help='全階層(all_levels)キャッシュ作成をスキップする（従来のLLTキャッシュのみ等に使う）。'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dummy_text = 'N/A'
    # 実行時オプションはconfigに埋め込んでmainへ渡す（既存呼び出し互換を保つ）
    config['_runtime'] = {
        'create_all_levels_cache': (not args.skip_all_levels),
        'create_llt_cache': (not args.all_levels_only),
        'create_shionogi_cache': (not args.all_levels_only),
    }
    main(config)
