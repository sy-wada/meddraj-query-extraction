# coding=utf-8
import argparse
from itertools import chain
import math
import pandas as pd
import numpy as np

import yaml
from typing import Any
from pathlib import Path
import joblib
import faiss
import torch
from tqdm.auto import tqdm

from text_processing_utils import get_embeddings_from_queries

def create_ranking_for_predictions(
    queries: list[str],
    mdra_embed_dict: dict[str, list[int]],
    faiss_index: faiss.IndexFlatIP,
    faiss_tbl: pd.DataFrame,
    n_tops: int,
    config: dict
    ) -> dict[str, pd.DataFrame]:
    """
    全クエリに対して、MedDRA/J用語集と類似最近傍探索を使って候補を提示する。
    MedDRA/J用語集に収録された表現であれば、返す候補数は一つ。
    収録されていない場合、最近傍探索を使って候補を複数返す。
    queries:
        検索対象の文字列リスト
    mdra_embed_dict:
        MedDRA/J用語集の辞書。
        keyは日本語カレンシーに採用されているllt_kanjiと日本語シノニム、
        valueは上記に対応したllt_code
    faiss_index;
        faissの内積計算用index
        configで`use_shionogi_db=True`の場合は、自社DBが追加された状態になる
    faiss_tbl:
        index_string, llt_codes
    n_tops:
        faissから抽出する候補数。
        LLTが重複する場合はより上位のスコアを採用するため、最終的に提示するのはこの値よりも少なくなる。
    """
    from transformers import AutoModel, AutoTokenizer
    model_path = config['model']['embeddings'].get('model_name_or_path', config['model']['embeddings']['embedding_model_path'])
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    mdra_embed_dict = {key: [value] if isinstance(value, int) else value
                        for key, value in mdra_embed_dict.items()}

    device = torch.device('cpu')

    if 'GLuCoSE-base-ja-v2' in model_path:
        q_prefix = 'query: '
        print(f'prefixが必要なモデル{model_path}を検出しました。\n全クエリにprefix`query: `を付与して埋め込み表現を獲得します。')
    else:
        q_prefix = ''
    unique_queries = list(set(queries))
    result = {}

    if not config['evaluate']['apply_similarity_search_to_all_queries']:
        result.update(
                {q: pd.Series({'index_string': q,
                            'llt_codes': mdra_embed_dict[q],
                            'score': 1.0}).to_frame().T
                    for q in unique_queries if q in mdra_embed_dict.keys()}
        )
        queries_for_faiss = [q for q in unique_queries if q not in mdra_embed_dict.keys()]
    else:
        queries_for_faiss = unique_queries
    q_embeddings = get_embeddings_from_queries(
                        queries=queries_for_faiss,
                        model_input_keys=config['model']['embeddings']['model_input_keys'],
                        model=model, tokenizer=tokenizer,
                        prefix=q_prefix, batch_size=config['model']['embeddings']['batch_size'], device=device
                        )
    distances, indices = faiss_index.search(q_embeddings, n_tops)
    result.update(
        {q: faiss_tbl.loc[indices[i]].assign(score=distances[i])
         for i, q in enumerate(queries_for_faiss)}
    )
    return {query: (result[query].explode(column='llt_codes')
                                .drop_duplicates(subset='llt_codes', keep='first')
                    ) for query in queries}

def average_precision(relevant_labels, predicted_labels):
    relevant_labels = set(relevant_labels)
    if len(relevant_labels) == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, label in enumerate(predicted_labels):
        if label in relevant_labels:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i

    if hits == 0:
        return 0.0
    else:
        return sum_precisions / len(relevant_labels)

def average_precision_at_k(relevant_labels, predicted_labels, k_eval: int | None):
    if k_eval is None:
        return average_precision(relevant_labels, predicted_labels)
    return average_precision(relevant_labels, predicted_labels[:k_eval])

def reciprocal_rank(relevant_labels, predicted_labels):
    relevant_labels = set(relevant_labels)
    for i, label in enumerate(predicted_labels):
        if label in relevant_labels:
            return 1.0 / (i + 1)
    return 0.0

def reciprocal_rank_at_k(relevant_labels, predicted_labels, k_eval: int | None):
    if k_eval is None:
        return reciprocal_rank(relevant_labels, predicted_labels)
    return reciprocal_rank(relevant_labels, predicted_labels[:k_eval])

def recall(relevant_labels, predicted_labels):
    relevant_labels = set(relevant_labels)
    if len(relevant_labels) == 0:
        return 0.0
    predicted_labels = set(predicted_labels)
    hits = len(relevant_labels.intersection(predicted_labels))
    return hits / len(relevant_labels)

def recall_at_k(relevant_labels, predicted_labels, k_eval: int | None):
    if k_eval is None:
        return recall(relevant_labels, predicted_labels)
    return recall(relevant_labels, predicted_labels[:k_eval])
    
def _gain_from_grade(grade: float | int | None, gain_fn):
    if grade is None:
        return 1.0
    return gain_fn(grade)

def ndcg(relevant_labels, predicted_labels, relevant_gains: dict | None = None, gain_fn=None):
    """
    graded対応のnDCG。relevant_gains が与えられない場合は二値ゲイン(1/0)で評価する。
    gain_fn が与えられない場合は gain(r)=2^r-1 を使用。
    """
    if gain_fn is None:
        gain_fn = lambda r: (2 ** r) - 1

    if relevant_gains is None:
        relevant_set = set(relevant_labels)
        if len(relevant_set) == 0:
            return 0.0
        dcg = 0.0
        for i, label in enumerate(predicted_labels):
            if label in relevant_set:
                dcg += 1.0 / math.log2(i+2)
        R = len(relevant_set)
        idcg = sum([1.0 / math.log2(rank+2) for rank in range(R)])
        return 0.0 if idcg == 0.0 else dcg / idcg

    # graded: relevant_gains は {label: grade}
    if len(relevant_gains) == 0:
        return 0.0
    # DCG
    dcg = 0.0
    for i, label in enumerate(predicted_labels):
        if label in relevant_gains:
            dcg += _gain_from_grade(relevant_gains[label], gain_fn) / math.log2(i+2)
    # IDCG
    ideal_gains = sorted([_gain_from_grade(g, gain_fn) for g in relevant_gains.values()], reverse=True)
    idcg = 0.0
    for rank, g in enumerate(ideal_gains):
        idcg += g / math.log2(rank+2)
    return 0.0 if idcg == 0.0 else dcg / idcg

def ndcg_at_k(relevant_labels, predicted_labels, k_eval: int | None, relevant_gains: dict | None = None, gain_fn=None):
    if k_eval is None:
        return ndcg(relevant_labels, predicted_labels, relevant_gains=relevant_gains, gain_fn=gain_fn)
    if gain_fn is None:
        gain_fn = lambda r: (2 ** r) - 1
    if relevant_gains is None:
        relevant_set = set(relevant_labels)
        if len(relevant_set) == 0:
            return 0.0
        truncated = predicted_labels[:k_eval]
        dcg = 0.0
        for i, label in enumerate(truncated):
            if label in relevant_set:
                dcg += 1.0 / math.log2(i+2)
        R = len(relevant_set)
        up_to = min(k_eval, R)
        idcg = sum([1.0 / math.log2(rank+2) for rank in range(up_to)])
        return 0.0 if idcg == 0.0 else dcg / idcg
    # graded
    if len(relevant_gains) == 0:
        return 0.0
    truncated = predicted_labels[:k_eval]
    dcg = 0.0
    for i, label in enumerate(truncated):
        if label in relevant_gains:
            dcg += _gain_from_grade(relevant_gains[label], gain_fn) / math.log2(i+2)
    ideal_gains = sorted([_gain_from_grade(g, gain_fn) for g in relevant_gains.values()], reverse=True)[:k_eval]
    idcg = 0.0
    for rank, g in enumerate(ideal_gains):
        idcg += g / math.log2(rank+2)
    return 0.0 if idcg == 0.0 else dcg / idcg

def calculate_ir_metrics(
    relevant_codes: set | list,
    predicted_codes: list,
    k_eval: int | None = None,
    relevant_gains: dict | None = None,
    gain_fn=None
    ) -> dict:
    ap = average_precision_at_k(relevant_codes, predicted_codes, k_eval)
    rr = reciprocal_rank_at_k(relevant_codes, predicted_codes, k_eval)
    rec = recall_at_k(relevant_codes, predicted_codes, k_eval)
    my_ndcg = ndcg_at_k(relevant_codes, predicted_codes, k_eval, relevant_gains=relevant_gains, gain_fn=gain_fn)
    return {
        'AP': ap,
        'RR': rr,
        'Recall': rec,
        'nDCG': my_ndcg
    }



# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--reference_path',
#     required=True
# )
# parser.add_argument(
#     '--output_path',
#     required=True
# )
# parser.add_argument(
#     '--use_shionogi_db',
#     help='塩野義DB検索拡張を適用するかどうか。',
#     action='store_true'
# )
# parser.add_argument(
#     '--apply_similarity_search_to_all_queries',
#     help='MedDRA/J用語集に収録されていても、設定閾値に従ってそれ以外の類似候補を提示するか。',
#     action='store_true'
# )
# parser.add_argument(
#     '--calculate_metrics_with_PTlevel',
#     help='評価指標の算出をPTレベルで行う（デフォルトはLLTレベル）。',
#     action='store_true'
# )
# parser.add_argument(
#     '--restrict-to-jp-currency',
#     help='true_codesを日本語カレンシーに限定するかどうか。',
#     action='store_true'
# )
# parser.add_argument(
#     '--k_eval',
#     type=int,
#     default=20,
#     help='評価カットオフ K_eval（既定: 20）。評価は常に先頭K_evalのみで実施する。'
# )
# args = parser.parse_args()

def load_config(
    path: str
    ) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# MedDRA/Jのコードから日本語カレンシーの判定を得ているLLT codeを選択する関数
def select_jp_currency_llt(
    codes: list | str | int | float,
    code2info: dict[int, dict[str, Any]]
    ) -> list[str]:
    if isinstance(codes, str):
        codes = str(codes).split(',')
    elif isinstance(codes, float):
        return codes    # pandasのNaNをそのまま帰す
    result = []
    for code in codes:
        if 'Y' in code2info[int(code)]['llt_jcurr']:
            result.append(code)
    if result:
        return ','.join(result)
    else:
        return np.nan

def main(config, args):
    # 引数で上書き
    config['evaluate']['use_shionogi_db'] = args.use_shionogi_db
    config['evaluate']['apply_similarity_search_to_all_queries'] = args.apply_similarity_search_to_all_queries
    reference_path = Path(args.reference_path)
    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    top_k = config['evaluate']['top_k']
    score_threshold = config['evaluate']['threshold']
    k_eval = int(args.k_eval)

    # MedDRA 関連
    mdra_ver = Path(config['mdra']['path']).name.lower()
    mdra_cache_dir = Path('./cache') / mdra_ver
    mdra_cache_path = mdra_cache_dir / 'mdra_dict.joblib'
    model_path = config['model']['embeddings'].get('model_name_or_path', config['model']['embeddings']['embedding_model_path'])
    md_hier, code2name, name2code, cd_llt2pt, term2info, code2info = joblib.load(mdra_cache_path)

    print('Load files..')
    df = pd.read_excel(reference_path, dtype=str)

    # true_codes を日本語カレンシーに限定
    if args.restrict_to_jp_currency and 'true_codes' in df.columns:
        df['true_codes'] = df['true_codes'].apply(lambda x: select_jp_currency_llt(x, code2info))

    # Faiss / Embeddings 準備
    mdra_embed_dict = joblib.load(mdra_cache_dir / f'{model_path.replace("/", "_")}.embed.dict.joblib')
    mdra_faiss_index = faiss.read_index(str(mdra_cache_dir / f'{model_path.replace("/", "_")}.embed.faiss_index'))
    faiss_index = mdra_faiss_index
    faiss_tbl = pd.DataFrame({
        'index_string': mdra_embed_dict.keys(),
        'llt_codes': [[code] for code in mdra_embed_dict.values()]
    })

    use_shionogi_db = config['evaluate']['use_shionogi_db']
    if use_shionogi_db:
        sheet_path = Path(config['mdra']['shionogi_path'])
        cache_dir = mdra_cache_dir.parent / f'{mdra_ver}_{sheet_path.stem}'
        exception_list_path = config['mdra']['exceptions_for_preventing_data_leak']
        stem_name = f'{model_path.replace("/", "_")}.exceptions_for_preventing_data_leak={int(bool(exception_list_path))}'
        additional_embeds = np.load(cache_dir / f'{stem_name}.embed.npy')
        additional_embeds_dict = joblib.load(cache_dir / f'{stem_name}.embed.dict.joblib')
        faiss_index.add(additional_embeds)
        faiss_tbl = pd.concat([
            faiss_tbl, pd.DataFrame({
                'index_string': additional_embeds_dict.keys(),
                'llt_codes': additional_embeds_dict.values()
            })],
            ).reset_index(drop=True)

    # ae_preds（改行区切り）からユニーククエリを抽出
    queries = list(set(chain.from_iterable(df['ae_preds'].dropna().apply(lambda x: x.split('\n')).tolist())))
    ranking_results = create_ranking_for_predictions(
            queries=queries,
            mdra_embed_dict=mdra_embed_dict, faiss_index=faiss_index, faiss_tbl=faiss_tbl, n_tops=200, config=config
        )

    def modify_ranking(
        df_ranking: pd.DataFrame | None,
        top_k: int,
        score_threshold: float,
        k_eval: int | None
    ) -> pd.DataFrame:
        if df_ranking is None:
            return pd.DataFrame(columns=['index_string', 'llt_codes', 'score'])
        else:
            return (df_ranking[df_ranking['score'] >= score_threshold]
                        .drop_duplicates(subset='llt_codes')
                        .head(min(top_k, k_eval) if k_eval is not None else top_k))
    
    results = []
    for keys in df['ae_preds'].fillna('').apply(lambda x: x.split('\n')).to_list():
        items = [modify_ranking(ranking_results.get(item, None), top_k, score_threshold, k_eval) for item in keys]
        results.append(pd.concat(items).sort_values(by='score', ascending=False).drop_duplicates(subset='llt_codes'))

    doc_evals = []
    for (_, row), result in zip(df.iterrows(), results):
        true_codes = row['true_codes'] if 'true_codes' in row else ''
        if isinstance(true_codes, float):
            true_codes = set()
        else:
            true_codes = set([c for c in true_codes.split(',') if c != ''])
        doc_evals.append({
            'doc_id': row['doc_id'],
            'entity_strings': row['entity_strings'] if 'entity_strings' in row and isinstance(row['entity_strings'], str) else '',
            'true_codes': true_codes,
            'pred_codes': result['llt_codes'].astype(str).to_list()
        })

    df_doc = pd.DataFrame(doc_evals)
    df_doc['matched_codes'] = [true_codes.intersection(set(pred_codes))
                                for true_codes, pred_codes in df_doc[['true_codes', 'pred_codes']].values]
    df_doc['missing_codes'] = df_doc['true_codes'] - df_doc['pred_codes'].apply(set)
    df_doc['extra_codes'] = df_doc['pred_codes'].apply(set) - df_doc['true_codes']

    df_doc['TP'] = df_doc['matched_codes'].apply(len)
    df_doc['FP'] = df_doc['extra_codes'].apply(len)
    df_doc['FN'] = df_doc['missing_codes'].apply(len)
    df_doc = df_doc.join(pd.DataFrame([calculate_ir_metrics(relevant_codes=true_codes,
                                        predicted_codes=pred_codes, k_eval=k_eval)
                for true_codes, pred_codes in df_doc[['true_codes', 'pred_codes']].values]))

    cm = df_doc[['TP', 'FP', 'FN']].sum()
    tp = cm['TP']
    fp = cm['FP']
    fn = cm['FN']
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    doc_metrics = {
        'top_k': top_k,
        'score_thoreshold': f'{score_threshold:.2f}',
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': prec,
        'recall': rec,
        'f1': (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0,
    }
    # 参考: fβ 派生指標
    for beta, key in [(2.0, 'f2'), (0.5, 'f05')]:
        if prec + rec == 0:
            val = 0.0
        else:
            beta2 = beta * beta
            val = (1 + beta2) * (prec * rec) / (beta2 * prec + rec)
        doc_metrics[key] = val
    doc_metrics.update(df_doc[['AP', 'RR', 'nDCG']].mean().rename({'AP': 'MAP', 'RR': 'MRR'}).to_dict())
    for f_name in ['f1', 'f2', 'f05']:
        if f_name in doc_metrics:
            doc_metrics[f'{f_name}-nDCG'] = math.sqrt(doc_metrics[f_name] * doc_metrics['nDCG']) if doc_metrics['nDCG'] > 0 else 0.0
            doc_metrics[f'{f_name}-MAP'] = math.sqrt(doc_metrics[f_name] * doc_metrics['MAP']) if doc_metrics['MAP'] > 0 else 0.0

    # Excel出力
    with pd.ExcelWriter(output_path / \
                        ('metrics_for_llm' + \
                            f'.y_pred={reference_path.stem}' + \
                            f'.K_eval={k_eval}' + \
                            f'.UseShionogiDB={bool(use_shionogi_db)}' + \
                            f'.ApplySimilaritySearchToAllQueries={bool(args.apply_similarity_search_to_all_queries)}' + \
                            f'.CalculateMetricsWithPTlevel={bool(args.calculate_metrics_with_PTlevel)}' + \
                            f'.RestrictToJpCurrency={bool(args.restrict_to_jp_currency)}' + \
                            '.xlsx')
                            ) as writer:
        pd.Series(doc_metrics).to_excel(writer, sheet_name='summary-document_level')
        # ドキュメント単位の出力
        for col_name in ['true_codes', 'pred_codes', 'matched_codes', 'missing_codes', 'extra_codes']:
            df_doc[col_name] = df_doc[col_name].apply(lambda x: ','.join(sorted(list(x))))
        df_doc.to_excel(writer, index=False, sheet_name='result-document_level')

if __name__ == "__main__":
    config = load_config('./config.yml')
    main(config, args)
