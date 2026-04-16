#!/usr/bin/env python3
"""
研究計画書（documents/研究計画書.md）に沿って、testの評価指標を集計する。

要点:
- 入力は seed_*/metrics_test_list.csv（= 文書単位の指標）を使用する
- 集約は「文書単位で seed 平均 → 文書平均」（研究計画書 6.2 / 7.1）
- H1/H2 の差分 Δ について、paired bootstrap（必要に応じ階層bootstrap）で 95% CI を算出（研究計画書 7.2 / 7.3）

出力:
- <out_dir>/model_summary.csv
- <out_dir>/delta_summary.csv

使い方（デフォルト設定で実行）:
  python scripts/summarize_test_metrics.py --out_dir outputs/summary_test_metrics --bootstrap 10000

※ 公開版では相対パスの例をデフォルトにしている。実運用では --config_json を使う。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


Metric = Literal["AP", "RR", "Recall", "nDCG"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    run_dirs: list[Path]
    seeds: list[int]


def _default_model_specs() -> list[ModelSpec]:
    # 公開版では相対パスの例をデフォルトにし、実運用では --config_json を推奨する。
    base = Path("./outputs/llm_eval/base")
    base_large = Path("./outputs/llm_eval/base_large")
    peftrl = Path("./outputs/llm_eval/peft_rl")

    ner_refit = Path("./outputs/distill_ner/refit")
    ner_final = Path("./outputs/distill_ner/final")

    return [
        ModelSpec(name="Base", run_dirs=[base], seeds=list(range(0, 30))),
        ModelSpec(name="Base_Large", run_dirs=[base_large], seeds=list(range(0, 30))),
        ModelSpec(name="PEFT_RL", run_dirs=[peftrl], seeds=list(range(0, 30))),
        ModelSpec(name="Distill_NER_filtered", run_dirs=[ner_refit, ner_final], seeds=list(range(0, 10))),
    ]


def _load_config_json(path: Path) -> list[ModelSpec]:
    """
    config_json 形式:
    {
      "models": [
        {"name": "...", "run_dirs": ["...","..."], "seeds": [0,1,2]}
      ]
    }
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    models: list[ModelSpec] = []
    for m in obj["models"]:
        models.append(
            ModelSpec(
                name=str(m["name"]),
                run_dirs=[Path(x) for x in m["run_dirs"]],
                seeds=[int(x) for x in m["seeds"]],
            )
        )
    return models


def _find_seed_csv(run_dirs: list[Path], seed: int) -> Path:
    rel = Path(f"seed_{seed}") / "metrics_test_list.csv"
    for d in run_dirs:
        cand = d / rel
        if cand.exists():
            return cand
    # fallback: 念のため再帰探索（トップ直下にない構造にも対応）
    # ただし遅くなり得るので最後の手段。
    for d in run_dirs:
        if not d.exists():
            continue
        hits = sorted(d.rglob(rel.as_posix()))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"missing metrics_test_list.csv for seed={seed} in run_dirs={run_dirs}")


def _load_seed_df(csv_path: Path, *, model_name: str, seed: int, metrics: list[Metric]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "guid" not in df.columns:
        raise ValueError(f"[{model_name} seed={seed}] 'guid' column missing: {csv_path}")
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"[{model_name} seed={seed}] metric columns missing {missing}: {csv_path}")
    out = df[["guid", *metrics]].copy()
    out.insert(1, "seed", seed)
    return out


def _collect_model_docs(model: ModelSpec, *, metrics: list[Metric]) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns:
      - long dataframe: guid, seed, metrics...
      - ordered guid list (for stable array conversion)
    """
    dfs: list[pd.DataFrame] = []
    for s in model.seeds:
        csv_path = _find_seed_csv(model.run_dirs, s)
        dfs.append(_load_seed_df(csv_path, model_name=model.name, seed=s, metrics=metrics))

    long_df = pd.concat(dfs, ignore_index=True)

    # guid整合性チェック（seed間で一致することを期待）
    guid_sets = [set(d["guid"].tolist()) for d in dfs]
    base = guid_sets[0]
    for i, gs in enumerate(guid_sets[1:], start=1):
        if gs != base:
            raise ValueError(
                f"[{model.name}] guid set mismatch across seeds: seed={model.seeds[0]} vs seed={model.seeds[i]}"
            )

    guids = sorted(base)
    return long_df, guids


def _to_metric_matrix(long_df: pd.DataFrame, *, guids: list[str], seeds: list[int], metric: Metric) -> np.ndarray:
    """
    shape: (n_docs, n_seeds) in (guid, seed) order.
    """
    pivot = long_df.pivot(index="guid", columns="seed", values=metric)
    # ensure ordering
    pivot = pivot.reindex(index=guids, columns=seeds)
    if pivot.isna().any().any():
        # 欠損は異常（本タスクの出力では起きない想定）
        raise ValueError(f"NaNs found in pivot for metric={metric}")
    return pivot.to_numpy(dtype=np.float64)


def _point_estimate_from_matrix(mat: np.ndarray) -> float:
    # 文書ごとに seed 平均、その後 文書平均
    return float(mat.mean(axis=1).mean(axis=0))


def _bootstrap_mean_ci(
    *,
    mat: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    モデル単体の指標 mean（文書→seed平均→文書平均）の 95% CI を階層bootstrapで推定する。

    外側: 文書を復元抽出
    内側: seed を復元抽出（文書内平均に反映）

    mat: (n_docs, n_seeds)
    """
    n_docs, n_seeds = mat.shape
    stats = np.empty((n_bootstrap,), dtype=np.float64)
    for t in range(n_bootstrap):
        doc_idx = rng.integers(0, n_docs, size=n_docs)
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        stats[t] = mat[doc_idx][:, seed_idx].mean(axis=1).mean(axis=0)

    mean_hat = float(_point_estimate_from_matrix(mat))
    lo, hi = np.quantile(stats, [0.025, 0.975]).astype(float).tolist()
    return mean_hat, lo, hi


def _bootstrap_delta_ci(
    *,
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    paired_seeds: bool,
) -> tuple[float, float, float]:
    """
    階層bootstrap:
      外側: 文書を復元抽出
      内側: seed を復元抽出して文書内平均

    a, b: (n_docs, n_seedsA/B)
    paired_seeds:
      - True: seed resample のインデックスを a と b で共有（H1向け）
      - False: a と b を独立に seed resample（H2向け）
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"n_docs mismatch: a={a.shape}, b={b.shape}")
    n_docs = a.shape[0]
    n_sa = a.shape[1]
    n_sb = b.shape[1]

    deltas = np.empty((n_bootstrap,), dtype=np.float64)
    for t in range(n_bootstrap):
        doc_idx = rng.integers(0, n_docs, size=n_docs)
        if paired_seeds:
            seed_idx = rng.integers(0, min(n_sa, n_sb), size=min(n_sa, n_sb))
            a_doc = a[doc_idx][:, seed_idx].mean(axis=1)
            b_doc = b[doc_idx][:, seed_idx].mean(axis=1)
        else:
            seed_idx_a = rng.integers(0, n_sa, size=n_sa)
            seed_idx_b = rng.integers(0, n_sb, size=n_sb)
            a_doc = a[doc_idx][:, seed_idx_a].mean(axis=1)
            b_doc = b[doc_idx][:, seed_idx_b].mean(axis=1)

        deltas[t] = b_doc.mean() - a_doc.mean()

    # 点推定は観測データの平均との差（bootstrap平均との差ではなく）
    delta_hat = float(_point_estimate_from_matrix(b) - _point_estimate_from_matrix(a))
    lo, hi = np.quantile(deltas, [0.025, 0.975]).astype(float).tolist()
    return delta_hat, lo, hi


def _print_table(title: str, df: pd.DataFrame) -> None:
    print("")
    print("=" * 80)
    print(title)
    print("=" * 80)
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
        print(df.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True, help="出力先ディレクトリ")
    p.add_argument("--bootstrap", type=int, default=10000, help="bootstrap反復回数（0でCI計算なし）")
    p.add_argument("--rng_seed", type=int, default=1, help="乱数seed（bootstrap用）")
    p.add_argument(
        "--metrics",
        type=str,
        default="AP,RR,Recall,nDCG",
        help="集計する指標（CSV列名）, e.g. 'Recall,nDCG'",
    )
    p.add_argument("--config_json", type=str, default="", help="モデル定義のJSON（指定時はデフォルトを上書き）")
    p.add_argument(
        "--ni_delta",
        type=float,
        default=float("nan"),
        help=(
            "非劣勢マージン δ（H2: Distill_NER_filtered - PEFT_RL を想定）。"
            "指定すると delta_summary.csv に非劣勢判定列を追加する（lower_CI >= -δ）。"
        ),
    )
    p.add_argument(
        "--ni_retain",
        type=float,
        default=float("nan"),
        help=(
            "効果保持率 ρ（0<ρ<=1）。指定すると δ を (1-ρ)*(ref_b - ref_a) で自動算出して非劣勢判定に使う。"
            "（例: ρ=0.9 は『PEFT_RLがBaseから得た改善の90%%を保持』）"
        ),
    )
    p.add_argument(
        "--ni_ref_a",
        type=str,
        default="Base",
        help="δ自動算出の参照モデルA（ref_b - ref_a を改善量とみなす）。デフォルト: Base",
    )
    p.add_argument(
        "--ni_ref_b",
        type=str,
        default="PEFT_RL",
        help="δ自動算出の参照モデルB（ref_b - ref_a を改善量とみなす）。デフォルト: PEFT_RL",
    )
    p.add_argument(
        "--ni_contrast",
        type=str,
        default="H2_NER_minus_PEFTRl",
        help="非劣勢判定を適用する contrast 名（デフォルトはH2）。",
    )
    p.add_argument(
        "--ni_metric",
        type=str,
        default="nDCG",
        help="非劣勢判定を適用する metric 名。カンマ区切りで複数指定可（例: 'nDCG,Recall'）。デフォルトはnDCG。",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    # type narrowing
    metric_list: list[Metric] = []
    for m in metrics:
        if m not in ("AP", "RR", "Recall", "nDCG"):
            raise ValueError(f"unsupported metric: {m} (supported: AP,RR,Recall,nDCG)")
        metric_list.append(m)  # type: ignore[arg-type]

    if args.config_json:
        models = _load_config_json(Path(args.config_json))
    else:
        models = _default_model_specs()

    # 1) 読み込み & モデル別点推定
    model_long: dict[str, pd.DataFrame] = {}
    model_guids: dict[str, list[str]] = {}
    for ms in models:
        long_df, guids = _collect_model_docs(ms, metrics=metric_list)
        model_long[ms.name] = long_df
        model_guids[ms.name] = guids

    # 2) guid集合の一致チェック（研究計画書の paired 差分の前提）
    guid0 = set(model_guids[models[0].name])
    for ms in models[1:]:
        g = set(model_guids[ms.name])
        if g != guid0:
            raise ValueError(f"guid set mismatch across models: {models[0].name} vs {ms.name}")
    common_guids = sorted(guid0)

    rng = np.random.default_rng(int(args.rng_seed))

    model_rows: list[dict[str, Any]] = []
    model_mats: dict[tuple[str, Metric], np.ndarray] = {}
    for ms in models:
        for met in metric_list:
            mat = _to_metric_matrix(model_long[ms.name], guids=common_guids, seeds=ms.seeds, metric=met)
            model_mats[(ms.name, met)] = mat
            if int(args.bootstrap) > 0:
                mean_hat, lo, hi = _bootstrap_mean_ci(mat=mat, n_bootstrap=int(args.bootstrap), rng=rng)
            else:
                mean_hat = _point_estimate_from_matrix(mat)
                lo, hi = float("nan"), float("nan")
            model_rows.append(
                {
                    "model": ms.name,
                    "metric": met,
                    "n_docs": mat.shape[0],
                    "n_seeds": mat.shape[1],
                    "mean": mean_hat,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap": int(args.bootstrap),
                }
            )
    model_summary = pd.DataFrame(model_rows).sort_values(["metric", "model"]).reset_index(drop=True)

    # 3) Δ（H1/H2） + bootstrap CI
    # 研究計画書 7.1:
    #   H1: Δ = PEFT_RL - Base （paired seed）
    #   H2: Δ = Distill_NER_filtered - PEFT_RL （seed独立）
    contrasts = [
        {"name": "H1_PEFTRl_minus_Base", "a": "Base", "b": "PEFT_RL", "paired_seeds": True},
        {"name": "H2_NER_minus_PEFTRl", "a": "PEFT_RL", "b": "Distill_NER_filtered", "paired_seeds": False},
    ]

    delta_rows: list[dict[str, Any]] = []
    for c in contrasts:
        for met in metric_list:
            a = model_mats[(c["a"], met)]
            b = model_mats[(c["b"], met)]
            if int(args.bootstrap) > 0:
                delta_hat, lo, hi = _bootstrap_delta_ci(
                    a=a,
                    b=b,
                    n_bootstrap=int(args.bootstrap),
                    rng=rng,
                    paired_seeds=bool(c["paired_seeds"]),
                )
            else:
                delta_hat = float(_point_estimate_from_matrix(b) - _point_estimate_from_matrix(a))
                lo, hi = float("nan"), float("nan")
            delta_rows.append(
                {
                    "contrast": c["name"],
                    "metric": met,
                    "a": c["a"],
                    "b": c["b"],
                    "paired_seeds": bool(c["paired_seeds"]),
                    "delta": delta_hat,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap": int(args.bootstrap),
                }
            )
    delta_summary = pd.DataFrame(delta_rows).sort_values(["contrast", "metric"]).reset_index(drop=True)

    # 3.5) 非劣勢判定（研究計画書 H2 を想定）
    # 判定: lower_CI(Δ) >= -δ
    ni_contrast = str(args.ni_contrast)
    ni_metrics = [m.strip() for m in str(args.ni_metric).split(",") if m.strip()]
    ni_delta = float(args.ni_delta)
    ni_retain = float(args.ni_retain)
    ni_ref_a = str(args.ni_ref_a)
    ni_ref_b = str(args.ni_ref_b)

    # δを効果保持率から自動算出（明示δがあればそれを優先）
    ni_delta_by_metric: dict[str, float] = {}
    ni_ref_value_by_metric: dict[str, float] = {}
    if not np.isfinite(ni_delta) and np.isfinite(ni_retain):
        if not (0.0 < ni_retain <= 1.0):
            raise ValueError(f"--ni_retain must satisfy 0<rho<=1, got: {ni_retain}")
        for met in ni_metrics:
            if (ni_ref_a, met) not in model_mats or (ni_ref_b, met) not in model_mats:
                raise ValueError(
                    f"missing ref models for ni margin: ({ni_ref_a},{met}) or ({ni_ref_b},{met}). "
                    f"available models: {sorted(set(k[0] for k in model_mats.keys()))}"
                )
            ref_a = model_mats[(ni_ref_a, met)]
            ref_b = model_mats[(ni_ref_b, met)]
            ref_value = float(_point_estimate_from_matrix(ref_b) - _point_estimate_from_matrix(ref_a))
            if ref_value <= 0:
                raise ValueError(
                    f"reference improvement (ref_b-ref_a) must be >0 to define NI margin from retention; got {ref_value}. "
                    f"(ref_a={ni_ref_a}, ref_b={ni_ref_b}, metric={met})"
                )
            ni_ref_value_by_metric[met] = ref_value
            ni_delta_by_metric[met] = float((1.0 - ni_retain) * ref_value)

    # 明示δの場合は全指定metricに同一δを適用（必要ならmetricごとに別δを与える拡張も可能だが、現時点では単一値）
    if np.isfinite(ni_delta):
        for met in ni_metrics:
            ni_delta_by_metric[met] = ni_delta

    if ni_delta_by_metric:
        delta_summary["ni_delta"] = np.nan
        delta_summary["ni_retain"] = np.nan
        delta_summary["ni_ref_a"] = ""
        delta_summary["ni_ref_b"] = ""
        delta_summary["ni_ref_value"] = np.nan
        delta_summary["ni_rule"] = ""
        delta_summary["ni_pass"] = np.nan  # bool相当だがCSV互換のため数値/NaNで保持

        for met, d in ni_delta_by_metric.items():
            mask = (delta_summary["contrast"] == ni_contrast) & (delta_summary["metric"] == met)
            delta_summary.loc[mask, "ni_delta"] = float(d)
            if np.isfinite(ni_retain):
                delta_summary.loc[mask, "ni_retain"] = ni_retain
                delta_summary.loc[mask, "ni_ref_a"] = ni_ref_a
                delta_summary.loc[mask, "ni_ref_b"] = ni_ref_b
                delta_summary.loc[mask, "ni_ref_value"] = float(ni_ref_value_by_metric.get(met, np.nan))
            delta_summary.loc[mask, "ni_rule"] = "lower_CI(delta) >= -delta"

            # bootstrap=0 の場合はCIがNaNなので判定しない
            can_judge = mask & delta_summary["ci_low"].notna()
            delta_summary.loc[can_judge, "ni_pass"] = (
                delta_summary.loc[can_judge, "ci_low"].astype(float) >= -float(d)
            ).astype(int)

    # 4) 保存 + 表示
    model_csv = out_dir / "model_summary.csv"
    delta_csv = out_dir / "delta_summary.csv"
    model_summary.to_csv(model_csv, index=False)
    delta_summary.to_csv(delta_csv, index=False)

    _print_table("Model summary (doc->seed mean -> doc mean)", model_summary)
    _print_table("Delta summary (H1/H2) with 95% CI (hierarchical bootstrap)", delta_summary)
    print("")
    print(f"Wrote: {model_csv}")
    print(f"Wrote: {delta_csv}")


if __name__ == "__main__":
    # pandasのスレッド過多を抑える（環境によっては不要だが安全）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
