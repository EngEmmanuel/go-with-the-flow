from typing import Any, Dict, Tuple, Iterable, Optional, List
from pathlib import Path
import re
import yaml
import pandas as pd
import numpy as np
import wandb
from collections import defaultdict

def _iter_variant_paths(subtree: Dict[str, Any], prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[str, ...]]:
    if isinstance(subtree, dict) and "stylegan_results" in subtree:
        yield prefix
        return
    if isinstance(subtree, dict):
        for k, v in subtree.items():
            yield from _iter_variant_paths(v, prefix + (k,))


def _get_node_at_path(subtree: Dict[str, Any], path: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    cur = subtree
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    if isinstance(cur, dict) and "stylegan_results" in cur:
        return cur
    return None


def _extract_metric_value_from_node(node: Dict[str, Any], metric: str) -> Optional[float]:
    if not isinstance(node, dict):
        return None
    sg = node.get("stylegan_results", {}) or {}
    if metric in sg and isinstance(sg[metric], (int, float)):
        return float(sg[metric])
    sm = node.get("summary", {}) or {}
    if metric in sm:
        val = sm[metric]
        if isinstance(val, dict) and "mean" in val and isinstance(val["mean"], (int, float)):
            return float(val["mean"])
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _nmf_sort_key(nmf_key: str) -> float:
    if not isinstance(nmf_key, str):
        return float("inf")
    m = re.search(r"(\d+)", nmf_key)
    if m:
        return float(int(m.group(1)))
    if "max" in nmf_key.lower():
        return float("inf")
    return float("inf")


def yaml_metrics_to_nested_tables(
    yaml_path_or_dict: Any,
    *,
    nmf_order: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, wandb.Table]]]:
    # load yaml if necessary
    if isinstance(yaml_path_or_dict, (str, Path)):
        with open(yaml_path_or_dict, "r") as fh:
            data = yaml.safe_load(fh)
    else:
        data = yaml_path_or_dict

    if not isinstance(data, dict):
        raise ValueError("Expected top-level dict mapping nmf keys to subtrees.")

    nmf_keys = [k for k in data.keys()]
    if nmf_order is None:
        nmf_order = sorted(nmf_keys, key=lambda k: (_nmf_sort_key(k), k))

    # discover all variant paths across nmf entries
    all_paths = set()
    for nmf in nmf_keys:
        subtree = data.get(nmf, {}) or {}
        for path in _iter_variant_paths(subtree, prefix=()):
            all_paths.add(path)

    # containers to return
    mode_to_dfs: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
    mode_to_wtabs: Dict[str, Dict[str, wandb.Table]] = defaultdict(dict)

    for path in sorted(all_paths):
        if len(path) == 0:
            continue
        mode = path[-1]
        bin_key = "" if len(path) == 1 else "/".join(path[:-1])

        # collect union of metric names for this path across nmfs
        metrics = set()
        for nmf in nmf_order:
            node = _get_node_at_path(data.get(nmf, {}), path)
            if not node:
                continue
            sg = node.get("stylegan_results", {}) or {}
            if isinstance(sg, dict):
                metrics.update([k for k, v in sg.items() if isinstance(v, (int, float))])
            sm = node.get("summary", {}) or {}
            if isinstance(sm, dict):
                metrics.update(sm.keys())

        metrics = sorted(metrics)
        # build rows
        rows: List[Dict[str, float]] = []
        for nmf in nmf_order:
            node = _get_node_at_path(data.get(nmf, {}), path)
            row = {m: np.nan for m in metrics}
            if node:
                for m in metrics:
                    val = _extract_metric_value_from_node(node, m)
                    if val is not None:
                        row[m] = float(val)
            rows.append(row)

        df = pd.DataFrame(rows, index=list(nmf_order))
        df.index.name = "nmf"
        mode_to_dfs[mode][bin_key] = df

        # create wandb table
        cols = ["nmf"] + list(df.columns)
        wtab = wandb.Table(columns=[str(c) for c in cols])
        for nmf in df.index:
            row_vals = [nmf]
            for c in df.columns:
                v = df.at[nmf, c]
                row_vals.append(None if pd.isna(v) else float(v))
            wtab.add_data(*row_vals)
        mode_to_wtabs[mode][bin_key] = wtab

    return dict(mode_to_dfs), dict(mode_to_wtabs)

