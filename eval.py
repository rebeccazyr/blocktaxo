#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _normalize_relationship_list(raw_relationships: Sequence[object]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for rel in raw_relationships or []:
        parent = child = None
        if isinstance(rel, dict):
            lowered = {str(key).lower(): value for key, value in rel.items() if key is not None}
            parent = (
                rel.get("parent")
                or rel.get("parent_name")
                or rel.get("parentId")
                or lowered.get("parent")
                or lowered.get("parent_name")
                or lowered.get("parentid")
                or lowered.get("parent_id")
            )
            child = (
                rel.get("child")
                or rel.get("child_name")
                or rel.get("childId")
                or lowered.get("child")
                or lowered.get("child_name")
                or lowered.get("childid")
                or lowered.get("child_id")
            )
        elif isinstance(rel, (list, tuple)) and len(rel) >= 2:
            parent, child = rel[0], rel[1]
        parent = parent.strip() if isinstance(parent, str) else None
        child = child.strip() if isinstance(child, str) else None
        if parent and child:
            pairs.append((parent, child))
    return pairs


def _label_from_node(node: object) -> str | None:
    if isinstance(node, str):
        return node.strip() or None
    if isinstance(node, dict):
        label = node.get("label") or node.get("name") or node.get("title")
        if isinstance(label, str):
            label = label.strip()
        return label or None
    return None


def _children_from_node(node: object) -> List[object]:
    if isinstance(node, dict):
        children = node.get("children")
        if isinstance(children, list):
            return children
    return []


def extract_edges_from_taxonomy(taxonomy: object) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []

    def _walk(parent: object) -> None:
        parent_label = _label_from_node(parent)
        for child in _children_from_node(parent):
            child_label = _label_from_node(child)
            if parent_label and child_label:
                edges.append((parent_label, child_label))
            _walk(child)

    _walk(taxonomy)
    return edges


def load_relationships_from_csv(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, object]] = []
        for row in reader:
            if not row:
                continue
            normalized: Dict[str, object] = {}
            for key, value in row.items():
                if key is None:
                    continue
                clean_key = key.strip().lstrip("\ufeff")
                normalized[clean_key] = value
            rows.append(normalized)
    return _normalize_relationship_list(rows)


def load_predicted_edges(path: Path) -> List[Tuple[str, str]]:
    if path.suffix.lower() == ".csv":
        return load_relationships_from_csv(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and "taxonomy" in raw:
        return extract_edges_from_taxonomy(raw["taxonomy"])
    if isinstance(raw, list):
        return _normalize_relationship_list(raw)
    if isinstance(raw, dict):
        return _normalize_relationship_list(raw.values())
    return []


def load_ground_truth_edges(path: Path) -> List[Tuple[str, str]]:
    if path.suffix.lower() == ".csv":
        return load_relationships_from_csv(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return _normalize_relationship_list(raw)


def compute_ancestor_pairs(relationship_set: Iterable[Tuple[str, str]]) -> set[Tuple[str, str]]:
    adjacency: Dict[str, List[str]] = {}
    for parent, child in relationship_set:
        adjacency.setdefault(parent, []).append(child)

    ancestors: set[Tuple[str, str]] = set()
    for ancestor in adjacency:
        stack = list(adjacency.get(ancestor, []))
        visited: set[str] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            ancestors.add((ancestor, node))
            stack.extend(adjacency.get(node, []))
    return ancestors


def _calc_metrics(pred_set: set, gt_set: set) -> Dict[str, float]:
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "pred_total": float(len(pred_set)),
        "gt_total": float(len(gt_set)),
    }


def evaluate(predicted_pairs: Iterable[Tuple[str, str]],
             ground_truth_pairs: Iterable[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
    pred_set = {tuple(pair) for pair in predicted_pairs}
    gt_set = {tuple(pair) for pair in ground_truth_pairs}
    node_pred = {node for edge in pred_set for node in edge}
    node_gt = {node for edge in gt_set for node in edge}
    pred_anc = compute_ancestor_pairs(pred_set)
    gt_anc = compute_ancestor_pairs(gt_set)
    return {
        "edge": _calc_metrics(pred_set, gt_set),
        "node": _calc_metrics(node_pred, node_gt),
        "ancestor": _calc_metrics(pred_anc, gt_anc),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted taxonomy against ground-truth relationships.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predicted",
        type=Path,
        required=True,
        help="Predicted taxonomy JSON or CSV (child/parent columns)",
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        required=True,
        help="Ground-truth relationships JSON or CSV",
    )
    return parser.parse_args()


def _print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"{title}:")
    print(f"  precision: {metrics['precision']:.6f}")
    print(f"  recall:    {metrics['recall']:.6f}")
    print(f"  f1:        {metrics['f1']:.6f}")
    print(f"  tp:        {int(metrics['tp'])}")
    print(f"  fp:        {int(metrics['fp'])}")
    print(f"  fn:        {int(metrics['fn'])}")
    print(f"  pred_total:{int(metrics['pred_total'])}")
    print(f"  gt_total:  {int(metrics['gt_total'])}")


def main() -> int:
    args = parse_args()
    predicted_path = args.predicted.resolve()
    ground_truth_path = args.ground_truth.resolve()

    if not predicted_path.exists():
        print(f"Predicted file not found: {predicted_path}")
        return 1
    if not ground_truth_path.exists():
        print(f"Ground-truth file not found: {ground_truth_path}")
        return 1

    predicted_pairs = load_predicted_edges(predicted_path)
    ground_truth_pairs = load_ground_truth_edges(ground_truth_path)

    metrics = evaluate(predicted_pairs, ground_truth_pairs)
    _print_metrics("edge", metrics["edge"])
    _print_metrics("node", metrics["node"])
    _print_metrics("ancestor", metrics["ancestor"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
