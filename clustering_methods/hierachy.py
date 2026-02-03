#!/usr/bin/env python3
"""
Hierarchical clustering without overlap.
Supports CLI I/O: --input ... --output ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage, fcluster

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"[{self.name}] took {elapsed:.4f}s", file=sys.stderr)


class EmbeddingBackend:
    def __init__(self, model_name: str = "allenai/specter2", batch_size: int = 128, device: str = "auto"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        with Timer("Model Loading"):
            from transformers import AutoTokenizer
            from adapters import AutoAdapterModel

            dev = self.device
            if dev == "auto":
                dev = "cuda" if (torch and torch.cuda.is_available()) else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
            self._model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
            self._model.load_adapter("allenai/specter2", source="hf", set_active=True)
            self._model.to(dev)
            self._model.eval()

    def encode(self, items: Sequence[str]) -> NDArray[np.float32]:
        if not items:
            return np.empty((0, 0), dtype=np.float32)
        if self._model is None:
            self._load_model()

        with Timer(f"Encoding {len(items)} items"):
            all_embs = []
            for i in range(0, len(items), self.batch_size):
                batch = list(items[i : i + self.batch_size])
                toks = self._tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
                if torch:
                    toks = {k: v.to(self._model.device) for k, v in toks.items()}
                with torch.no_grad():
                    out = self._model(**toks)
                    emb = out.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embs.append(emb)

            res = np.vstack(all_embs)
            norms = np.linalg.norm(res, axis=1, keepdims=True)
            return (res / np.maximum(norms, 1e-12)).astype(np.float32)


def _get_root_label(record: dict, entities: Sequence[str]) -> Optional[str]:
    for key in ("root", "root_label", "root_name", "root_node"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    for candidate in ("[ROOT]", "ROOT", "root"):
        if candidate in entities:
            return candidate
    return None


def _ensure_root_in_entities(entities: Sequence[str], root_label: Optional[str]) -> List[str]:
    if root_label and root_label not in entities:
        return [root_label] + list(entities)
    return list(entities)


def _candidate_ks(count: int) -> List[int]:
    if count <= 1:
        return []
    k_star = max(1, math.ceil(count / 100))
    upper = min(math.ceil(1.3 * k_star), count - 1)
    if upper < 2:
        return []
    lower = min(max(2, math.floor(0.7 * k_star)), upper)
    return list(range(lower, upper + 1))


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


def _load_relationships(path: Optional[Path]) -> List[Tuple[str, str]]:
    if path is None:
        return []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row]
        return _normalize_relationship_list(rows)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):
        data = list(data.values())
    return _normalize_relationship_list(data)


def _compute_edge_recall(blocks: Sequence[Sequence[str]], relationships: Sequence[Tuple[str, str]]) -> dict:
    total_edges = len(relationships)
    if total_edges == 0:
        return {"recall": 0.0, "covered": 0, "total": 0}

    entity_to_blocks: dict[str, set[int]] = {}
    for idx, block in enumerate(blocks):
        for entity in block:
            entity_to_blocks.setdefault(entity, set()).add(idx)

    covered = 0
    for parent, child in relationships:
        blocks_parent = entity_to_blocks.get(parent)
        blocks_child = entity_to_blocks.get(child)
        if blocks_parent and blocks_child and (blocks_parent & blocks_child):
            covered += 1

    return {"recall": covered / total_edges, "covered": covered, "total": total_edges}


def _compute_intra_edge_metric(
    blocks: Sequence[Sequence[str]],
    relationships: Sequence[Tuple[str, str]],
) -> dict:
    denom = 0.0
    numer = 0.0
    for block in blocks:
        n_b = len(block)
        if n_b <= 0:
            continue
        denom += n_b * math.log(n_b) if n_b > 1 else 0.0
        block_set = set(block)
        intra = 0
        for parent, child in relationships:
            if parent in block_set and child in block_set:
                intra += 1
        numer += intra
    metric = (numer / denom) if denom > 0 else 0.0
    return {"metric": metric, "numerator": numer, "denominator": denom}


def hierarchical_partitioning_no_overlap(
    entities: Sequence[str],
    embeddings: NDArray[np.float32],
    linkage_method: str = "ward",
) -> List[List[str]]:
    """
    自适应层次聚类划分（无重叠版本）
    1. 使用 Ward 准则构建树状图
    2. 通过合并距离的激增点（Distance Jumps）自动确定切分线
    """
    n = len(entities)
    if n <= 2:
        return [list(entities)]

    def _labels_to_blocks(labels: NDArray[np.int_]) -> List[List[str]]:
        unique_labels = np.unique(labels)
        blocks = [[] for _ in range(len(unique_labels))]
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        for ent_idx, lbl in enumerate(labels):
            blocks[label_to_idx[lbl]].append(entities[ent_idx])
        blocks.sort(key=len, reverse=True)
        return blocks

    metric = "euclidean" if linkage_method == "ward" else "cosine"
    Z = linkage(embeddings, method=linkage_method, metric=metric)

    distances = Z[:, 2]
    jumps = np.diff(distances)
    if jumps.size == 0:
        labels = np.ones(n, dtype=int)
        return _labels_to_blocks(labels)

    candidate_ks = _candidate_ks(n)
    best_jump_idx = None
    if candidate_ks:
        lower, upper = candidate_ks[0], candidate_ks[-1]
        best_jump = -np.inf
        for i, jump in enumerate(jumps):
            k = n - (i + 1)
            if lower <= k <= upper and jump > best_jump:
                best_jump = jump
                best_jump_idx = i

    if best_jump_idx is None:
        look_back = max(1, int(len(jumps) * 0.3))
        best_jump_idx = np.argmax(jumps[-look_back:]) + (len(jumps) - look_back)
        partition_height = distances[best_jump_idx]
        print(f"  -> Detected distance jump: {jumps[best_jump_idx]:.4f}", file=sys.stderr)
        print(f"  -> Partition Line set at height: {partition_height:.4f}", file=sys.stderr)
        labels = fcluster(Z, t=partition_height, criterion="distance")
        return _labels_to_blocks(labels)

    target_k = n - (best_jump_idx + 1)
    print(f"  -> Detected distance jump: {jumps[best_jump_idx]:.4f}", file=sys.stderr)
    print(f"  -> Selected k in range: {target_k}", file=sys.stderr)
    labels = fcluster(Z, t=target_k, criterion="maxclust")
    return _labels_to_blocks(labels)


def partition_entities_hac_no_overlap(
    entities: Sequence[str],
    backend: EmbeddingBackend,
    root_label: Optional[str] = None,
    linkage_method: str = "ward",
) -> List[List[str]]:
    print(f"\n--- HAC Partitioning ({linkage_method}) for {len(entities)} entities ---", file=sys.stderr)

    embs = backend.encode(entities)
    blocks = hierarchical_partitioning_no_overlap(
        entities,
        embs,
        linkage_method=linkage_method,
    )

    if root_label and root_label in entities:
        for block in blocks:
            if root_label not in block:
                block.append(root_label)

    sizes = [len(b) for b in blocks]
    print(f"  -> Resulting {len(blocks)} blocks. Sizes: {sizes}", file=sys.stderr)

    return blocks


def _load_records(path: Path) -> Iterable[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, list):
        if all(isinstance(i, str) for i in data):
            yield {"entity_list": data}
        else:
            yield from data
    else:
        yield data


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hierarchical clustering without overlap.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--relationships", type=Path, default=None)
    parser.add_argument("--linkage-method", type=str, default="ward")
    parser.add_argument("--root-label", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)

    backend = EmbeddingBackend(batch_size=args.batch_size, device=args.device)
    relationships = _load_relationships(args.relationships)

    records = list(_load_records(args.input))
    if not records:
        if args.output:
            args.output.write_text("[]", encoding="utf-8")
        return 0

    output_records = []

    if isinstance(records[0], dict) and "entity_list" in records[0]:
        for rec_idx, rec in enumerate(records):
            entities = rec.get("entity_list", [])
            root_label = args.root_label or _get_root_label(rec, entities)
            entities = _ensure_root_in_entities(entities, root_label)
            blocks = partition_entities_hac_no_overlap(
                entities,
                backend,
                root_label=root_label,
                linkage_method=args.linkage_method,
            )
            if relationships:
                metrics = _compute_edge_recall(blocks, relationships)
                intra = _compute_intra_edge_metric(blocks, relationships)
                title = "Edge Recall" if len(records) == 1 else f"Edge Recall (record {rec_idx})"
                print(f"\n{title}:")
                print(f"  covered_edges: {metrics['covered']}")
                print(f"  total_edges:   {metrics['total']}")
                print(f"  recall_edge:   {metrics['recall']:.6f}")
                print("Intra-Block Metric:")
                print(f"  numerator:    {int(intra['numerator'])}")
                print(f"  denominator:  {intra['denominator']:.6f}")
                print(f"  metric:       {intra['metric']:.6f}")
            new_rec = dict(rec)
            new_rec["entity_list"] = entities
            new_rec["entity_blocks"] = blocks
            output_records.append(new_rec)
    else:
        entities = [r.get("name") for r in records if isinstance(r, dict) and r.get("name")]
        if not entities:
            args.output.write_text("[]", encoding="utf-8")
            return 0
        root_label = args.root_label
        if root_label is None:
            for rec in records:
                if isinstance(rec, dict):
                    root_label = _get_root_label(rec, entities)
                    if root_label:
                        break
        entities = _ensure_root_in_entities(entities, root_label)
        blocks = partition_entities_hac_no_overlap(
            entities,
            backend,
            root_label=root_label,
            linkage_method=args.linkage_method,
        )
        if relationships:
            metrics = _compute_edge_recall(blocks, relationships)
            intra = _compute_intra_edge_metric(blocks, relationships)
            print("\nEdge Recall:")
            print(f"  covered_edges: {metrics['covered']}")
            print(f"  total_edges:   {metrics['total']}")
            print(f"  recall_edge:   {metrics['recall']:.6f}")
            print("Intra-Block Metric:")
            print(f"  numerator:    {int(intra['numerator'])}")
            print(f"  denominator:  {intra['denominator']:.6f}")
            print(f"  metric:       {intra['metric']:.6f}")
        output_records.append({"entity_list": entities, "entity_blocks": blocks})

    if args.output:
        payload = output_records[0] if len(output_records) == 1 else output_records
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
