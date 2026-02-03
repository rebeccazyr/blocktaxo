#!/usr/bin/env python3
"""
Spectral clustering without overlap.
Supports CLI I/O: --input ... --output ...
Directory input scans */test_sets/size_*/sample_*/entities.json.
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
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances

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


def spectral_partitioning_no_overlap(
    entities: Sequence[str],
    embeddings: NDArray[np.float32],
    n_clusters: Optional[int] = None,
    gamma: float = 1.0,
    seed: int = 42,
) -> List[List[str]]:
    """
    纯净的谱切分逻辑（不允许重叠）
    """
    n = len(entities)
    if n <= 1:
        return [list(entities)]

    if n_clusters is None:
        dist_matrix = pairwise_distances(embeddings, metric="cosine")
        affinity = np.exp(-gamma * dist_matrix ** 2)
        L = laplacian(affinity, normed=True)
        vals, _ = eigh(L)
        candidate_ks = _candidate_ks(n)
        if candidate_ks:
            min_k = candidate_ks[0]
            max_k = min(candidate_ks[-1], len(vals) - 1)
            if max_k >= min_k:
                gaps = np.diff(vals[: max_k + 1])
                best_k = None
                best_gap = -np.inf
                for k in range(min_k, max_k + 1):
                    gap = gaps[k - 1]
                    if gap > best_gap:
                        best_gap = gap
                        best_k = k
                n_clusters = best_k
            else:
                n_clusters = None
        else:
            n_clusters = None

        if n_clusters is None:
            n_clusters = np.argmax(np.diff(vals[: min(n, 20)])) + 1
            n_clusters = max(2, n_clusters)
        print(f"  -> Spectral Eigengap selected K: {n_clusters}", file=sys.stderr)

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        assign_labels="discretize",
        random_state=seed,
        n_neighbors=min(n - 1, 10),
    )
    labels = sc.fit_predict(embeddings)

    blocks_dict = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        blocks_dict[label].append(entities[idx])

    return [b for b in blocks_dict.values() if b]


def partition_entities_spectral_no_overlap(
    entities: Sequence[str],
    backend: EmbeddingBackend,
    seed: int,
    root_label: Optional[str] = None,
    n_clusters: Optional[int] = None,
    gamma: float = 1.0,
) -> List[List[str]]:
    print(f"\n--- Spectral Clustering {len(entities)} entities (No Overlap) ---", file=sys.stderr)
    if len(entities) <= 1:
        return [list(entities)]

    embs = backend.encode(entities)
    blocks = spectral_partitioning_no_overlap(
        entities,
        embs,
        n_clusters=n_clusters,
        gamma=gamma,
        seed=seed,
    )

    if root_label and root_label in entities:
        for block in blocks:
            if root_label not in block:
                block.append(root_label)

    sizes = [len(b) for b in blocks]
    print(f"  -> Final Block Sizes: {sorted(sizes, reverse=True)}", file=sys.stderr)
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


def _iter_entity_files(root: Path) -> Iterable[Path]:
    yield from root.glob("*/test_sets/size_*/sample_*/entities.json")


def _build_output_record(
    record: dict,
    entities: Sequence[str],
    blocks: List[List[str]],
    output_format: str,
) -> dict:
    if output_format == "clustered":
        return {"entity_list": list(entities), "entity_blocks": blocks}
    new_rec = dict(record)
    new_rec["entity_list"] = list(entities)
    new_rec["entity_blocks"] = blocks
    return new_rec


def _cluster_records(
    records: Sequence[dict],
    backend: EmbeddingBackend,
    seed: int,
    root_label: Optional[str],
    n_clusters: Optional[int],
    gamma: float,
    relationships: Sequence[Tuple[str, str]],
    output_format: str,
) -> List[dict]:
    output_records: List[dict] = []
    if not records:
        return output_records

    if isinstance(records[0], dict) and "entity_list" in records[0]:
        total_records = len(records)
        for rec_idx, rec in enumerate(records):
            entities = rec.get("entity_list", [])
            rec_root_label = root_label or _get_root_label(rec, entities)
            entities = _ensure_root_in_entities(entities, rec_root_label)
            blocks = partition_entities_spectral_no_overlap(
                entities,
                backend,
                seed,
                root_label=rec_root_label,
                n_clusters=n_clusters,
                gamma=gamma,
            )
            if relationships:
                metrics = _compute_edge_recall(blocks, relationships)
                intra = _compute_intra_edge_metric(blocks, relationships)
                title = "Edge Recall" if total_records == 1 else f"Edge Recall (record {rec_idx})"
                print(f"\n{title}:")
                print(f"  covered_edges: {metrics['covered']}")
                print(f"  total_edges:   {metrics['total']}")
                print(f"  recall_edge:   {metrics['recall']:.6f}")
                print("Intra-Block Metric:")
                print(f"  numerator:    {int(intra['numerator'])}")
                print(f"  denominator:  {intra['denominator']:.6f}")
                print(f"  metric:       {intra['metric']:.6f}")
            output_records.append(_build_output_record(rec, entities, blocks, output_format))
        return output_records

    entities = [r.get("name") for r in records if isinstance(r, dict) and r.get("name")]
    if not entities:
        return output_records
    rec_root_label = root_label
    if rec_root_label is None:
        for rec in records:
            if isinstance(rec, dict):
                rec_root_label = _get_root_label(rec, entities)
                if rec_root_label:
                    break
    entities = _ensure_root_in_entities(entities, rec_root_label)
    blocks = partition_entities_spectral_no_overlap(
        entities,
        backend,
        seed,
        root_label=rec_root_label,
        n_clusters=n_clusters,
        gamma=gamma,
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
    output_records.append(_build_output_record({}, entities, blocks, output_format))
    return output_records


def _write_output(output_path: Path, output_records: Sequence[dict]) -> None:
    payload: object
    if not output_records:
        payload = []
    else:
        payload = output_records[0] if len(output_records) == 1 else output_records
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spectral clustering without overlap.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--relationships", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--root-label", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-format",
        choices=("full", "clustered"),
        default="full",
        help="Output record shape. 'clustered' keeps only entity_list and entity_blocks.",
    )
    args = parser.parse_args(argv)

    backend = EmbeddingBackend(batch_size=args.batch_size, device=args.device)
    relationships = _load_relationships(args.relationships)

    if args.input.is_dir():
        if args.output is None:
            print("error: --output is required when --input is a directory", file=sys.stderr)
            return 2
        if args.relationships:
            print("warning: --relationships ignored when --input is a directory", file=sys.stderr)
            relationships = []
        input_root = args.input
        output_root = args.output
        for input_path in sorted(_iter_entity_files(input_root)):
            rel_path = input_path.relative_to(input_root)
            output_path = output_root / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            records = list(_load_records(input_path))
            if not records:
                _write_output(output_path, [])
                continue
            output_records = _cluster_records(
                records,
                backend,
                args.seed,
                args.root_label,
                args.n_clusters,
                args.gamma,
                relationships,
                args.output_format,
            )
            _write_output(output_path, output_records)
        return 0

    records = list(_load_records(args.input))
    if not records:
        if args.output:
            _write_output(args.output, [])
        return 0

    output_records = _cluster_records(
        records,
        backend,
        args.seed,
        args.root_label,
        args.n_clusters,
        args.gamma,
        relationships,
        args.output_format,
    )

    if args.output:
        _write_output(args.output, output_records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
