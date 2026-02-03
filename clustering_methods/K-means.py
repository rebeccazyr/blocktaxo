#!/usr/bin/env python3
"""
K-Means clustering without KNN expansion.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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


def _candidate_ks(count: int) -> List[int]:
    if count <= 1:
        return []
    k_star = max(1, math.ceil(count / 100))
    upper = min(math.ceil(1.3 * k_star), count - 1)
    if upper < 2:
        return []
    lower = min(max(2, math.floor(0.7 * k_star)), upper)
    return list(range(lower, upper + 1))


def _select_k(
    embeddings: NDArray[np.float32],
    seed: int,
) -> int:
    n_terms = embeddings.shape[0]
    if n_terms <= 2:
        return 1

    with Timer("PCA Dimension Reduction"):
        n_comp = min(32, n_terms)
        pca = PCA(n_components=n_comp, random_state=seed)
        reduced_data = pca.fit_transform(embeddings)

    candidate_ks = _candidate_ks(n_terms)
    if not candidate_ks:
        return 1

    best_k = 2
    max_score = -np.inf

    with Timer(f"Searching Optimal K (range: 2-{max(candidate_ks)})"):
        for k in candidate_ks:
            kmeans = KMeans(n_clusters=k, n_init=5, random_state=seed)
            labels = kmeans.fit_predict(reduced_data)
            score = silhouette_score(reduced_data, labels)
            if score > max_score:
                max_score = score
                best_k = k

    print(f"  -> Selected optimal K: {best_k} (Silhouette Score: {max_score:.4f})", file=sys.stderr)
    return best_k


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


def partition_entities_no_knn(
    entities: Sequence[str],
    backend: EmbeddingBackend,
    seed: int,
    root_label: Optional[str] = None,
) -> List[List[str]]:
    """
    执行纯净的语义分块，不进行 KNN 扩展（No Overlap）。
    每个实体被唯一分配到一个 Block。
    """
    print(f"\n--- Hard Clustering {len(entities)} entities (No KNN) ---", file=sys.stderr)
    if len(entities) <= 1:
        return [list(entities)]

    embs = backend.encode(entities)

    k = _select_k(embs, seed)

    with Timer(f"Final K-Means Clustering (k={k})"):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = kmeans.fit_predict(embs)

    blocks = [[] for _ in range(k)]
    for ent, lbl in zip(entities, labels):
        blocks[lbl].append(ent)

    if root_label and root_label in entities:
        for block in blocks:
            if root_label not in block:
                block.append(root_label)

    blocks = [block for block in blocks if block]
    sizes = [len(block) for block in blocks]
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="K-Means clustering without KNN expansion.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--relationships", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=13)
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
            blocks = partition_entities_no_knn(entities, backend, args.seed, root_label=root_label)
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
        blocks = partition_entities_no_knn(entities, backend, args.seed, root_label=root_label)
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
