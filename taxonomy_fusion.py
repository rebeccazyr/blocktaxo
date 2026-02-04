#!/usr/bin/env python3
"""
Multi-granular taxonomy fusion across block-level taxonomies.

Stage 1: Node merge (exact label overlap across blocks)
Stage 2: Node attachment (hierarchy construction via embeddings + LLM)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from llm_client import LLMClient, LLMConfig
except ImportError:  # Optional dependency
    LLMClient = None
    LLMConfig = None

from prompts import get_parent_selection_prompt


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class NodeRecord:
    node_id: str
    block_id: int
    label: str
    parent_id: Optional[str]
    children_ids: List[str] = field(default_factory=list)
    path: Tuple[str, ...] = field(default_factory=tuple)
    depth: int = 0
    embedding_text: str = ""
    source_file: Optional[Path] = None


@dataclass
class CanonicalNode:
    node_id: str
    label: str
    source_nodes: List[Dict[str, object]]
    source_blocks: List[int]
    embedding: np.ndarray
    depth: int
    merge_size: int
    parent_candidates: set[str] = field(default_factory=set)
    child_candidates: set[str] = field(default_factory=set)

    def to_dict(self, children: List[Dict[str, object]]) -> Dict[str, object]:
        return {
            "id": self.node_id,
            "label": self.label,
            "merge_size": self.merge_size,
            "source_blocks": self.source_blocks,
            "source_nodes": self.source_nodes,
            "children": children,
        }


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse block-level taxonomies from CSV format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-label", type=str, default="Computing Classification System")
    parser.add_argument("--root-description", type=str, default="Fused taxonomy root")
    parser.add_argument("--embedding-model", type=str, default="allenai/specter2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--merge-threshold", type=float, default=0.95)
    parser.add_argument("--parent-top-k", type=int, default=10)
    parser.set_defaults(use_llm=True)
    parser.add_argument("--no-llm", dest="use_llm", action="store_false",
                        help="Disable LLM-assisted parent selection")
    # LLM configuration
    parser.add_argument("--llm-provider", type=str, default="openai",
                        help="LLM provider (e.g., 'openai')")
    parser.add_argument("--llm-model", type=str, default="gpt-5",
                        help="LLM model name (e.g., 'gpt-5', 'gpt-4o', 'gpt-4o-mini')")
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="API key for LLM provider (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--llm-api-base", type=str, default=None,
                        help="API base URL for LLM provider (optional)")
    parser.add_argument("--llm-max-tokens", type=int, default=None,
                        help="Maximum tokens for LLM response (not needed for GPT-5)")
    return parser.parse_args()


# =============================================================================
# Loading & flattening
# =============================================================================










def _build_hierarchy_from_relations(entities: List[str], relations: List[Tuple[str, str]]) -> List[dict]:
    children_map = defaultdict(list)
    parents_map: Dict[str, str] = {}
    
    # Build parent-child relationships, ensuring each child has only ONE parent
    # If a child appears with multiple parents, only keep the first occurrence
    for parent, child in relations:
        if child not in parents_map:  # Only add if this child hasn't been assigned a parent yet
            children_map[parent].append(child)
            parents_map[child] = parent
        # else: skip duplicate parent-child relationship

    root_entities = [e for e in entities if e not in parents_map]
    if not root_entities:
        def resolve_root(label: str) -> str:
            seen: set[str] = set()
            curr = label
            while True:
                parent = parents_map.get(curr)
                if not parent or parent == curr:
                    return curr
                if parent in seen:
                    return parent
                seen.add(curr)
                curr = parent

        root_entities = sorted({resolve_root(e) for e in entities})

    def build_node(label: str, ancestors: Tuple[str, ...]) -> dict:
        if label in ancestors:
            return {"label": label, "children": []}
        next_ancestors = ancestors + (label,)
        child_nodes: List[dict] = []
        seen: set[str] = set()
        for child in children_map.get(label, []):
            if child in seen:
                continue
            seen.add(child)
            child_nodes.append(build_node(child, next_ancestors))
        return {"label": label, "children": child_nodes}

    roots = root_entities or sorted(set(entities))
    return [build_node(root, tuple()) for root in roots]
def load_flat_taxonomies(file_path: Path) -> List[Tuple[int, List[dict], Path]]:
    block_data = defaultdict(lambda: {"entities": set(), "relations": {}, "duplicate_parents": defaultdict(list)})
    with file_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            block_id = int(row["block_id"])
            child = row["child"].strip()
            parent = row["parent"].strip()
            block_data[block_id]["entities"].add(child)
            block_data[block_id]["entities"].add(parent)
            
            # Track if a child has multiple parents - only keep the first one
            if child not in block_data[block_id]["relations"]:
                block_data[block_id]["relations"][child] = parent
            else:
                # Log duplicate parent relationship
                existing_parent = block_data[block_id]["relations"][child]
                if existing_parent != parent:
                    block_data[block_id]["duplicate_parents"][child].append(parent)

    records: List[Tuple[int, List[dict], Path]] = []
    for block_id in sorted(block_data.keys()):
        entities = list(block_data[block_id]["entities"])
        
        # Convert dict back to list of tuples, filtering out duplicates
        relations = [(parent, child) for child, parent in block_data[block_id]["relations"].items()]
        
        # Report duplicate parents if any
        if block_data[block_id]["duplicate_parents"]:
            print(f"[Warning] Block {block_id}: Found nodes with multiple parents (keeping first occurrence):")
            for child, extra_parents in block_data[block_id]["duplicate_parents"].items():
                kept_parent = block_data[block_id]["relations"][child]
                print(f"  - '{child}': kept parent='{kept_parent}', ignored parents={extra_parents}")
        
        hierarchy = _build_hierarchy_from_relations(entities, relations)
        records.append((block_id, hierarchy, file_path))
    return records


def flatten_taxonomy(block_id: int, nodes: List[dict], source_file: Path) -> List[NodeRecord]:
    flattened: List[NodeRecord] = []
    counter = 0

    def visit(node: dict, parent_id: Optional[str], path_prefix: Tuple[str, ...]) -> str:
        nonlocal counter
        node_id = f"b{block_id:02d}:n{counter:04d}"
        counter += 1
        label = str(node.get("label") or "").strip() or f"[Unnamed {node_id}]"
        path = path_prefix + (label,)
        record = NodeRecord(
            node_id=node_id,
            block_id=block_id,
            label=label,
            parent_id=parent_id,
            path=path,
            depth=len(path),
            source_file=source_file,
        )
        record.embedding_text = " > ".join(path) or label
        flattened.append(record)
        for child in node.get("children", []):
            if isinstance(child, dict):
                child_id = visit(child, node_id, path)
                record.children_ids.append(child_id)
        return node_id

    for root in nodes:
        if isinstance(root, dict):
            visit(root, None, tuple())
    return flattened


# =============================================================================
# Embeddings
# =============================================================================


def _auto_select_device() -> str:
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


class _Specter2Encoder:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.device = _auto_select_device()
        self._tokenizer = None
        self._model = None

    def _ensure_backend(self) -> None:
        if self._model is not None:
            return
        if torch is None:
            raise RuntimeError("allenai/specter2 requires PyTorch.")
        try:
            from transformers import AutoTokenizer
            from adapters import AutoAdapterModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("allenai/specter2 requires transformers[adapters].") from exc
        self._tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self._model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self._model.load_adapter("allenai/specter2", source="hf", set_active=True)
        self._model.to(self.device)
        self._model.eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_backend()
        batches = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start:start + self.batch_size])
            inputs = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                batches.append(embeddings)
        matrix = np.vstack(batches)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


def build_embeddings(records: Sequence[NodeRecord], model_name: str, batch_size: int) -> Tuple[np.ndarray, str]:
    texts = [r.embedding_text for r in records]
    if model_name.lower().startswith("allenai/specter2"):
        encoder = _Specter2Encoder(batch_size)
        return encoder.encode(texts), encoder.device
    from sentence_transformers import SentenceTransformer
    device = _auto_select_device()
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32), device


# =============================================================================
# Fusion helpers
# =============================================================================


class CanonicalStore:
    """Manage canonical nodes while merging blocks sequentially."""

    def __init__(self, node_embeddings: Dict[str, np.ndarray]):
        self.node_embeddings = node_embeddings
        self.canonical_nodes: Dict[str, CanonicalNode] = {}
        self.node_to_canonical: Dict[str, str] = {}
        self.label_to_ids: Dict[str, List[str]] = defaultdict(list)
        self._counter = 0

    def find_merge_candidate(self, record: NodeRecord) -> Optional[str]:
        """Check if a node can be merged with existing canonical nodes (same label, different block)."""
        for cid in self.label_to_ids.get(record.label, []):
            canonical = self.canonical_nodes[cid]
            if record.block_id not in canonical.source_blocks:
                return cid
        return None

    def create_node(self, record: NodeRecord) -> str:
        """Create a new canonical node for the given record."""
        cid = self._new_canonical(record)
        self.node_to_canonical[record.node_id] = cid
        return cid

    def merge_node(self, record: NodeRecord, canonical_id: str) -> None:
        """Merge a record into an existing canonical node."""
        self._merge_into(canonical_id, record)
        self.node_to_canonical[record.node_id] = canonical_id

    # ------------------------------------------------------------------
    def _new_canonical(self, record: NodeRecord) -> str:
        cid = f"c{self._counter:05d}"
        self._counter += 1
        payload = self._make_source_payload(record)
        embedding = self._embedding_for(record.node_id)
        canonical = CanonicalNode(
            node_id=cid,
            label=record.label,
            source_nodes=[payload],
            source_blocks=[record.block_id],
            embedding=embedding,
            depth=record.depth,
            merge_size=1,
        )
        self.canonical_nodes[cid] = canonical
        self.label_to_ids[record.label].append(cid)
        return cid

    def _merge_into(self, canonical_id: str, record: NodeRecord) -> None:
        canonical = self.canonical_nodes[canonical_id]
        canonical.source_nodes.append(self._make_source_payload(record))
        canonical.source_blocks = sorted({*canonical.source_blocks, record.block_id})
        canonical.merge_size += 1
        canonical.depth = min(canonical.depth, record.depth)
        canonical.embedding = self._recompute_embedding(canonical)
        # Logging moved to caller for better context

    def _make_source_payload(self, record: NodeRecord) -> Dict[str, object]:
        return {
            "node_id": record.node_id,
            "block_id": record.block_id,
            "path": list(record.path),
            "label": record.label,
            "source_file": str(record.source_file) if record.source_file else None,
        }

    def _embedding_for(self, node_id: str) -> np.ndarray:
        vec = self.node_embeddings.get(node_id)
        if vec is None:
            return np.empty((0,), dtype=np.float32)
        return vec.astype(np.float32)

    def _recompute_embedding(self, canonical: CanonicalNode) -> np.ndarray:
        vectors = []
        for src in canonical.source_nodes:
            vec = self.node_embeddings.get(src["node_id"])
            if vec is not None and vec.size > 0:
                vectors.append(vec)
        if not vectors:
            return np.empty((0,), dtype=np.float32)
        matrix = np.vstack(vectors)
        centroid = matrix.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid.astype(np.float32)

    def populate_structure_candidates(self, records: Sequence[NodeRecord]) -> None:
        for record in records:
            canonical_id = self.node_to_canonical.get(record.node_id)
            if canonical_id is None:
                continue
            if record.parent_id:
                parent_canonical = self.node_to_canonical.get(record.parent_id)
                if parent_canonical and parent_canonical != canonical_id:
                    self.canonical_nodes[canonical_id].parent_candidates.add(parent_canonical)
                    self.canonical_nodes[parent_canonical].child_candidates.add(canonical_id)
            for child_id in record.children_ids:
                child_canonical = self.node_to_canonical.get(child_id)
                if child_canonical and child_canonical != canonical_id:
                    self.canonical_nodes[canonical_id].child_candidates.add(child_canonical)
                    self.canonical_nodes[child_canonical].parent_candidates.add(canonical_id)


# =============================================================================
# Stage 2: Merge blocks + attach nodes
# =============================================================================


def resolve_parent(
    node: CanonicalNode,
    candidate_ids: Sequence[str],
    canonical_nodes: Dict[str, CanonicalNode],
    node_paths: Dict[str, List[str]],
    children_map: Dict[str, List[str]],
    parent_map: Dict[str, Optional[str]],
    top_k: int,
    llm_client: Optional[LLMClient],
    query_children: Optional[List[str]] = None,
    root_label: str = "ROOT",
    original_had_parent: bool = False,
) -> Tuple[Optional[str], str]:
    ordered = dict.fromkeys(candidate_ids or [])
    candidates = [cid for cid in ordered if cid in canonical_nodes]
    if not candidates:
        print(f"    [resolve_parent] No valid candidates")
        return None, "no-candidates"

    scored = [
        (cid, float(np.dot(node.embedding, canonical_nodes[cid].embedding)))
        for cid in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    limit = top_k if top_k and top_k > 0 else len(scored)
    top_candidates = scored[:limit]
    if not top_candidates:
        print(f"    [resolve_parent] No candidates after scoring")
        return None, "no-candidates"
    
    print(f"    [resolve_parent] Top {len(top_candidates)} candidates by similarity:")
    for i, (cid, score) in enumerate(top_candidates[:5], 1):
        children_list = children_map.get(cid, [])
        children_labels = [canonical_nodes[child_id].label for child_id in children_list[:3]]
        children_preview = ", ".join(children_labels)
        if len(children_list) > 3:
            children_preview += f", ... (+{len(children_list) - 3} more)"
        print(f"      {i}. {cid} ('{canonical_nodes[cid].label}') - similarity: {score:.4f}, children: [{children_preview}]")

    # Always use LLM if available, even for single candidate (but ROOT only if node had no parent originally)
    if llm_client:
        if original_had_parent:
            print(f"    [resolve_parent] Using LLM to select best parent from {len(top_candidates)} candidates (ROOT not allowed - node had parent in original block)")
        else:
            print(f"    [resolve_parent] Using LLM to select best parent from {len(top_candidates)} candidates (including ROOT option)")
        
        query_source = node.source_nodes[0] if node.source_nodes else {}
        query = {
            "id": node.node_id,
            "label": node.label,
            "path_hint": query_source.get("path", []),
            "children": query_children or [],
        }
        
        candidates_payload = []
        
        # Only add ROOT option if the original node had no parent
        if not original_had_parent:
            # Prepare ROOT as first candidate - present it as a regular node option
            # Root children are nodes that currently have no parent (are at the top level)
            root_children_ids = [cid for cid, parent in parent_map.items() if parent is None]
            root_children_labels = [canonical_nodes[cid].label for cid in root_children_ids if cid in canonical_nodes]
            
            # Build candidates payload with ROOT at the front (presented as a normal candidate)
            candidates_payload.append({
                "id": "ROOT",
                "label": root_label,  # Use the actual root label like "Computing Classification System"
                "path": root_label,  # Just the root label as string
                "merge_size": len(root_children_ids),
                "children": root_children_labels,  # Show its current children like any other node
                "num_children": len(root_children_labels),
            })
            print(f"    [resolve_parent] Added '{root_label}' as first candidate option with {len(root_children_labels)} current children: {root_children_labels}")
        else:
            print(f"    [resolve_parent] ROOT option excluded (node had parent in original taxonomy)")
        
        # Add top-k candidates after ROOT
        for cid, score in top_candidates:
            children_list = children_map.get(cid, [])
            children_labels = [canonical_nodes[child_id].label for child_id in children_list]
            # Ensure path starts from ROOT and format as "A -> B -> C"
            node_path = node_paths.get(cid, [canonical_nodes[cid].label])
            full_path_list = [root_label] + node_path
            full_path_str = " -> ".join(full_path_list)
            payload = {
                "id": cid,
                "label": canonical_nodes[cid].label,
                "path": full_path_str,
                "merge_size": canonical_nodes[cid].merge_size,
                "children": children_labels,
                "num_children": len(children_labels),
            }
            candidates_payload.append(payload)
        
        try:
            prompt = get_parent_selection_prompt(query, candidates_payload)
            
            # Log LLM input
            print(f"\n{'='*80}")
            print(f"[LLM-INPUT] Query node: {node.node_id} ('{node.label}')")
            print(f"[LLM-INPUT] Number of candidates: {len(candidates_payload)}")
            print(f"[LLM-INPUT] Full prompt:")
            print(f"{'-'*80}")
            print(prompt)
            print(f"{'-'*80}")
            
            response = llm_client.complete(prompt)
            if response is None:
                print(f"    [resolve_parent] LLM response is None, falling back to similarity")
                print(f"    [resolve_parent] LLM error: LLMClient returned None (API may have failed or timed out)")
                # 直接 fallback，不再继续后续处理
            else:
                # Log LLM output
                print(f"\n[LLM-OUTPUT] Raw response:")
                print(f"{'-'*80}")
                print(response)
                print(f"{'-'*80}")
                print(f"{'='*80}\n")
                # Clean response - remove any leading/trailing text and markdown
                response_clean = response.strip()
                
                # Remove markdown code blocks if present
                if response_clean.startswith("```"):
                    lines = response_clean.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    response_clean = "\n".join(lines).strip()
                
                # Find JSON object boundaries (handle leading text like "After analyzing...")
                json_start = response_clean.find('{')
                json_end = response_clean.rfind('}')
                
                if json_start == -1 or json_end == -1 or json_end <= json_start:
                    print(f"    [resolve_parent] No valid JSON found in response")
                    print(f"    [resolve_parent] Raw response: {response_clean[:500]}")
                    print(f"    [resolve_parent] Falling back to similarity")
                else:
                    # Extract only the JSON portion
                    json_str = response_clean[json_start:json_end + 1]
                    
                    print(f"    [resolve_parent] LLM raw response: {response_clean[:200]}...")
                    print(f"    [resolve_parent] Extracted JSON: {json_str}")
                    
                    decision = json.loads(json_str)
                    selected = decision.get("selected_parent_id")
                    print(f"    [resolve_parent] LLM selected: {selected}")
                    print(
                        "[Stage2-LLM] Decision:",
                        json.dumps({"node": query, "decision": decision}, ensure_ascii=False),
                    )
                    
                    if selected == "ROOT":
                        if original_had_parent:
                            print(f"    [resolve_parent] LLM selected ROOT but node had parent - rejecting, falling back to similarity")
                        else:
                            print(f"    [resolve_parent] LLM chose ROOT")
                            return None, "llm-root"
                    elif selected in {cid for cid, _ in top_candidates}:
                        return selected, "llm"
                    else:
                        print(f"    [resolve_parent] LLM selection '{selected}' not in candidates, falling back")
        except json.JSONDecodeError as e:
            print(f"    [resolve_parent] JSON decode failed: {str(e)}")
            print(f"    [resolve_parent] Attempted to parse: {json_str if 'json_str' in locals() else response[:500]}")
            print(f"    [resolve_parent] Falling back to similarity")
        except Exception as e:
            print(f"    [resolve_parent] LLM error: {str(e)}")
            print(f"    [resolve_parent] Falling back to similarity")

    best_candidate = top_candidates[0][0]
    print(f"    [resolve_parent] Selected by similarity: {best_candidate} ('{canonical_nodes[best_candidate].label}')")
    return best_candidate, "similarity"


def determine_sibling_or_parent_relationship(
    query_node: CanonicalNode,
    query_block_id: int,
    selected_parent_id: str,
    canonical_nodes: Dict[str, CanonicalNode],
    children_map: Dict[str, List[str]],
    node_paths: Dict[str, List[str]],
    llm_client: Optional[LLMClient],
    root_label: str = "ROOT",
) -> Tuple[str, List[str], str]:
    """Determine if query node should be sibling or parent of existing children.
    
    Args:
        query_node: The node being placed
        selected_parent_id: The ID of the selected parent node
        canonical_nodes: All canonical nodes
        children_map: Parent -> children mapping
        node_paths: Node paths for building full path strings
        llm_client: LLM client for decision making
        root_label: Root label for path construction
        
    Returns:
        Tuple of (relationship_type, affected_children_ids, reasoning)
        - relationship_type: "sibling" or "parent"
        - affected_children_ids: list of child IDs to move under query (empty if sibling)
        - reasoning: explanation of the decision
    """
    from prompts import get_sibling_vs_parent_relationship_prompt
    
    existing_children_ids = children_map.get(selected_parent_id, [])
    cross_block_children_ids = []
    for cid in existing_children_ids:
        child_blocks = set(canonical_nodes[cid].source_blocks)
        if query_block_id not in child_blocks:
            cross_block_children_ids.append(cid)

    # If parent has no children, query must be a sibling (direct child)
    if not existing_children_ids:
        print(f"    [sibling-or-parent] Parent has no children - Q will be direct child (sibling level)")
        return "sibling", [], "parent-has-no-children"
    if not cross_block_children_ids:
        print("    [sibling-or-parent] No cross-block children - defaulting to sibling")
        return "sibling", [], "same-block-children"
    
    print(f"\n[Sibling-or-Parent Analysis] Query: {query_node.node_id} ('{query_node.label}')")
    print(f"  - Selected parent: {selected_parent_id} ('{canonical_nodes[selected_parent_id].label}')")
    print(f"  - Parent has {len(existing_children_ids)} existing children")
    print(f"  - Cross-block children considered: {len(cross_block_children_ids)}")
    
    # Prepare query node info
    query_source = query_node.source_nodes[0] if query_node.source_nodes else {}
    query_children_labels = []
    # Get children from original block structure if available
    if "path" in query_source:
        # Children info might not be directly available, use empty list
        pass
    
    query_payload = {
        "id": query_node.node_id,
        "label": query_node.label,
        "path_hint": query_source.get("path", []),
        "children": query_children_labels,
    }
    
    # Prepare parent info
    parent_node = canonical_nodes[selected_parent_id]
    parent_path = node_paths.get(selected_parent_id, [parent_node.label])
    parent_full_path = " -> ".join([root_label] + parent_path)
    
    parent_payload = {
        "id": selected_parent_id,
        "label": parent_node.label,
        "path": parent_full_path,
        "children": [canonical_nodes[cid].label for cid in cross_block_children_ids],
    }
    
    # Prepare existing children info
    children_payload = []
    for child_id in cross_block_children_ids:
        child_node = canonical_nodes[child_id]
        child_children_ids = children_map.get(child_id, [])
        child_children_labels = [canonical_nodes[ccid].label for ccid in child_children_ids]
        
        children_payload.append({
            "id": child_id,
            "label": child_node.label,
            "children": child_children_labels,
            "num_children": len(child_children_labels),
        })
    
    print(f"  - Query node children: {query_children_labels}")
    print(f"  - Existing children (cross-block only): {[c['label'] for c in children_payload]}")
    
    # Use LLM to determine relationship
    if llm_client:
        try:
            prompt = get_sibling_vs_parent_relationship_prompt(
                query_payload,
                parent_payload,
                children_payload
            )
            
            print(f"\n{'='*80}")
            print(f"[LLM-INPUT-SiblingOrParent] Query: {query_node.node_id} ('{query_node.label}')")
            print(f"[LLM-INPUT-SiblingOrParent] Full prompt:")
            print(f"{'-'*80}")
            print(prompt)
            print(f"{'-'*80}")
            
            response = llm_client.complete(prompt)
            
            print(f"\n[LLM-OUTPUT-SiblingOrParent] Raw response:")
            print(f"{'-'*80}")
            print(response)
            print(f"{'-'*80}")
            print(f"{'='*80}\n")
            
            # Parse response
            response_clean = response.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_clean = "\n".join(lines).strip()
            
            json_start = response_clean.find('{')
            json_end = response_clean.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = response_clean[json_start:json_end + 1]
                decision = json.loads(json_str)
                
                relationship = decision.get("relationship", "sibling")
                confidence = decision.get("confidence", 0.0)
                affected_children = decision.get("affected_children", [])
                
                print(f"    [sibling-or-parent] LLM decision: {relationship}")
                print(f"    [sibling-or-parent] Confidence: {confidence}")
                
                if relationship == "parent" and affected_children:
                    # Validate affected children exist
                    valid_affected = [cid for cid in affected_children if cid in canonical_nodes]
                    query_blocks = set(query_node.source_blocks)
                    filtered_affected = []
                    for cid in valid_affected:
                        child_blocks = set(canonical_nodes[cid].source_blocks)
                        if query_blocks & child_blocks:
                            continue
                        filtered_affected.append(cid)
                    if filtered_affected:
                        print(f"    [sibling-or-parent] Affected children: {[canonical_nodes[cid].label for cid in filtered_affected]}")
                        return "parent", filtered_affected, f"llm-parent (confidence: {confidence:.2f})"
                    print("    [sibling-or-parent] Affected children filtered out (same block as query)")
                else:
                    return "sibling", [], f"llm-sibling (confidence: {confidence:.2f})"
                    
        except json.JSONDecodeError as e:
            print(f"    [sibling-or-parent] JSON decode failed: {str(e)}")
            print(f"    [sibling-or-parent] Falling back to heuristic")
        except Exception as e:
            print(f"    [sibling-or-parent] LLM error: {str(e)}")
            print(f"    [sibling-or-parent] Falling back to heuristic")
    
    # LLM unavailable or failed: default to sibling without heuristic inference
    print(f"    [sibling-or-parent] LLM unavailable/failed: defaulting to sibling")
    return "sibling", [], "llm-fallback-sibling"


def build_fused_hierarchy(
    node_records: Sequence[NodeRecord],
    embeddings: np.ndarray,
    top_k: int,
    llm_client: Optional[LLMClient],
    root_label: str = "ROOT",
) -> Tuple[Dict[str, CanonicalNode], Dict[str, Optional[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    node_embeddings: Dict[str, np.ndarray] = {}
    for idx, record in enumerate(node_records):
        vector = embeddings[idx] if embeddings.size else np.empty((0,), dtype=np.float32)
        node_embeddings[record.node_id] = vector

    store = CanonicalStore(node_embeddings)
    parent_map: Dict[str, Optional[str]] = {}
    children_map: Dict[str, List[str]] = defaultdict(list)
    node_paths: Dict[str, List[str]] = {}
    placed: set[str] = set()

    block_nodes: Dict[int, List[NodeRecord]] = defaultdict(list)
    for record in node_records:
        block_nodes[record.block_id].append(record)
    if not block_nodes:
        return store.canonical_nodes, parent_map, children_map, node_paths

    block_order = sorted(block_nodes.keys(), key=lambda bid: (-len(block_nodes[bid]), bid))
    start_block = block_order[0]
    remaining_blocks = block_order[1:]

    def bfs_records(nodes: List[NodeRecord]) -> List[NodeRecord]:
        lookup = {r.node_id: r for r in nodes}
        roots = sorted([r for r in nodes if r.parent_id is None], key=lambda r: r.label.lower())
        queue: deque[NodeRecord] = deque(roots)
        seen: set[str] = set()
        order: List[NodeRecord] = []
        while queue:
            current = queue.popleft()
            if current.node_id in seen:
                continue
            seen.add(current.node_id)
            order.append(current)
            children = [lookup[cid] for cid in current.children_ids if cid in lookup]
            children.sort(key=lambda r: r.label.lower())
            queue.extend(children)
        return order

    def attach_node(node_id: str, parent_id: Optional[str], reason: str) -> None:
        parent_map[node_id] = parent_id
        if parent_id:
            if node_id not in children_map[parent_id]:
                children_map[parent_id].append(node_id)
            base = node_paths.get(parent_id, [store.canonical_nodes[parent_id].label])
            node_paths[node_id] = base + [store.canonical_nodes[node_id].label]
        else:
            node_paths[node_id] = [store.canonical_nodes[node_id].label]
        placed.add(node_id)
        parent_label = store.canonical_nodes[parent_id].label if parent_id else "[ROOT]"
        stage_label = "Stage1" if reason.startswith("seed") else "Stage2"
        print(
            f"[{stage_label}-Attach] {node_id} ('{store.canonical_nodes[node_id].label}') "
            f"→ {parent_label} (via {reason})"
        )

    # Seed hierarchy with the largest block
    print(f"[Stage1] Seeding hierarchy with block {start_block} ({len(block_nodes[start_block])} nodes)")
    for record in bfs_records(block_nodes[start_block]):
        canonical_id = store.create_node(record)
        if canonical_id in placed:
            continue
        
        parent_canonical = store.node_to_canonical.get(record.parent_id) if record.parent_id else None
        attach_parent = parent_canonical if parent_canonical in placed else None
        reason = "seed-parent" if attach_parent else "seed-root"
        
        # Print detailed decision flow for seed nodes too
        if not attach_parent:
            print(f"\n[Stage1-Decision] Seeding node {canonical_id} ('{record.label}')")
            print(f"  - Original block: {record.block_id}")
            print(f"  - Original parent: {record.parent_id}")
            print(f"  - Path: {' > '.join(record.path)}")
            if parent_canonical:
                print(f"  - Parent canonical: {parent_canonical} ('{store.canonical_nodes[parent_canonical].label}')")
                print(f"  - Parent in placed set: {parent_canonical in placed}")
                print(f"  - Decision: Attaching to [ROOT] (parent not yet placed)")
            else:
                print(f"  - Parent canonical: None")
                print(f"  - Decision: Attaching to [ROOT] (node is root in seed block)")
        
        attach_node(canonical_id, attach_parent, reason)

    def candidate_pool(parent_canonical: Optional[str]) -> List[str]:
        if parent_canonical and parent_canonical in placed:
            ordered = [parent_canonical]
            ordered.extend(children_map.get(parent_canonical, []))
            unique: List[str] = []
            for cid in ordered:
                if cid in placed and cid not in unique:
                    unique.append(cid)
            return unique
        # When parent_canonical is not available/placed, all placed nodes are candidates
        return list(placed)

    # Merge remaining blocks
    print(f"[Stage2] Merging {len(remaining_blocks)} remaining blocks into hierarchy")
    for block_id in remaining_blocks:
        print(f"[Stage2] Processing block {block_id} ({len(block_nodes[block_id])} nodes)")
        record_lookup = {r.node_id: r for r in block_nodes[block_id]}
        
        for record in bfs_records(block_nodes[block_id]):
            # Skip if this node was already processed (e.g., in a batch-attach)
            if record.node_id in store.node_to_canonical:
                continue
            
            # Merge disabled: always create a new canonical node and attach
            canonical_id = store.create_node(record)
            if canonical_id in placed:
                continue

            # Process node individually (batch attach removed)
            # Print detailed decision flow
            print(f"\n[Stage2-Decision] Attaching node {canonical_id} ('{record.label}')")
            print(f"  - Original block: {record.block_id}")
            print(f"  - Original parent: {record.parent_id}")
            print(f"  - Path: {' > '.join(record.path)}")
            
            parent_canonical = store.node_to_canonical.get(record.parent_id) if record.parent_id else None
            if parent_canonical:
                print(f"  - Parent canonical: {parent_canonical} ('{store.canonical_nodes[parent_canonical].label}')")
                print(f"  - Parent in placed set: {parent_canonical in placed}")
            else:
                print(f"  - Parent canonical: None (original node was root in its block)")
            
            candidates = candidate_pool(parent_canonical)
            print(f"  - Candidate pool size: {len(candidates)}")
            if candidates:
                candidate_labels = [f"{cid} ('{store.canonical_nodes[cid].label}')" for cid in candidates[:3]]
                print(f"  - Candidates: {candidate_labels}")
            
            chosen_parent: Optional[str] = None
            strategy = "no-candidates"
            if parent_canonical and parent_canonical in placed:
                existing_children_ids = children_map.get(parent_canonical, [])
                if existing_children_ids:
                    same_block_children = True
                    for cid in existing_children_ids:
                        child_blocks = set(store.canonical_nodes[cid].source_blocks)
                        if child_blocks != {record.block_id}:
                            same_block_children = False
                            break
                    if same_block_children:
                        chosen_parent = parent_canonical
                        strategy = "same-block-parent"
                        print("  - Using original parent (all existing children from same block)")
            if chosen_parent is None and len(candidates) == 1:
                chosen_parent = candidates[0]
                strategy = "single-candidate"
                print(f"  - Single candidate: {chosen_parent} ('{store.canonical_nodes[chosen_parent].label}')")
            elif chosen_parent is None and candidates:
                print(f"  - Resolving parent via embeddings/LLM...")
                # Determine if original node had a parent
                original_had_parent = record.parent_id is not None
                # Provide the query node's child labels to the LLM to give context about its local subtree
                query_child_labels = [
                    record_lookup[cid].label for cid in record.children_ids if cid in record_lookup
                ]
                chosen_parent, strategy = resolve_parent(
                    store.canonical_nodes[canonical_id],
                    candidates,
                    store.canonical_nodes,
                    node_paths,
                    children_map,
                    parent_map,
                    top_k,
                    llm_client,
                    query_children=query_child_labels,
                    root_label=root_label,
                    original_had_parent=original_had_parent,
                )
                if chosen_parent:
                    print(f"  - Resolved parent: {chosen_parent} ('{store.canonical_nodes[chosen_parent].label}') via {strategy}")
                else:
                    print(f"  - Resolved parent: None (no suitable candidate found)")
            
            if chosen_parent is None:
                if parent_canonical and parent_canonical in placed:
                    chosen_parent = parent_canonical
                    strategy = "direct-parent"
                    print(f"  - Fallback: Using direct parent {chosen_parent}")
                else:
                    chosen_parent = None
                    strategy = "root"
                    print(f"  - Fallback: Attaching to [ROOT] (no valid parent available)")
                    if parent_canonical and parent_canonical not in placed:
                        print(f"    Reason: Parent {parent_canonical} not yet placed in hierarchy")
                    elif not parent_canonical:
                        print(f"    Reason: Node was root in original block, no other suitable parent found")
            
            # Determine if query node should be sibling or parent of existing children
            if chosen_parent is not None and llm_client:
                existing_children_ids = children_map.get(chosen_parent, [])
                if not existing_children_ids:
                    relationship = "sibling"
                    affected_children = []
                    rel_reasoning = "parent-has-no-children"
                    print("  - Skipping sibling/parent decision (parent has no children)")
                else:
                    all_same_block = True
                    for cid in existing_children_ids:
                        child_blocks = set(store.canonical_nodes[cid].source_blocks)
                        if child_blocks != {record.block_id}:
                            all_same_block = False
                            break
                    if all_same_block:
                        relationship = "sibling"
                        affected_children = []
                        rel_reasoning = "same-block-children"
                        print("  - Skipping sibling/parent decision (children from same block)")
                    else:
                        relationship, affected_children, rel_reasoning = determine_sibling_or_parent_relationship(
                            store.canonical_nodes[canonical_id],
                            record.block_id,
                            chosen_parent,
                            store.canonical_nodes,
                            children_map,
                            node_paths,
                            llm_client,
                            root_label,
                        )
                
                if relationship == "parent" and affected_children:
                    # Query node should become parent of some existing children
                    print(f"  - Relationship decision: PARENT of {len(affected_children)} existing children")
                    print(f"  - Affected children: {[store.canonical_nodes[cid].label for cid in affected_children]}")
                    
                    # First attach query node to the selected parent
                    attach_node(canonical_id, chosen_parent, f"{strategy}+parent-layer")
                    
                    # Then re-attach affected children under query node
                    for child_id in affected_children:
                        if child_id in children_map.get(chosen_parent, []):
                            # Remove from old parent's children
                            children_map[chosen_parent].remove(child_id)
                            # Update parent map
                            parent_map[child_id] = canonical_id
                            # Add to new parent's children
                            if canonical_id not in children_map:
                                children_map[canonical_id] = []
                            if child_id not in children_map[canonical_id]:
                                children_map[canonical_id].append(child_id)
                            # Update path
                            base = node_paths.get(canonical_id, [store.canonical_nodes[canonical_id].label])
                            node_paths[child_id] = base + [store.canonical_nodes[child_id].label]
                            
                            print(f"    [Stage2-Restructure] Moved {child_id} ('{store.canonical_nodes[child_id].label}') under {canonical_id}")
                else:
                    # Query node is a sibling
                    print(f"  - Relationship decision: SIBLING of existing children")
                    attach_node(canonical_id, chosen_parent, f"{strategy}+sibling")
            else:
                # No LLM or no parent - just attach directly
                attach_node(canonical_id, chosen_parent, strategy)

    # Sort children for deterministic output
    for children in children_map.values():
        children.sort(key=lambda cid: store.canonical_nodes[cid].label.lower())

    # Populate candidate sets for downstream inspection
    store.populate_structure_candidates(node_records)

    return store.canonical_nodes, parent_map, children_map, node_paths


# =============================================================================
# Materialisation
# =============================================================================


def materialise_taxonomy(
    canonical_nodes: Dict[str, CanonicalNode],
    parent_map: Dict[str, Optional[str]],
    children_map: Dict[str, List[str]],
    root_label: str,
    root_description: str,
) -> Dict[str, object]:
    def build_subtree(node_id: str) -> Dict[str, object]:
        node = canonical_nodes[node_id]
        children = [build_subtree(cid) for cid in children_map.get(node_id, [])]
        return node.to_dict(children)

    roots = sorted([nid for nid, parent in parent_map.items() if parent is None],
                   key=lambda cid: canonical_nodes[cid].label.lower())
    return {
        "label": root_label,
        "description": root_description,
        "children": [build_subtree(root) for root in roots],
    }


# =============================================================================
# LLM setup
# =============================================================================


def build_llm_client(args: argparse.Namespace) -> Optional[LLMClient]:
    if not args.use_llm:
        return None
    if LLMClient is None or LLMConfig is None:
        raise RuntimeError(
            "LLM support requires llm_client.py to be available, but it could not be imported."
        )
    
    # Get API key from args or environment variable
    import os
    api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not provided. Set --llm-api-key or OPENAI_API_KEY environment variable."
        )
    
    # Build config - only include max_new_tokens if specified (not needed for GPT-5)
    config_kwargs = {
        "provider": args.llm_provider,
        "model": args.llm_model,
        "api_base": args.llm_api_base,
        "api_key": api_key,
        "device": "auto",
    }
    
    # Only add max_new_tokens if explicitly provided (GPT-5 doesn't need it)
    if args.llm_max_tokens is not None:
        config_kwargs["max_new_tokens"] = args.llm_max_tokens
    
    config = LLMConfig(**config_kwargs)
    
    print(f"[LLM Config] Provider: {args.llm_provider}")
    print(f"[LLM Config] Model: {args.llm_model}")
    if args.llm_api_base:
        print(f"[LLM Config] API Base: {args.llm_api_base}")
    if args.llm_max_tokens is not None:
        print(f"[LLM Config] Max Tokens: {args.llm_max_tokens}")
    else:
        print(f"[LLM Config] Max Tokens: Not specified (using model default)")
    
    return LLMClient(config)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = parse_args()
    print(f"Loading taxonomies from {args.input_file}...")
    block_records = load_flat_taxonomies(args.input_file)
    flattened_nodes: List[NodeRecord] = []
    for block_id, taxonomy, source_file in block_records:
        flattened_nodes.extend(flatten_taxonomy(block_id, taxonomy, source_file))
    print(f"Loaded {len(flattened_nodes)} nodes from {len(block_records)} blocks")

    print(f"Generating embeddings using {args.embedding_model}...")
    embeddings, device = build_embeddings(flattened_nodes, args.embedding_model, args.batch_size)
    print(f"Generated embeddings on {device}")

    llm_client = build_llm_client(args)
    if llm_client:
        print("Stage 1: Seed largest block as base hierarchy")
        print("Stage 2: Merge taxonomies - merge identical labels, attach unique nodes with LLM assistance...")
    else:
        print("Stage 1: Seed largest block as base hierarchy")
        print("Stage 2: Merge taxonomies - merge identical labels, attach unique nodes via structural heuristics...")

    canonical_nodes, parent_map, children_map, _ = build_fused_hierarchy(
        flattened_nodes,
        embeddings,
        args.parent_top_k,
        llm_client,
        args.root_label,
    )

    taxonomy = materialise_taxonomy(
        canonical_nodes,
        parent_map,
        children_map,
        args.root_label,
        args.root_description,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "source_file": str(args.input_file),
            "blocks_processed": sorted(list({block_id for block_id, _, _ in block_records})),
            "embedding_model": args.embedding_model,
            "merge_threshold": args.merge_threshold,
            "parent_top_k": args.parent_top_k,
            "use_llm": bool(llm_client),
            "total_input_nodes": len(flattened_nodes),
            "canonical_nodes": len(canonical_nodes),
        },
        "taxonomy": taxonomy,
    }

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ Unified taxonomy written to {args.output}")
    print(f"✓ Processed {len(block_records)} blocks")
    print(f"✓ Merged {len(flattened_nodes)} nodes → {len(canonical_nodes)} canonical nodes")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
