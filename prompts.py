#!/usr/bin/env python3
"""
Prompts for in-block taxonomy building (Stage-G, Stage-A) and recursion control.
"""

import json
from textwrap import dedent
from typing import Any, Dict, List, Optional, Sequence


def _sanitize_for_prompt(value: Any, max_len: int = 320) -> Any:
    """Normalize objects for inclusion in prompt JSON blocks."""
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        cleaned = " ".join(value.replace("\r", " ").replace("\n", " ").split())
        if max_len and len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 3].rstrip() + "..."
        return cleaned
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, val in value.items():
            if val is None:
                continue
            sanitized[str(_sanitize_for_prompt(key, max_len=max_len))] = _sanitize_for_prompt(
                val, max_len=max_len
            )
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_prompt(item, max_len=max_len) for item in value if item is not None]
    return str(value)

def get_chunk_direction_analysis_prompt(terms: list) -> str:
    """DEPRECATED: no longer used."""
    return "{}"

def get_stage_g_parent_generation_prompt(
    terms: list,
    min_parents: Optional[int] = 3,
    max_parents: Optional[int] = 7,
    parent_label: Optional[str] = None,
    ancestor_labels: Optional[List[str]] = None,
    forbidden_labels: Optional[List[str]] = None,
) -> str:
    """Create a prompt for Stage-G: generate parent categories for a set of method terms.

    If min_parents/max_parents is None, let the LLM decide a reasonable width.
    Returns JSON with categories: [{label, description, relation_to_parent}]
    """
    # Ensure UTF-8 strings
    clean_terms = []
    for term in terms:
        if isinstance(term, str):
            clean_terms.append(term.encode('utf-8').decode('utf-8'))
        else:
            clean_terms.append(str(term).encode('utf-8').decode('utf-8'))

    parent_label = (parent_label or "").encode('utf-8').decode('utf-8')
    ancestor_labels = ancestor_labels or []
    ancestor_labels = [
        (a if isinstance(a, str) else str(a)).encode('utf-8').decode('utf-8')
        for a in ancestor_labels
    ]
    forbidden_labels = forbidden_labels or []
    forbidden_labels = [
        (a if isinstance(a, str) else str(a)).encode('utf-8').decode('utf-8')
        for a in forbidden_labels
    ]

    # Width guidance text
    if min_parents is None or max_parents is None:
        width_text = "a reasonable number of"
        width_rule = "- Choose as many categories as necessary to achieve coherent grouping, but avoid over-fragmentation."
    else:
        width_text = f"{min_parents} to {max_parents}"
        width_rule = f"- Propose between {min_parents} and {max_parents} categories (inclusive); prefer the fewest that still maintain coherence."

    return f"""
You are building a hierarchical taxonomy for research "methods". Given a list of method-related terms{(' under the parent category: ' + parent_label) if parent_label else ''}, propose {width_text} high-quality child categories that best organize these terms.

Requirements:
- Categories must be specific methodological areas (e.g., "Deep Learning Architectures", "Optimization & Search", "Representation Learning", "Retrieval & Indexing", "Evaluation & Analysis Methods").
- Avoid duplicates and excessive overlap; use concise, precise labels.
 - Each category must include a 1–2 sentence description.
 - For each child, include a short "relation_to_parent" string (e.g., "is-a specialization of <parent>", "methodological sub-type of <parent>"). Be explicit about how the child refines the parent.
- Cover the term space as much as possible with minimal overlap. If some terms don’t fit, they can be handled later.
{width_rule}
- Child categories must be proper subcategories of the parent (if given). Do not reuse the parent label verbatim.
- Do not repeat any ancestor labels: {json.dumps(ancestor_labels, ensure_ascii=False)}
- Do not use any of the forbidden labels: {json.dumps(forbidden_labels, ensure_ascii=False)}
- Prefer fewer, coherent child categories whose assigned terms will be meaningful groups; aim for each child to collect at least 5 terms. If a term doesn’t clearly fit any child, leave it at the parent.
 - Keep sibling categories at comparable granularity (avoid mixing overly coarse and overly fine siblings in the same split).
 - Output must include ONLY the fields specified (label, description, relation_to_parent). Do NOT include any other fields like "terms" or counts.

Input terms (method-related):
{json.dumps(clean_terms, ensure_ascii=False, indent=2)}

Return strictly in this JSON format (no extra text):
{{
  "categories": [
    {{
      "label": "Deep Learning Architectures",
      "description": "Neural network model families and architectural variations used for NLP/ML.",
      "relation_to_parent": "is-a specialization of methods"
    }}
  ]
}}
"""


def get_stage_a_assignment_prompt(terms: list, categories: list, allow_multi_assign: bool = False, compact_mode: bool = True) -> str:
    """Create a prompt for Stage-A: assign terms to the given parent categories.

    categories is a list of dicts with label and description
    Returns JSON with assignments per label; optionally an "unassigned" array.
    """
    # Ensure UTF-8
    clean_terms = []
    for term in terms:
        if isinstance(term, str):
            clean_terms.append(term.encode('utf-8').decode('utf-8'))
        else:
            clean_terms.append(str(term).encode('utf-8').decode('utf-8'))

    # Lightweight sanitization of categories
    clean_categories = []
    for cat in categories:
        clean_categories.append({
            "label": str(cat.get("label", "")).encode('utf-8').decode('utf-8'),
            "description": str(cat.get("description", "")).encode('utf-8').decode('utf-8')
        })

    multi_text = "one or more" if allow_multi_assign else "exactly one"

    if compact_mode:
        indexed = [f"{i}: {t}" for i, t in enumerate(clean_terms)]
        return f"""
Assign each input term to {multi_text} of the given parent categories.

Rules:
- Use the category labels and descriptions as guidance.
- Use only the provided category labels; do not invent new ones.
- Use the index-based format to avoid long outputs.
- Do not repeat any index more than once per label.
- If a term does not fit any category, list its index under "unassigned_indices".
- Do not withhold assignment due to batch size; decide per term based on semantics.
- Use "unassigned_indices" sparingly — only when a term clearly does not fit any label.
- Evaluate ALL indexed terms and include all that clearly fit each category.
- Coverage requirement (when multi-assign is {str(allow_multi_assign).lower()}): every index from 0..{len(clean_terms)-1} must appear either under exactly one label (if {str(allow_multi_assign).lower()} == false) or one or more labels (if true), OR in "unassigned_indices". Do not omit any index.
- Return strictly valid JSON. No extra text.

Categories:
{json.dumps(clean_categories, ensure_ascii=False, indent=2)}

Indexed terms (index: term):
{json.dumps(indexed, ensure_ascii=False, indent=2)}

Return strictly in this JSON format:
{{
  "assignments_indexed": [
    {{"label": "Deep Learning Architectures", "indices": [0, 5, 9]}},
    {{"label": "Optimization & Search", "indices": [1, 2]}}
  ],
  "unassigned_indices": [7]
}}
"""
    else:
        return f"""
Assign each input term to {multi_text} of the given parent categories.

Rules:
- Use the category labels and descriptions as guidance.
- Prefer balanced, coherent groups; avoid dumping everything into a single category. Do not withhold assignment due to batch size.
- Use only the provided category labels; do not invent new ones.
- Include all terms in assignments unless a term clearly does not fit anywhere.
- If a term does not fit any category, put it into "unassigned".
- Do not repeat any term more than once per label.
- Return strictly valid JSON.

Categories:
{json.dumps(clean_categories, ensure_ascii=False, indent=2)}

Terms to assign:
{json.dumps(clean_terms, ensure_ascii=False, indent=2)}

Return strictly in this JSON format (no extra text):
{{
  "assignments": [
    {{"label": "Deep Learning Architectures", "terms": ["term1", "term2"]}},
    {{"label": "Optimization & Search", "terms": ["term3"]}}
  ],
  "unassigned": ["termX"]
}}
"""


def get_stage_a_batch_prompt(indexed_terms: List[dict], categories: list, allow_multi_assign: bool = False) -> str:
    """Create a prompt for Stage-A in mini-batches.

    indexed_terms: list of objects {"index": int, "term": str}
    categories: list of dicts with label and description
    Returns JSON:
    {
      "results": [ {"index": 0, "labels": ["LabelA", "LabelB"]}, ...],
      "unassigned_indices": [ ... ]
    }
    """
    clean_categories = []
    for cat in categories:
        clean_categories.append({
            "label": str(cat.get("label", "")).encode('utf-8').decode('utf-8'),
            "description": str(cat.get("description", "")).encode('utf-8').decode('utf-8')
        })

    clean_batch = []
    for it in indexed_terms:
        idx = int(it.get("index", -1))
        t = it.get("term", "")
        t = (t if isinstance(t, str) else str(t)).encode('utf-8').decode('utf-8')
        clean_batch.append({"index": idx, "term": t})

    multi_text = "one or more" if allow_multi_assign else "exactly one"

    return f"""
You are classifying a small batch of research method terms into the provided parent categories.

Rules:
- Assign each input term to {multi_text} of the given parent category labels.
- Use the category labels and descriptions as guidance.
- Do NOT invent new labels; use only the provided label_options.
- Avoid tiny groups: prefer child categories that collect at least ~5 terms globally; if unsure, leave the term unassigned in this batch.
- Coverage requirement: every input index must appear either in "results" (with one or more labels if multi-assign is allowed, otherwise exactly one) or in "unassigned_indices".
- Return strictly valid JSON. No extra text.

label_options:
{json.dumps([c['label'] for c in clean_categories], ensure_ascii=False, indent=2)}

Categories (with descriptions):
{json.dumps(clean_categories, ensure_ascii=False, indent=2)}

Batch to classify (index, term):
{json.dumps(clean_batch, ensure_ascii=False, indent=2)}

Return strictly in this JSON format:
{{
  "results": [
    {{"index": 0, "labels": ["Deep Learning Architectures"]}},
    {{"index": 1, "labels": ["Optimization & Search", "Representation Learning"]}}
  ],
  "unassigned_indices": [3]
}}
"""

def get_recursion_decision_prompt(terms: list, parent_label: str, parent_description: str, depth: int) -> str:
    """Ask the LLM whether to further subdivide this subset under a parent category.

    Returns strictly JSON: {"decision": "split"|"stop", "rationale": "...", "confidence": 0.0-1.0}
    """
    clean_terms = []
    for term in terms:
        if isinstance(term, str):
            clean_terms.append(term.encode('utf-8').decode('utf-8'))
        else:
            clean_terms.append(str(term).encode('utf-8').decode('utf-8'))

    parent_label = (parent_label or "").encode('utf-8').decode('utf-8')
    parent_description = (parent_description or "").encode('utf-8').decode('utf-8')

    return f"""
You are controlling recursive taxonomy expansion for research methods.
Given a parent category and its assigned terms, decide whether to further subdivide (split) or stop.

Consider:
- Term diversity and heterogeneity within this subset
- Whether splitting would produce coherent, meaningful subcategories
- Avoid over-fragmentation; stop if the subset is already cohesive or too small to split meaningfully
- Prefer not to split if resulting child categories would each cover fewer than ~5 terms; in such cases keep terms at the parent.
 - Only split if you can propose at least two child categories at comparable granularity with clear parent→child relations (each child is a refinement/specialization of the parent).

Parent category: {json.dumps(parent_label, ensure_ascii=False)}
Description: {json.dumps(parent_description, ensure_ascii=False)}
Depth: {depth}

Assigned terms:
{json.dumps(clean_terms, ensure_ascii=False, indent=2)}

Return strictly JSON with fields:
{{
  "decision": "split" | "stop",
  "rationale": "one or two sentences",
  "confidence": 0.0
}}
"""


def get_merge_decision_prompt(group_id: str, nodes: Sequence[Dict[str, object]], edges: Sequence[Dict[str, object]]) -> str:
    """Prompt the LLM to evaluate whether similar taxonomy nodes should be merged."""

    def _safe_join(items: Optional[Sequence[str]]) -> str:
        if not items:
            return "(none)"
        if isinstance(items, str):
            return items
        cleaned = []
        for item in items:
            if item is None:
                continue
            cleaned.append(str(item))
        return ", ".join(cleaned) if cleaned else "(none)"

    node_blocks = []
    for idx, node in enumerate(nodes, start=1):
        label = str(node.get("label") or "[missing label]")
        path = node.get("path") or []
        if not isinstance(path, list):
            path = [str(path)]
        path_str = " > ".join(str(token) for token in path if token) or "[root]"
        description = str(node.get("description") or "")
        relation = str(node.get("relation_to_parent") or "")
        child_labels = _safe_join(node.get("child_labels"))
        snippet = str(node.get("document") or "")[:400]
        node_blocks.append(
            {
                "index": idx,
                "id": str(node.get("id")),
                "label": label,
                "path": path_str,
                "description": description,
                "relation": relation,
                "child_labels": child_labels,
                "snippet": snippet,
            }
        )

    edge_blocks = []
    for edge in edges:
        edge_blocks.append(
            {
                "source": str(edge.get("source")),
                "target": str(edge.get("target")),
                "distance": float(edge.get("distance", 0.0)),
            }
        )

    return f"""
You are an expert taxonomy curator. Review the following group of potentially duplicate categories and decide whether they should be merged into a single node.

Consider:
- How aligned the category scopes are (labels, descriptions, parent paths)
- Whether a merged node would remain coherent and properly placed in the hierarchy
- Whether any node should stay separate (e.g., materially different focus or parent path)

Candidate group ID: {json.dumps(group_id, ensure_ascii=False)}

Nodes:
{json.dumps(node_blocks, ensure_ascii=False, indent=2)}

Pairwise distances (lower = more similar):
{json.dumps(edge_blocks, ensure_ascii=False, indent=2)}

If a merge is appropriate, propose a consolidated label, description, relation_to_parent, and target parent path. If not, explain briefly.

Return STRICT JSON:
{{
  "merge": true,
  "group_id": {json.dumps(group_id, ensure_ascii=False)},
  "merged_label": "...",
  "merged_description": "...",
  "merged_relation_to_parent": "...",
  "target_parent_path": ["Parent", "Path"],
  "justification": "",
  "notes": ["optional"]
}}

If you decline to merge, respond with:
{{
  "merge": false,
  "group_id": {json.dumps(group_id, ensure_ascii=False)},
  "justification": "",
  "suggested_merges": [
    {{
      "members": ["label1", "label2"],
      "merged_label": "...",
      "merged_description": "...",
      "merged_relation_to_parent": "...",
      "target_parent_path": ["Parent", "Path"]
    }}
  ],
  "notes": ["optional"]
}}
If you have no pairing recommendations, return an empty array for "suggested_merges".
"""
def get_parent_selection_prompt(query_node: Dict[str, object], candidates: Sequence[Dict[str, object]]) -> str:
    """Prompt the LLM to pick the best parent path for an unmerged node."""

    def _clean_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    clean_query = {
        "id": _clean_text(query_node.get("id")),
        "label": _clean_text(query_node.get("label")),
    }
    path_hint = [
        _clean_text(token) for token in query_node.get("path_hint", []) if token
    ]
    if path_hint:
        clean_query["path_hint"] = path_hint
    if query_node.get("description"):
        clean_query["description"] = _clean_text(query_node.get("description"))
    if query_node.get("relation_to_parent"):
        clean_query["relation_to_parent"] = _clean_text(query_node.get("relation_to_parent"))
    representative_terms = [
        _clean_text(term)
        for term in query_node.get("representative_terms", [])
        if term
    ]
    if representative_terms:
        clean_query["representative_terms"] = representative_terms
    # Include query node's children to help LLM understand its granularity
    children = [
        _clean_text(child) for child in query_node.get("children", []) if child
    ]
    if children:
        clean_query["children"] = children
        clean_query["num_children"] = len(children)

    clean_candidates = []
    for candidate in candidates:
        payload = {
            "id": _clean_text(candidate.get("id")),
            "label": _clean_text(candidate.get("label")),
            "merge_size": int(candidate.get("merge_size", 1)),
            "num_children": int(candidate.get("num_children", 0)),
        }
        # Handle path - it's now a string like "A -> B -> C", not a list
        path_value = candidate.get("path")
        if path_value:
            if isinstance(path_value, str):
                payload["path"] = _clean_text(path_value)
            else:
                # Fallback for old list format
                path_tokens = [_clean_text(token) for token in path_value if token]
                if path_tokens:
                    payload["path"] = path_tokens
        if candidate.get("description"):
            payload["description"] = _clean_text(candidate.get("description"))
        children = [
            _clean_text(child) for child in candidate.get("children", []) if child
        ]
        if children:
            payload["children"] = children
        clean_candidates.append(payload)

    return f"""
Attach the query concept to the most appropriate parent in the unified taxonomy.

Query concept:
{json.dumps(clean_query, ensure_ascii=False, indent=2)}

Candidate parents:
{json.dumps(clean_candidates, ensure_ascii=False, indent=2)}

You are an expert in constructing a STRICT HIERARCHICAL TAXONOMY.

Your task is to select the SINGLE BEST parent for the query concept Q
from the given candidate parents.

IMPORTANT: Do NOT restructure the taxonomy.
Your task is ONLY to select the most appropriate parent for Q.

--------------------------------------------------
Decision Priority (highest → lowest):

1. Subsumption correctness (MUST PASS - CRITICAL):
   - P subsumes Q means: ALL major subareas of Q must fit semantically under P.
   - Use Q's children as PRIMARY EVIDENCE: Check EACH child of Q individually.
   - If even ONE child of Q does NOT belong under P, then P FAILS subsumption.
   - Examples:
     * "Applied computing" has children like "Aerospace", "Robotics", "Electronics"
     * These do NOT fit under "Mathematics of computing" → FAIL subsumption
     * They DO fit under "Computing Classification System" (ROOT) → PASS subsumption
   - STRICT RULE: If no non-ROOT candidate passes subsumption for ALL children, you MUST choose ROOT.
   - Do not rationalize or make exceptions - check every child explicitly.

2. Granularity alignment using children:
   - Use BOTH Q's children (if any) AND each candidate parent's existing children.
   - If Q has children, Q should be placed so that its children are peers of
     (or slightly more specific than) the candidate parent's children.
   - If Q's children are much broader/more diverse than P's children, Q doesn't belong under P.

3. Depth and level continuity:
   - Prefer placements that avoid skipping hierarchy levels.
   - Maintain a clean and meaningful depth progression.

4. Breadth of the candidate parent:
   - Very broad parents with many children are appropriate for general concepts.
   - Prefer the parent that best semantically subsumes Q, not just the closest by name.

--------------------------------------------------
CRITICAL CHECK before selecting ANY non-ROOT parent P:
1. List ALL children of Q
2. For EACH child, ask: "Does this child semantically belong under P?"
3. If ANY answer is "NO", then P is INVALID - choose ROOT instead.

--------------------------------------------------
Failure handling:

- If NONE of the candidate parents properly subsume Q,
  select ROOT as the parent.
- When in doubt about subsumption, choose ROOT (safer than incorrect placement).

--------------------------------------------------
Output format (STRICT JSON, no extra text):

{{
  "selected_parent_id": "<candidate_id or ROOT>"
}}
"""


def get_parent_selection_prompt_(query_node: Dict[str, object], candidates: Sequence[Dict[str, object]]) -> str:
    """
    Backward-compatible alias for get_parent_selection_prompt.
    
    Older scripts referenced this helper; keep behavior identical to the
    canonical interface so Stage-2 fusion logic can reuse a single prompt.
    """
    return get_parent_selection_prompt(query_node, candidates)


def get_parent_selection_prompt_1(query_node: Dict[str, object], candidates: Sequence[Dict[str, object]]) -> str:
    """Legacy alias retained for compatibility with historical pipelines."""
    return get_parent_selection_prompt(query_node, candidates)


def get_sibling_vs_parent_relationship_prompt(
    query_node: Dict[str, object],
    parent_node: Dict[str, object],
    existing_children: Sequence[Dict[str, object]],
) -> str:
    """Determine if query node should be a sibling or parent of existing children.
    
    Args:
        query_node: The node being placed, with label, children, path_hint
        parent_node: The selected parent node with label, path, children
        existing_children: Current children of parent_node
        
    Returns:
        A prompt asking LLM to classify relationships
    """
    def _clean_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)
    
    clean_query = {
        "id": _clean_text(query_node.get("id")),
        "label": _clean_text(query_node.get("label")),
        "children": [_clean_text(c) for c in query_node.get("children", []) if c],
        "num_children": len(query_node.get("children", [])),
    }
    if query_node.get("path_hint"):
        clean_query["path_hint"] = [_clean_text(p) for p in query_node.get("path_hint", []) if p]
    if query_node.get("description"):
        clean_query["description"] = _clean_text(query_node.get("description"))
    
    clean_parent = {
        "id": _clean_text(parent_node.get("id")),
        "label": _clean_text(parent_node.get("label")),
        "path": _clean_text(parent_node.get("path", parent_node.get("label"))),
    }
    
    clean_children = []
    for child in existing_children:
        clean_child = {
            "id": _clean_text(child.get("id")),
            "label": _clean_text(child.get("label")),
            "children": [_clean_text(c) for c in child.get("children", []) if c],
            "num_children": len(child.get("children", [])),
        }
        if child.get("description"):
            clean_child["description"] = _clean_text(child.get("description"))
        clean_children.append(clean_child)
    
    return f"""
You are determining the hierarchical relationship between a query node Q and existing children of its selected parent P.

Query node Q:
{json.dumps(clean_query, ensure_ascii=False, indent=2)}

Selected parent P:
{json.dumps(clean_parent, ensure_ascii=False, indent=2)}

Existing children of P:
{json.dumps(clean_children, ensure_ascii=False, indent=2)}

Your task:
Determine if Q should be:
1. A SIBLING: Q is at the same level as some/all existing children (direct child of P)
2. A PARENT: Q should become the parent of some existing children (intermediate layer between P and those children)

Analysis guidelines:
- Compare Q's granularity with existing children based on:
  * Label specificity and scope
  * Number and nature of Q's children vs existing children's children
  * Semantic relationships and domain coverage
  
- Q should be a SIBLING if:
  * Q's scope is comparable to existing children
  * Q covers a distinct area not overlapping with existing children
  * Q and existing children are at similar abstraction levels
  
- Q should be a PARENT (to some children) if:
  * Q's label is more general and subsumes some existing children
  * Q represents a broader category that naturally groups some existing children
  * Inserting Q as intermediate layer improves hierarchy clarity
  
- When Q is a PARENT, identify which specific existing children should be moved under Q

Output format (STRICT JSON):
{{
  "relationship": "sibling" | "parent",
  "affected_children": ["child_id1", "child_id2"]
}}

If relationship is "sibling", affected_children should be empty [].
If relationship is "parent", affected_children should list IDs of children that should move under Q.
"""


def get_sibling_harmonization_prompt(parent_node: Dict[str, object], siblings: Sequence[Dict[str, object]]) -> str:
    """Prompt the LLM to harmonise sibling labels/descriptions under a parent."""

    def _clean_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    parent_payload = {
        "id": _clean_text(parent_node.get("id")),
        "label": _clean_text(parent_node.get("label")),
        "path": [
            _clean_text(token) for token in parent_node.get("path", []) if token
        ],
    }
    sibling_payload = []
    for sibling in siblings:
        sibling_payload.append(
            {
                "id": _clean_text(sibling.get("id")),
                "label": _clean_text(sibling.get("label")),
                "description": _clean_text(sibling.get("description")),
                "path": [
                    _clean_text(token) for token in sibling.get("path", []) if token
                ],
                "terms": [
                    _clean_text(term) for term in sibling.get("terms", []) if term
                ],
            }
        )

    return f"""
You are harmonising the labels and descriptions of sibling categories so that they exhibit consistent granularity and non-overlapping scopes.

Parent context:
{json.dumps(parent_payload, ensure_ascii=False, indent=2)}

Sibling categories:
{json.dumps(sibling_payload, ensure_ascii=False, indent=2)}

Return STRICT JSON:
{{
  "updated_labels": [
    {{"id": "child_id", "label": "refined label", "description": "optional revision"}}
  ],
  "summary": "brief rationale"
}}
If no adjustments are required, return an empty list for "updated_labels".
"""


def get_dimension_prompt(dimension: str, terms: list) -> str:
    """DEPRECATED: no longer used."""
    return "[]"


def get_hierarchical_assignment_prompt(
    items_summary: Sequence[Dict],
    categories: Sequence[Dict],
    parent_label: str,
) -> str:
    sanitized_items = [_sanitize_for_prompt(item) for item in items_summary]
    sanitized_categories = [_sanitize_for_prompt(cat) for cat in categories]
    sanitized_parent = _sanitize_for_prompt(parent_label or "ROOT")

    return f"""
Assign each existing subtree to the most appropriate child category under "{sanitized_parent}".

Instructions:
- Use only the provided category labels; do not invent new categories.
- Assign every subtree to exactly one category unless it clearly does not belong; such cases go to "unassigned".
- Default to placing each subtree into its best-fit category. Treat "unassigned" as a last resort reserved for cases that would substantially distort every available category.
- Prefer fitting subtrees into existing categories even if the match is imperfect; choose the closest meaningful peer rather than abandoning the item.
- Consider descriptions, relations, child previews, and "terms_count" statistics when deciding.
- Balance the distribution so that categories remain meaningful peer groups.
- Never expose raw term strings — rely on summaries only.

Categories:
{json.dumps(sanitized_categories, ensure_ascii=False, indent=2)}

Subtrees to place:
{json.dumps(sanitized_items, ensure_ascii=False, indent=2)}

Return strictly valid JSON:
{{
  "assignments": [
    {{
      "category": "Category Label",
      "item_ids": ["subtree_id"]
    }}
  ],
  "unassigned": ["subtree_id"]
}}
"""


def get_hierarchical_single_assignment_prompt(
    item_summary: Dict,
    categories: Sequence[Dict],
    parent_label: str,
) -> str:
    sanitized_item = _sanitize_for_prompt(item_summary)
    sanitized_categories = [_sanitize_for_prompt(cat) for cat in categories]
    sanitized_parent = _sanitize_for_prompt(parent_label or "ROOT")

    return f"""
You are assigning a single existing subtree to the most appropriate child category under "{sanitized_parent}".

Instructions:
- Choose the closest matching category from the provided list; do not invent new categories.
- Default to assigning the subtree even if the match is imperfect, provided the category description can plausibly cover it.
- Only set "category" to null when every option would clearly misrepresent the subtree.
- Provide a concise justification referencing distinguishing metadata (descriptions, relations, child previews, counts).
- Include a confidence score between 0.0 and 1.0 reflecting how well the subtree fits the selected category.

Candidate categories:
{json.dumps(sanitized_categories, ensure_ascii=False, indent=2)}

Single subtree summary:
{json.dumps(sanitized_item, ensure_ascii=False, indent=2)}

Return strictly valid JSON:
{{
  "item_id": {json.dumps(sanitized_item.get('id', ''), ensure_ascii=False)},
  "category": "Category Label or null",
  "confidence": 0.0,
  "justification": "..."
}}

If no category fits, set "category" to null and explain why in "justification".
"""


def get_hierarchical_expansion_prompt(
    parent_label: str,
    category_label: str,
    category_description: str,
    assigned_items: Sequence[Dict],
    depth: int,
    aggregate_stats: Optional[Dict] = None,
) -> str:
    sanitized_parent = _sanitize_for_prompt(parent_label or "ROOT")
    sanitized_label = _sanitize_for_prompt(category_label)
    sanitized_description = _sanitize_for_prompt(category_description or "")
    sanitized_items = [_sanitize_for_prompt(item) for item in assigned_items]
    sanitized_stats = _sanitize_for_prompt(aggregate_stats or {})

    return f"""
Decide whether the category "{sanitized_label}" (child of "{sanitized_parent}") should be further subdivided.

Category context:
- Description: {sanitized_description}
- Depth within taxonomy: {depth} (root depth = 0)
- Aggregate stats: {json.dumps(sanitized_stats, ensure_ascii=False)}

Assigned subtrees (aggregated metadata only):
{json.dumps(sanitized_items, ensure_ascii=False, indent=2)}

Decision guidelines:
- Split further only if the assigned subtrees show multiple distinct methodological themes that benefit from another layer.
- Stop if there are too few subtrees, if they already form a cohesive focus, or if term coverage is modest.
- Prefer shallower hierarchies unless another layer adds substantial clarity.

Respond strictly in JSON:
{{
  "decision": "expand" | "stop",
  "confidence": 0.0,
  "justification": "One concise sentence"
}}
"""



def get_hierarchical_extension_prompt(
    parent_label: str,
    existing_categories: Sequence[Dict],
    leftover_items: Sequence[Dict],
    depth: int,
    ancestor_labels: Sequence[str],
    leftover_terms_total: int,
) -> str:
    sanitized_parent = _sanitize_for_prompt(parent_label or "ROOT")
    sanitized_existing = [_sanitize_for_prompt(cat) for cat in existing_categories]
    sanitized_leftovers = [_sanitize_for_prompt(item) for item in leftover_items]
    sanitized_ancestors = [_sanitize_for_prompt(a) for a in ancestor_labels]
    leftover_total = int(leftover_terms_total)
    leftover_labels_preview = []
    for item in sanitized_leftovers[:10]:
        if isinstance(item, dict):
            lbl = item.get("label")
            if lbl:
                leftover_labels_preview.append(lbl)
        elif isinstance(item, str):
            leftover_labels_preview.append(item)

    return f"""
You are an expert taxonomy curator ensuring comprehensive coverage.

Context:
- Parent category: {sanitized_parent}
- Current depth: {depth} (root depth = 0)
- Ancestor labels to avoid: {json.dumps(sanitized_ancestors, ensure_ascii=False)}
- Unassigned subtree count: {len(sanitized_leftovers)}
- Total terms in leftovers: {leftover_total}

Existing child categories:
{json.dumps(sanitized_existing, ensure_ascii=False, indent=2)}

Unassigned subtrees that still need placement:
{json.dumps(sanitized_leftovers, ensure_ascii=False, indent=2)}

For reference, notable leftover labels include: {json.dumps(leftover_labels_preview, ensure_ascii=False)}.

Requirements:
- Decide purely from the perspective of maintaining a professional, comprehensive taxonomy for "{sanitized_parent}"; do not rely on simple numeric thresholds.
- First ask whether the leftovers could be absorbed by tweaking the scope of existing categories; if so, respond with action "stop" and let the parent handle minor outliers manually.
- If the leftovers form a cohesive micro-domain that should stay together, respond with action "stop".
- Only propose new categories when multiple substantial leftovers share a theme that cannot be represented by existing children without harming clarity.
- Any new category must include a clear "label", 1–2 sentence "description", and a precise "relation_to_parent" referencing "{sanitized_parent}".
- Ensure new labels harmonise with the existing taxonomy style, keep categories fine-grained yet information-dense, and prefer nodes that cover multiple related subtrees without becoming overly broad.
- Treat tiny or idiosyncratic leftovers as acceptable to leave unassigned (action "stop").
- Ensure the expanded set of categories remains professional, concise, and mutually exclusive.

Return strictly valid JSON:
{{
  "action": "augment" | "stop",
  "justification": "One sentence explaining the decision",
  "categories": [
    {{
      "label": "...",
      "description": "...",
      "relation_to_parent": "..."
    }}
  ]
}}

If action is "stop", return an empty list for "categories".
"""

def get_merge_decision_prompt(group_id: str, nodes: Sequence[Dict[str, object]], edges: Sequence[Dict[str, object]]) -> str:
    """Prompt the LLM to evaluate whether similar taxonomy nodes should be merged."""

    def _safe_join(items: Optional[Sequence[str]]) -> str:
        if not items:
            return "(none)"
        if isinstance(items, str):
            return items
        cleaned = []
        for item in items:
            if item is None:
                continue
            cleaned.append(str(item))
        return ", ".join(cleaned) if cleaned else "(none)"

    node_blocks = []
    for idx, node in enumerate(nodes, start=1):
        label = str(node.get("label") or "[missing label]")
        path = node.get("path") or []
        if not isinstance(path, list):
            path = [str(path)]
        path_str = " > ".join(str(token) for token in path if token) or "[root]"
        description = str(node.get("description") or "")
        relation = str(node.get("relation_to_parent") or "")
        child_labels = _safe_join(node.get("child_labels"))
        snippet = str(node.get("document") or "")[:400]
        node_blocks.append(
            {
                "index": idx,
                "id": str(node.get("id")),
                "label": label,
                "path": path_str,
                "description": description,
                "relation": relation,
                "child_labels": child_labels,
                "snippet": snippet,
            }
        )

    edge_blocks = []
    for edge in edges:
        edge_blocks.append(
            {
                "source": str(edge.get("source")),
                "target": str(edge.get("target")),
                "distance": float(edge.get("distance", 0.0)),
            }
        )

    return f"""
You are an expert taxonomy curator. Review the following group of potentially duplicate categories and decide whether they should be merged into a single node.

Consider:
- How aligned the category scopes are (labels, descriptions, parent paths)
- Whether a merged node would remain coherent and properly placed in the hierarchy
- Whether any node should stay separate (e.g., materially different focus or parent path)

Candidate group ID: {json.dumps(group_id, ensure_ascii=False)}

Nodes:
{json.dumps(node_blocks, ensure_ascii=False, indent=2)}

Pairwise distances (lower = more similar):
{json.dumps(edge_blocks, ensure_ascii=False, indent=2)}

If a merge is appropriate, propose a consolidated label, description, relation_to_parent, and target parent path. If not, explain briefly.

Return STRICT JSON:
{{
  "merge": true,
  "group_id": {json.dumps(group_id, ensure_ascii=False)},
  "merged_label": "...",
  "merged_description": "...",
  "merged_relation_to_parent": "...",
  "target_parent_path": ["Parent", "Path"],
  "justification": "",
  "notes": ["optional"]
}}

If you decline to merge, respond with:
{{
  "merge": false,
  "group_id": {json.dumps(group_id, ensure_ascii=False)},
  "justification": "",
  "suggested_merges": [
    {{
      "members": ["label1", "label2"],
      "merged_label": "...",
      "merged_description": "...",
      "merged_relation_to_parent": "...",
      "target_parent_path": ["Parent", "Path"]
    }}
  ],
  "notes": ["optional"]
}}
If you have no pairing recommendations, return an empty array for "suggested_merges".
"""


def get_dimension_prompt(dimension: str, terms: list) -> str:
    """DEPRECATED: no longer used."""
    return "[]"


def get_hierarchical_category_prompt(
    items_summary: Sequence[Dict],
    parent_label: str,
    ancestor_labels: Sequence[str],
    depth: int,
    reference_labels: Optional[Sequence[str]] = None,
    min_categories: Optional[int] = 3,
    max_categories: Optional[int] = 6,
    dimension_context: Optional[str] = None,
) -> str:
    sanitized_items = [_sanitize_for_prompt(item) for item in items_summary]
    sanitized_parent = _sanitize_for_prompt(parent_label or "ROOT")
    sanitized_ancestors = [_sanitize_for_prompt(a) for a in ancestor_labels]
    sanitized_references = [_sanitize_for_prompt(label) for label in (reference_labels or [])]

    if min_categories is None or max_categories is None:
        width_text = "a well-structured set of"
        width_rule = "- Choose the smallest number of categories that still captures the thematic diversity."
    else:
        width_text = f"between {min_categories} and {max_categories}"
        width_rule = f"- Propose {width_text} coherent categories."

    return f"""
You are reorganising an existing professional taxonomy of research methods. Design crisp, non-overlapping child categories for the parent "{sanitized_parent}" at depth {depth} (root depth = 0).
{dimension_context or ""}
Input subtrees (no raw term strings):
{json.dumps(sanitized_items, ensure_ascii=False, indent=2)}

Design goals:
- Produce {width_text} high-quality child categories tailored to "{sanitized_parent}".
- Each category must have: a unique "label", a 1–2 sentence "description", and a precise "relation_to_parent" string that references "{sanitized_parent}" explicitly.
- Avoid repeating any ancestor labels: {json.dumps(sanitized_ancestors, ensure_ascii=False)}
- When possible, align new labels with the observed themes: {json.dumps(sanitized_references, ensure_ascii=False)}. Reuse or merge related concepts rather than inventing entirely new terminology.
- Avoid overly narrow or overly broad buckets; strive for comparable granularity across siblings.
- Respect the provided counts ("terms_count"). Aim for balanced coverage while acknowledging existing concentrations.
- Do not list term names anywhere — rely only on the aggregate metadata.
{width_rule}

Return strictly in this JSON format (no extra keys):
{{
  "categories": [
    {{
      "label": "...",
      "description": "...",
      "relation_to_parent": "..."
    }}
  ]
}}
"""
