"""
GPT-5 Prompt-based Taxonomy Construction
This script uses few-shot prompting with OpenAI's GPT models to construct taxonomies.
"""

from langchain.prompts import Prompt, BaseChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from tqdm import tqdm
import re
import pandas as pd
import dotenv
import random
import networkx as nx
from langchain.schema import (
    BaseMessage, 
    HumanMessage, 
    SystemMessage,
    AIMessage
)
import numpy as np
import json
import os
import argparse
import concurrent.futures
from together import Together

# Load environment variables (for OpenAI API key)
dotenv.load_dotenv()


TOGETHER_CHAT_MODELS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
}

_together_client = None


def get_together_client():
    """Return a cached Together client instance."""
    global _together_client
    if _together_client is None:
        _together_client = Together()
    return _together_client


def convert_messages_to_together_payload(messages):
    """Convert LangChain message objects into Together-compatible dicts."""
    payload = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = 'system'
        elif isinstance(msg, HumanMessage):
            role = 'user'
        elif isinstance(msg, AIMessage):
            role = 'assistant'
        else:
            role = 'user'
        payload.append({'role': role, 'content': msg.content})
    return payload


def call_together_chat_completion(model_name, messages, num_retries=5):
    """Call Together's chat completion endpoint using the provided messages."""
    payload = convert_messages_to_together_payload(messages)
    client = get_together_client()
    last_error = None
    for attempt in range(num_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=payload,
            )
            return response.choices[0].message.content
        except Exception as exc:
            print(f"[Together] Attempt {attempt + 1}/{num_retries} failed: {exc}")
            last_error = exc
    raise RuntimeError(f"[Together] Failed to generate response after {num_retries} attempts: {last_error}")


def format_model_tag(model_name):
    """Return a filesystem-friendly identifier for a model name."""
    safe = model_name.lower().replace(' ', '_').replace('/', '__')
    safe = re.sub(r'[^a-z0-9_\-]+', '_', safe)
    safe = safe.strip('_')
    return safe or 'model'


def _normalize_entity_list(raw_entities):
    names = []
    for item in raw_entities or []:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = item.get('name')
            name = name.strip() if isinstance(name, str) else None
        else:
            name = None
        if name:
            names.append(name)
    return names


def _normalize_relationship_list(raw_relationships):
    pairs = []
    for rel in raw_relationships or []:
        parent = child = None
        if isinstance(rel, dict):
            parent = rel.get('parent') or rel.get('parent_name') or rel.get('parentId')
            child = rel.get('child') or rel.get('child_name') or rel.get('childId')
        elif isinstance(rel, (list, tuple)) and len(rel) >= 2:
            parent, child = rel[0], rel[1]
        if isinstance(parent, str):
            parent = parent.strip()
        else:
            parent = None
        if isinstance(child, str):
            child = child.strip()
        else:
            child = None
        if parent and child:
            pairs.append([parent, child])
    return pairs


def extract_entities_and_blocks(raw_entities):
    """Extract entity list and entity blocks from raw JSON payloads."""
    entity_list_raw = None
    blocks_raw = None

    if isinstance(raw_entities, dict):
        entity_list_raw = raw_entities.get('entity_list') or raw_entities.get('entities')
        blocks_raw = raw_entities.get('entity_blocks') or raw_entities.get('blocks')
    elif isinstance(raw_entities, list):
        entity_list_raw = raw_entities

    entities = _normalize_entity_list(entity_list_raw)
    blocks = []
    if isinstance(blocks_raw, list):
        for block in blocks_raw:
            if isinstance(block, list):
                block_entities = _normalize_entity_list(block)
                if block_entities:
                    blocks.append(block_entities)

    if not blocks and entities:
        blocks = [entities]

    return entities, blocks


def load_entities_with_blocks(entities_file):
    """Load and normalize entities plus their blocks from disk."""
    with open(entities_file, 'r', encoding='utf-8') as f:
        raw_entities = json.load(f)
    return extract_entities_and_blocks(raw_entities)


def load_entities_and_relationships(entities_file, relationships_file=None):
    """Load and normalize entity and relationship data from disk."""
    entities, _ = load_entities_with_blocks(entities_file)

    raw_relationships = []
    if relationships_file and os.path.exists(relationships_file):
        with open(relationships_file, 'r', encoding='utf-8') as f:
            raw_relationships = json.load(f)

    relationships = _normalize_relationship_list(raw_relationships)
    return entities, relationships


def load_relationships(relationships_file):
    """Load and normalize relationship data from disk."""
    if not relationships_file or not os.path.exists(relationships_file):
        return []
    with open(relationships_file, 'r', encoding='utf-8') as f:
        raw_relationships = json.load(f)
    return _normalize_relationship_list(raw_relationships)


def get_nodes_edges(pairs_df, group):
    """Extract nodes and edges for a specific group from the pairs dataframe."""
    pairs = pairs_df[pairs_df['group'] == group]
    edges = []
    nodes = set()
    for i, row in pairs.iterrows():
        parent, child = row['parent'], row['child']
        parent = parent.replace('_', ' ')
        child = child.replace('_', ' ')
        edges.append({
            'parent': parent,
            'child': child,
        })
        nodes.add(parent)
        nodes.add(child)

    nodes = list(nodes)    
    # randomly shuffle the nodes and relations to avoid learning pattern
    random.shuffle(edges)
    random.shuffle(nodes)

    return nodes, edges


def get_groups(pairs_df, split='train'):
    """Get unique groups for a specific data split."""
    groups = pairs_df[pairs_df['type'] == split]['group'].unique()
    return groups


def build_prompt_examples(entities, relationships):
    """Convert raw entities and relationships into prompt-ready nodes/edges."""
    seen = set()
    nodes = []
    for entity in entities:
        clean = entity.replace('_', ' ').strip()
        if clean and clean not in seen:
            seen.add(clean)
            nodes.append(clean)

    edges = []
    for parent, child in relationships:
        parent_clean = parent.replace('_', ' ').strip()
        child_clean = child.replace('_', ' ').strip()
        if not parent_clean or not child_clean or parent_clean == child_clean:
            continue
        edges.append({'parent': parent_clean, 'child': child_clean})

    random.shuffle(nodes)
    random.shuffle(edges)
    return nodes, edges


class TaxomomyPrompt(BaseChatPromptTemplate):
    """Custom prompt template for taxonomy construction with few-shot examples."""
    
    def format_messages(self, **kwargs) -> list[BaseMessage]:
        group_examples = kwargs.get('group_examples', [])
        concepts = kwargs['concepts']

        prefix_prompt = (
            "You are an expert constructing a taxonomy from a list of concepts. Given a list of concepts,"
            "construct a taxonomy by creating a list of their parent-child relationships.\n\n"
        )

        prefix_message = SystemMessage(content=prefix_prompt)

        example_messages = []
        for nodes, edges in group_examples:
            node_prompt = '; '.join(nodes)
            edge_prompt = '; '.join([f"{edge['child']} is a subtopic of {edge['parent']}" for edge in edges])

            example_messages.append(
                HumanMessage(content=f"Concepts: {node_prompt}\nRelationships: ")
            )

            example_messages.append(
                AIMessage(content=f"{edge_prompt}\n\n")
            )
        
        concepts = '; '.join(concepts)
        question_message = HumanMessage(content=f"Concepts: {concepts}\nRelationships: ")

        return [prefix_message] + example_messages + [question_message]


class TaxonomyParser(BaseOutputParser):
    """Parser for extracting parent-child relationships from model output."""
    
    raw_output: str = None  # Declare as class field for Pydantic
    
    def parse(self, output):
        # Store raw output for later logging
        self.raw_output = output
        
        print(f"\n{'='*80}")
        print("RAW GPT OUTPUT:")
        print(f"{'='*80}")
        print(output)
        print(f"{'='*80}\n")
        
        result = []
        
        # Try multiple parsing strategies
        # Strategy 1: Split by semicolon and look for "is a subtopic of"
        lines = output.split(';')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove leading numbers, bullets, and markdown formatting
            line = re.sub(r'^\d+\.\s*\*?\*?', '', line)
            line = re.sub(r'^\*\*?', '', line)
            line = re.sub(r'\*\*?$', '', line)
            line = line.strip()
            
            # Try different patterns
            patterns = [
                r'(.+?)\s+is a subtopic of\s+(.+)',
                r'(.+?)\s+is a sub-topic of\s+(.+)',
                r'(.+?)\s+is subtopic of\s+(.+)',
                r'(.+?)\s+is a parent of\s+(.+)',
                r'(.+?),\s*(.+)',  # Comma-separated
            ]
            
            matched = False
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    part1 = match.group(1).strip()
                    part2 = match.group(2).strip()
                    
                    # Clean up
                    part1 = part1.rstrip('.,;:').strip()
                    part2 = part2.rstrip('.,;:').strip()
                    
                    # Determine which is parent and which is child
                    if 'is a subtopic of' in line.lower() or 'is subtopic of' in line.lower():
                        child, parent = part1, part2
                    elif 'is a parent of' in line.lower():
                        parent, child = part1, part2
                    else:
                        # Assume second is parent of first
                        child, parent = part1, part2
                    
                    if child and parent and child != parent:
                        result.append({
                            'child': child,
                            'parent': parent
                        })
                        matched = True
                        break
        
        print(f"\n{'='*60}")
        print(f"✓ Successfully parsed {len(result)} relationships")
        if len(result) > 0:
            print("Sample parsed relationships (first 5):")
            for rel in result[:5]:
                print(f"  - '{rel['child']}' → '{rel['parent']}'")
        else:
            print("❌ WARNING: No relationships were parsed!")
            print("   The output format may not match expected patterns.")
        print(f"{'='*60}\n")
        
        return result


def call_chain(chain, concepts, group_examples, num_retries=5):
    """Call the LLM chain with retry logic."""
    count = 0
    while count < num_retries:
        try:
            result = chain.run(
                concepts=concepts,
                group_examples=group_examples,
            )
            return result
        except Exception as e:
            print(e)
            count += 1
    raise Exception("Failed to generate result")


def generate_train_examples(pairs_df, train_groups, num_examples=5):
    """Generate few-shot training examples."""
    random.shuffle(train_groups)
    train_group_examples = []
    for group in train_groups[:num_examples]:
        nodes, edges = get_nodes_edges(pairs_df, group)
        train_group_examples.append((nodes, edges))
    return train_group_examples


def convert_to_ancestor_graph(G):
    """Converts a (parent) tree to a graph with edges for all ancestor relations in the tree."""
    G_anc = nx.DiGraph()
    for node in G.nodes():
        for anc in nx.ancestors(G, node):
            G_anc.add_edge(anc, node)
    return G_anc


def compute_ancestor_pairs(relationship_set):
    """Return all ancestor-descendant pairs implied by a relationship set."""
    if not relationship_set:
        return set()
    G = nx.DiGraph()
    G.add_edges_from(relationship_set)
    G_anc = convert_to_ancestor_graph(G)
    return set(G_anc.edges())


def load_few_shot_examples(base_dir, relative_path=None, num_examples=5):
    """Load few-shot examples from ``base_dir/relative_path`` if available."""
    if relative_path is None:
        return []
    target_dir = os.path.join(base_dir, relative_path)
    return load_few_shot_examples_from_directory(target_dir, num_examples)


def load_few_shot_examples_from_directory(target_dir, num_examples=5):
    """Load up to ``num_examples`` few-shot samples from ``target_dir``."""
    if not os.path.exists(target_dir):
        print(f"[Few-shot] Directory not found: {target_dir}")
        return []

    sample_names = sorted(
        [name for name in os.listdir(target_dir) if name.startswith('sample_')]
    )

    if not sample_names:
        print(f"[Few-shot] No sample_* folders in {target_dir}")
        return []

    group_examples = []
    for sample_name in sample_names[:num_examples]:
        sample_dir = os.path.join(target_dir, sample_name)
        entities_file = os.path.join(sample_dir, 'entities.json')
        relationships_file = os.path.join(sample_dir, 'relationships.json')

        if not (os.path.exists(entities_file) and os.path.exists(relationships_file)):
            print(f"[Few-shot] Missing JSON files in {sample_dir}")
            continue

        entities, relationships = load_entities_and_relationships(entities_file, relationships_file)
        nodes, edges = build_prompt_examples(entities, relationships)
        group_examples.append((nodes, edges))
        print(
            f"[Few-shot] Loaded {sample_name} from {target_dir}: "
            f"{len(nodes)} nodes, {len(edges)} edges"
        )

    return group_examples


def load_few_shot_examples_for_test(dataset_root, requested_size, num_examples=5):
    """Load few-shot examples for a particular dataset/size combination."""
    fewshot_root = os.path.join(dataset_root, 'fewshot_examples')
    if not os.path.isdir(fewshot_root):
        print(f"[Few-shot] Directory not found: {fewshot_root}")
        return []

    preferred_dir = os.path.join(fewshot_root, requested_size)
    chosen_dir = None

    if os.path.isdir(preferred_dir):
        chosen_dir = preferred_dir
    else:
        size_dirs = sorted(
            [
                os.path.join(fewshot_root, name)
                for name in os.listdir(fewshot_root)
                if os.path.isdir(os.path.join(fewshot_root, name))
            ]
        )
        if size_dirs:
            chosen_dir = size_dirs[0]
            print(
                f"[Few-shot] No examples for {requested_size}; "
                f"falling back to {os.path.basename(chosen_dir)}"
            )
        else:
            # fewshot_root might directly contain sample_* directories
            sample_present = any(name.startswith('sample_') for name in os.listdir(fewshot_root))
            if sample_present:
                chosen_dir = fewshot_root
            else:
                print(f"[Few-shot] No few-shot samples available under {fewshot_root}")
                return []

    return load_few_shot_examples_from_directory(chosen_dir, num_examples)


def numeric_sort_key(value):
    match = re.search(r'(\d+)', value)
    return (int(match.group(1)) if match else float('inf'), value)


def discover_processed_samples(processed_root, dataset_map, num_samples_per_size=5, relationships_required=True):
    """Collect entities/relationships paths for every dataset/size/sample combo."""
    samples = []
    for dataset_name, folder_name in dataset_map.items():
        dataset_root = os.path.join(processed_root, folder_name)
        test_root = os.path.join(dataset_root, 'test_sets')
        if not os.path.isdir(test_root):
            print(f"[Data] Missing test_sets under {dataset_root}")
            continue

        size_dirs = [
            name for name in os.listdir(test_root)
            if os.path.isdir(os.path.join(test_root, name))
        ]
        for size_name in sorted(size_dirs, key=numeric_sort_key):
            size_dir = os.path.join(test_root, size_name)
            sample_dirs = [
                name for name in os.listdir(size_dir)
                if os.path.isdir(os.path.join(size_dir, name)) and name.startswith('sample_')
            ]
            if not sample_dirs:
                continue

            for sample_name in sorted(sample_dirs, key=numeric_sort_key)[:num_samples_per_size]:
                sample_dir = os.path.join(size_dir, sample_name)
                entities_file = os.path.join(sample_dir, 'entities.json')
                relationships_file = os.path.join(sample_dir, 'relationships.json')
                if not os.path.exists(entities_file):
                    print(f"[Data] Skipping {sample_dir}, missing entities.json")
                    continue
                if relationships_required and not os.path.exists(relationships_file):
                    print(f"[Data] Skipping {sample_dir}, missing relationships.json")
                    continue
                if not os.path.exists(relationships_file):
                    relationships_file = None

                samples.append({
                    'dataset': dataset_name,
                    'dataset_root': dataset_root,
                    'size': size_name,
                    'sample': sample_name,
                    'sample_dir': sample_dir,
                    'entities_file': entities_file,
                    'relationships_file': relationships_file,
                })

    samples.sort(key=lambda item: (
        item['dataset'], numeric_sort_key(item['size']), numeric_sort_key(item['sample'])
    ))
    return samples
def run_taxonomy_generation(entities_file='/home/yirui/ijcai/data/taxonomies_by_size/subtaxonomy_50/entities.json',
                           relationships_file='/home/yirui/ijcai/data/taxonomies_by_size/subtaxonomy_50/relationships.json',
                           output_file='./results_block_taxo/gpt-5/subtaxonomy_50/gpt5.csv',
                           model_name='gpt-5',
                           few_shot_numbers=0,
                           few_shot_examples=None,
                           entities_override=None,
                           ground_truth_relationships=None):
    """
    Main function to run taxonomy generation using GPT prompting.
    
    Args:
        entities_file: Path to the input entities.json file
        relationships_file: Path to the ground truth relationships.json file (for reference)
        output_file: Path to save the results
        model_name: OpenAI model to use (gpt-4 or gpt-3.5-turbo)
        few_shot_numbers: Number of few-shot examples to include (0 for zero-shot)
        few_shot_examples: Pre-loaded few-shot examples (optional)
        
    Returns:
        df: DataFrame with predicted relationships
        ground_truth_relationships: List of ground truth relationships
        token_stats: Dictionary with 'input_tokens' and 'output_tokens'
    """
    # Load entities
    if entities_override is None:
        print(f"Loading entities from {entities_file}...")
        entities, ground_truth_relationships = load_entities_and_relationships(
            entities_file,
            relationships_file,
        )
    else:
        entities = entities_override
        if ground_truth_relationships is None:
            ground_truth_relationships = []

    print(f"Loaded {len(entities)} entities")
    print(f"Ground truth has {len(ground_truth_relationships)} relationships")
    
    # Initialize prompt and LLM
    prompt = TaxomomyPrompt(
        input_variables=['concepts', 'group_examples'],
    )
    parser = TaxonomyParser()  # Create parser instance to access raw output later
    use_together_model = model_name in TOGETHER_CHAT_MODELS
    llm = None
    chain = None
    if use_together_model:
        print(f"Using Together model: {model_name}")
    else:
        # For GPT-5, temperature must be 1.0 (default)
        llm = ChatOpenAI(model=model_name, temperature=1.0)
        chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    
    # Generate results
    print("\nGenerating taxonomy...")
    
    # Use provided few-shot examples or create empty list
    if few_shot_examples is not None and few_shot_numbers > 0:
        group_examples = few_shot_examples[:few_shot_numbers]
        print(f"Using {len(group_examples)} few-shot examples")
    else:
        group_examples = []
        print("Using zero-shot learning")
    
    # Log the input prompt
    log_dir = os.path.dirname(output_file)
    os.makedirs(log_dir, exist_ok=True)
    log_file = output_file.replace('.csv', '_log.txt')
    
    # Save prompt for debugging
    formatted_messages = prompt.format_messages(
        concepts=entities,
        group_examples=group_examples,
    )
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("INPUT PROMPT TO LLM\n")
        f.write("="*80 + "\n\n")
        for i, msg in enumerate(formatted_messages):
            f.write(f"--- Message {i+1} ({msg.__class__.__name__}) ---\n")
            f.write(msg.content + "\n\n")
        f.write("="*80 + "\n")
        f.write(f"Number of entities: {len(entities)}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Few-shot examples: {few_shot_numbers}\n")
        f.write("="*80 + "\n\n")
    
    # Get token count using method 1: get_num_tokens_from_messages
    token_stats = {'input_tokens': 0, 'output_tokens': 0}
    if llm is not None:
        try:
            if hasattr(llm, 'get_num_tokens_from_messages'):
                num_tokens = llm.get_num_tokens_from_messages(formatted_messages)
                token_stats['input_tokens'] = num_tokens
                print(f"Input tokens: {num_tokens}")
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Input tokens: {num_tokens}\n\n")
            else:
                print("Warning: LLM does not support get_num_tokens_from_messages method")
        except Exception as e:
            print(f"Failed to count input tokens: {e}")
    else:
        print("Input token count is unavailable for Together-hosted models")
    
    # Call LLM
    print("Calling LLM...")
    if use_together_model:
        raw_output = call_together_chat_completion(
            model_name,
            formatted_messages,
            num_retries=5,
        )
        result = parser.parse(raw_output)
    else:
        result = call_chain(chain, entities, group_examples)
    
    # Get output token count using method 1
    if llm is not None:
        try:
            if hasattr(parser, 'raw_output') and parser.raw_output and hasattr(llm, 'get_num_tokens'):
                output_tokens = llm.get_num_tokens(parser.raw_output)
                token_stats['output_tokens'] = output_tokens
                print(f"Output tokens: {output_tokens}")
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Output tokens: {output_tokens}\n\n")
            else:
                print("Warning: Cannot count output tokens")
        except Exception as e:
            print(f"Failed to count output tokens: {e}")
    else:
        print("Output token count is unavailable for Together-hosted models")
    
    # Log the raw output
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAW OUTPUT FROM LLM\n")
        f.write("="*80 + "\n")
        if hasattr(parser, 'raw_output') and parser.raw_output:
            f.write(parser.raw_output + "\n")
        else:
            f.write("(Raw output not captured)\n")
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("="*80 + "\n")
        f.write("PARSED RELATIONSHIPS\n")
        f.write("="*80 + "\n")
        f.write(f"Number of relationships parsed: {len(result)}\n\n")
        if len(result) > 0:
            f.write("Parsed relationships:\n")
            for rel in result:
                f.write(f"  {rel['child']} → {rel['parent']}\n")
        else:
            f.write("WARNING: No relationships were parsed!\n")
        f.write("\n" + "="*80 + "\n")
    
    # Convert results to DataFrame
    result_pairs = []
    for edge in result:
        result_pairs.append({
            'child': edge['child'],
            'parent': edge['parent'],
        })
    
    # Save results
    if len(result_pairs) > 0:
        df = pd.DataFrame(result_pairs)
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")
        print(f"✓ Generated {len(result_pairs)} relationships")
        print(f"✓ Log saved to {log_file}")
    else:
        # Still save empty file with headers
        df = pd.DataFrame(columns=['child', 'parent'])
        df.to_csv(output_file, index=False)
        print(f"\n⚠️  WARNING: No relationships generated!")
        print(f"⚠️  Empty file saved to {output_file}")
        print(f"⚠️  Check log file: {log_file}")
    
    return df, ground_truth_relationships, token_stats


def evaluate_results(predicted_relationships_file='./results/gpt-5/subtaxonomy_50/gpt5.csv',
                     ground_truth_relationships=None):
    """Evaluate edge, node, and ancestor metrics for a predicted taxonomy."""

    def _calc_metrics(pred_set, gt_set):
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'pred_total': len(pred_set),
            'gt_total': len(gt_set),
        }

    try:
        df_pred = pd.read_csv(predicted_relationships_file)
    except Exception as e:
        print(f"\n❌ Error loading predictions: {e}")
        return {
            'edge': _calc_metrics(set(), set()),
            'node': _calc_metrics(set(), set()),
            'ancestor': _calc_metrics(set(), set()),
        }

    predicted_set = set()
    for _, row in df_pred.iterrows():
        parent = str(row['parent']).strip()
        child = str(row['child']).strip()
        if parent and child:
            predicted_set.add((parent, child))

    ground_truth_set = set()
    for rel in ground_truth_relationships or []:
        parent = str(rel[0]).strip()
        child = str(rel[1]).strip()
        if parent and child:
            ground_truth_set.add((parent, child))

    metrics = {
        'edge': _calc_metrics(predicted_set, ground_truth_set)
    }

    predicted_nodes = {node for edge in predicted_set for node in edge}
    ground_nodes = {node for edge in ground_truth_set for node in edge}
    metrics['node'] = _calc_metrics(predicted_nodes, ground_nodes)

    predicted_ancestors = compute_ancestor_pairs(predicted_set)
    ground_ancestors = compute_ancestor_pairs(ground_truth_set)
    metrics['ancestor'] = _calc_metrics(predicted_ancestors, ground_ancestors)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print("Edge metrics:")
    print(f"  Ground Truth Relationships: {metrics['edge']['gt_total']}")
    print(f"  Predicted Relationships: {metrics['edge']['pred_total']}")
    print(f"  True Positives: {metrics['edge']['tp']}")
    print(f"  False Positives: {metrics['edge']['fp']}")
    print(f"  False Negatives: {metrics['edge']['fn']}")
    print(f"  Precision: {metrics['edge']['precision']:.4f}")
    print(f"  Recall: {metrics['edge']['recall']:.4f}")
    print(f"  F1 Score: {metrics['edge']['f1']:.4f}")

    print("\nNode metrics:")
    print(f"  Ground Truth Nodes: {metrics['node']['gt_total']}")
    print(f"  Predicted Nodes: {metrics['node']['pred_total']}")
    print(f"  True Positives: {metrics['node']['tp']}")
    print(f"  False Positives: {metrics['node']['fp']}")
    print(f"  False Negatives: {metrics['node']['fn']}")
    print(f"  Precision: {metrics['node']['precision']:.4f}")
    print(f"  Recall: {metrics['node']['recall']:.4f}")
    print(f"  F1 Score: {metrics['node']['f1']:.4f}")

    print("\nAncestor metrics:")
    print(f"  Ground Truth Ancestor pairs: {metrics['ancestor']['gt_total']}")
    print(f"  Predicted Ancestor pairs: {metrics['ancestor']['pred_total']}")
    print(f"  True Positives: {metrics['ancestor']['tp']}")
    print(f"  False Positives: {metrics['ancestor']['fp']}")
    print(f"  False Negatives: {metrics['ancestor']['fn']}")
    print(f"  Precision: {metrics['ancestor']['precision']:.4f}")
    print(f"  Recall: {metrics['ancestor']['recall']:.4f}")
    print(f"  F1 Score: {metrics['ancestor']['f1']:.4f}")
    print("=" * 60)

    if metrics['edge']['tp'] > 0:
        print("\nSome correct predictions:")
        for rel in list(predicted_set & ground_truth_set)[:5]:
            print(f"  ✓ {rel[1]} is a subtopic of {rel[0]}")

    if metrics['edge']['fp'] > 0:
        print("\nSome incorrect predictions:")
        for rel in list(predicted_set - ground_truth_set)[:5]:
            print(f"  ✗ {rel[1]} is a subtopic of {rel[0]} (not in ground truth)")

    if metrics['edge']['fn'] > 0:
        print("\nSome missed relationships:")
        for rel in list(ground_truth_set - predicted_set)[:5]:
            print(f"  ⊗ {rel[1]} is a subtopic of {rel[0]} (should be predicted)")

    return metrics


def evaluate_results_from_pairs(predicted_pairs, ground_truth_relationships=None):
    """Evaluate metrics for predicted (parent, child) pairs."""
    predicted_set = set()
    for parent, child in predicted_pairs or []:
        parent = str(parent).strip()
        child = str(child).strip()
        if parent and child:
            predicted_set.add((parent, child))

    ground_truth_set = set()
    for rel in ground_truth_relationships or []:
        parent = str(rel[0]).strip()
        child = str(rel[1]).strip()
        if parent and child:
            ground_truth_set.add((parent, child))

    def _calc_metrics(pred_set, gt_set):
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'pred_total': len(pred_set),
            'gt_total': len(gt_set),
        }

    metrics = {
        'edge': _calc_metrics(predicted_set, ground_truth_set)
    }
    predicted_nodes = {node for edge in predicted_set for node in edge}
    ground_nodes = {node for edge in ground_truth_set for node in edge}
    metrics['node'] = _calc_metrics(predicted_nodes, ground_nodes)
    predicted_ancestors = compute_ancestor_pairs(predicted_set)
    ground_ancestors = compute_ancestor_pairs(ground_truth_set)
    metrics['ancestor'] = _calc_metrics(predicted_ancestors, ground_ancestors)
    return metrics


def is_output_ready(output_file):
    """Return True if the output file exists and is non-empty."""
    try:
        return os.path.isfile(output_file) and os.path.getsize(output_file) > 0
    except OSError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run taxonomy generation across datasets.")
    parser.add_argument(
        "--model-name",
        type=str,
        default='gpt-5',
        help="Model identifier to use (e.g., gpt-5 or Together model id)",
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=5,
        help="Number of few-shot examples to include per run",
    )
    parser.add_argument(
        "--samples-per-size",
        type=int,
        default=5,
        help="Number of samples per dataset size to evaluate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples that already have an output CSV",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=30,
        help="Max number of concurrent taxonomy calls",
    )
    parser.add_argument(
        "--single-entities",
        type=str,
        default=None,
        help="Run a single entities JSON (optionally with entity_blocks) instead of dataset sweep",
    )
    parser.add_argument(
        "--single-relationships",
        type=str,
        default=None,
        help="Ground-truth relationships JSON for single-run evaluation",
    )
    parser.add_argument(
        "--single-output-dir",
        type=str,
        default=None,
        help="Output directory for single-run CSV/log files",
    )
    parser.add_argument(
        "--few-shot-dir",
        type=str,
        default=None,
        help="Directory containing few-shot examples (sample_*/entities.json, relationships.json)",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help="Root directory for clustered processed data (overrides default)",
    )
    parser.add_argument(
        "--gt-root",
        type=str,
        default=None,
        help="Root directory for ground-truth processed data (overrides default)",
    )
    args = parser.parse_args()

    model_name = args.model_name
    few_shot_numbers = args.few_shot
    num_samples_per_size = args.samples_per_size
    resume = args.resume
    max_workers = args.max_workers

    print(f"=== Running Taxonomy Generation with {model_name} ===\n")

    processed_root = (
        os.path.abspath(args.input_root)
        if args.input_root
        else os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'clustered_processed_data_spectural')
        )
    )
    ground_truth_root = (
        os.path.abspath(args.gt_root)
        if args.gt_root
        else os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data')
        )
    )
    # dataset_map = {
    #     'ccs': 'output_ccs',
    #     'food': 'output_food',
    #     'google': 'output_google',
    # }
    dataset_map = {
        'ccs': 'output_ccs'
    }
    model_tag = format_model_tag(model_name)

    if args.single_entities:
        entities_file = os.path.abspath(args.single_entities)
        relationships_file = (
            os.path.abspath(args.single_relationships)
            if args.single_relationships
            else None
        )
        output_root = args.single_output_dir or os.path.join(
            'results', model_tag, 'single_run'
        )
        output_root = os.path.abspath(output_root)
        os.makedirs(output_root, exist_ok=True)

        entities, entity_blocks = load_entities_with_blocks(entities_file)
        if not entity_blocks:
            entity_blocks = [entities]

        ground_truth_relationships = (
            load_relationships(relationships_file)
            if relationships_file
            else []
        )

        few_shot_examples = []
        if few_shot_numbers > 0 and args.few_shot_dir:
            few_shot_examples = load_few_shot_examples_from_directory(
                args.few_shot_dir,
                num_examples=few_shot_numbers,
            )
            if len(few_shot_examples) < few_shot_numbers:
                print(
                    f"[Few-shot] WARNING: Requested {few_shot_numbers} examples but "
                    f"found {len(few_shot_examples)} in {args.few_shot_dir}"
                )
        elif few_shot_numbers > 0:
            print("[Few-shot] No --few-shot-dir provided; running zero-shot")

        def process_single_block(task):
            block_index, block_entities = task
            block_id = f"block_{block_index}"
            output_file = os.path.join(output_root, block_id, f"{model_tag}.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            df, _, token_stats = run_taxonomy_generation(
                entities_file=entities_file,
                relationships_file=relationships_file,
                output_file=output_file,
                model_name=model_name,
                few_shot_numbers=few_shot_numbers,
                few_shot_examples=few_shot_examples,
                entities_override=block_entities,
                ground_truth_relationships=ground_truth_relationships,
            )
            predicted_pairs = []
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    parent = str(row.get('parent', '')).strip()
                    child = str(row.get('child', '')).strip()
                    if parent and child:
                        predicted_pairs.append((parent, child))

            result_row = {
                'block': block_id,
                'block_size': len(block_entities),
                'entities_file': entities_file,
                'relationships_file': relationships_file,
                'output_file': output_file,
                'precision': np.nan,
                'recall': np.nan,
                'f1': np.nan,
                'node_precision': np.nan,
                'node_recall': np.nan,
                'node_f1': np.nan,
                'ancestor_precision': np.nan,
                'ancestor_recall': np.nan,
                'ancestor_f1': np.nan,
                'input_tokens': token_stats['input_tokens'],
                'output_tokens': token_stats['output_tokens'],
                'total_tokens': token_stats['input_tokens'] + token_stats['output_tokens'],
            }
            return result_row, predicted_pairs

        all_results = []
        merged_predicted_pairs = []
        tasks = [(idx, block_entities) for idx, block_entities in enumerate(entity_blocks)]
        if tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_block, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    result_row, predicted_pairs = future.result()
                    all_results.append(result_row)
                    merged_predicted_pairs.extend(predicted_pairs)

        merged_metrics = evaluate_results_from_pairs(
            predicted_pairs=merged_predicted_pairs,
            ground_truth_relationships=ground_truth_relationships,
        )
        merged_row = {
            'block': 'ALL',
            'block_size': sum(len(b) for b in entity_blocks),
            'entities_file': entities_file,
            'relationships_file': relationships_file,
            'output_file': output_root,
            'precision': merged_metrics['edge']['precision'],
            'recall': merged_metrics['edge']['recall'],
            'f1': merged_metrics['edge']['f1'],
            'node_precision': merged_metrics['node']['precision'],
            'node_recall': merged_metrics['node']['recall'],
            'node_f1': merged_metrics['node']['f1'],
            'ancestor_precision': merged_metrics['ancestor']['precision'],
            'ancestor_recall': merged_metrics['ancestor']['recall'],
            'ancestor_f1': merged_metrics['ancestor']['f1'],
            'input_tokens': sum(item['input_tokens'] for item in all_results),
            'output_tokens': sum(item['output_tokens'] for item in all_results),
            'total_tokens': sum(item['total_tokens'] for item in all_results),
        }
        all_results.append(merged_row)

        summary_df = pd.DataFrame(all_results)
        summary_file = os.path.join(output_root, f"summary_{model_tag}_fewshot{few_shot_numbers}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to {summary_file}\n")
        raise SystemExit(0)

    samples = discover_processed_samples(
        processed_root,
        dataset_map,
        num_samples_per_size=num_samples_per_size,
        relationships_required=False,
    )

    if not samples:
        print(f"No taxonomy inputs found under {processed_root}")
        raise SystemExit(1)

    print(f"Found {len(samples)} test samples across {len(dataset_map)} datasets\n")

    output_root = os.path.join('results_block_taxo', model_tag, 'clustered_processed_data')
    all_results = []
    global_few_shot_examples = None
    if few_shot_numbers > 0 and args.few_shot_dir:
        global_few_shot_examples = load_few_shot_examples_from_directory(
            args.few_shot_dir,
            num_examples=few_shot_numbers,
        )
        if len(global_few_shot_examples) < few_shot_numbers:
            print(
                f"[Few-shot] WARNING: Requested {few_shot_numbers} examples but "
                f"found {len(global_few_shot_examples)} in {args.few_shot_dir}"
            )

    def process_block(task):
        sample, block_id, block_entities, output_file, few_shot_examples = task
        block_sample_id = f"{sample['dataset']}/{sample['size']}/{sample['sample']}/{block_id}"
        ground_truth_dir = os.path.join(
            ground_truth_root,
            os.path.relpath(sample['sample_dir'], processed_root),
        )
        ground_truth_file = os.path.join(ground_truth_dir, 'relationships.json')
        ground_truth_relationships = load_relationships(ground_truth_file)
        try:
            _, _, token_stats = run_taxonomy_generation(
                entities_file=sample['entities_file'],
                relationships_file=sample['relationships_file'],
                output_file=output_file,
                model_name=model_name,
                few_shot_numbers=few_shot_numbers,
                few_shot_examples=few_shot_examples,
                entities_override=block_entities,
                ground_truth_relationships=ground_truth_relationships,
            )
            metrics = evaluate_results(
                predicted_relationships_file=output_file,
                ground_truth_relationships=ground_truth_relationships,
            )
            return {
                'dataset': sample['dataset'],
                'size': sample['size'],
                'sample': sample['sample'],
                'block': block_id,
                'block_size': len(block_entities),
                'entities_file': sample['entities_file'],
                'relationships_file': sample['relationships_file'],
                'output_file': output_file,
                'precision': metrics['edge']['precision'],
                'recall': metrics['edge']['recall'],
                'f1': metrics['edge']['f1'],
                'node_precision': metrics['node']['precision'],
                'node_recall': metrics['node']['recall'],
                'node_f1': metrics['node']['f1'],
                'ancestor_precision': metrics['ancestor']['precision'],
                'ancestor_recall': metrics['ancestor']['recall'],
                'ancestor_f1': metrics['ancestor']['f1'],
                'input_tokens': token_stats['input_tokens'],
                'output_tokens': token_stats['output_tokens'],
                'total_tokens': token_stats['input_tokens'] + token_stats['output_tokens'],
            }
        except Exception as e:
            print(f"❌ Error processing {block_sample_id}: {e}")
            import traceback

            traceback.print_exc()
            return {
                'dataset': sample['dataset'],
                'size': sample['size'],
                'sample': sample['sample'],
                'block': block_id,
                'block_size': len(block_entities),
                'entities_file': sample['entities_file'],
                'relationships_file': sample['relationships_file'],
                'output_file': output_file,
                'precision': np.nan,
                'recall': np.nan,
                'f1': np.nan,
                'node_precision': np.nan,
                'node_recall': np.nan,
                'node_f1': np.nan,
                'ancestor_precision': np.nan,
                'ancestor_recall': np.nan,
                'ancestor_f1': np.nan,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'error': str(e),
            }

    tasks = []
    for sample in samples:
        sample_id = f"{sample['dataset']}/{sample['size']}/{sample['sample']}"
        entities_file = sample['entities_file']
        relationships_file = sample['relationships_file']

        print(f"\n{'='*80}")
        print(f"Processing input: {sample_id}")
        print(f"Entities file: {entities_file}")
        print(f"Relationships file: {relationships_file}")
        print(f"{'='*80}\n")

        _, entity_blocks = load_entities_with_blocks(entities_file)
        if not entity_blocks:
            print(f"[Data] No entity blocks found for {sample_id}, skipping.")
            continue

        few_shot_examples = []
        if few_shot_numbers > 0:
            if global_few_shot_examples is not None:
                few_shot_examples = global_few_shot_examples
            else:
                fewshot_dataset_root = os.path.join(
                    ground_truth_root,
                    os.path.relpath(sample['dataset_root'], processed_root),
                )
                few_shot_examples = load_few_shot_examples_for_test(
                    fewshot_dataset_root,
                    sample['size'],
                    num_examples=few_shot_numbers,
                )
                if len(few_shot_examples) < few_shot_numbers:
                    print(
                        f"[Few-shot] WARNING: Requested {few_shot_numbers} examples but "
                        f"found {len(few_shot_examples)} for {sample_id}"
                    )

        for block_index, block_entities in enumerate(entity_blocks):
            block_id = f"block_{block_index}"
            output_file = os.path.join(
                output_root,
                sample['dataset'],
                sample['size'],
                sample['sample'],
                block_id,
                f"{model_tag}.csv",
            )
            print(f"--- Output will be written to {output_file} ---")

            if resume and is_output_ready(output_file):
                print(f"[Resume] Found existing output, skipping generation: {output_file}")
                all_results.append({
                    'dataset': sample['dataset'],
                    'size': sample['size'],
                    'sample': sample['sample'],
                    'block': block_id,
                    'block_size': len(block_entities),
                    'entities_file': entities_file,
                    'relationships_file': relationships_file,
                    'output_file': output_file,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1': np.nan,
                    'node_precision': np.nan,
                    'node_recall': np.nan,
                    'node_f1': np.nan,
                    'ancestor_precision': np.nan,
                    'ancestor_recall': np.nan,
                    'ancestor_f1': np.nan,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                })
                continue

            tasks.append((sample, block_id, block_entities, output_file, few_shot_examples))

    if tasks:
        print(f"\nLaunching {len(tasks)} block jobs with up to {max_workers} workers...\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_block, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                all_results.append(future.result())

    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(
        output_root,
        f"summary_{model_tag}_fewshot{few_shot_numbers}.csv",
    )
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to {summary_file}\n")

    if len(summary_df) > 0:
        print(summary_df.to_string(index=False))
        print(f"\n{'='*80}")
        print("OVERALL AVERAGE:")
        print(f"{'='*80}")
        if summary_df['precision'].notna().any():
            print(f"Edge Precision: {summary_df['precision'].mean():.4f}")
            print(f"Edge Recall: {summary_df['recall'].mean():.4f}")
            print(f"Edge F1 Score: {summary_df['f1'].mean():.4f}")
            print(f"Node Precision: {summary_df['node_precision'].mean():.4f}")
            print(f"Node Recall: {summary_df['node_recall'].mean():.4f}")
            print(f"Node F1 Score: {summary_df['node_f1'].mean():.4f}")
            print(f"Ancestor Precision: {summary_df['ancestor_precision'].mean():.4f}")
            print(f"Ancestor Recall: {summary_df['ancestor_recall'].mean():.4f}")
            print(f"Ancestor F1 Score: {summary_df['ancestor_f1'].mean():.4f}")
        else:
            print("Ground truth relationships unavailable; skipping accuracy metrics.")
        print(f"Average Input Tokens: {summary_df['input_tokens'].mean():.0f}")
        print(f"Average Output Tokens: {summary_df['output_tokens'].mean():.0f}")
        print(f"Average Total Tokens: {summary_df['total_tokens'].mean():.0f}")
