# BlockTaxo

Tools for block-level taxonomy generation and fusion. The repo supports two main workflows:
1) Clustering
2) Generate block taxonomies with LLMs (`build_block_taxonomy.py`).
3) Fuse multiple block taxonomies into a single hierarchy (`taxonomy_fusion.py`).

## Repository Layout
- `build_block_taxonomy.py`: LLM + few-shot taxonomy generation with optional evaluation.
- `taxonomy_fusion.py`: Multi-block taxonomy fusion using embeddings with optional LLM decisions.
- `clustering_methods/`: Clustering utilities (hierarchical, K-means, etc.) for upstream blocking.
- `processed_data/`: Example data (few-shot pools, test sets, etc.).

## Environment & Dependencies
Recommended Python 3.9+. Install as needed:
- `build_block_taxonomy.py`: `langchain`, `langchain-community`, `tqdm`, `pandas`,
  `networkx`, `python-dotenv`, `together`, `numpy`.
- `taxonomy_fusion.py`: `numpy`, `torch`, `transformers`, `adapters`,
  `sentence-transformers`.
- `clustering_methods/`: `numpy`, `scipy`, `torch`, `transformers`, `adapters`.

Example install:
```bash
pip install numpy pandas torch transformers adapters sentence-transformers \
  langchain langchain-community networkx tqdm python-dotenv together scipy
```

## Usage

### 1) Cluster entities into blocks
Use a clustering script in `clustering_methods/` to split large entity sets into
block-sized subsets. Example (spectral clustering):
```bash
python clustering_methods/spectural.py \
  --input processed_data/output_ccs/test_sets/size_1000/sample_3/entities.json \
  --output processed_data/output_ccs/test_sets/size_1000/sample_3/blocks.json
```

Notes:
- Scripts accept `--input` and `--output`; some support directory input (see file headers).
- Output blocks are then used as inputs for block taxonomy generation.

### 2) Generate block taxonomies
Single run (one entities file):
```bash
python build_block_taxonomy.py \
  --single-entities processed_data/output_ccs/test_sets/size_1000/sample_3/entities.json \
  --single-relationships processed_data/output_ccs/test_sets/size_1000/sample_3/relationships.json \
  --few-shot 5 \
  --few-shot-dir processed_data/output_ccs/fewshot_examples/size_10 \
  --single-output-dir results/ccs_sample_3
```

Notes:
- Default model is `gpt-5`; you can pass a Together model id via `--model-name`.
- OpenAI models require `OPENAI_API_KEY`; Together models typically use `TOGETHER_API_KEY`.
- For dataset sweeps, provide `--input-root` and `--gt-root` to point to your local data.

### 3) Fuse block taxonomies
Input CSV must include columns: `block_id`, `parent`, `child` (order doesn't matter).

Example:
```csv
block_id,parent,child
0,Computer science,Artificial intelligence
0,Artificial intelligence,Machine learning
1,Computing,Artificial intelligence
```

Run fusion:
```bash
python taxonomy_fusion.py \
  --input-file path/to/blocks.csv \
  --output outputs/fused_taxonomy.json
```

Options:
- `--no-llm`: disable LLM decisions, use embeddings + structural heuristics only.
- `--embedding-model`: default `allenai/specter2` (requires `torch/transformers/adapters`).

Notes:
- `taxonomy_fusion.py` expects `prompts.py` (with `get_parent_selection_prompt` and
  `get_sibling_vs_parent_relationship_prompt`) and optional `llm_client.py`
  (with `LLMClient`/`LLMConfig`) in the project.
- If LLM is enabled, set `OPENAI_API_KEY` or pass `--llm-api-key`.

Output `outputs/fused_taxonomy.json` includes `meta` (params, stats) and `taxonomy` (tree).

## Troubleshooting
- Missing `prompts.py` / `llm_client.py`: add these modules or disable LLM with `--no-llm`.
- Default input paths not found: use `--input-root` / `--gt-root` to point to your data.
