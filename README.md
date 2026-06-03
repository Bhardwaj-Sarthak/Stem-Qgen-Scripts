# Stem-Qgen-Scripts

Generate STEM question–answer pairs with a **local Hugging Face Transformers model** while using a remote **MCP endpoint** for retrieval and cognitive-level scoring.

This repository is designed for automatic item/question generation experiments in Maths, Physics, and Chemistry. It supports baseline local generation, retrieval-augmented generation, classifier-guided revision, and a combined RAG + tool-feedback pipeline.

## Repository contents

```text
.
├── generate_stem_questions_local_transformers_mcp.py  # Main generation script
├── stem_prompt_templates.py                           # Pipeline-specific prompt templates
├── requirements.txt                                   # Python dependencies
└── README.md
```

## What the script does

The main script loads a local causal language model with Hugging Face Transformers and connects to the STEM-QA-Contextualize MCP endpoint. The local model writes the questions; the MCP endpoint is used for retrieval and/or classification depending on the selected pipeline.

Supported pipelines:

| Pipeline | Description |
|---|---|
| `default` | Local model generates a question directly. |
| `rag` | MCP retrieves benchmark-style examples, then the local model generates using those examples as context. |
| `tool` | Local model generates, MCP classifies Bloom/DOK alignment, and the local model revises if needed. |
| `rag_tool` | MCP retrieves examples, local model generates, MCP classifies, and the local model revises if needed. |

The full sweep covers:

```text
3 subjects × 4 grades/classes × 2 topics × 6 Bloom levels × 4 DOK levels
= 576 generated rows per pipeline
```

Subjects and topics are defined inside `generate_stem_questions_local_transformers_mcp.py`.

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

You may also install the core dependencies manually:

```bash
pip install "mcp>=1.9" "transformers>=4.45" accelerate torch tqdm pandas
```

For 4-bit or 8-bit quantized loading:

```bash
pip install bitsandbytes
```

## Hugging Face access

Some models, especially Llama models, require a Hugging Face access token.

Set your token as an environment variable:

```bash
export HF_TOKEN="your_huggingface_token"
```

Do not paste your token directly into shared logs, notebooks, screenshots, GitHub issues, or README files.

## MCP endpoint

By default, the script uses:

```text
https://bhardwaj08sarthak-stem-qa-contextualize-mcp.hf.space/gradio_api/mcp/
```

To override it:

```bash
export MCP_URL="https://your-space-or-server/gradio_api/mcp/"
```

The MCP endpoint is expected to expose tools for:

- retrieving benchmark-style question examples
- classifying/scoring generated questions for Bloom's Taxonomy and Depth of Knowledge alignment

## Quick start

Run a small test with the default Qwen model:

```bash
python generate_stem_questions_local_transformers_mcp.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --pipelines rag_tool \
  --limit-combos 10 \
  --out test_local_questions.jsonl \
  --csv-out test_local_questions.csv
```

This generates 10 question–answer records using the `rag_tool` pipeline and writes both JSONL and CSV outputs.

## Running all pipelines

```bash
python generate_stem_questions_local_transformers_mcp.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --pipelines default rag tool rag_tool \
  --out stem_questions_local.jsonl \
  --csv-out stem_questions_local.csv
```

## Running with a gated Llama model

```bash
export HF_TOKEN="your_huggingface_token"

python generate_stem_questions_local_transformers_mcp.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --hf-token "$HF_TOKEN" \
  --torch-dtype bf16 \
  --device-map auto \
  --pipelines rag_tool \
  --limit-combos 10
```

## Quantized model loading

For lower memory usage, use 4-bit loading:

```bash
python generate_stem_questions_local_transformers_mcp.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --hf-token "$HF_TOKEN" \
  --load-in-4bit \
  --device-map auto \
  --pipelines rag_tool \
  --limit-combos 10
```

Or 8-bit loading:

```bash
python generate_stem_questions_local_transformers_mcp.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --load-in-8bit \
  --device-map auto \
  --pipelines rag_tool \
  --limit-combos 10
```

Use only one of `--load-in-4bit` or `--load-in-8bit` at a time.

## Loading without quantization

To load a model normally, do not pass `--load-in-4bit` or `--load-in-8bit`.

Example:

```bash
python generate_stem_questions_local_transformers_mcp.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --torch-dtype bf16 \
  --device-map auto \
  --pipelines default \
  --limit-combos 5
```

Approximate memory use depends on the model size, dtype, context length, and GPU setup. For large models, start with `--load-in-4bit` unless you are sure your GPU has enough VRAM.

## Important command-line arguments

| Argument | Default | Description |
|---|---:|---|
| `--model-id` | `Qwen/Qwen2.5-3B-Instruct` | Hugging Face model ID or local model path. |
| `--mcp-url` | default HF Space MCP URL | Remote MCP endpoint. |
| `--hf-token` | `$HF_TOKEN` | Hugging Face token for gated models or private Spaces. |
| `--torch-dtype` | `auto` | One of `auto`, `bf16`, `fp16`, `fp32`. |
| `--device-map` | `auto` | Transformers device map, for example `auto`, `cuda:0`, or `cpu`. |
| `--load-in-4bit` | disabled | Load model with bitsandbytes 4-bit quantization. |
| `--load-in-8bit` | disabled | Load model with bitsandbytes 8-bit quantization. |
| `--pipelines` | `rag_tool` | One or more of `default`, `rag`, `tool`, `rag_tool`. |
| `--attempts` | `3` | Revision attempts for `tool` and `rag_tool`. |
| `--limit-combos` | `0` | Limit number of generated combinations for testing. `0` means full sweep. |
| `--out` | `stem_questions_local.jsonl` | JSONL output file. |
| `--csv-out` | `stem_questions_local.csv` | CSV output file. |
| `--sleep` | `0.0` | Delay between generations to reduce MCP/Space load. |

## Output files

The script writes a JSONL file and optionally a CSV file.

Each generated record contains:

- model ID
- pipeline name
- subject
- grade/class
- topic
- target Bloom level
- target DOK level
- generated question
- generated answer
- reasoning/explanation
- retrieved examples, if applicable
- MCP classification result, if applicable
- number of attempts
- success flag
- error message, if any

## Prompt templates

Pipeline-specific prompts are stored in:

```text
stem_prompt_templates.py
```

The main script imports:

```python
from stem_prompt_templates import build_pipeline_prompt
```

and uses it inside the local generator's `_build_prompt(...)` method.

To customize generation behavior, edit the relevant prompt in `stem_prompt_templates.py`:

- `DEFAULT_PROMPT`
- `RAG_PROMPT`
- `TOOL_FIRST_ATTEMPT_PROMPT`
- `TOOL_REVISION_PROMPT`
- `RAG_TOOL_FIRST_ATTEMPT_PROMPT`
- `RAG_TOOL_REVISION_PROMPT`

## Suggested workflows

### 1. Check MCP connectivity

Before running a large generation job, open the Hugging Face Space in a browser or run a small test:

```bash
python generate_stem_questions_local_transformers_mcp.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --pipelines rag_tool \
  --limit-combos 1
```

If the Space was sleeping, the first run may take longer.

### 2. Debug with a small run

```bash
python generate_stem_questions_local_transformers_mcp.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --pipelines default \
  --limit-combos 3
```

### 3. Run the full thesis-style grid

```bash
python generate_stem_questions_local_transformers_mcp.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --pipelines default rag tool rag_tool \
  --out full_generation.jsonl \
  --csv-out full_generation.csv
```

## Troubleshooting

### MCP timeout or `httpx.ReadTimeout`

This usually means the remote MCP server did not respond quickly enough. Try:

- opening the Hugging Face Space in a browser to wake it up
- running with `--limit-combos 1` first
- adding `--sleep 1`
- checking that `--mcp-url` is correct
- verifying that your cluster allows outbound HTTPS/streaming HTTP connections

### CUDA out of memory

Try one or more of the following:

```bash
--load-in-4bit
--load-in-8bit
--torch-dtype fp16
--max-new-tokens 512
```

Also use a smaller model for debugging.

### Gated model access error

Make sure:

- you accepted the model license on Hugging Face
- `HF_TOKEN` is set
- the token has permission to read gated models

### Invalid JSON from the model

The script tries to extract the first JSON object from the model response. If JSON errors persist, reduce temperature:

```bash
--temperature 0.1
```

or strengthen the output-format instructions in `stem_prompt_templates.py`.

## Notes

- The local model performs generation.
- The MCP endpoint performs retrieval and/or Bloom/DOK classification.
- RAG and tool-based pipelines require the MCP endpoint to be reachable.
- The `default` pipeline still initializes the MCP client in the current script, so the endpoint should be available even if you only generate with `default`.

## License

Add a license file if you plan to share or reuse the repository publicly.
