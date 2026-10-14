# StructLLM

StructLLM starts by discovering the **authorized, live MCP tools** exposed by a
StructSub server. Its first command creates an immutable JSON snapshot containing
the server-provided tool descriptions, input/output schemas, annotations,
metadata, server identity, MCP protocol version, capture time, and a catalogue
content hash.

This step only calls MCP `tools/list`; it does not submit scientific work or
read any training datasets. Tool listings can be scoped to the authenticated
user and workspace, so the snapshot records what that caller can discover at
capture time, not a universal list of every capability configured in StructSub.

## Requirements

- Python 3.11 or newer
- An authorized StructSub MCP bearer token with the `science:execute` scope
- Network access to the configured MCP endpoint

Install the project with `uv sync`. Configure credentials in the process
environment; do not put tokens in command-line arguments or snapshot files:

```bash
export MCP_RESOURCE_URL="https://your-structsub-host/mcp"
export MCP_ACCESS_TOKEN="..."
# Optional when selecting a specific workspace:
export MCP_WORKSPACE_ID="..."
uv run structllm --output-dir catalogues
```

The command writes a filename of the form
`mcp-catalogue-<UTC timestamp>-<catalogue hash>.json`. It fails rather than
overwriting a snapshot with the same name. Each snapshot includes only the
endpoint URL without user information, query parameters, or fragments.

For local development, `http://localhost/mcp`, `http://127.0.0.1/mcp`, and
`http://[::1]/mcp` are accepted. Other remote endpoints must use HTTPS.

## Schema-only dataset generation

Generate a versioned dataset from a saved snapshot without contacting the MCP
server or executing operations:

```bash
uv run structllm generate-dataset \
  --catalogue catalogues/mcp-catalogue-<timestamp>-<hash>.json \
  --output-dir /Volumes/X9/scratch/structsub-mcp-dev/datasets \
  --seed 17 \
  --holdout-fraction 0.2
```

The output directory is named with the generator version, source catalogue
hash, and seed. It contains:

- `train.jsonl`: schema-conditioned tool-routing examples and missing-argument
  clarification examples
- `heldout-tools.jsonl`: tools split entirely out of training, including their
  schemas in held-out candidate contexts
- `quarantine.jsonl`: tools whose schemas or illustrative arguments could not
  be validated
- `quality-report.json` and `manifest.json`: validation counts, limitations,
  provenance, and file hashes

Every generated example contains only a small candidate set (up to five tool
definitions), rather than the full catalogue. Illustrative positive arguments
are checked against the JSON Schema but are explicitly **not executable** and
do not establish that the backend will accept them or that an operation is
scientifically valid. Negative examples ask for required information rather
than inventing values. No scientific operations are executed.

The held-out split is grouped by tool name, so a capability and its versions
cannot leak between training and held-out records. This is a schema-use
prototype, not a chemistry reasoning benchmark; it does not measure
natural-language chemistry understanding, backend execution success, or
scientific accuracy. Per-request approval depends on price and exact inputs,
not a static tool flag. For each later dataset build, take a fresh capability
snapshot first so additions and deprecations are reflected.

One inexpensive real negative example can be captured from the explicitly
configured local development server:

```bash
uv run structllm capture-invalid-smiles \
  --output-dir /Volumes/X9/scratch/structsub-mcp-dev/observed-failures
```

This command creates a local project, submits one intentionally invalid SMILES
to `build_structure`, waits briefly for the real failure response, and stores a
sanitized example without project/job identifiers or credentials. It never
approves a job and fails if the backend does not return a `failed` status with
errors. Keep these observed failures separate from schema-only examples and
label them as local-development evidence, not production OAuth validation.

## Tool-grounded conversations (teacher trajectories)

Install the extra with `uv sync --extra datagen` and set Anthropic credentials
in the environment (`ANTHROPIC_API_KEY` or an `ant auth login` profile), never
on the command line or in files.

1. Write user requests from the scenario specs in
   `structllm/resources/scenarios.yaml`. The writer sees only each scenario's
   brief and the slot values a request must use, never tool names. Requests that
   miss a slot value, name a tool or duplicate another request are dropped and
   listed in `<output>.dropped.json`.

   ```bash
   uv run structllm generate-requests \
     --catalogue catalogues/mcp-catalogue-<timestamp>-<hash>.json \
     --output /Volumes/X9/scratch/structsub-mcp-dev/requests/requests-seed17.jsonl
   ```

2. Answer them with a teacher model through the orchestrator against the
   development server (`MCP_RESOURCE_URL`, `MCP_ACCESS_TOKEN`). Use `--limit`
   to try a few requests first.

   ```bash
   uv run structllm generate-trajectories \
     --catalogue catalogues/mcp-catalogue-<timestamp>-<hash>.json \
     --requests /Volumes/X9/scratch/structsub-mcp-dev/requests/requests-seed17.jsonl \
     --limit 5
   ```

The command refuses to start if the live catalogue hash differs from the
snapshot. Each request gets a fresh project and runs the scenario's cheap setup
calls. Expensive work is never sent: this includes any capability whose base
price reaches its approval threshold, an ML-potential engine (as reported by
`list_engines`), a run longer than the step limits, or a workflow containing any
of these. Such calls get a locally created `awaiting_approval` envelope, and
every tool result in a record says whether it was executed or mocked. Assistant
turns whose calls the orchestrator refused are kept for context with
`train: false`.

Many scenarios are multi-turn and can only be answered correctly from
context. Follow-ups refer back ("now export it", "the first one", "at 273 K
instead"). Some turns have forbidden tools, so "don't relax it, just export
it" fails if it relaxes. Revision checks require "it" to resolve to a
structure revision already seen in the conversation. Some missing information
is supplied by earlier turns ("what's the uptake?" means ask in a fresh chat,
but run after the gas and conditions were given). Use
`--shard K/N` to run N worker processes in parallel; each writes its own
resumable shard directory.

Records go to `accepted.jsonl` or `rejected.jsonl` after a structural check
against the scenario's expected behaviour (execute, clarify, approval_hold,
unsupported, failure). That check confirms the conversation has the right
shape, not that its science is correct. Setup failures go to `errors.jsonl`.
Whole scenarios are assigned to `train` or `eval`. An interrupted run resumes
where it stopped.

### Export for fine-tuning

```bash
uv run structllm export-sft \
  --trajectories /Volumes/X9/scratch/structsub-mcp-dev/trajectories/trajectories-<hash>-seed17
```

This writes `train.jsonl`, `eval.jsonl` and `manifest.json` (with counts and
file hashes) under `/Volumes/X9/scratch/structsub-mcp-dev/sft/`. Each line holds
`messages` in the Hugging Face chat-template shape (`reasoning_content`,
`tool_calls` with object arguments, `tool` messages), the `tools` offered, and
`meta`. Assistant messages carry `train: true|false`; take loss only on
`true`. The files are self-contained: training needs only these files, the
tokenizer's chat template, and no StructSub installation or server.
Conversations render to roughly 5–15K tokens with Qwen3/Qwen3.5 templates,
so use a maximum sequence length of at least 16K (32K keeps 99% of the turns of the final dataset). Qwen3's template keeps
`<think>` only for the latest assistant turn, so train each assistant turn as
its own prefix-and-target example to keep earlier reasoning.

### Building the training set

Each catalogue is exported separately (`export-sft` refuses to mix catalogues, and
adds the passing opening turns of rejected conversations and drops ungrounded
ones), then the exports are joined:

```bash
uv run structllm export-sft --trajectories <dir of all runs for one catalogue> --output-dir <out>
uv run structllm assemble-sft --exports <export A> <export B> --output-dir sft/structllm-sft-FINAL-v3
```

`assemble-sft` rewrites the system prompt to the current one, keeps any scenario that
is in train out of eval, and checks that conversation ids are unique.

### Scenario files

| File | What it is |
| --- | --- |
| `structllm/resources/scenarios.yaml` | Everything: hand-written, generated multi-turn, coverage (`cov_`, `cov2_`). About 1,350 scenarios. |
| `scenarios_budget.yaml` | An execute-heavy subset (about 400 conversations), cheap to run. |
| `scenarios_cov2.yaml` | Second coverage pass for tools the first run missed. |
| `scenarios_curation.yaml` | Upload, find overlapping atoms, delete, re-check, export. Needs the catalogue that contains `find_overlaps`. |
| `scenarios_reasoning.yaml` | Reasoning-heavy science questions. Expensive per usable conversation on Sonnet; see the cost note below. |

Run any of them with `--scenarios <file>`. Check a file against a catalogue first:
loading fails on an unknown tool, and setup calls are schema-checked.

**Cost.** Use `claude-haiku-5-5` for `generate-requests` and as the teacher
(`--model claude-haiku-5-5 --effort low`). Measured on this repository: about
$0.002 to $0.004 per conversation. Sonnet 5.5 cost about $0.09 per conversation
and few were accepted. Almost all of the cost is cache writes, because each
conversation offers a different tool subset.

**Grounding.** `export-sft` drops any conversation whose reply states a number
that no tool result, tool definition or user message supports (derived
differences, percent versus fraction and scientific notation are allowed). The
count is recorded in `manifest.json`. Pass `--allow-ungrounded` to keep them.

**New tools.** A route added to StructSub is published as a tool automatically
(`find_overlaps` is an example). The model finds it at run time through
`find_tools` and reads its schema, so a new tool does not require retraining.
Retrain only if the model fumbles it or you want it used in multi-step recipes:
capture a new catalogue, add scenarios, regenerate, then concatenate the new
export with the old one (every row carries its own `tools` list).

### Fine-tuning

Training needs no API key and no StructSub server. It needs only the export and
the open Qwen3 weights from Hugging Face.

1. Put the data on the cluster, from the repository root:

   ```bash
   rsync -a /Volumes/X9/scratch/structsub-mcp-dev/sft/structllm-sft-FINAL-v3 <cluster>:<repo>/data/sft/
   rsync -a --exclude .venv --exclude __pycache__ ./ <cluster>:<repo>/
   ```

2. Optional but recommended, on a login node, to fetch the weights and build the
   environment once (compute nodes may have no internet):

   ```bash
   SETUP_ONLY=1 SFT=data/sft/structllm-sft-FINAL-v3 bash hpc/submit.slurm
   ```

3. Submit from the repository root:

   ```bash
   sbatch --export=ALL,SFT=data/sft/structllm-sft-FINAL-v3 hpc/submit.slurm
   ```

   Useful overrides: `QLORA=1` (smaller GPU), `EPOCHS=3`, `MAX_LENGTH=32768`,
   `MODEL=Qwen/Qwen3-8B`, `OUT=runs/<name>`, `CUDA_TAG=cu126`. Sequences reach
   32K tokens, so request an 80 GB GPU, or use `MAX_LENGTH=24576` on 40 to 48 GB, or `QLORA=1`.

The job builds a venv on first run, checks that a GPU is visible, tokenises the
data and prints length statistics, then trains and writes the LoRA adapter, the
tokenizer and `training.json` to `OUT`.

Evaluation runs inside the loop, at the end of every epoch (or every
`EVAL_EVERY` steps). Each time it computes eval loss on all held-out turns and
generates a fixed sample of `BEHAVIOUR_SAMPLES` held-out turns (default 32),
scoring them against the teacher: the right decision (call a tool or answer),
the right tools, schema-valid arguments, exact arguments including copied
revision ids, no invented tool names, and numbers grounded in the conversation.
It needs no StructSub server. Results are appended to `OUT/metrics.jsonl`; the
adapter is saved to `OUT/last` (with optimiser state) and the best one, by
behaviour score then eval loss, to `OUT/best`. `PATIENCE=n` stops after n
evaluations without improvement. Resubmitting the same command after a
wall-clock kill continues from `OUT/last`. Use `OUT/best` for the next step,
not necessarily the final adapter. Each behaviour evaluation takes a few
minutes on an 8B model, so lower `BEHAVIOUR_SAMPLES` to speed a run up.

To check the data without a GPU:

```bash
uv run structllm-train --data <export> --model Qwen/Qwen3-8B --output-dir /tmp/x --prepare-only
```

Each assistant turn marked `train: true` becomes one example whose prompt is the
whole conversation before it; the loss covers that turn only. Examples longer
than `--max-length` (default 16384, 32768 in the Slurm script) are skipped and counted in the output.

### After training

1. Copy `OUT` back and read `metrics.jsonl`. Check that the behaviour score
   rose across evaluations and take `OUT/best`. That is a check on a small
   sample of held-out turns, not a full evaluation.
2. Measure behaviour on the full held-out set and on whole conversations, not
   one turn at a time. Serve the adapter with vLLM
   (`--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`),
   replay the held-out eval requests through the orchestrator against the dev
   server, and score them with the same acceptance and grounding checks used for
   the data. `adapters.py` has an OpenAI-compatible adapter for this; the
   `generate-trajectories` command does not yet have a switch to use it.
3. Compare against the untuned Qwen3-8B and against the Haiku teacher on the same
   requests, so the gain is measured.
4. Turn the failures into scenarios, regenerate a little data, retrain.

## Tests

```bash
uv run pytest
```

The live catalogue can only be retrieved with valid MCP credentials. If no
authorized endpoint is available, no capability inventory should be inferred
from source code or fabricated.
