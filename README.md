# Woa

Woa (the Woa-Awe-oo project) is a local-first assistant and the beginning of a
project-owned language model. The browser, desktop, CLI, training process, and
service share one SQLite database. Chat and training can run without cloud APIs.

## What this model is (and is not)

The project contains its own decoder-only Transformer implementation:
byte-level tokens, conversation boundary tokens, RoPE, RMSNorm, causal
self-attention, and SwiGLU. It initializes random weights and trains them from
text you supply. It is not pretrained and will not be a capable assistant after
a tiny demo run; useful quality requires a large, well-curated, licensed corpus
and substantial training compute. Training continuously or “without limits” is
not possible on finite hardware.

At inference, Woa keeps its local-assistant guidance and newest user message
inside the checkpoint's context window, retaining recent turns before older
history. Long messages are shortened safely to fit. The byte tokenizer is still
the current format, so context size is measured in bytes rather than words;
prompt guidance does not substitute for training or factual evaluation.

The model architecture is recorded in each checkpoint, and
`woa.model.register_architecture()` provides an extension point for future
experimental architectures. Changing architecture code does not automatically
improve or self-modify the model.

The trainer supports local `.txt` and `.md` files/directories. An opt-in
ingestion command can fetch one explicitly allowlisted HTTPS endpoint returning
plain text, JSON, or CSV at a time:

```sh
python -m woa ingest \
  --url https://data.example.org/public-corpus.jsonl \
  --allow-host data.example.org \
  --output ./my-corpus/source.txt
```

Ingestion has a 32 MiB response limit, rejects redirects and non-public DNS
addresses, and does not crawl pages or run in the background. Only fetch data
you are authorized to use; obey source licensing, terms, rate limits, and
privacy requirements. Training events are recorded locally and served at
`/api/training/events`.

### Curated subject knowledge

The optional starter collection includes sourced texts for philosophy,
historical science, art history, technology history, conversation, and English
prose style, alongside the existing scripture, esoteric, economics, aerospace,
and forensics sources. For example:

```sh
python -m woa download-starter \
  --category philosophy \
  --category science \
  --category art \
  --category technology \
  --category communication \
  --category language \
  --output-dir ./subject-corpus
python -m woa digest ./subject-corpus
```

The download manifest records source URLs, rights notes, hashes, and scope
caveats. Several texts are historical: they are useful for context, not a
substitute for current scientific or engineering references. Indexing adds
retrievable, cited local knowledge; it does not change model weights. A
capable conversational model still requires a suitably licensed instruction
dataset, substantial training, and evaluation.

The installed app also bundles six Project Gutenberg texts and two NASA NTRS
abstract collections. `woa start` and `woa serve` seed these into the local
SQLite knowledge index without network access; repeated starts are idempotent.
“Mewtronic” is the project’s name for composing the one local model, bundled
knowledge, documents the user explicitly indexes, and local tools. `/about`
reports those available components. The assistant is not conscious and does
not scan the computer or use a hosted model API.

The runtime detects available CUDA, Apple Metal (MPS), or CPU. It can select an
available device; it cannot create a virtual GPU or upgrade physical hardware.
No model weights or training datasets are bundled; the packaged reference texts
are used only by local retrieval.

Chat inference and local training do not call a hosted model or remote API. The
web, desktop, and CLI clients use a loopback-only HTTP service to share
conversations; that local API is not an external dependency. The optional
`download-starter` and `ingest` commands need internet access only when you
explicitly run them. The service rejects cross-origin POST requests.

The assistant also has deterministic local tools for bounded degree-two
polynomial equations (`/solve x**2 - 5*x + 6 = 0`) and sample Shannon entropy
(`/entropy some text`). Entropy describes the provided sample; it does not
measure model intelligence or make the model autonomous.

## Install

Python 3.10+ is required. Install the project and the optional PyTorch model
runtime:

```sh
python -m pip install -e ".[train]"
```

PyTorch platform-specific installation instructions may vary. Choose a build
compatible with your operating system and GPU; CPU training is supported but
will be much slower.

## Train the project-owned model

Put permitted training text in one or more `.txt` or `.md` files, then run:

```sh
python -m woa train --data ./my-corpus --steps 1000
```

The command auto-selects an available compute device, logs training progress,
and checkpoints every 100 steps (and at completion) to `model.pt` plus a
versioned, matching optimizer-state sidecar in `~/.woa-awe-oo/`. You can tune
architecture size and training parameters, for example:

```sh
python -m woa train \
  --data ./my-corpus \
  --steps 1000 \
  --context-length 256 \
  --width 128 \
  --layers 4 \
  --heads 4 \
  --batch-size 2
```

For an offline experiment with only the packaged reference texts, write to a
separate checkpoint so the current assistant is not replaced:

```sh
python -m woa train \
  --data ./woa/data/knowledge \
  --steps 1000 \
  --output ~/.woa-awe-oo/experiments/model-pretrained-bundled.pt
```

This is next-token pretraining, not chat instruction tuning. The bundled
corpus is small, historical, and not a substitute for a large licensed
training set; this run should not be assumed to improve conversational
quality. Review and evaluate a candidate checkpoint before selecting it with
`WOA_MODEL_PATH`.

To train a model with exactly 1,000,000 unique trainable parameters on local
text, pass every desired corpus directory to `--data`. The command reads its
`.txt` and `.md` files, deduplicating identical files, and writes a candidate
checkpoint without replacing the active model:

```sh
python -m woa train --exact-million \
  --data ~/.woa-awe-oo/corpora ./woa/data/knowledge \
  --steps 5000 --batch-size 1 \
  --output ~/.woa-awe-oo/experiments/woa-1m-pretrained.pt
```

`--exact-million` uses a 100-wide, 7-layer, 5-head decoder with a 204-token
context and learned position embeddings. The count excludes duplicate state-dict
entries for tied input/output embeddings. Random-window training draws from
the whole supplied corpus, but a finite run does not guarantee every byte was
seen. Byte-level tokens are inefficient and this small model is a research
baseline, not a match for production language models. Next-token pretraining
learns text continuation, not reliable instruction following. Follow it with
authorized, high-quality SFT data and held-out conversation evaluations; do not
promote a checkpoint based only on training loss.

Continue from the saved model and optimizer state with:

```sh
python -m woa train --data ./additional-text --steps 500 --resume
```

Resume keeps the checkpoint architecture and optimizer state and adds training
steps; do not pass architecture-size overrides with `--resume`. The optimizer
sidecar is written first, then the model checkpoint is atomically replaced;
resume verifies the matching checkpoint ID. The current trainer loads the
corpus into host memory, so the selected text files must fit available RAM.

### Supervised fine-tuning

Pretraining learns next-token patterns; it does not, by itself, teach the
assistant to follow instructions. Prepare JSONL with one conversation per line:

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

For a small, original starter set covering Woa's identity, limitations,
constructive clarification, and local tools, generate JSONL locally:

```sh
python -m woa make-synthetic \
  --output ~/.woa-awe-oo/corpora/woa-synthetic-sft.jsonl \
  --arithmetic-examples 32
```

This teaches response patterns and tool-call format, not general world
knowledge. Review it and combine it with authorized, higher-quality examples
before training.

Then fine-tune only on assistant response tokens (the prompt tokens are
context, not target labels):

```sh
python -m woa sft --data ./instructions.jsonl --init-from ~/.woa-awe-oo/model.pt
```

This writes `model-sft.pt` beside the base checkpoint. Continue SFT with the
same dataset using `--resume --output PATH`.

The locally prepared OASST1 subset uses the dataset's reviewed English
rank-0/1 replies and preserves its Apache-2.0 attribution. It can be applied to
a trained candidate with:

```sh
python -m woa sft \
  --data ~/.woa-awe-oo/corpora/oasst1/oasst1-en-short-sft.jsonl \
  --init-from ~/.woa-awe-oo/experiments/woa-1m-all-text.pt \
  --output ~/.woa-awe-oo/experiments/woa-1m-oasst1-sft.pt \
  --steps 1000 --batch-size 1
```

The local assistant can add heuristic labels for message intent, register,
tone words, and possible rhetorical markers. Use `/cues TEXT` to inspect them.
They are transient wording cues used to guide response style; they are not a
reliable mood or personality assessment, are not stored as a user profile, and
cannot identify every metaphor, joke, or language pattern. Woa may adapt its
formality and directness, but does not claim exact knowledge of a user's mood.

### Preference tuning

After SFT, prepare JSONL preference pairs:

```json
{"prompt":[{"role":"user","content":"What is 2 + 2?"}],"chosen":"4","rejected":"5"}
```

Run DPO against a frozen reference model:

```sh
python -m woa dpo \
  --data ./preferences.jsonl \
  --reference ~/.woa-awe-oo/model-sft.pt
```

This writes `model-sft-aligned.pt` and can be continued with `--resume
--reference BASE --output CHECKPOINT`. Preference data should be reviewed and
authorized; DPO is an optimization method, not a guarantee of harmlessness or
truthfulness.

### Evaluate

Measure next-token cross-entropy and perplexity on a separate local text file:

```sh
python -m woa eval --model ./model-sft-aligned.pt --data ./held-out.txt
```

Perplexity is a language-modeling metric, not a measure of assistant quality or
safety. Keep evaluation data separate from training data.

The initial configuration is intentionally small enough to experiment with.
Increase model size only when the detected device has enough memory. To use
another local data directory, set `WOA_DATA_DIR`.
Set `WOA_MODEL_PATH` if the server should load a checkpoint written outside
that data directory.

## Run the synchronized clients

Check the local service and integrations:

```sh
python -m woa doctor
```

The diagnostic checks local Python/runtime files, the writable SQLite/FTS5
store, packaged knowledge, checkpoint presence, and service status. It reports
local readiness and the start command even when the service is stopped.

For a one-command launch, start a client and automatically start the shared
service if needed:

```sh
python -m woa start --client web
# Or select desktop or chat:
python -m woa start --client desktop
python -m woa start --client chat
```

When `start --client web` starts the service itself, keep that terminal open;
press Ctrl+C to stop it. If the service is already running, `start` reuses it.
Alternatively, start the service separately:

```sh
python -m woa serve
```

Then launch any combination of clients:

```sh
python -m woa web
python -m woa desktop
python -m woa chat
```

The Woa web app is available at <http://127.0.0.1:8765>. Keep the service running
for clients to synchronize conversations. Use `WOA_HOST` and `WOA_PORT` to
change its loopback bind address/port. The service is intended for one local
user and has no authentication; do not bind it to a public network.

The browser chat includes suggested starter prompts, distinct message bubbles,
and an expanding composer. Press Enter to send or Shift+Enter for a newline.
In the desktop client, Enter inserts a newline and Ctrl+Enter sends.

## Local knowledge and agent tools

The shared local service includes a bounded tool bus, a safe arithmetic tool,
and a searchable local knowledge index. Index permitted UTF-8 `.txt`, `.md`,
`.rst`, `.py`, `.json`, and `.csv` files without sending them to a remote API:

```sh
python -m woa digest ./notes ./src
python -m woa search "supply chain logistics"
```

The web app also accepts explicitly selected files up to 400 KiB each. Chat
commands are `/calc 2 + 2`, `/solve x**2 - 5*x + 6 = 0`, `/entropy TEXT`,
`/search supply chain`, `/digest supply chain`, `/about`, `/cues TEXT`, `/tools`, and
`/tool calculator {"expression":"2 + 2"}`. Search results include local source
labels and chunk numbers. Re-indexing a source replaces its previous chunks.
The index is stored in the shared SQLite database and is bounded to 100 MiB;
per-file and batch limits are enforced.

The chat router also recognizes plain requests such as “calculate 2 plus 2,”
“search my files for logistics,” and “summarize my documents about trade.”
These common requests use deterministic routing. For other prompts, the model
may request one local tool using a constrained call format. The tool bus checks
the tool allowlist and declared argument schema before invocation; results are
returned directly. The allowlist contains only local calculation, algebra,
knowledge lookup/digest, entropy, communication-cue, and environment tools.
There is no shell, arbitrary command execution, or file-writing tool, and the
small model may fail to select a tool reliably.

`/digest QUERY` ranks and returns verbatim sentences from matching local
sources; it is extractive rather than a model-generated summary, so citations
remain tied to the source text.

Tools are registered through `woa.tools.ToolBus`; their declared JSON schemas
are checked before invocation. The agent can route explicit user commands or
accept one allowlisted tool request from the model. It does not execute
arbitrary shell commands, modify files, or autonomously plan actions. The
extractive search tool can return matching source passages, but generated
model answers are not guaranteed accurate merely because those passages exist.
PDF, image, audio, video, and general-purpose autonomous-agent support are not
implemented.

## Current limitations and next stages

- A byte tokenizer is simple and fully self-contained, but much less efficient
  than a trained BPE tokenizer.
- From-scratch pretraining alone does not teach instruction following. The
  SFT and DPO code are small-model research foundations, not production-scale
  RLHF. Human preference collection, distributed training, automatic scheduled
  ingestion, and website-specific scrapers are not included.
- Chat inference uses the project checkpoint directly; no third-party model
  endpoint is required.
- Adding sources should start with explicit allowlists and bounded ingestion,
  not unrestricted scraping. Source-specific connectors need separate
  licensing and quality checks.

These are deliberate first milestones, not claims that this is already a
production-scale or generally capable LLM.
