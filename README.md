# Woa-Awe-oo

Woa-Awe-oo is a local-first assistant and the beginning of a project-owned
language model. The browser, desktop, CLI, training process, and service share
one SQLite database. Chat and training can run without cloud APIs.

## What this model is (and is not)

The project contains its own decoder-only Transformer implementation:
byte-level tokens, conversation boundary tokens, RoPE, RMSNorm, causal
self-attention, and SwiGLU. It initializes random weights and trains them from
text you supply. It is not pretrained and will not be a capable assistant after
a tiny demo run; useful quality requires a large, well-curated, licensed corpus
and substantial training compute. Training continuously or “without limits” is
not possible on finite hardware.

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

The runtime detects available CUDA, Apple Metal (MPS), or CPU. It can select an
available device; it cannot create a virtual GPU or upgrade physical hardware.
No model weights or datasets are bundled.

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

Continue from the saved model and optimizer state with:

```sh
python -m woa train --data ./additional-text --steps 500 --resume
```

Resume keeps the checkpoint architecture and optimizer state and adds training
steps; do not pass architecture-size overrides with `--resume`. The optimizer
sidecar is written first, then the model checkpoint is atomically replaced;
resume verifies the matching checkpoint ID. The current trainer loads the
corpus into host memory, so the selected text files must fit available RAM.

The initial configuration is intentionally small enough to experiment with.
Increase model size only when the detected device has enough memory. To use
another local data directory, set `WOA_DATA_DIR`.
Set `WOA_MODEL_PATH` if the server should load a checkpoint written outside
that data directory.

## Run the synchronized clients

Start the local service:

```sh
python -m woa serve
```

Then launch any combination of clients:

```sh
python -m woa web
python -m woa desktop
python -m woa chat
```

The web app is available at <http://127.0.0.1:8765>. Keep the service running
for clients to synchronize conversations. Use `WOA_HOST` and `WOA_PORT` to
change its loopback bind address/port. The service is intended for one local
user and has no authentication; do not bind it to a public network.

## Current limitations and next stages

- A byte tokenizer is simple and fully self-contained, but much less efficient
  than a trained BPE tokenizer.
- From-scratch pretraining alone does not teach instruction following. The
  first version does not yet include SFT, preference collection, DPO/RLHF,
  distributed training, automatic scheduled ingestion, or website-specific
  scrapers.
- Chat inference uses the project checkpoint directly; no third-party model
  endpoint is required.
- Adding sources should start with explicit allowlists and bounded ingestion,
  not unrestricted scraping. Source-specific connectors need separate
  licensing and quality checks.

These are deliberate first milestones, not claims that this is already a
production-scale or generally capable LLM.
