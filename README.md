# Qwen3.8 Flash Next on Lenovo PGX / NVIDIA DGX Spark

A reproducible serving setup for **Qwen3.8 Flash Next** — INT4 AutoRound weights
served by **vLLM** with **InstantTensor** loading — on a single **Lenovo PGX /
NVIDIA DGX Spark (GB10)**, together with the validation and benchmark results
that establish it.

Validation uses both RigMark and sgbench.

## Model

- Model: `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`
- Model revision: `e4f732c0f817e3904a96ffadb3402522fb1a7b82`
- Quantisation: INT4 GPTQ-Marlin (group 128) weights; FP8 E4M3
  attention/shared-expert projections; BF16 embeddings, routers, and norms

## Hardware

- 1x NVIDIA GB10 (Blackwell), 128 GB unified memory
- Tensor parallelism TP=1, pipeline parallelism PP=1

## Current image

`qwen38-flash-next:v1` —
`sha256:320d1bcf0dc46f913cfa58a30b29131bb26fadb650d646bbf13b943a5e324156`

A single consolidated build that replaces the historical three-stage chain. It
pins the upstream source commit, the vLLM base digest, and the InstantTensor
version, and applies eleven patches in one `docker build`.

## Architecture

```text
container/   one docker build  ->  qwen38-flash-next:v1
             (pinned vLLM base + 11 patches + InstantTensor)
mods/        chat-template fix, applied at runtime — NOT baked into the image
recipes/     Sparkrun recipe that launches the server
benchmarks/  RigMark and sgbench results validating the setup
docs/        provenance and validation evidence
```

The recipe and external chat template provide the runtime configuration used for
the validated setup. Several custom features are activated by environment
variables or command-line flags, while other compatibility fixes are built
directly into the image.

## Layout

| Path | Contents |
| --- | --- |
| `container/` | `Dockerfile`, `build.sh`, `patches/`, build documentation |
| `mods/fix-qwen3.8-flash-chat-template/` | external runtime chat-template mod |
| `recipes/0-qwen3.8-flash-a5b.yaml` | current serving recipe for `qwen38-flash-next:v1` |
| `recipes/historical/` | the recipe that launched the published historical run |
| `benchmarks/` | RigMark and sgbench results, and how to read them |
| `docs/provenance.md` | source chain, pinned digests, B12X status, attribution |
| `docs/validation.md` | build, runtime, static, and RigMark validation evidence |

## Container source

`container/build.sh` stages the patch sources from the pinned upstream commit,
verifies each staged file's SHA-256, runs one `docker build`, and then runs a
14-check verification pass against the result. Details in
[`container/README.md`](container/README.md).

## Serving recipe

[`recipes/0-qwen3.8-flash-a5b.yaml`](recipes/0-qwen3.8-flash-a5b.yaml) is the
publishable recipe. It launches `qwen38-flash-next:v1` as container
`qwen38-flash-next`, serving model `qwen3.8` on `0.0.0.0:8000` with TP=1, a
262,144-token context, 8,192 max batched tokens, 8 max sequences, a 9 GB KV
cache, MTP with 3 speculative tokens, InstantTensor loading, and the external
chat template.

## Chat template

The chat-template fix stays outside the image so it can be corrected without
rebuilding or revalidating it:

```text
mods/fix-qwen3.8-flash-chat-template/run.sh
  -> copies chat_template.jinja to $WORKSPACE_DIR/fixed_chat_template.jinja
recipes/0-qwen3.8-flash-a5b.yaml
  -> --chat-template fixed_chat_template.jinja
```

## Validation status

- Build verification: **14/14 PASS**
- Runtime smoke test: `qwen3.8` visible via `/v1/models`; `reasoning_effort: none`
  returns `reasoning: null` and `reasoning_tokens = 0`
- Static comparison against the historical image: 7,071 comparable files checked,
  two permitted cosmetic source-tag differences, base first 32 layers
  byte-identical
- RigMark v1 warm run: **15/15 basic output gates passed**

High-concurrency throughput is effectively unchanged from the historical run,
while single-stream code and prose decode measured somewhat lower. No exact
performance equivalence is claimed, and no throughput result here is offered as
evidence about output quality.

Full evidence: [`docs/validation.md`](docs/validation.md).

## Benchmarks

- [`benchmarks/README.md`](benchmarks/README.md) — how the two harnesses differ,
  and how to reproduce a run
- [`benchmarks/rigmark/v1/`](benchmarks/rigmark/v1/) — consolidated-image results
- [`benchmarks/rigmark/historical/`](benchmarks/rigmark/historical/) — the
  published historical result
- [`benchmarks/sgbench/README.md`](benchmarks/sgbench/README.md) — sgbench passes

## Build and use

```bash
# 1. Build and verify the consolidated image
cd container && ./build.sh

# 2. Apply the chat-template mod, then launch the server with
#    recipes/0-qwen3.8-flash-a5b.yaml
```

## Provenance and attribution

The source chain, pinned digests, the B12X status, and component attribution are
in [`docs/provenance.md`](docs/provenance.md).
