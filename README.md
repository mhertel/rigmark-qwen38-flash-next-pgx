# Qwen3.8 Flash Next INT4 on a Single NVIDIA GB10 / Lenovo PGX

This repository contains a RigMark receipt and share card for
`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` running on one NVIDIA
GB10. The JSON receipt and text card are the original generated artifacts and
have not been edited.

## Model

- Model: `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`
- Model revision: `e4f732c0f817e3904a96ffadb3402522fb1a7b82`
- Quantisation: INT4 GPTQ-Marlin (group 128) weights; FP8 E4M3
  attention/shared-expert projections; BF16 embeddings, routers, and norms

## Hardware

- 1x NVIDIA GB10 (Blackwell)
- 128 GB unified memory
- Tensor parallelism: TP=1
- Pipeline parallelism: PP=1

## Serving configuration

- vLLM `0.1.dev20073+g8e685d198`
- MTP with 3 speculative tokens
- 262144-token context
- 9 GB KV cache
- BF16 KV cache
- Prefix caching enabled
- Chunked prefill enabled
- No meaningful competing GPU traffic during the benchmark

## Serving recipe

The exact [Sparkrun recipe](recipe/0-qwen3.8-flash-a5b.yaml) used to launch the
benchmarked server is included for serving-configuration reproducibility. The
RigMark JSON receipt remains the authoritative benchmark artifact; the recipe
documents how the model server was configured and launched.

## Additional benchmark comparison

The original azampatti single-Spark sgbench harness was also run against this
server. See [SGBENCH.md](SGBENCH.md) for all six passes and the comparison.

## RigMark result

- Protocol: `1.1.0`
- Reasoning effort: `none`
- Basic output gates: **15/15 passed**

| Measurement | Median | Range |
| --- | ---: | ---: |
| Code | 82.1 tok/s | 81.3–82.8 tok/s |
| Prose | 44.9 tok/s | 44.4–45.9 tok/s |
| Structured* | 86.5 tok/s | 84.3–87.0 tok/s |
| 64K cold prefill | 2,483 tok/s | — |
| 64K immediate replay | 46,636 tok/s | — |
| C1 aggregate | 71.9 tok/s | — |
| C2 aggregate | 114.7 tok/s | — |
| C4 aggregate | 182.4 tok/s | — |

*Structured output is RigMark's predictable-output / speculative-decoding ceiling and is not a proxy for coding-agent speed.

## Basic gates versus code audit

RigMark's basic output gate and `audit-code` measure different things.

The **15/15 basic output gates passed** result means that all benchmark outputs
completed and met RigMark's basic output requirements. It is not a claim that
generated code is correct.

The separate code audit replays each retained generated Go implementation with
its own model-generated tests in RigMark's locked-down Go 1.25 environment.
Only **1/5 model-supplied Go test suites passed**:

1. **Run 1 — build failed.** `ratelimit.go:28:11: undefined: sync`. The
   generated implementation uses `sync.Mutex` but does not import `sync`; no
   tests ran.
2. **Run 2 — build failed.** `ratelimit.go:29:11: undefined: sync`. The
   generated implementation uses `sync.Mutex` but does not import `sync`; no
   tests ran.
3. **Run 3 — tests failed.** The first failure is
   `ratelimit_test.go:117: AllowN(6) = false, want true` in
   `TestAllowN/refill_capped`. The generated test gives the bucket capacity 5
   and immediately requests 6 tokens, yet expects success. The same generated
   suite also reports `AllowN(5) = false, want true` in `multiple_calls`: after
   two successful 5-token withdrawals from capacity 10, it advances only one
   second at one token per second and incorrectly expects another 5-token
   withdrawal to succeed. The implementation and its generated tests disagree;
   12 tests/subtests were started.
4. **Run 4 — passed.** All 13 model-generated tests/subtests passed.
5. **Run 5 — build failed.** `ratelimit.go:23:13: undefined: sync`. The
   generated implementation uses `sync.Mutex` but does not import `sync`; no
   tests ran.

Generated code was not repaired or altered for this publication.

## Benchmark command

```bash
./rigmark run \
  --base-url http://localhost:8000 \
  --model qwen3.8 \
  --label qwen38-flash-next-int4-pgx-none-public \
  --comparison-id qwen38-flash-next-pgx-20260915 \
  --metadata metadata.json \
  --extra-body '{"reasoning_effort":"none"}'
```

## Artifacts and integrity

- [JSON receipt](results/qwen38-flash-next-int4-pgx-none-public-20260916T004709Z.json)
- [Share card](results/qwen38-flash-next-int4-pgx-none-public-20260916T004709Z.card.txt)
- [Run metadata](metadata.json)

Full SHA-256 of the JSON receipt:

```text
74de0d6a135b5aee5f911746e1f0a30c83b0998ae8a610c5ca610c3ac12888ff
```
