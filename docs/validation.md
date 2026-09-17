# Validation

Evidence that `qwen38-flash-next:v1` is a faithful successor to the historical
benchmark image.

Everything below reports throughput and output-gate results. **Throughput is not
evidence about output quality**, and RigMark's basic output gates are completion
checks, not correctness checks. No result here is offered as a claim that the
model produces better output than any other configuration.

## Build verification — 14/14 PASS

`container/build.sh` runs a 14-check verification pass against the built image.
All 14 checks passed.

| # | Check |
|---|---|
| 1 | vLLM version == `0.1.dev20073+g8e685d198` |
| 2 | InstantTensor version == `0.1.9` |
| 3 | `spark-fla-shmem` marker present |
| 4 | `spark-fla-warps` marker present |
| 5 | PLE mmap hook present |
| 6 | FP8 hybrid hook present |
| 7 | LM-head quantisation present in `model.py` |
| 8 | LM-head quantisation present in `mtp.py` |
| 9 | Speculator prefill CUDA-graph fix present |
| 10 | Guarded `mamba_utils.py` present |
| 11 | `speculator.py` not vendored (no `.orig`) |
| 12 | Chat template **not** baked in |
| 13 | B12X unchanged from the pinned base (file set and every hash) |
| 14 | Build-time patch scripts removed from `/tmp` |

## Runtime smoke test

Against the running container:

- container name `qwen38-flash-next`, image `qwen38-flash-next:v1`;
- `qwen3.8` visible via `/v1/models`;
- `reasoning_effort: none` returns `reasoning: null`;
- `reasoning_tokens = 0`.

## Static comparison against the historical image

- **7,071** comparable files checked.
- Only **two** differences were found in the patched vLLM source, both
  permitted cosmetic source-tag differences.
- The base image's first **32** layers are byte-identical.
- InstantTensor's Python sources are identical.
- The compiled InstantTensor `.so` differs. That difference is attributable to
  ephemeral build paths and the build ID embedded at compile time, not to source
  behaviour.

## RigMark — v1 warm run (primary)

`benchmarks/rigmark/v1/qwen38-flash-next-v1-none-warm-20260916T053329Z.json`

Protocol `1.1.0`, reasoning effort `none`, **15/15 basic output gates passed**.

| Measurement | v1 warm | Historical |
| --- | ---: | ---: |
| Code | 78.5 tok/s | 82.1 tok/s |
| Prose | 42.0 tok/s | 44.9 tok/s |
| Structured* | 85.0 tok/s | 86.5 tok/s |
| 64K cold prefill | 2,403 tok/s | 2,483 tok/s |
| 64K immediate replay | 45,911 tok/s | 46,636 tok/s |
| C1 aggregate | 72.9 tok/s | 71.9 tok/s |
| C2 aggregate | 115.6 tok/s | 114.7 tok/s |
| C4 aggregate | 181.6 tok/s | 182.4 tok/s |

\* Structured output is RigMark's predictable-output / speculative-decoding
ceiling. It is not a proxy for coding-agent speed.

## RigMark — v1 first run (cold / first-run evidence)

`benchmarks/rigmark/v1/qwen38-flash-next-v1-none-20260916T052134Z.json`

Also **15/15 basic output gates passed**. Retained because it includes additional
cache warm-up effects and is therefore not the headline number.

| Measurement | v1 first run |
| --- | ---: |
| Code | 77.3 tok/s |
| Prose | 40.3 tok/s |
| Structured | 85.5 tok/s |
| 64K cold prefill | 2,401 tok/s |
| 64K immediate replay | 46,395 tok/s |
| C1 aggregate | 65.3 tok/s |
| C2 aggregate | 114.4 tok/s |
| C4 aggregate | 182.3 tok/s |

## Basic output gates versus code audit

RigMark's basic output gate and a code audit measure different things.

The **15/15 basic output gates passed** result means that all benchmark outputs
completed and met RigMark's basic output requirements. It is **not** a claim that
generated code is correct.

The code audit carried out for the published historical run replays each retained
generated Go implementation with its own model-generated tests in RigMark's
locked-down Go 1.25 environment. Only **1/5** model-supplied Go test suites
passed:

1. **Run 1 — build failed.** `ratelimit.go:28:11: undefined: sync`. The
   generated implementation uses `sync.Mutex` but does not import `sync`; no
   tests ran.
2. **Run 2 — build failed.** `ratelimit.go:29:11: undefined: sync`. Same defect;
   no tests ran.
3. **Run 3 — tests failed.** The first failure is
   `ratelimit_test.go:117: AllowN(6) = false, want true` in
   `TestAllowN/refill_capped`. The generated test gives the bucket capacity 5 and
   immediately requests 6 tokens, yet expects success. The same suite also
   reports `AllowN(5) = false, want true` in `multiple_calls`: after two
   successful 5-token withdrawals from capacity 10 it advances only one second at
   one token per second and incorrectly expects another 5-token withdrawal to
   succeed. The implementation and its generated tests disagree; 12
   tests/subtests were started.
4. **Run 4 — passed.** All 13 model-generated tests/subtests passed.
5. **Run 5 — build failed.** `ratelimit.go:23:13: undefined: sync`. Same defect;
   no tests ran.

Generated code was not repaired or altered for publication. This audit was run
against the historical published run; it has not been re-run against the v1 runs
documented above.

## What the v1 evidence supports, and what it does not

Supported:

- functional gates remain **15/15**;
- high-concurrency throughput is effectively unchanged
  (C1 72.9 vs 71.9, C2 115.6 vs 114.7, C4 181.6 vs 182.4);
- the warm v1 run measured somewhat lower single-stream code and prose decode
  than the historical run (78.5 vs 82.1 and 42.0 vs 44.9 tok/s).

**Not** supported, and not claimed:

- exact performance equivalence with the historical image;
- any statement about output quality inferred from throughput;
- that the historical digest can be reproduced bit-for-bit (see
  [`provenance.md`](provenance.md)).
