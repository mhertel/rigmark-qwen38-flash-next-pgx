# Benchmarks

Validation and benchmark output produced against the Qwen3.8 Flash Next serving
setup. Two independent harnesses are used. They measure different things and
their numbers must not be mixed.

## Harnesses

| Harness | What it measures | Where |
| --- | --- | --- |
| [RigMark](https://github.com/alexellis/rigmark) | Agent-style serving workloads: single-stream decode across code / prose / structured output, long-context prefill and replay, and capped concurrent generation with per-workload output gates. | `rigmark/` |
| sgbench | A separate throughput harness: five fixed single-stream workloads (Q&A, Code, JSON, Math, LongCode) driven against the OpenAI-compatible endpoint. | `sgbench/` |

RigMark's decode estimate, its basic output gates, and sgbench's per-workload
tok/s come from different prompts, different output handling, and different
aggregation. A number from one is not comparable with a number from the other.

## Runs

| Directory | Image | Role |
| --- | --- | --- |
| `rigmark/historical/` | the historical three-stage image, `sha256:1eb74c6e…` | published result, retained for comparison |
| `rigmark/v1/` | `qwen38-flash-next:v1`, `sha256:320d1bcf…` | validates the consolidated image |
| `sgbench/README.md` | both | six passes per image |

Within `rigmark/v1/`, the **warm** run (`…-none-warm-…`) is the primary v1
validation result. The earlier v1 run is retained as first-run / cold evidence:
it carries additional cache warm-up effects and is not the headline number.

## Reading the results

- Every RigMark run carries a JSON receipt and a share card. The receipt is the
  authoritative artifact; the card renders it.
- Throughput says nothing about output quality. A higher tok/s is not evidence
  that the model is better, and RigMark's basic output gates are completion
  checks, not correctness checks. See
  [`../docs/validation.md`](../docs/validation.md).
- Benchmark artifacts in this directory are generated output. Their contents,
  filenames, hashes, and timestamps are provenance and are not edited.

## Reproducing a run

Neither harness is vendored in this repository.

RigMark is run from a local RigMark checkout against the running server, passing
the run metadata that accompanies each result:

```bash
./rigmark run \
  --base-url http://localhost:8000 \
  --model qwen3.8 \
  --label <label> \
  --comparison-id <comparison-id> \
  --metadata <this repository>/benchmarks/rigmark/v1/metadata.json \
  --extra-body '{"reasoning_effort":"none"}'
```

The historical run used `benchmarks/rigmark/historical/metadata.json`; the v1
runs used `benchmarks/rigmark/v1/metadata.json`.

The sgbench harness comes from the upstream
[azampatti/GB10-3.8-Flash-Next](https://github.com/azampatti/GB10-3.8-Flash-Next)
repository. The exact local procedure is recorded in
[`sgbench/README.md`](sgbench/README.md).
