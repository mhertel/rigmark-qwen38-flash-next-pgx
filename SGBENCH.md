# sgbench comparison: Qwen3.8 Flash Next on a single GB10

This is a separate sgbench comparison, not a RigMark result. Do not mix the
numbers in this document with the RigMark headline numbers: the harnesses,
workloads, measurements, and output handling differ.

## Benchmark source and procedure

The comparison uses the original
[azampatti/GB10-3.8-Flash-Next](https://github.com/azampatti/GB10-3.8-Flash-Next)
sgbench harness at commit
`2dee9f3ba6ef00d5b46411ba5cb1fae047d77015`.

The only script change was replacing `http://localhost:30000` with
`http://localhost:8000`, directing the unmodified workloads and prompts to the
current vLLM server. The exact local procedure was:

```bash
git clone https://github.com/azampatti/GB10-3.8-Flash-Next.git gb10-original-bench
cd ~/gb10-original-bench
git checkout 2dee9f3ba6ef00d5b46411ba5cb1fae047d77015
cp sgbench.sh sgbench-vllm.sh
sed -i 's|localhost:30000|localhost:8000|g' sgbench-vllm.sh
bash sgbench-vllm.sh
```

`~` is a generic shell home-directory shorthand in the procedure above; no
user-specific path is required.

## Original published single-stream Run 2

| Workload | Original |
| --- | ---: |
| Q&A | 35.2 tok/s |
| Code | 38.4 tok/s |
| JSON | 43.9 tok/s |
| Math | 42.7 tok/s |
| LongCode | 43.6 tok/s |

## Current vLLM single-stream passes

| Pass | Q&A | Code | JSON | Math | LongCode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 43.7 | 59.8 | 72.0 | 58.1 | 57.6 |
| 2 | 53.1 | 69.6 | 73.2 | 66.6 | 47.4 |
| 3 | 50.8 | 72.0 | 75.1 | 58.7 | 52.6 |
| 4 | 56.7 | 71.9 | 79.9 | 62.1 | 61.8 |
| 5 | 54.1 | 71.0 | 77.8 | 63.3 | 53.5 |
| 6 | 56.1 | 72.8 | 79.2 | 64.6 | 57.5 |

All values are tok/s. The six-run median is the arithmetic mean of the third
and fourth sorted values; percentage improvement is `(median - original) /
original`.

| Workload | Original | Six-run median | Range | Median improvement |
| --- | ---: | ---: | ---: | ---: |
| Q&A | 35.2 tok/s | 53.60 tok/s | 43.7–56.7 tok/s | +52.3% |
| Code | 38.4 tok/s | 71.45 tok/s | 59.8–72.8 tok/s | +86.1% |
| JSON | 43.9 tok/s | 76.45 tok/s | 72.0–79.9 tok/s | +74.1% |
| Math | 42.7 tok/s | 62.70 tok/s | 58.1–66.6 tok/s | +46.8% |
| LongCode | 43.6 tok/s | 55.50 tok/s | 47.4–61.8 tok/s | +27.3% |

## Interpretation

- This compares the same single-GB10 class of hardware with the same sgbench
  script and prompts.
- The current setup uses vLLM with the published INT4 AutoRound recipe.
- The original setup used SGLang/NVFP4.
- The cleanest workload comparison is Code: original **38.4 tok/s** versus the
  current six-run median of **71.45 tok/s** (**71.5 tok/s** rounded to one
  decimal).
- This comparison does not prove better model quality.
- Do not mix sgbench results with RigMark numbers because the harnesses differ.
