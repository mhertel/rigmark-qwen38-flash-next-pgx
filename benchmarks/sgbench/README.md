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

## Historical image: six vLLM passes

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

## Consolidated image (`qwen38-flash-next:v1`) passes

The same unmodified sgbench script and the same prompts were run against the
consolidated image, for six passes.

| Pass | Q&A | Code | JSON | Math | LongCode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.7 | 57.5 | 64.9 | 52.8 | 50.0 |
| 2 | 54.4 | 67.1 | 73.8 | 60.9 | 65.9 |
| 3 | 53.9 | 72.9 | 73.7 | 65.3 | 54.2 |
| 4 | 48.0 | 71.3 | 73.6 | 62.7 | 59.6 |
| 5 | 51.3 | 65.3 | 74.5 | 59.2 | 53.8 |
| 6 | 54.8 | 71.2 | 76.2 | 60.3 | 54.0 |

All values are tok/s. Medians use the same method as above.

| Workload | Original | Six-run median | Range | Median improvement |
| --- | ---: | ---: | ---: | ---: |
| Q&A | 35.2 tok/s | 52.60 tok/s | 35.7–54.8 tok/s | +49.4% |
| Code | 38.4 tok/s | 69.15 tok/s | 57.5–72.9 tok/s | +80.1% |
| JSON | 43.9 tok/s | 73.75 tok/s | 64.9–76.2 tok/s | +68.0% |
| Math | 42.7 tok/s | 60.60 tok/s | 52.8–65.3 tok/s | +41.9% |
| LongCode | 43.6 tok/s | 54.10 tok/s | 50.0–65.9 tok/s | +24.1% |

**Pass 1 shows additional cold / cache warm-up effects.** It is the lowest
reading in every workload and is retained in the six-run summary for a
consistent comparison with the historical six-pass results. The subsequent
passes are more representative of warmed steady-state behavior.

## Interpretation

- This compares the same single-GB10 class of hardware with the same sgbench
  script and prompts.
- The current setup uses vLLM with the published INT4 AutoRound recipe.
- The original setup used SGLang/NVFP4.
- The cleanest workload comparison is Code: original **38.4 tok/s** versus the
  current six-run median of **71.45 tok/s** (**71.5 tok/s** rounded to one
  decimal).
- The consolidated image (`qwen38-flash-next:v1`) was measured with the same
  script and prompts, six passes, and is reported separately above.
- Pass 1 of the consolidated-image runs shows additional cold / warm-up
  effects; subsequent passes better represent warmed steady-state behavior.
- This comparison does not prove better model quality.
- Do not mix sgbench results with RigMark numbers because the harnesses differ.
