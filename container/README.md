# `qwen38-flash-next:v1` — consolidated container build

## What this is

A **consolidated successor** to the historical image that produced the published
RigMark result. That image was assembled as a three-stage chain, each stage
adding one layer:

| Stage | Tag | Created (PDT) | Layers | Image ID |
|---|---|---|---|---|
| 1 | `qwen38-flash-dgx` | 2026-09-07 18:51:41 | 50 | `d526e34e289d…` |
| 2 | `qwen38-flash-dgx-specfix` | 2026-09-08 19:35:10 | 51 | `cdd0ab3f1492…` |
| 3 | `qwen38-flash-dgx-specfix-instanttensor` | 2026-09-09 21:54:15 | 52 | `1eb74c6ef412…` |

```
historical benchmark image:
  sha256:1eb74c6ef412a63da256f2f0f75fdb664cdb3ab0c529c9c0bc36fe5ce40dcebc

new image:
  qwen38-flash-next:v1
  built image ID: sha256:320d1bcf0dc46f913cfa58a30b29131bb26fadb650d646bbf13b943a5e324156
```

This directory builds the same content in **one** `docker build`. The
historical image is left untouched; nothing here deletes, retags, or rebuilds
it.

## Why the historical digest cannot be reproduced bit-for-bit

Two independent reasons, neither fixable:

1. **Build timestamps.** BuildKit stamps `Created` in the image config at build
   time. The config digest is therefore different for every build, so no
   rebuild can ever yield `1eb74c6ef412…` again. Only the *content* is
   reproducible, not the digest.
2. **Layer grouping differs.** The chain's root is
   `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
   which is now pulled and retained locally (32 layers, `arm64/linux`). Those 32
   base layers are byte-identical in the base, the historical image, and this
   build. The 20 locally-added layers of the chain become 21 here, because the
   whole chain collapses into one 22-step build, so the added layer digests
   differ even where the resulting file content is identical.

A third, smaller difference: the historical step 3 ran
`pip install --no-cache-dir instanttensor` **unpinned**, so it installed
whatever PyPI served on 2026-09-09 (0.1.9). This build pins
`instanttensor==0.1.9` explicitly.

## The chat template stays external — deliberately

**No chat template is baked into this image.** The fix remains a runtime mod:

```
mods/fix-qwen3.8-flash-chat-template/run.sh
  -> copies chat_template.jinja to $WORKSPACE_DIR/fixed_chat_template.jinja
serving recipe
  -> --chat-template fixed_chat_template.jinja
```

Rationale: the template is the part most likely to need correction (tool-call
and reasoning-block formatting), and keeping it external means it can be fixed
without rebuilding or revalidating the image. The historical image behaved the
same way — `docker diff` of the running container shows the template arriving
at runtime, and the image history contains no chat-template layer.

`container/` intentionally does **not** duplicate the mod. The mod lives in
`mods/fix-qwen3.8-flash-chat-template/`.

> **Status:** the mod is now vendored in this repository at
> `mods/fix-qwen3.8-flash-chat-template/` (`run.sh`, `chat_template.jinja`,
> `chat_template.jinja.medium`), byte-identical to the copy `docker cp`'d into
> the benchmarked container. The recipe is therefore runnable as published.
> The active template is `chat_template.jinja`
> (`sha256 a6e1c0921795725e3d649347912d136fa6915ac60804345ce1755fba41df458e`);
> `chat_template.jinja.medium` is the same file with `reasoning_effort`
> defaulting to `medium` instead of `high`.

## Exact baked patches

Eleven patches are baked in, all proven present in the historical image and
verified byte-for-byte inside the running `vllm_node` container.

| # | Patch | Mechanism | Runtime switch | Source |
|---|---|---|---|---|
| 1 | PLE n-gram table served by `mmap` | `COPY vllm_ple_mmap.py` + hook appended to `ple_layer.py` | `VLLM_PLE_MMAP=1` | upstream `src/vllm_ple_mmap.py` |
| 2 | FLA shared-memory gate `102400 → 101376` | `sed` | always on | upstream `Dockerfile` |
| 3 | FLA `num_warps [2,4] → [2]` | `sed` | always on | upstream `Dockerfile` |
| 4 | INT4 + FP8 hybrid dispatch | `COPY vllm_fp8_hybrid.py` + hook appended to `auto_gptq.py` | `VLLM_FP8_HYBRID=1` | upstream `src/vllm_fp8_hybrid.py` |
| 5 | never-evict prompt pinning | `python3 patch_never_evict.py` | `--never-evict-kv-cache-prompt-includes` | upstream `src/patch_never_evict.py` |
| 6 | INT8 GPTQ LM head | `sed` on `model.py` **and** `mtp.py` | always on | upstream `Dockerfile` |
| 7 | guarded `mamba_utils.py` | whole-file replace (base + vllm#50729 + bounds guard) | always on | upstream `src/mamba_utils_guarded.py` |
| 8 | prefix-cache hit debug | `python3 patch_hit_debug.py` | `VLLM_HIT_DEBUG=1` | upstream `src/patch_hit_debug.py` |
| 9 | mamba align chunk split (1600) | `python3 patch_mamba_align_split.py` | always on | upstream `src/patch_mamba_align_split.py` |
| 10 | on-demand step profiler | `python3 patch_step_profile.py` | `VLLM_STEP_PROFILE=1` | upstream `src/patch_step_profile.py` |
| 11 | MTP speculator PIECEWISE prefill graph disable | exact-match replace in `speculator.py` | always on | `patches/patch_speculator_cudagraph.py` (local) |

Patches 2, 3, 11 are GB10 / `sm_121` correctness fixes. Patch 2 is GB10
performance (99 KiB shared memory per block reads as ADA, so the big GDN tiles
fit — but the gate demanded 100 KiB and all 36 GDN layers fell back to small
tiles). Patch 3 fixes a Blackwell `tl.dot` race
([flash-linear-attention#953](https://github.com/fla-org/flash-linear-attention/issues/953))
that corrupts GDN state along the prefix-cache-resume path. Patch 11 exists
because the PLE `mmap` lookup does a GPU→CPU transfer, which is unsafe during
MTP/speculator PIECEWISE prefill graph capture on DGX Spark.

### B12X

B12X-related modules are shipped by the pinned upstream vLLM base image. This
custom build does not add or modify B12X and does not use the separate local
B12X integration from the other vLLM image lineage
(`vllm-029-qwen38-plemmap-hybridfp8-lmhead-mtp`,
`B12X_REPO=https://github.com/lukealonso/b12x.git`).

Ten `*b12x*` files exist under `dist-packages` in the base image, in the
historical benchmark image, and in `qwen38-flash-next:v1` — byte-identical in
all three. They are therefore **not** part of this build's local optimization
work, and they are **not** absent. `build.sh` verifies this by comparing the
set and every per-file hash against the pinned base.

## Exact upstream provenance

The same chain, with pinned digests and component attribution, is recorded in
[`../docs/provenance.md`](../docs/provenance.md).

| Item | Value |
|---|---|
| Source repo | `https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound.git` |
| Pinned commit | `45c4635cd21865825dec6b95c6c88cd186d15aff` |
| Base image | `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |
| Base source | `https://github.com/vllm-project/vllm` |
| vLLM | `0.1.dev20073+g8e685d198` (git `8e685d198`) — **not updated** |
| InstantTensor | `instanttensor==0.1.9` |
| Model | `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` @ `e4f732c0f817e3904a96ffadb3402522fb1a7b82` |

Staged file digests (git blob at the pinned commit, verified by `build.sh`):

```
vllm_ple_mmap.py           6ffc537effd05ec6515bd8578c28d292cb376d7eb4a87697466206dfd57748b5
vllm_fp8_hybrid.py         d272b63ff72402a0ade73256363269fa493ee4cd8c1ed577fc05c02a4f87b569
mamba_utils_guarded.py     18be29f43147d93b9ac50b38a640bb7fdb9da1926a312b985cba00d1353bcddb
patch_never_evict.py       39b434d84a58f5aae048b0d586b58fabf3a82de1c39a58f1eee2eb701b2159a3
patch_hit_debug.py         1c000b0996509240a5d74cb13ad42c2d2adf0ae30211f6ba5e10a73349cba77f
patch_mamba_align_split.py 8850ddab02527a64a17f1694e658a2b682a3ff7422f15b732b98c9c2026e24a7
patch_step_profile.py      f898231ae3834f821ca494cdcdf7911281aeb9485f2f7f2616ad7a2743f6a560
```

`patches/patch_speculator_cudagraph.py` is the only file that does **not**
exist upstream. It is derived from the local `patch-flash-speculator.sh` that
produced the historical `specfix` layer.

## Build

```bash
./build.sh
```

Equivalent to:

```bash
git clone https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound.git .build/src
git -C .build/src checkout --detach 45c4635cd21865825dec6b95c6c88cd186d15aff
# stage src/*.py -> .build/context/patches/  (sha256-verified)
docker build -f Dockerfile -t qwen38-flash-next:v1 .build/context
```

`build.sh` also runs a post-build verification pass: vLLM version,
InstantTensor version, all 8 patch markers, absence of the chat template,
B12X unchanged from the pinned base (file set and every hash), and absence of
leftover `/tmp/patch_*.py` scripts.

## Expected runtime flags

The image is inert without these. They come from the serving recipe, not the
image:

```
VLLM_PLE_MMAP=1
VLLM_PLE_MMAP_WORKERS=32
VLLM_PLE_MMAP_PREWARM=0
VLLM_PLE_MMAP_PREFETCH=0
VLLM_PLE_MMAP_MADV_RANDOM=0
VLLM_PLE_MMAP_DIR=<model dir>/ple-table
VLLM_FP8_HYBRID=1
VLLM_MARLIN_USE_ATOMIC_ADD=1
VLLM_USE_DEEP_GEMM=0
VLLM_USE_FLASHINFER_SAMPLER=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```
--load-format instanttensor
--chat-template fixed_chat_template.jinja      # supplied by the external mod
--enable-prefix-caching
--enable-chunked-prefill
--no-enable-flashinfer-autotune
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE","splitting_ops":[…,"vllm::ple_mmap_lookup"]}'
```

Note: `VLLM_PLE_MMAP*` and `VLLM_FP8_HYBRID` are read directly by the baked-in
patch modules and are **not** registered vLLM environment variables — the
server logs `Unknown vLLM environment variable detected` warnings for them.
That is expected and matches the historical image.

## Revalidation

Content equivalence to the historical chain is necessary but not sufficient:
layer structure differs, the speculator fix is applied in place rather than via
a vendored file copy, and the InstantTensor install is now pinned. The full
RigMark protocol has since been re-run against this image and compared against
the published receipt.

**Status: revalidated.** Build verification is 14/14 PASS, and the warm v1
RigMark run passed 15/15 basic output gates. High-concurrency throughput is
effectively unchanged from the historical run; single-stream code and prose
decode measured somewhat lower. No exact performance equivalence is claimed.

Full evidence: [`../docs/validation.md`](../docs/validation.md). Source chain:
[`../docs/provenance.md`](../docs/provenance.md).
