# Provenance

Source chain for the image served by
[`recipes/0-qwen3.8-flash-a5b.yaml`](../recipes/0-qwen3.8-flash-a5b.yaml).

## Chain

| Item | Value |
| --- | --- |
| Upstream custom source | <https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound.git> |
| Pinned upstream commit | `45c4635cd21865825dec6b95c6c88cd186d15aff` |
| Pinned vLLM base | `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |
| Final consolidated image | `qwen38-flash-next:v1` |
| Final image ID | `sha256:320d1bcf0dc46f913cfa58a30b29131bb26fadb650d646bbf13b943a5e324156` |
| vLLM | `0.1.dev20073+g8e685d198` (git `8e685d198`) — **not** updated by this build |
| InstantTensor | `instanttensor==0.1.9` |
| Model | `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` |
| Model revision | `e4f732c0f817e3904a96ffadb3402522fb1a7b82` |

## How the consolidated image is assembled

`container/build.sh` clones the upstream source at the pinned commit, stages the
patch sources into a build context, verifies each staged file's SHA-256 against
the pinned value, and runs **one** `docker build` against the pinned base digest.

Eleven patches are applied in that single build:

- seven patch sources are staged from the pinned upstream commit and
  SHA-256-verified (`vllm_ple_mmap.py`, `vllm_fp8_hybrid.py`,
  `mamba_utils_guarded.py`, `patch_never_evict.py`, `patch_hit_debug.py`,
  `patch_mamba_align_split.py`, `patch_step_profile.py`);
- one patch source, `patches/patch_speculator_cudagraph.py`, does **not** exist
  upstream and is carried locally;
- three patches are applied by `sed` directly in the `Dockerfile`.

The full patch list, the mechanism of each patch, and the runtime switch each
one honours are documented in
[`container/README.md`](../container/README.md).

## Historical image

The consolidated image succeeds a three-stage chain. That chain, not this build,
produced the published historical RigMark result.

| Stage | Tag | Created (PDT) | Layers | Image ID |
|---|---|---|---|---|
| 1 | `qwen38-flash-dgx` | 2026-09-07 18:51:41 | 50 | `d526e34e289d…` |
| 2 | `qwen38-flash-dgx-specfix` | 2026-09-08 19:35:10 | 51 | `cdd0ab3f1492…` |
| 3 | `qwen38-flash-dgx-specfix-instanttensor` | 2026-09-09 21:54:15 | 52 | `1eb74c6ef412…` |

```text
historical benchmark image:
  sha256:1eb74c6ef412a63da256f2f0f75fdb664cdb3ab0c529c9c0bc36fe5ce40dcebc

consolidated image:
  qwen38-flash-next:v1
  built image ID:
  sha256:320d1bcf0dc46f913cfa58a30b29131bb26fadb650d646bbf13b943a5e324156
```

The historical digest cannot be reproduced bit-for-bit, for two reasons that are
not fixable:

1. **Build timestamps.** BuildKit stamps `Created` in the image config at build
   time, so the config digest differs for every build. Only content is
   reproducible, not the digest.
2. **Layer grouping.** The 20 locally-added layers of the three-stage chain
   collapse into one build, so the added-layer digests differ even where the
   resulting file content is identical. The base's first 32 layers are
   byte-identical across the base, the historical image, and this build.

A third, smaller difference: the historical step 3 installed `instanttensor`
unpinned and therefore took whatever PyPI served on 2026-09-09 (0.1.9). This
build pins `instanttensor==0.1.9` explicitly.

## B12X

The pinned base image **contains** B12X-related modules. The custom/consolidated
patch chain does **not** add, modify, or enable a separate B12X integration.
Verification showed those base-provided files were unchanged — status:

> **base-provided-not-modified**

This is **not** a claim that B12X is absent from the image. Ten `*b12x*` files
are present under `dist-packages` in the base image, in the historical benchmark
image, and in `qwen38-flash-next:v1`, byte-identical in all three. `build.sh`
verifies this by comparing both the file set and every per-file hash against the
pinned base.

The separate local B12X integration belongs to a different image lineage
(`vllm-029-qwen38-plemmap-hybridfp8-lmhead-mtp`,
`B12X_REPO=https://github.com/lukealonso/b12x.git`) and is **not** used here.

## Chat template

The chat-template fix is deliberately **not** baked into the image. It stays an
external runtime mod:

```text
mods/fix-qwen3.8-flash-chat-template/run.sh
  -> copies chat_template.jinja to $WORKSPACE_DIR/fixed_chat_template.jinja
recipes/0-qwen3.8-flash-a5b.yaml
  -> --chat-template fixed_chat_template.jinja
```

Keeping it external means the template can be corrected without rebuilding or
revalidating the image. The historical image behaved the same way: `docker diff`
of the running container shows the template arriving at runtime, and the image
history contains no chat-template layer.

The active template is `mods/fix-qwen3.8-flash-chat-template/chat_template.jinja`
(`sha256 a6e1c0921795725e3d649347912d136fa6915ac60804345ce1755fba41df458e`).
`chat_template.jinja.medium` is the same file with `reasoning_effort` defaulting
to `medium` instead of `high`.

## Attribution

| Component | Source | Licence |
| --- | --- | --- |
| Custom patch chain and build | <https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound>, itself forked from <https://github.com/blazux/qwen3.8-Flash-DGX> | Apache-2.0 (`Copyright 2026 blazux`) |
| vLLM base image | <https://github.com/vllm-project/vllm> | Apache-2.0 |
| InstantTensor loader | `instanttensor==0.1.9` on PyPI | Apache-2.0 (`Copyright 2026 Yitao Yuan`) |
| Model checkpoint | `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`, quantised with Intel's W4A16 AutoRound workflow | see the model card |
| RigMark | <https://github.com/alexellis/rigmark> | MIT (`Copyright (c) 2026 Alex Ellis, OpenFaaS Ltd`) |
| sgbench harness | <https://github.com/azampatti/GB10-3.8-Flash-Next> | see upstream |

No licence terms are asserted here beyond what the sources themselves state.
The upstream source tree is fetched at build time into `container/.build/src`,
which is gitignored and therefore not redistributed from this repository.

## Reproducibility

`container/build.sh` is the reproducible entry point. It pins the source commit,
the base digest, and the InstantTensor version, and it fails the build if any
staged file's SHA-256 does not match the pinned value. Its verification pass is
summarised in [`validation.md`](validation.md).
