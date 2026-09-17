# =============================================================================
# Qwen3.8 Flash Next — consolidated single-stage image, DGX Spark / GB10
#
# SUCCESSOR TO the historical three-stage chain:
#
#   qwen38-flash-dgx
#     -> qwen38-flash-dgx-specfix
#       -> qwen38-flash-dgx-specfix-instanttensor
#         == sha256:1eb74c6ef412a63da256f2f0f75fdb664cdb3ab0c529c9c0bc36fe5ce40dcebc
#
# This build collapses all 20 locally-added layers of that chain into ONE
# docker build, producing qwen38-flash-next:v1.
#
# The chat-template fix is deliberately NOT baked in. It stays an external
# runtime mod so it can be edited without rebuilding the image:
#   mods/fix-qwen3.8-flash-chat-template  ->  --chat-template fixed_chat_template.jinja
#
# Build:  ./build.sh
# =============================================================================

# -----------------------------------------------------------------------------
# Base: the official Qwen3.8-Flash-Next vLLM image, pinned by digest.
#
# Identical base to the historical chain, so the vLLM lineage is preserved
# exactly: 0.1.dev20073+g8e685d198  (git 8e685d198, torch 2.13 cu130,
# numpy 2.2.6, CUDA 13.0.1, arm64/sbsa). vLLM is NOT updated here.
#
# Source of the base's own build: https://github.com/vllm-project/vllm
# -----------------------------------------------------------------------------
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

# -----------------------------------------------------------------------------
# Provenance labels.
#
# The historical image carried ai.vllm.build.commit=unknown and
# org.opencontainers.image.revision=unknown, so the vLLM commit was only
# recoverable by exec'ing into a container. These labels fix that.
# -----------------------------------------------------------------------------
ARG SP=/usr/local/lib/python3.12/dist-packages

LABEL org.opencontainers.image.title="qwen38-flash-next" \
      org.opencontainers.image.source="https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound" \
      org.opencontainers.image.revision="45c4635cd21865825dec6b95c6c88cd186d15aff" \
      ai.qwen38flash.base.image="vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8" \
      ai.qwen38flash.vllm.version="0.1.dev20073+g8e685d198" \
      ai.qwen38flash.vllm.commit="8e685d198" \
      ai.qwen38flash.instanttensor="0.1.9" \
      ai.qwen38flash.b12x="base-provided-not-modified" \
      ai.qwen38flash.chat-template="external-runtime-mod"

# All patch targets, overridable at build time. Defaults match the historical
# image exactly; no --build-arg is required for a normal build.
ARG PLE=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
ARG GPTQ_PY=${SP}/vllm/model_executor/layers/quantization/auto_gptq.py
ARG MODEL_PY=${SP}/vllm/models/qwen3_8_flash_next/nvidia/model.py
ARG MTP_PY=${SP}/vllm/models/qwen3_8_flash_next/nvidia/mtp.py
ARG MAMBA_UTILS=${SP}/vllm/v1/worker/mamba_utils.py
ARG SPECULATOR_PY=${SP}/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py

# -----------------------------------------------------------------------------
# 1. PLE mmap — serve the 51B-parameter n-gram ("Engram") table from disk via
#    np.memmap instead of keeping it resident in the 128 GB unified pool.
#    No-op unless VLLM_PLE_MMAP=1 at runtime.
#
#    Source: qwen3.8-flash-next@45c4635  src/vllm_ple_mmap.py
#            sha256 6ffc537effd05ec6515bd8578c28d292cb376d7eb4a87697466206dfd57748b5
# -----------------------------------------------------------------------------
COPY patches/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py
RUN cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-flash-next: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read()); print('ple_layer.py patched OK')"

# -----------------------------------------------------------------------------
# 2. spark-fla-shmem — GB10 / sm_121 reports 99 KiB of shared memory per block
#    (same as ADA, where the big GDN tiles fit) but the flash-linear-attention
#    gate demands 100 KiB, so all 36 GDN layers silently fell back to small
#    Triton tiles. Lower the gate to 99 KiB so GB10 takes the big-tile path.
#
#    Source: qwen3.8-flash-next@45c4635  Dockerfile (sed block)
#    Upstream origin: Entrpi/qwen3.5-122B-A10B-on-spark  patch_fla_shmem.py
# -----------------------------------------------------------------------------
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem: GB10 99KiB = ADA, big GDN tiles fit|' ${FLA_UTILS} \
 && grep -q "spark-fla-shmem" ${FLA_UTILS} && echo "fla shmem gate patched"

# -----------------------------------------------------------------------------
# 3. spark-fla-warps — fla-org/flash-linear-attention#953: the chunked
#    delta-rule state kernel races on Blackwell when autotune picks
#    num_warps=4 (a tl.dot recurrence race), yielding nondeterministic
#    h/v_new, i.e. corrupt GDN state. Corruption tracks the cached-block
#    (prefix-cache resume) path. Upstream pins num_warps=2 on Blackwell;
#    this image only ever runs on GB10, so pin unconditionally.
#
#    Source: qwen3.8-flash-next@45c4635  Dockerfile (sed block)
# -----------------------------------------------------------------------------
RUN sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps: fla#953 Blackwell tl.dot race|' ${FLA_CDH} \
 && grep -q "spark-fla-warps" ${FLA_CDH} && echo "fla num_warps pinned"

# -----------------------------------------------------------------------------
# 4. int4 + fp8 hybrid dispatch — vLLM's GPTQ config quantizes listed layers
#    and leaves the rest to UnquantizedLinearMethod, with no notion of "this
#    excluded layer is actually fp8 in the checkpoint". This shim wraps
#    AutoGPTQConfig to dispatch blockwise-fp8 side layers.
#    No-op unless VLLM_FP8_HYBRID=1 at runtime.
#
#    Source: qwen3.8-flash-next@45c4635  src/vllm_fp8_hybrid.py
#            sha256 d272b63ff72402a0ade73256363269fa493ee4cd8c1ed577fc05c02a4f87b569
# -----------------------------------------------------------------------------
COPY patches/vllm_fp8_hybrid.py ${SP}/vllm_fp8_hybrid.py
RUN printf '\n\n# --- qwen38-flash-next: int4+fp8 hybrid dispatch (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> ${GPTQ_PY} \
 && python3 -c "import ast; ast.parse(open('${GPTQ_PY}').read()); print('auto_gptq.py patched OK')"

# -----------------------------------------------------------------------------
# 5. never-evict prompt pinning — --never-evict-kv-cache-prompt-includes pins
#    the HA system prompt's KV blocks against eviction (pin-only port of the
#    122B recipe's arc_pin2). No-op unless the flag is passed at runtime.
#
#    Source: qwen3.8-flash-next@45c4635  src/patch_never_evict.py
# -----------------------------------------------------------------------------
COPY patches/patch_never_evict.py /tmp/patch_never_evict.py
RUN python3 /tmp/patch_never_evict.py && rm /tmp/patch_never_evict.py

# -----------------------------------------------------------------------------
# 6. int8 GPTQ LM head — upstream constructs ParallelLMHead without
#    quant_config, forcing a bf16 head (1.27 GiB, plus a bf16 GEMV per token
#    over a 248320 vocab). Passing quant_config in BOTH model.py and mtp.py
#    lets the head pick up the checkpoint's int8 GPTQ packing. Without the
#    mtp.py half, MTP >= 3 crashes at load with
#    "no module or parameter named 'lm_head.qweight'".
#
#    Source: qwen3.8-flash-next@45c4635  Dockerfile (sed block)
# -----------------------------------------------------------------------------
RUN cp ${MODEL_PY} ${MODEL_PY}.orig && cp ${MTP_PY} ${MTP_PY}.orig \
 && sed -i 's|prefix=maybe_prefix(prefix, "lm_head"),|quant_config=vllm_config.quant_config,\n            prefix=maybe_prefix(prefix, "lm_head"),|' ${MODEL_PY} \
 && sed -i 's|prefix=maybe_prefix(prefix, "lm_head"),|quant_config=vllm_config.quant_config,\n                    prefix=maybe_prefix(prefix, "lm_head"),|' ${MTP_PY} \
 && grep -c 'quant_config=vllm_config.quant_config' ${MODEL_PY} ${MTP_PY} \
 && python3 -c "import ast; [ast.parse(open(p).read()) for p in ('${MODEL_PY}','${MTP_PY}')]; print('lm_head patched OK in model.py + mtp.py')"

# -----------------------------------------------------------------------------
# 7. guarded mamba_utils — the "Xid 31 / illegal memory access under load"
#    crash with PREFIX_CACHE=1 + MTP (also blazux/qwen3.8-Flash-DGX#2).
#    CUDA_LAUNCH_BLOCKING=1 caught the fault synchronously inside
#    precopy_mamba_align_fused_kernel reading a wild address derived from a
#    bad block id. This file is the base image's stock mamba_utils.py plus:
#      (a) upstream a02cfccbc6 "[Bugfix][Mamba] Fix overlapping state copy
#          race" (vllm#50729, landed after this image's vLLM snapshot), and
#      (b) a bounds guard in _copy_mamba_state_block validating block ids
#          against each state pool; an out-of-range id skips the copy and
#          bumps a counter instead of taking down the CUDA context.
#
#    Source: qwen3.8-flash-next@45c4635  src/mamba_utils_guarded.py
#            sha256 18be29f43147d93b9ac50b38a640bb7fdb9da1926a312b985cba00d1353bcddb
# -----------------------------------------------------------------------------
RUN cp ${MAMBA_UTILS} ${MAMBA_UTILS}.orig
COPY patches/mamba_utils_guarded.py ${MAMBA_UTILS}
RUN python3 -c "import ast; ast.parse(open('${MAMBA_UTILS}').read()); print('mamba_utils.py guarded OK')"

# -----------------------------------------------------------------------------
# 8. prefix-cache diagnosis logging — per-group hit breakdown, mamba
#    boundary-state publication, cached-block eviction, prefill chunk stops.
#    Enabled by VLLM_HIT_DEBUG=1.
#
#    Source: qwen3.8-flash-next@45c4635  src/patch_hit_debug.py
# -----------------------------------------------------------------------------
COPY patches/patch_hit_debug.py /tmp/patch_hit_debug.py
RUN python3 /tmp/patch_hit_debug.py && rm /tmp/patch_hit_debug.py

# -----------------------------------------------------------------------------
# 9. mamba align chunk split — prefill chunks must end at MAMBA block
#    boundaries (1600), not the scheduler minimum block size (8); otherwise
#    cold requests publish no mamba state and repeated prompts only hit the
#    prefix cache from the 3rd request on.
#
#    Source: qwen3.8-flash-next@45c4635  src/patch_mamba_align_split.py
# -----------------------------------------------------------------------------
COPY patches/patch_mamba_align_split.py /tmp/patch_mamba_align_split.py
RUN python3 /tmp/patch_mamba_align_split.py && rm /tmp/patch_mamba_align_split.py

# -----------------------------------------------------------------------------
# 10. on-demand step profiler — torch.profiler around engine steps via
#     VLLM_STEP_PROFILE=1 + touch /tmp/profile_trigger. This vLLM predates
#     VLLM_TORCH_PROFILER_DIR.
#
#     Source: qwen3.8-flash-next@45c4635  src/patch_step_profile.py
# -----------------------------------------------------------------------------
COPY patches/patch_step_profile.py /tmp/patch_step_profile.py
RUN python3 /tmp/patch_step_profile.py && rm /tmp/patch_step_profile.py

# -----------------------------------------------------------------------------
# 11. MTP speculator PIECEWISE prefill CUDA-graph fix.
#
#     The PLE mmap lookup performs a GPU->CPU transfer and is NOT safe during
#     MTP/speculator PIECEWISE prefill CUDA graph capture on DGX Spark. This
#     keeps PIECEWISE CUDA graphs enabled for the target model but disables
#     graph capture for speculator prefill only.
#
#     The historical chain produced this by docker cp-ing speculator.py out of
#     qwen38-flash-dgx, applying an exact-match string replacement, and COPYing
#     the result back in a second image. Here the same replacement is applied
#     in place, so the file is never vendored and stays base-derived.
#
#     Source: patches/patch_speculator_cudagraph.py
#             (derived from /home/mike/patch-flash-speculator.sh)
# -----------------------------------------------------------------------------
COPY patches/patch_speculator_cudagraph.py /tmp/patch_speculator_cudagraph.py
RUN SPECULATOR_PY=${SPECULATOR_PY} python3 /tmp/patch_speculator_cudagraph.py && rm /tmp/patch_speculator_cudagraph.py

# -----------------------------------------------------------------------------
# 12. InstantTensor — ultra-fast distributed Safetensors loader.
#
#     --load-format instanttensor is supported NATIVELY by the base image's
#     vLLM (vllm/config/load.py:38 documents it; model_loader/* references
#     it), so no source patch is required — only the package.
#
#     Pinned to 0.1.9, the exact version in the benchmarked image. The
#     historical build used an unpinned "pip install instanttensor", so the
#     version it resolved was whatever PyPI served on 2026-09-09.
#
#     B12X is NOT installed or modified here. B12X-related modules ARE already
#     shipped by the pinned upstream vLLM base image (10 files under
#     dist-packages, verified byte-identical to the base by build.sh). This
#     build does not add, change, or enable any of them. The separate local B12X
#     integration belongs to a different lineage
#     (vllm-029-qwen38-plemmap-hybridfp8-lmhead-mtp, B12X_REPO=
#     https://github.com/lukealonso/b12x.git) and is NOT used here.
# -----------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir instanttensor==0.1.9

# -----------------------------------------------------------------------------
# 13. Chat template — intentionally NOT here.
#
#     The chat-template fix stays a runtime mod:
#       mods/fix-qwen3.8-flash-chat-template/run.sh
#         -> copies chat_template.jinja to $WORKSPACE_DIR/fixed_chat_template.jinja
#     and the serving recipe passes --chat-template fixed_chat_template.jinja.
#
#     Keeping it external means the template can be corrected without
#     rebuilding or revalidating the image.
# -----------------------------------------------------------------------------
