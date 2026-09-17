#!/usr/bin/env bash
# =============================================================================
# Consolidated build for qwen38-flash-next:v1
#
# Stages the seven upstream patch files from the pinned source commit, runs ONE
# docker build, then verifies the result against the historical benchmark image.
#
# The chat-template mod is NOT staged and NOT copied. It stays external:
#   mods/fix-qwen3.8-flash-chat-template
#
# Usage:
#   ./build.sh              stage + build + verify
#   ./build.sh --verify-only
#   ./build.sh --stage-only
#   ./build.sh --no-verify
# =============================================================================
set -euo pipefail

# --- pinned provenance -------------------------------------------------------
UPSTREAM_URL="https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound.git"
UPSTREAM_COMMIT="45c4635cd21865825dec6b95c6c88cd186d15aff"
BASE_IMAGE="vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
INSTANTTENSOR_VERSION="0.1.9"
EXPECTED_VLLM="0.1.dev20073+g8e685d198"

IMAGE_NAME="${IMAGE_NAME:-qwen38-flash-next}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# sha256 of every staged file, taken from git blob at UPSTREAM_COMMIT.
declare -A PINNED=(
  [vllm_ple_mmap.py]="6ffc537effd05ec6515bd8578c28d292cb376d7eb4a87697466206dfd57748b5"
  [vllm_fp8_hybrid.py]="d272b63ff72402a0ade73256363269fa493ee4cd8c1ed577fc05c02a4f87b569"
  [mamba_utils_guarded.py]="18be29f43147d93b9ac50b38a640bb7fdb9da1926a312b985cba00d1353bcddb"
  [patch_never_evict.py]="39b434d84a58f5aae048b0d586b58fabf3a82de1c39a58f1eee2eb701b2159a3"
  [patch_hit_debug.py]="1c000b0996509240a5d74cb13ad42c2d2adf0ae30211f6ba5e10a73349cba77f"
  [patch_mamba_align_split.py]="8850ddab02527a64a17f1694e658a2b682a3ff7422f15b732b98c9c2026e24a7"
  [patch_step_profile.py]="f898231ae3834f821ca494cdcdf7911281aeb9485f2f7f2616ad7a2743f6a560"
)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/.build"
SRC="$STAGE/src"
CTX="$STAGE/context"

DO_BUILD=1
DO_VERIFY=1
DO_STAGE=1

for arg in "$@"; do
  case "$arg" in
    --verify-only) DO_BUILD=0; DO_STAGE=0 ;;
    --stage-only)  DO_VERIFY=0 ;;
    --no-verify)   DO_VERIFY=0 ;;
    -h|--help)     sed -n '1,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. obtain the pinned upstream source ------------------------------------
stage_sources() {
  log "Staging upstream sources at ${UPSTREAM_COMMIT:0:7}"
  mkdir -p "$SRC" "$CTX/patches"

  if [[ -d "$SRC/.git" ]]; then
    git -C "$SRC" fetch --quiet origin "$UPSTREAM_COMMIT" 2>/dev/null \
      || git -C "$SRC" fetch --quiet origin
    git -C "$SRC" checkout --quiet --detach "$UPSTREAM_COMMIT"
  else
    git clone --quiet "$UPSTREAM_URL" "$SRC"
    git -C "$SRC" checkout --quiet --detach "$UPSTREAM_COMMIT"
  fi

  local head; head="$(git -C "$SRC" rev-parse HEAD)"
  [[ "$head" == "$UPSTREAM_COMMIT" ]] \
    || die "source commit mismatch: got $head, want $UPSTREAM_COMMIT"

  local f want got
  for f in "${!PINNED[@]}"; do
    want="${PINNED[$f]}"
    git -C "$SRC" show "HEAD:src/$f" > "$CTX/patches/$f" \
      || die "missing src/$f at $UPSTREAM_COMMIT"
    got="$(sha256sum "$CTX/patches/$f" | awk '{print $1}')"
    [[ "$got" == "$want" ]] \
      || die "sha256 mismatch for $f: got $got, want $want"
    printf '  %-28s %s\n' "$f" "${got:0:16}…"
  done

  # The one patch that does NOT exist upstream.
  cp "$HERE/patches/patch_speculator_cudagraph.py" "$CTX/patches/"
  printf '  %-28s %s\n' "patch_speculator_cudagraph.py" "(local, derived from patch-flash-speculator.sh)"
}

# --- 2. build ----------------------------------------------------------------
build() {
  log "Preflight: base image"
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 \
    || { echo "  base image not local; pulling"; docker pull "$BASE_IMAGE"; }

  log "Building $IMAGE"
  docker build -f "$HERE/Dockerfile" -t "$IMAGE" "$CTX"
}

# --- 3. verify ----------------------------------------------------------------
verify() {
  log "Verifying $IMAGE"
  local sp=/usr/local/lib/python3.12/dist-packages
  local fail=0

  check() {
    local label="$1" ok="$2"
    if [[ "$ok" == "PASS" ]]; then printf '  \033[32mPASS\033[0m  %s\n' "$label"
    else printf '  \033[31mFAIL\033[0m  %s\n' "$label"; fail=1; fi
  }

  local v
  v=$(docker run --rm --entrypoint python3 "$IMAGE" -c 'import vllm;print(vllm.__version__)' 2>/dev/null || true)
  check "vllm version == $EXPECTED_VLLM (got: ${v:-<none>})" \
        "$([[ "$v" == "$EXPECTED_VLLM" ]] && echo PASS || echo FAIL)"

  v=$(docker run --rm --entrypoint python3 "$IMAGE" -c 'import importlib.metadata as m;print(m.version("instanttensor"))' 2>/dev/null || true)
  check "instanttensor == $INSTANTTENSOR_VERSION (got: ${v:-<none>})" \
        "$([[ "$v" == "$INSTANTTENSOR_VERSION" ]] && echo PASS || echo FAIL)"

  local markers=(
    "spark-fla-shmem:$sp/vllm/third_party/flash_linear_attention/ops/utils.py"
    "spark-fla-warps:$sp/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py"
    "ple-mmap-hook:$sp/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
    "fp8-hybrid-hook:$sp/vllm/model_executor/layers/quantization/auto_gptq.py"
    "lm-head-quant:$sp/vllm/models/qwen3_8_flash_next/nvidia/model.py"
    "lm-head-quant-mtp:$sp/vllm/models/qwen3_8_flash_next/nvidia/mtp.py"
    "speculator-prefill-fix:$sp/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    "guarded-mamba:$sp/vllm/v1/worker/mamba_utils.py"
  )
  local entry label file
  for entry in "${markers[@]}"; do
    label="${entry%%:*}"; file="${entry#*:}"
    case "$label" in
      speculator-prefill-fix) pat="prefill_cudagraph_mode" ;;
      guarded-mamba)          pat="mamba state-copy guard" ;;
      lm-head-quant)          pat="quant_config=vllm_config.quant_config" ;;
      lm-head-quant-mtp)      pat="quant_config=vllm_config.quant_config" ;;
      ple-mmap-hook)          pat="VLLM_PLE_MMAP" ;;
      fp8-hybrid-hook)        pat="VLLM_FP8_HYBRID" ;;
      *)                      pat="$label" ;;
    esac
    ok=$(docker run --rm --entrypoint grep "$IMAGE" -c "$pat" "$file" >/dev/null 2>&1 && echo PASS || echo FAIL)
    check "$label" "$ok"
  done

  # Must NOT be present.
  ok=$(docker run --rm --entrypoint bash "$IMAGE" -c \
        "ls $sp/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py.orig >/dev/null 2>&1 && exit 1 || exit 0" \
        >/dev/null 2>&1 && echo PASS || echo FAIL)
  check "speculator.py not vendored (no .orig)" "$ok"

  ok=$(docker run --rm --entrypoint bash "$IMAGE" -c \
        "ls /vllm-workspace/fixed_chat_template.jinja >/dev/null 2>&1 && exit 1 || exit 0" \
        >/dev/null 2>&1 && echo PASS || echo FAIL)
  check "chat template NOT baked in" "$ok"

  # B12X: shipped by the pinned upstream vLLM base image. This build neither
  # installs, modifies, nor enables it, so the file set AND every per-file hash
  # must be identical to the base. "absent" is the wrong assertion — the base
  # ships 10 b12x modules; what must hold is that this build changed none of them.
  local bx_base bx_new bx_n bx_match
  bx_base="$(docker run --rm --entrypoint bash "$BASE_IMAGE" -c \
      "find $sp -type f -iname '*b12x*' | sort | xargs -r -d \"\\n\" sha256sum" 2>/dev/null || true)"
  bx_new="$(docker run --rm --entrypoint bash "$IMAGE" -c \
      "find $sp -type f -iname '*b12x*' | sort | xargs -r -d \"\\n\" sha256sum" 2>/dev/null || true)"
  bx_n="$(printf '%s\n' "$bx_base" | grep -c . || true)"
  [[ "$bx_base" == "$bx_new" ]] && bx_match=yes || bx_match=no
  if [[ -n "$bx_base" && "$bx_match" == "yes" ]]; then
    check "B12X unchanged from pinned base ($bx_n files, all hashes match)" "PASS"
  else
    check "B12X unchanged from pinned base (base $bx_n files, match=$bx_match)" "FAIL"
  fi

  # Leftover build-time scripts must be gone.
  ok=$(docker run --rm --entrypoint bash "$IMAGE" -c \
        "ls /tmp/patch_*.py /tmp/patch_speculator_cudagraph.py >/dev/null 2>&1 && exit 1 || exit 0" \
        >/dev/null 2>&1 && echo PASS || echo FAIL)
  check "build-time patch scripts removed from /tmp" "$ok"

  echo
  [[ "$fail" == 0 ]] || die "verification failed — do NOT treat $IMAGE as benchmark-equivalent"
  echo "  All checks passed. $IMAGE is content-equivalent to the historical chain"
  echo "  EXCEPT for layer structure and build timestamps; revalidate before use."
}

# --- run ---------------------------------------------------------------------
[[ "$DO_STAGE" == 1 ]] && stage_sources
[[ "$DO_BUILD"  == 1 ]] && build
[[ "$DO_VERIFY" == 1 ]] && verify
echo
echo "Done: $IMAGE"
