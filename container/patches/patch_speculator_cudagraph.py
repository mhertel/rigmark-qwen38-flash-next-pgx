#!/usr/bin/env python3
"""
Disable CUDA-graph capture for MTP/speculator PREFILL only.

The PLE mmap lookup performs a GPU->CPU transfer and is not safe during
MTP/speculator PIECEWISE prefill CUDA graph capture on DGX Spark. This keeps
PIECEWISE CUDA graphs enabled for the target model but turns them off for
speculator prefill.

Applied in place inside the image, so speculator.py is never vendored and
stays base-derived. The exact-match block below was found exactly once in the
base image's speculator.py, which is what the historical
qwen38-flash-dgx-specfix layer shipped.

Derived from /home/mike/patch-flash-speculator.sh.
"""

import os
import sys
from pathlib import Path

DEFAULT_TARGET = (
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/"
    "spec_decode/autoregressive/speculator.py"
)

OLD = '''    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )

        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft decodes (draft positions > 0).
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )
'''

NEW = '''    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # PLE mmap lookup performs a GPU->CPU transfer and is not safe during
        # MTP/speculator PIECEWISE prefill CUDA graph capture on DGX Spark.
        #
        # Keep PIECEWISE CUDA graphs enabled for the target model, but disable
        # CUDA graph capture for speculator prefill only.
        prefill_cudagraph_mode = cudagraph_mode
        if cudagraph_mode == CUDAGraphMode.PIECEWISE:
            prefill_cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            prefill_cudagraph_mode,
            self.num_speculative_steps + 1,
        )

        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft decodes (draft positions > 0).
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )
'''


def main() -> int:
    path = Path(os.environ.get("SPECULATOR_PY", DEFAULT_TARGET))
    if not path.is_file():
        print(f"ERROR: speculator.py not found at {path}", file=sys.stderr)
        return 1

    text = path.read_text()
    count = text.count(OLD)

    if count != 1:
        print(
            "ERROR: expected exactly one init_cudagraph_manager block; "
            f"found {count}. Image NOT modified.",
            file=sys.stderr,
        )
        return 1

    path.write_text(text.replace(OLD, NEW, 1))
    print("speculator.py PIECEWISE prefill graph fix applied OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
