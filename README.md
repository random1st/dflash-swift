# DFlashKit

Swift port of the **DFlash 2** block-diffusion speculative-decoding drafter, for
MLX on Apple Silicon. No Python.

## Why

Decoding a large model is bound by reading its weights: one token costs a full
sweep over them. Verifying a whole block of drafted tokens reads those same
weights once, so throughput scales with how many drafted tokens survive
verification. A paired MTP head drafts autoregressively and lands around 2.3-2.6
accepted tokens per step on Qwen3.8-27B; DFlash 2 proposes an eight-token block
in a single forward pass and reaches 5.3-5.8 on the same model and machine.

Apple's `mlx-swift-lm` ships MTP speculation. This package adds the DFlash 2
drafter alongside it.

## Status

Early. The port is validated against the Python reference token-for-token.

## Requirements

- macOS 14+, Apple Silicon
- A DFlash 2 checkpoint matching the target model, e.g.
  `incoai/Qwen3.8-27B-DFlash2` for `Qwen3.8-27B`.

## Licensing

Derivative of MIT and Apache-2.0 sources - see [NOTICE](NOTICE).
