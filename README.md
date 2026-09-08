# DFlashKit

A Swift port of the **DFlash 2** block-diffusion speculative-decoding drafter for
MLX on Apple Silicon. No Python anywhere in the loop.

Apple's `mlx-swift-lm` ships MTP speculation. This adds the faster drafter next to it.

## Why

Decoding a large model is bound by reading its weights, not by arithmetic: one token
costs a full sweep. Verifying a block of drafted tokens reads those same weights **once**,
so throughput follows how many drafted tokens survive verification.

An MTP head drafts one token at a time and lands 2.3–2.6 accepted tokens per round on
Qwen3.8-27B. DFlash 2 proposes an eight-token block in a single forward pass and lands
4.1–5.7, depending on the prompt. Output is unchanged: the target model confirms every
token it emits.

That only pays if verifying eight rows costs about what verifying one costs. On stock MLX
kernels it does not — a quantised matmul re-reads the weights per row until the GEMM
tiling takes over around M=13:

| rows | stock | with the small-M kernel |
|---:|---:|---:|
| 1 | 1.00x | 1.00x |
| 2 | 1.05x | 1.06x |
| 4 | 1.63x | 1.64x |
| 8 | **3.07x** | **1.51x** |

`SmallMQuantizedMatmul` covers M ≤ 8 with one 8×8 `simdgroup_matrix` tile, so every
quantised group is dequantised once and reused by all rows. Flat from six rows up — which
moves the block cap back out to the full eight and is worth **1.45x** end to end against
the narrow block it replaced, and **2.06x** against the same full block on stock kernels.

## How it works

Each round feeds the drafter `[anchor] + 7 mask slots`. It reads the target's hidden
states from a ladder of layers — 5, 19, 33, 47 and 61 for Qwen3.8-27B — projects them
into its own width, and predicts every slot in parallel. A low-rank candidate selector
then walks one coherent path through the per-slot top-K, because block diffusion produces
slots that are individually plausible but not jointly consistent.

The target verifies the whole block in one pass, and the accepted prefix is committed.
The rollback is what makes this pay on a hybrid model: attention caches trim exactly,
but the gated-delta layers carry recurrent state that cannot be trimmed, and re-running
the model over the accepted prefix would spend a second full sweep — the entire saving.
Instead the round's recurrence inputs are captured, and since the recurrence is causal,
replaying it over the accepted prefix from the pre-round state reproduces a committed
forward. Measured cost: 0.08 s of a 12.6 s generation.

## Prefix cache

Prefill is one weight sweep per position, so re-sending a conversation costs more than
answering it. `PrefixCache` keeps the state of recent prompts — but the target is hybrid,
and gated-delta state accumulates over every token with no way to unwind, so entries are
snapshots reusable at exactly their own length rather than something rewindable.

That makes placement the whole design. One snapshot sits four tokens below the end of the
prompt (a chat template renders its trailing generation prompt differently once the turn
closes, so a snapshot at the exact end is never a prefix of the next turn), and one at the
end of the reply. The second turn of a conversation then reuses 1024 of its 1042 prompt
tokens and prefills **28x** faster than cold, producing output identical to a cold
control.

```swift
let generator = DFlashSpeculativeGenerator(
    target: target, drafter: drafter, prefixCache: PrefixCache(slots: 4))
```

## Use

```swift
let drafter = try DFlashDraftModel.load(directory: drafterDirectory)
let generator = DFlashSpeculativeGenerator(target: qwen35TextModel, drafter: drafter)

for await event in generator.stream(prompt: tokens, maximumTokens: 512, stopTokens: eos) {
    switch event {
    case .token(let id): …
    case .finished(let statistics): print(statistics.meanAcceptedPerRound)
    case .failed(let reason): …
    }
}
```

Greedy only for now — speculative sampling at `temperature > 0` is not implemented.

`dflash-bench` runs it end to end against a real checkpoint:

```sh
swift build -c release
.build/release/dflash-bench <model-dir> <drafter-dir> 200
```

## Requirements

- macOS 14+, Apple Silicon
- A Qwen3.5-family target (`model_type: qwen3_5`, dense or MoE)
- A DFlash 2 checkpoint trained against it, e.g. `incoai/Qwen3.8-27B-DFlash2`

The package depends on a fork of `mlx-swift-lm` carrying two patches the drafter needs:
hidden states from a chosen ladder of target layers, and capture/rollback of a round's
gated-delta recurrence. Both are written to be upstreamable — a multi-layer tap is what
EAGLE-3-class drafters need too — and when they land this becomes an ordinary dependency
on `ml-explore/mlx-swift-lm`.

## Tests

```sh
xcodebuild test -scheme dflash-swift-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation -skipMacroValidation
```

`swift test` cannot work here: `mlx-swift` does not find its metallib from a SwiftPM
test run.

The suite checks the convolution and the selector against tensors the Python reference
produced on the same deterministic inputs, loads the real checkpoint, proves the rollback
reproduces a committed forward, proves a restored prefix equals a cold prefill, and checks
the small-M kernel against `quantizedMatmul`.

Every one of those carries a negative control, because none of these components fails
loudly. A broken drafter does not crash — it produces fluent text the target quietly
rejects, which reads as "speculation does not help on this model" rather than as a bug. A
broken prefix cache answers a slightly different prompt, fluently. A kernel that silently
fell back to the stock op would pass every closeness test for the wrong reason. So the
suite also asserts that a rollback to the wrong position diverges, that a snapshot filed
one token off diverges, and that the kernel's output is *not* bit-identical to the op it
replaces.

## Licensing

Derivative of MIT and Apache-2.0 sources — see [NOTICE](NOTICE).
