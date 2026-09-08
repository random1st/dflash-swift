//
//  Qwen35Bridge.swift
//  DFlashKit
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Connects a Qwen3.5-family target to the DFlash drafter.
///
/// The drafter owns no embedding table and no output head. It borrows the
/// target's, and it consumes the target's hidden states from the layer ladder
/// named in the drafter's own config - for Qwen3.8-27B, layers 5, 19, 33, 47
/// and 61. Those are requested through `mtpLayerTapIndicesKey` and come back on
/// `LMOutput.state`.
public final class Qwen35Bridge: DFlashTargetBridge {
    private let target: Qwen35TextModel
    private let tapIndices: [Int]

    public init(target: Qwen35TextModel, tapIndices: [Int]) {
        self.target = target
        self.tapIndices = tapIndices
    }

    public func embed(_ tokens: MLXArray) -> MLXArray {
        target.model.embedTokens(tokens)
    }

    public func logits(_ hidden: MLXArray) -> MLXArray {
        // A tied checkpoint has no separate head and scores through the
        // embedding table instead.
        if let head = target.lmHead {
            return head(hidden)
        }
        return target.model.embedTokens.asLinear(hidden)
    }

    /// State that makes the target emit the tapped layers on the next call.
    public func requestState(from existing: LMOutput.State? = nil) -> LMOutput.State {
        var state = existing ?? LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpLayerTapIndicesKey] = tapIndices
        return state
    }

    /// Concatenates the tapped layers into the width the drafter's `fc` expects.
    ///
    /// Order follows `tapIndices`, which is the order the checkpoint was trained
    /// with; concatenating them in any other order silently mismatches `fc` and
    /// produces drafts the target rejects rather than an error.
    public func fuse(_ state: LMOutput.State) -> MLXArray? {
        guard let captured = state[mtpLayerHiddenStatesKey], captured.count == tapIndices.count
        else { return nil }
        return concatenated(captured, axis: -1)
    }
}
