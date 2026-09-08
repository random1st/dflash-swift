//
//  CandidateSelector.swift
//  DFlashKit
//
//  Port of the DFlash 2 candidate selector. See NOTICE.
//

import Foundation
import MLX
import MLXNN

/// Picks one coherent token path through the block's mask slots.
///
/// Block diffusion predicts every slot in parallel, which makes the slots
/// marginally plausible but not jointly consistent - the reason plain
/// block drafting accepts fewer tokens than it proposes. The selector keeps the
/// target head's top-K candidates per slot, scores the K x K transitions between
/// adjacent slots, and walks the lattice:
///
///     score[slot, pred, cand] = unary[slot, cand]
///                             + <A[pred] * proj(hidden[slot]), B[cand]>
///
/// The predecessors of slot 0 are all the verified anchor token, so the walk is
/// anchored in something the target actually emitted. Scores are computed in
/// fp32 (bf16 bilinear, fp32 unary) to match the reference.
final class CandidateSelector: Module {
    let topK: Int

    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: MLXArray
    @ModuleInfo(key: "successor_codebook") var successorCodebook: MLXArray
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    init(hiddenSize: Int, vocabularySize: Int, rank: Int, topK: Int) {
        self.topK = topK
        _predecessorCodebook.wrappedValue = MLXArray.zeros([vocabularySize, rank])
        _successorCodebook.wrappedValue = MLXArray.zeros([vocabularySize, rank])
        _hiddenProjection.wrappedValue = Linear(hiddenSize, rank, bias: false)
    }

    /// - Parameters:
    ///   - candidateIds: `[slots, K]`
    ///   - unaryLogits: `[slots, K]`, already transformed to the scale the
    ///     bilinear term was trained against
    ///   - hidden: `[slots, H]`, post-final-norm - the same rows the LM head reads
    ///   - anchorId: the verified token the block continues from
    /// - Returns: `[slots, K predecessors, K candidates]` in fp32
    func lattice(
        candidateIds: MLXArray, unaryLogits: MLXArray, hidden: MLXArray, anchorId: Int32
    ) -> MLXArray {
        let slots = candidateIds.dim(0)
        let k = candidateIds.dim(1)
        let projected = hiddenProjection(hidden)

        // Slot 0's predecessor row is the anchor repeated; slot s inherits
        // slot s-1's candidates.
        let anchorRow = MLXArray.full([1, k], values: MLXArray(anchorId))
            .asType(candidateIds.dtype)
        let predecessorIds = concatenated(
            [anchorRow, candidateIds[..<(slots - 1)]], axis: 0)

        let predecessor =
            predecessorCodebook[predecessorIds] * projected.expandedDimensions(axis: 1)
        let successor = successorCodebook[candidateIds]
        let bilinear = matmul(predecessor, successor.transposed(0, 2, 1))

        return unaryLogits.expandedDimensions(axis: 1).asType(.float32)
            + bilinear.asType(.float32)
    }

    /// Follows the best successor from the anchor. Stays in the graph - no
    /// host synchronisation per slot, which is what keeps drafting cheap.
    func walkGreedy(scores: MLXArray, candidateIds: MLXArray) -> MLXArray {
        let slots = scores.dim(0)
        var index = argMax(scores[0, 0])
        var picks = [candidateIds[0, index]]
        for slot in 1 ..< slots {
            index = argMax(scores[slot, index])
            picks.append(candidateIds[slot, index])
        }
        return stacked(picks)
    }
}
