//
//  DFlashDraftModel.swift
//  DFlashKit
//
//  Port of the DFlash / DFlash 2 drafter. See NOTICE.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// What the target model must lend the drafter.
///
/// DFlash owns no embedding table and no output head - it reuses the target's,
/// which is what keeps a 27B-class drafter down to a few gigabytes. The target
/// also supplies the fused hidden states the drafter attends over.
public protocol DFlashTargetBridge {
    /// Token embeddings, `[B, L] -> [B, L, H]`.
    func embed(_ tokens: MLXArray) -> MLXArray
    /// Output head over the drafter's hidden states, `[L, H] -> [L, V]`.
    func logits(_ hidden: MLXArray) -> MLXArray
}

/// The DFlash 2 block-diffusion drafter.
///
/// One forward pass proposes a whole block of tokens. The block is fed as
/// `[anchor] + (blockSize - 1) mask tokens`; logits are read at the mask
/// positions, which is why `forwardHidden` drops the anchor row via
/// `logitsStart`. With a selector present, the per-slot top-K candidates are
/// then walked into one coherent path instead of taken independently.
public final class DFlashDraftModel: Module {
    public let configuration: DFlashConfiguration

    /// Projects the concatenated target layers down to the drafter's width.
    @ModuleInfo var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo var layers: [DFlashDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var selector: CandidateSelector?

    /// Set through `bind(_:)` before any drafting.
    private var target: DFlashTargetBridge?

    public init(_ configuration: DFlashConfiguration) {
        self.configuration = configuration
        _fc.wrappedValue = Linear(
            configuration.fusedTargetWidth, configuration.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map {
            DFlashDecoderLayer(configuration, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _selector.wrappedValue =
            configuration.hasSelector
            ? CandidateSelector(
                hiddenSize: configuration.hiddenSize,
                vocabularySize: configuration.vocabularySize,
                rank: configuration.selectorRank,
                topK: configuration.selectorTopK)
            : nil
    }

    /// Lends the drafter the target's embedding table and output head.
    public func bind(_ target: DFlashTargetBridge) {
        self.target = target
    }

    /// One cache per layer. Sliding layers get a rotating cache sized to the
    /// window, matching how the checkpoint was trained.
    public func makeCache() -> [BaseKVCache] {
        configuration.layerTypes.map { type in
            if type == "sliding_attention", let window = configuration.slidingWindow {
                return RotatingKVCache(maxSize: window - 1, keep: 0)
            }
            return KVCacheSimple()
        }
    }

    /// Fused target hidden `[B, S, layers * H]` -> drafter context rows `[B, S, H]`.
    ///
    /// Per-position, so computing it over any slice matches the full-width
    /// result, which is what lets a prefix-cache restore rebuild context lazily.
    public func projectContext(_ fusedTargetHidden: MLXArray) -> MLXArray {
        hiddenNorm(fc(fusedTargetHidden))
    }

    /// Appends already-projected context rows to every layer's cache without
    /// drafting - the prefix-cache restore path.
    public func appendContext(_ contextRows: MLXArray, cache: [BaseKVCache]) {
        for (layer, layerCache) in zip(layers, cache) {
            layer.attention.appendContext(contextRows, cache: layerCache)
        }
    }

    /// Backbone forward to the post-final-norm hidden states.
    ///
    /// Slicing before the per-token norm is identical to slicing after it, and
    /// doing it first means the selector never normalises rows nobody reads.
    public func forwardHidden(
        _ inputs: MLXArray, fusedTargetHidden: MLXArray, cache: [BaseKVCache],
        logitsStart: Int = 0
    ) -> MLXArray {
        guard let target else {
            fatalError("DFlashDraftModel.bind(_:) must be called before drafting")
        }
        var h = target.embed(inputs)
        let context = projectContext(fusedTargetHidden)
        for (layer, layerCache) in zip(layers, cache) {
            h = layer(h, context: context, cache: layerCache)
        }
        if logitsStart > 0 {
            h = h[0..., logitsStart...]
        }
        return norm(h)
    }

    /// The DFlash 2 unary-logit transform.
    ///
    /// The bilinear selector term was trained against this scale, so feeding it
    /// raw logits would misweight every transition.
    private func transformUnary(_ logits: MLXArray) -> MLXArray {
        var out = logits.asType(.float32)
        if configuration.outputMultiplier != 1.0 {
            out = out * configuration.outputMultiplier
        }
        if let cap = configuration.finalLogitSoftcapping {
            out = tanh(out / cap) * cap
        }
        return out
    }

    /// Backbone forward plus the output head, restricted to the block's mask slots.
    ///
    /// A padded output head would let a candidate id index past the real
    /// vocabulary and gather garbage codebook rows, hence the trim.
    private func blockLogits(
        _ block: MLXArray, fusedTargetHidden: MLXArray, cache: [BaseKVCache], cap: Int
    ) -> (hidden: MLXArray, logits: MLXArray) {
        guard let target else {
            fatalError("DFlashDraftModel.bind(_:) must be called before drafting")
        }
        let hidden = forwardHidden(
            block, fusedTargetHidden: fusedTargetHidden, cache: cache, logitsStart: 1)[0][..<cap]
        return (hidden, target.logits(hidden)[0..., ..<configuration.vocabularySize])
    }

    /// Scores every transition in the block without choosing a path through them.
    ///
    /// ``selectBlock(_:fusedTargetHidden:cache:cap:anchorId:)`` walks this greedily and
    /// keeps one token per slot. A caller that wants to keep more than one branch - or
    /// to measure how often the target's token sat in a branch the walk did not take -
    /// needs the scores themselves. Returns nil for a DFlash 1 checkpoint, which has no
    /// selector and so no transitions to score.
    ///
    /// Runs the same forward and the same candidate selection as `selectBlock`, so what
    /// it reports is what the walk actually saw.
    public func draftLattice(
        _ block: MLXArray, fusedTargetHidden: MLXArray, cache: [BaseKVCache],
        cap: Int, anchorId: Int32
    ) -> DFlashBlockLattice? {
        guard let selector else { return nil }
        let (hidden, logits) = blockLogits(
            block, fusedTargetHidden: fusedTargetHidden, cache: cache, cap: cap)

        let k = selector.topK
        let partitioned = argPartition(logits, kth: logits.dim(-1) - k, axis: -1)
        let candidateIds = partitioned[0..., (logits.dim(-1) - k)...].asType(.int32)
        let unary = transformUnary(takeAlong(logits, candidateIds, axis: -1))
        let scores = selector.lattice(
            candidateIds: candidateIds, unaryLogits: unary, hidden: hidden, anchorId: anchorId)
        return DFlashBlockLattice(
            candidateIds: candidateIds, scores: scores, unaryLogits: unary)
    }

    /// Drafts one block greedily.
    ///
    /// - Parameters:
    ///   - block: `[1, blockSize]` - the anchor followed by mask tokens
    ///   - fusedTargetHidden: the target's concatenated tapped layers
    ///   - cap: how many of the drafted tokens to return
    ///   - anchorId: the verified token this block continues from
    /// - Returns: `[cap]` drafted token ids, still in the graph
    public func selectBlock(
        _ block: MLXArray, fusedTargetHidden: MLXArray, cache: [BaseKVCache],
        cap: Int, anchorId: Int32
    ) -> MLXArray {
        guard let selector else {
            // DFlash 1: no selector, each slot takes its own argmax.
            let (_, logits) = blockLogits(
                block, fusedTargetHidden: fusedTargetHidden, cache: cache, cap: cap)
            return argMax(logits, axis: -1).asType(.int32)
        }

        guard
            let lattice = draftLattice(
                block, fusedTargetHidden: fusedTargetHidden, cache: cache, cap: cap,
                anchorId: anchorId)
        else {
            fatalError("a checkpoint with a selector always produces a lattice")
        }
        return selector.walkGreedy(scores: lattice.scores, candidateIds: lattice.candidateIds)
    }
}

/// One drafted block before a path is chosen through it.
///
/// The drafter predicts every slot in parallel, so what it really produces is a lattice:
/// K candidates per slot and a score for every transition between adjacent slots. Greedy
/// decoding collapses that to a single chain; the lattice is what a tree would branch
/// over.
public struct DFlashBlockLattice {
    /// `[slots, K]` candidate token ids per mask slot.
    public let candidateIds: MLXArray
    /// `[slots, K predecessors, K candidates]` transition scores, fp32. Slot 0's
    /// predecessor rows are all the anchor, so they are identical to one another.
    public let scores: MLXArray
    /// `[slots, K]` the drafter's own transformed logits at those candidates - the half
    /// of the score that does not depend on the predecessor. Kept separate because the
    /// balance between it and the bilinear transition term is a trained constant, and
    /// the only way to ask whether that balance is right is to reweigh the two halves.
    public let unaryLogits: MLXArray
}
