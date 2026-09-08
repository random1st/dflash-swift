//
//  DFlashAttention.swift
//  DFlashKit
//
//  Port of the DFlash drafter attention. See NOTICE.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// Attention over the drafter's own context rows plus the current draft block.
///
/// The context rows are projections of the target's fused hidden states, injected
/// into this layer's K/V cache. They are appended first, then the block's rows
/// attend over context and block together. Splitting the context half out into
/// `appendContext` lets a prefix-cache restore rebuild the drafter's context
/// without running a draft pass.
final class DFlashAttention: Module {
    let heads: Int
    let kvHeads: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int?

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ config: DFlashConfiguration, layerIndex: Int) {
        let dim = config.hiddenSize
        let headDim = config.headDim
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.scale = pow(Float(headDim), -0.5)
        self.isSliding = config.layerTypes.indices.contains(layerIndex)
            && config.layerTypes[layerIndex] == "sliding_attention"
        self.slidingWindow = isSliding ? config.slidingWindow : nil

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self.rope = initializeRope(
            dims: headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    /// Projects context rows, appends them to `cache`, and returns the cache's
    /// full keys and values.
    ///
    /// A sliding layer keeps only the last `window - 1` rows and advances the
    /// cache offset past the ones it drops, so RoPE positions stay absolute.
    @discardableResult
    func appendContext(_ contextRows: MLXArray, cache: BaseKVCache) -> (MLXArray, MLXArray) {
        var x = contextRows
        var (B, S) = (x.dim(0), x.dim(1))
        if isSliding, let window = slidingWindow {
            let keep = window - 1
            if keep < S {
                let skip = S - keep
                x = x[0..., skip...]
                S = x.dim(1)
                cache.offset += skip
            }
        }
        var keys = wk(x)
        let values = wv(x).reshaped(B, S, kvHeads, -1).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, S, kvHeads, -1)).transposed(0, 2, 1, 3)
        keys = applyRotaryPosition(rope, to: keys, offset: cache.ropeOffset)
        return cache.update(keys: keys, values: values)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray, cache: BaseKVCache) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var (keys, values) = appendContext(context, cache: cache)

        // The cache offset now sits past the appended context, so the block rows
        // rope directly after it.
        var queries = qNorm(wq(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
        var blockKeys = kNorm(wk(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let blockValues = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, offset: cache.ropeOffset)
        blockKeys = applyRotaryPosition(rope, to: blockKeys, offset: cache.ropeOffset)

        let contextLength = keys.dim(2)
        keys = concatenated([keys, blockKeys], axis: 2)
        values = concatenated([values, blockValues], axis: 2)

        var mask = MLXFast.ScaledDotProductAttentionMaskMode.none
        if isSliding, let window = slidingWindow {
            mask =
                contextLength + L <= window
                ? .causal
                : .array(
                    createCausalMask(n: L, offset: contextLength, windowSize: window))
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}
