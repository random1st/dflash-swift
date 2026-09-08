//
//  DFlashDecoderLayer.swift
//  DFlashKit
//
//  Port of the DFlash drafter decoder layer. See NOTICE.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Feed-forward block of the drafter backbone - Qwen3-style SiLU gating.
final class DFlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

/// One drafter layer: pre-norm attention and MLP, each optionally wrapped by a
/// DFlash 2 dynamic convolution.
///
/// The convolution wraps a sublayer rather than sitting between sublayers: one
/// projection of the sublayer's input produces both the coefficients that
/// convolve that input and the ones that convolve the sublayer's output.
final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlashAttention
    @ModuleInfo var mlp: DFlashMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlashGroupedConv?
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlashGroupedConv?

    init(_ config: DFlashConfiguration, layerIndex: Int) {
        _attention.wrappedValue = DFlashAttention(config, layerIndex: layerIndex)
        _mlp.wrappedValue = DFlashMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.hasGroupedConvolution {
            _attentionConv.wrappedValue = DFlashGroupedConv(
                hiddenSize: config.hiddenSize,
                taps: config.convKernelSize,
                groupSize: config.convGroupSize)
            _mlpConv.wrappedValue = DFlashGroupedConv(
                hiddenSize: config.hiddenSize,
                taps: config.convKernelSize,
                groupSize: config.convGroupSize)
        } else {
            _attentionConv.wrappedValue = nil
            _mlpConv.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray, cache: BaseKVCache) -> MLXArray {
        var normed = inputLayerNorm(x)
        var attentionCoefficients: MLXArray? = nil
        if let attentionConv {
            let (convolved, coefficients) = attentionConv.prepare(normed)
            normed = convolved
            attentionCoefficients = coefficients
        }
        var attended = attention(normed, context: context, cache: cache)
        if let attentionConv, let coefficients = attentionCoefficients {
            attended = attentionConv.finish(attended, delta: coefficients)
        }
        let residual = x + attended

        var projected = postAttentionLayerNorm(residual)
        var mlpCoefficients: MLXArray? = nil
        if let mlpConv {
            let (convolved, coefficients) = mlpConv.prepare(projected)
            projected = convolved
            mlpCoefficients = coefficients
        }
        var expanded = mlp(projected)
        if let mlpConv, let coefficients = mlpCoefficients {
            expanded = mlpConv.finish(expanded, delta: coefficients)
        }
        return residual + expanded
    }
}
