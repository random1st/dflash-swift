//
//  DFlashConfiguration.swift
//  DFlashKit
//
//  Port of the DFlash / DFlash 2 drafter configuration. See NOTICE for the
//  MIT (z-lab/dflash, ARahim3/mlx-dspark) and Apache-2.0 (sgl-project/sglang)
//  sources this is derived from.
//

import Foundation
import MLXLMCommon

/// Layout of a published DFlash checkpoint's `config.json`.
///
/// The DFlash 2 fields default to the values that reproduce DFlash 1 exactly,
/// so a v1 checkpoint loads through the same type: `selectorRank == 0` disables
/// the candidate selector and `convKernelSize == 0` disables the grouped
/// convolution, which is how the reference gates them.
public struct DFlashConfiguration: Decodable, Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int

    /// Tokens proposed per forward pass. The backbone always drafts the full
    /// block width because it was trained at that width; a runtime cap only
    /// bounds how many of them get verified.
    public var blockSize: Int

    /// Zero-based target layers whose hidden states are concatenated and fed
    /// through `fc`. For Qwen3.8-27B DFlash 2 this is [5, 19, 33, 47, 61].
    public var targetLayerIds: [Int]

    /// Token id used to fill the block's mask slots.
    public var maskTokenId: Int

    public var ropeScaling: [String: StringOrNumber]? { ropeScalingResolved }
    private var ropeScalingResolved: [String: StringOrNumber]?
    public var layerTypes: [String]
    public var slidingWindow: Int?
    public var finalLogitSoftcapping: Float?

    // MARK: DFlash 2

    /// Rank of the selector's low-rank bilinear transition term. 0 = DFlash 1.
    public var selectorRank: Int
    /// Candidates kept per mask slot before the selector walk.
    public var selectorTopK: Int
    /// Taps of the dynamic depthwise convolution. 0 = DFlash 1; every published
    /// DFlash 2 head uses 2.
    public var convKernelSize: Int
    /// Channels sharing one content-adaptive coefficient correction.
    public var convGroupSize: Int
    /// Scales the selector's unary logits. The bilinear term was trained against
    /// the transformed logit scale, so this must be applied before scoring.
    public var outputMultiplier: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case dflashConfig = "dflash_config"
    }

    /// Published checkpoints nest RoPE under `rope_parameters` rather than
    /// exposing `rope_theta` at the top level; reading the flat key would
    /// silently substitute the 10000 default for the real 1e7.
    private enum RopeKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    /// The block-diffusion parameters live in a nested `dflash_config` object.
    private struct Nested: Decodable {
        var blockSize: Int?
        var targetLayerIds: [Int]?
        var maskTokenId: Int?
        var selectorRank: Int?
        var selectorTopK: Int?
        var convKernelSize: Int?
        var convGroupSize: Int?
        var outputMultiplier: Float?

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case targetLayerIds = "target_layer_ids"
            case maskTokenId = "mask_token_id"
            case selectorRank = "selector_rank"
            case selectorTopK = "selector_top_k"
            case convKernelSize = "conv_kernel_size"
            case convGroupSize = "conv_group_size"
            case outputMultiplier = "output_multiplier"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 4
        self.attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11008
        self.vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        let ropeContainer = try decoder.container(keyedBy: RopeKeys.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        if var ropeParameters {
            if ropeParameters["type"] == nil, let type = ropeParameters["rope_type"] {
                ropeParameters["type"] = type
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 10000.0
            self.ropeScalingResolved = ropeParameters
        } else {
            self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
            self.ropeScalingResolved = nil
        }
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        self.slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        self.headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / attentionHeads)

        let nested = try c.decodeIfPresent(Nested.self, forKey: .dflashConfig)
        self.blockSize = nested?.blockSize ?? 8
        self.targetLayerIds = nested?.targetLayerIds ?? []
        self.maskTokenId = nested?.maskTokenId ?? 0
        self.selectorRank = nested?.selectorRank ?? 0
        self.selectorTopK = nested?.selectorTopK ?? 0
        self.convKernelSize = nested?.convKernelSize ?? 0
        self.convGroupSize = nested?.convGroupSize ?? 16
        self.outputMultiplier = nested?.outputMultiplier ?? 1.0

        // An absent layer_types means every layer is full attention, matching
        // the reference default.
        let declared = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        self.layerTypes = declared ?? Array(repeating: "full_attention", count: hiddenLayers)
    }

    /// Number of target layers fused into one drafter context row.
    public var fusedTargetWidth: Int { targetLayerIds.count * hiddenSize }

    /// True when this checkpoint carries the DFlash 2 candidate selector.
    public var hasSelector: Bool { selectorRank > 0 }

    /// True when this checkpoint carries the DFlash 2 grouped convolution.
    public var hasGroupedConvolution: Bool { convKernelSize > 0 }
}
