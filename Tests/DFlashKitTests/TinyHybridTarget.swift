//
//  TinyHybridTarget.swift
//  DFlashKitTests
//

import Foundation
import MLXLLM

/// A hybrid Qwen3.5 small enough to run on synthetic tokens: 4 layers with
/// `full_attention_interval = 2`, so layers 0 and 2 are gated-DeltaNet and layers 1 and 3
/// are full attention - the same mix as the real target, at a size where a forward is
/// microseconds.
///
/// `linear_key_head_dim` is 32 so the fused gated-delta kernel is exercised; it truncates
/// key dimensions that are not a multiple of 32 and would otherwise route these tests to
/// the ops fallback.
enum TinyHybridTarget {
    static let vocabularySize = 64

    static let configurationJSON = """
        {
          "model_type": "qwen3_5",
          "hidden_size": 64,
          "num_hidden_layers": 4,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "linear_num_value_heads": 4,
          "linear_num_key_heads": 2,
          "linear_key_head_dim": 32,
          "linear_value_head_dim": 32,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": 64,
          "tie_word_embeddings": false,
          "full_attention_interval": 2,
          "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25
          }
        }
        """

    static func make() throws -> Qwen35TextModel {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(configurationJSON.utf8))
        return Qwen35TextModel(configuration)
    }
}
