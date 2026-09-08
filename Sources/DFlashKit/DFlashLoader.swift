//
//  DFlashLoader.swift
//  DFlashKit
//

import Foundation
import MLX
import MLXNN

public enum DFlashLoaderError: LocalizedError {
    case missingConfiguration(URL)
    case missingWeights(URL)

    public var errorDescription: String? {
        switch self {
        case .missingConfiguration(let url):
            "No config.json in \(url.path)"
        case .missingWeights(let url):
            "No .safetensors weights in \(url.path)"
        }
    }
}

extension DFlashDraftModel {
    /// Loads a published DFlash checkpoint directory.
    ///
    /// Parameter names are verified against the module tree rather than merged
    /// leniently: a renamed tensor would otherwise leave that submodule at its
    /// initialised values, and a drafter with one silently random layer still
    /// produces fluent-looking tokens - the target rejects nearly all of them,
    /// which reads as "speculation does not help here" rather than as a bug.
    /// - Parameter quantizeBits: quantise the drafter's linear layers after loading.
    ///   Off by default: measured here it made drafting slower, not faster (2.18 s -> 2.40 s
    ///   over 44 rounds), because this drafter's matrices are small enough that the
    ///   quantised matmul's overhead outweighs the traffic it saves. Kept as an option
    ///   since that balance flips on larger drafters.
    public static func load(directory: URL, quantizeBits: Int? = nil) throws -> DFlashDraftModel {
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DFlashLoaderError.missingConfiguration(directory)
        }
        let configuration = try JSONDecoder().decode(
            DFlashConfiguration.self, from: Data(contentsOf: configURL))

        let shards = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !shards.isEmpty else { throw DFlashLoaderError.missingWeights(directory) }

        var weights: [String: MLXArray] = [:]
        for shard in shards {
            for (key, value) in try loadArrays(url: shard) {
                weights[key] = value
            }
        }

        let model = DFlashDraftModel(configuration)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        if let bits = quantizeBits {
            // The codebooks are gathered by token id, not matmul'd, so quantising them
            // would corrupt the selector's scores; only the linear layers are eligible.
            quantize(model: model, groupSize: 64, bits: bits) { path, module in
                module is Linear && !path.contains("codebook")
            }
        }
        eval(model)
        return model
    }
}
