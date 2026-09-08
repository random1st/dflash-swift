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
    public static func load(directory: URL) throws -> DFlashDraftModel {
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
        eval(model)
        return model
    }
}
