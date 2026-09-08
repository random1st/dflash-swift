//
//  LoaderTests.swift
//  DFlashKitTests
//
//  Loads the real checkpoint if it is present on this machine. Skipped
//  elsewhere rather than failed - a missing 3.7 GB download is not a defect.
//

import Foundation
import MLX
import XCTest

@testable import DFlashKit

final class LoaderTests: XCTestCase {
    private static let checkpoint = URL(
        fileURLWithPath: NSHomeDirectory()
    ).appending(path: "Library/Application Support/Feynt/models/Qwen3.8-27B-DFlash2")

    func testLoadsRealCheckpoint() throws {
        guard FileManager.default.fileExists(atPath: Self.checkpoint.path) else {
            throw XCTSkip("checkpoint not present at \(Self.checkpoint.path)")
        }

        let model = try DFlashDraftModel.load(directory: Self.checkpoint)
        let config = model.configuration

        // Values read from the published config, not guessed.
        XCTAssertEqual(config.blockSize, 8)
        XCTAssertEqual(config.hiddenLayers, 5)
        XCTAssertEqual(config.hiddenSize, 5120)
        XCTAssertEqual(config.targetLayerIds, [5, 19, 33, 47, 61])
        XCTAssertEqual(config.selectorRank, 256)
        XCTAssertEqual(config.selectorTopK, 16)
        XCTAssertEqual(config.convKernelSize, 2)
        XCTAssertEqual(config.vocabularySize, 248_320)
        // The flat rope_theta default is 10000; the real value lives nested and
        // is three orders larger, so this asserts the decoder read the right one.
        XCTAssertEqual(config.ropeTheta, 10_000_000, accuracy: 1)
        XCTAssertEqual(config.slidingWindow, 2048)
        XCTAssertEqual(config.layerTypes.count, 5)
        XCTAssertTrue(config.layerTypes.allSatisfy { $0 == "sliding_attention" })
        XCTAssertEqual(config.fusedTargetWidth, 25_600)

        // update(verify: .all) already rejects a name mismatch; this checks the
        // weights actually carry checkpoint values rather than initialisation.
        let projection = model.fc.weight
        XCTAssertEqual(projection.shape, [5120, 25600])
        XCTAssertGreaterThan(abs(projection).max().item(Float.self), 0)

        XCTAssertEqual(model.layers.count, 5)
        XCTAssertNotNil(model.selector)
        XCTAssertEqual(model.makeCache().count, 5)
    }
}
