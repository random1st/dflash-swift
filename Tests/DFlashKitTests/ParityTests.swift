//
//  ParityTests.swift
//  DFlashKitTests
//
//  The port is checked against tensors produced by the Python reference
//  implementation on the same deterministic inputs. Shapes and plausible
//  outputs prove nothing here: a swapped axis or a mis-ordered top-K compiles
//  cleanly and produces confident garbage, which shows up only as an acceptance
//  rate near zero after the whole pipeline is assembled.
//

import Foundation
import MLX
import MLXNN
import XCTest

@testable import DFlashKit

final class ParityTests: XCTestCase {
    private var golden: [String: MLXArray] = [:]

    override func setUpWithError() throws {
        guard
            let url = Bundle.module.url(
                forResource: "golden", withExtension: "safetensors", subdirectory: "Golden")
        else {
            throw XCTSkip("golden.safetensors is missing - regenerate it with make_golden.py")
        }
        golden = try loadArrays(url: url)
    }

    /// Largest absolute difference, as a Float, so a failure message carries the
    /// magnitude rather than just "not equal".
    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = abs(a.asType(.float32) - b.asType(.float32)).max()
        return d.item(Float.self)
    }

    func testGroupedConvolutionMatchesReference() throws {
        let x = golden["conv/x"]!
        let hiddenSize = x.dim(2)
        let taps = golden["conv/base_kernel"]!.dim(1)
        let groupCount = golden["conv/out_coeff"]!.dim(3)

        let conv = DFlashGroupedConv(
            hiddenSize: hiddenSize, taps: taps, groupSize: hiddenSize / groupCount)
        conv.update(parameters: ModuleParameters.unflattened([
            "base_kernel": golden["conv/base_kernel"]!,
            "kernel_projection.weight": golden["conv/kernel_projection.weight"]!,
        ]))

        let (prepared, coefficients) = conv.prepare(x)
        XCTAssertLessThan(
            maxDifference(prepared, golden["conv/prepared"]!), 2e-3,
            "convolved input diverges from the reference")
        XCTAssertLessThan(
            maxDifference(coefficients, golden["conv/out_coeff"]!), 2e-3,
            "output coefficients diverge from the reference")

        let finished = conv.finish(prepared, delta: coefficients)
        XCTAssertLessThan(
            maxDifference(finished, golden["conv/finished"]!), 2e-3,
            "convolved output diverges from the reference")
    }

    func testCandidateSelectorMatchesReference() throws {
        let hidden = golden["selector/hidden"]!
        let candidateIds = golden["selector/candidate_ids"]!.asType(.int32)
        let codebook = golden["selector/predecessor_codebook"]!

        let selector = CandidateSelector(
            hiddenSize: hidden.dim(1),
            vocabularySize: codebook.dim(0),
            rank: codebook.dim(1),
            topK: candidateIds.dim(1))
        selector.update(parameters: ModuleParameters.unflattened([
            "predecessor_codebook": codebook,
            "successor_codebook": golden["selector/successor_codebook"]!,
            "hidden_projection.weight": golden["selector/hidden_projection.weight"]!,
        ]))

        let scores = selector.lattice(
            candidateIds: candidateIds,
            unaryLogits: golden["selector/unary"]!,
            hidden: hidden,
            anchorId: 11)
        XCTAssertLessThan(
            maxDifference(scores, golden["selector/scores"]!), 2e-3,
            "transition lattice diverges from the reference")

        // The walk is discrete: any difference at all is a different draft.
        let picks = selector.walkGreedy(scores: scores, candidateIds: candidateIds)
        let expected = golden["selector/picks"]!.asType(.int32)
        XCTAssertEqual(
            picks.asArray(Int32.self), expected.asArray(Int32.self),
            "greedy walk picked a different path")
    }
}
