//
//  AcceptLengthTests.swift
//  DFlashKitTests
//
//  The accept test decides how much of a round is kept, which is both what the
//  user sees as output and what the rollback is told to keep. An off-by-one here
//  is lossy generation, not slow generation, so it is pinned on hand-built
//  logits where the target's argmax at every position is known exactly.
//

import Foundation
import MLX
import XCTest

@testable import DFlashKit

final class AcceptLengthTests: XCTestCase {
    private let vocabulary = 10

    /// Logits whose argmax per row is exactly `tokens` - one peak per row.
    private func logits(argmax tokens: [Int32]) -> MLXArray {
        var values = [Float](repeating: 0, count: tokens.count * vocabulary)
        for (row, token) in tokens.enumerated() {
            values[row * vocabulary + Int(token)] = 1
        }
        return MLXArray(values, [tokens.count, vocabulary])
    }

    private func accept(target: [Int32], proposals: [Int32])
        -> DFlashSpeculativeGenerator.AcceptOutcome
    {
        DFlashSpeculativeGenerator.acceptedPrefix(
            targetLogits: logits(argmax: target), proposals: MLXArray(proposals))
    }

    func testFullAccept() {
        // The target's own continuation is the whole draft; position 3 is the
        // bonus token, which is committed whether or not anything was accepted.
        let outcome = accept(target: [3, 4, 5, 6], proposals: [3, 4, 5])
        XCTAssertEqual(outcome.accepted, 3)
        XCTAssertEqual(outcome.targetTokens[outcome.accepted], 6)
    }

    func testZeroAccept() {
        let outcome = accept(target: [9, 4, 5, 6], proposals: [3, 4, 5])
        XCTAssertEqual(outcome.accepted, 0)
        // The bonus token replaces the rejected draft, so a zero-accept round
        // still commits one token and never stalls.
        XCTAssertEqual(outcome.targetTokens[outcome.accepted], 9)
    }

    func testPartialAcceptStopsAtFirstMismatch() {
        // Position 2 disagrees; the agreement at position 3 is downstream of a
        // rejected token and must not be counted.
        let outcome = accept(target: [3, 4, 8, 6, 1], proposals: [3, 4, 5, 6])
        XCTAssertEqual(outcome.accepted, 2)
        XCTAssertEqual(outcome.targetTokens[outcome.accepted], 8)
    }

    func testProposalsAreReportedAsGiven() {
        let outcome = accept(target: [3, 4, 8, 6, 1], proposals: [3, 4, 5, 6])
        XCTAssertEqual(outcome.proposals, [3, 4, 5, 6])
    }
}
