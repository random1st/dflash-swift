//
//  GatedDeltaRollbackTests.swift
//  DFlashKitTests
//
//  The property the whole verify loop rests on: after a speculative round of
//  `width` positions is rolled back to `keep`, every cache must hold exactly
//  what a committed forward over those `keep` positions would have left.
//
//  Attention caches only drop rows, so they are easy. The gated-DeltaNet layers
//  are not: their state is an accumulation over all `width` positions and cannot
//  be trimmed, so it is replayed from the round's captured recurrence inputs. If
//  that replay is wrong the model still produces fluent text - it just disagrees
//  with its own drafts, so acceptance collapses silently. Hence a numeric
//  equality check against a real committed forward rather than a shape check.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import DFlashKit

final class GatedDeltaRollbackTests: XCTestCase {
    private func makeModel() throws -> Qwen35TextModel {
        try TinyHybridTarget.make()
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func assertClose(
        _ a: MLXArray, _ b: MLXArray, _ label: String, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, "\(label): shape", file: file, line: line)
        guard a.shape == b.shape else { return }
        // Replay and committed forward run the same ops over different sequence
        // lengths, so they agree to floating-point tiling, not bit-for-bit. The
        // bound is far tighter than bf16's ~8e-3 relative step: a wrong replay
        // is off by orders of magnitude, not by ulps.
        let scale = max(1, abs(b.asType(.float32)).max().item(Float.self))
        XCTAssertLessThan(
            maxDifference(a, b) / scale, 2e-3, "\(label): value", file: file, line: line)
    }

    /// Runs one round of `width` positions with capture and rolls it back to
    /// `rolledBackTo`, alongside a second cache that only ever saw the first
    /// `committedTo` positions of the same round.
    private func roundAndCommittedForward(
        width: Int, rolledBackTo: Int, committedTo: Int
    ) throws -> (speculative: [any KVCache], committed: [any KVCache]) {
        let model = try makeModel()
        // A prompt first, so the round starts from a non-trivial recurrent state
        // and a non-empty conv window - replaying from zeros would hide a
        // dropped pre-round state.
        let prompt = MLXArray((0 ..< 6).map { Int32($0 + 1) }).reshaped(1, 6)
        let round = MLXArray((0 ..< width).map { Int32(($0 * 7 + 3) % 64) }).reshaped(1, width)

        let speculative = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: prompt), cache: speculative, state: nil)
        var request = LMOutput.State()
        request[mtpGatedDeltaCaptureFlagKey] = true
        let verified = model(
            LMInput.Text(tokens: round), cache: speculative, state: request)
        let captures = try XCTUnwrap(
            verified.state?[mtpGatedDeltaCapturesKey], "capture was requested but not returned")
        XCTAssertEqual(captures.count, 2, "one capture per gated-delta layer, in layer order")

        rollbackGatedDeltaRound(
            cache: speculative, captures: captures, width: width, keep: rolledBackTo)

        let committed = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: prompt), cache: committed, state: nil)
        _ = model(
            LMInput.Text(tokens: round[0..., ..<committedTo]), cache: committed, state: nil)
        return (speculative, committed)
    }

    /// The property: a rollback to `keep` leaves every cache holding what a
    /// forward over those `keep` positions would have left.
    private func assertRollbackMatchesCommittedForward(keep: Int, width: Int) throws {
        let (speculative, committed) = try roundAndCommittedForward(
            width: width, rolledBackTo: keep, committedTo: keep)

        for (index, caches) in zip(speculative, committed).enumerated() {
            let (rolledBack, reference) = caches
            if let rolledBack = rolledBack as? MambaCache,
                let reference = reference as? MambaCache
            {
                let convA = try XCTUnwrap(rolledBack[0], "layer \(index): no conv window")
                let convB = try XCTUnwrap(reference[0], "layer \(index): no conv window")
                assertClose(convA, convB, "layer \(index) conv window")

                let stateA = try XCTUnwrap(rolledBack[1], "layer \(index): no recurrent state")
                let stateB = try XCTUnwrap(reference[1], "layer \(index): no recurrent state")
                assertClose(stateA, stateB, "layer \(index) recurrent state")
            } else {
                XCTAssertEqual(
                    rolledBack.offset, reference.offset, "layer \(index): attention offset")
                let a = rolledBack.state
                let b = reference.state
                XCTAssertEqual(a.count, b.count, "layer \(index): attention state count")
                for (slot, pair) in zip(a, b).enumerated() {
                    assertClose(pair.0, pair.1, "layer \(index) attention slot \(slot)")
                }
            }
        }
    }

    func testRollbackAfterPartialAcceptMatchesCommittedForward() throws {
        try assertRollbackMatchesCommittedForward(keep: 3, width: 8)
    }

    func testRollbackAfterZeroAcceptMatchesCommittedForward() throws {
        // keep == 1: only the anchor survives, the deepest rebuild there is.
        try assertRollbackMatchesCommittedForward(keep: 1, width: 8)
    }

    func testRollbackOfLastPositionMatchesCommittedForward() throws {
        try assertRollbackMatchesCommittedForward(keep: 7, width: 8)
    }

    /// Negative controls: the comparison above has to be able to fail, or the
    /// three tests prove nothing. Two ways to be wrong, both of which a real bug
    /// would produce - no rollback at all, and a rollback that keeps one
    /// position too many (an off-by-one in the accept length or in a slice
    /// bound).
    private func assertRecurrentStatesDiffer(
        width: Int, rolledBackTo: Int, committedTo: Int, _ label: String
    ) throws {
        let (speculative, committed) = try roundAndCommittedForward(
            width: width, rolledBackTo: rolledBackTo, committedTo: committedTo)

        let recurrentIndices = speculative.indices.filter { speculative[$0] is MambaCache }
        XCTAssertFalse(recurrentIndices.isEmpty, "the fixture must have gated-delta layers")
        for index in recurrentIndices {
            let a = try XCTUnwrap((speculative[index] as? MambaCache)?[1])
            let b = try XCTUnwrap((committed[index] as? MambaCache)?[1])
            let scale = max(1, abs(b.asType(.float32)).max().item(Float.self))
            XCTAssertGreaterThan(
                maxDifference(a, b) / scale, 2e-3, "layer \(index): \(label)")
        }
    }

    func testRoundWithoutRollbackDivergesFromCommittedForward() throws {
        // rolledBackTo == width is the no-op rollback: the state still carries
        // the whole rejected tail.
        try assertRecurrentStatesDiffer(
            width: 8, rolledBackTo: 8, committedTo: 3,
            "an un-rolled-back state must not look committed")
    }

    func testRollbackToTheWrongKeepDivergesFromCommittedForward() throws {
        try assertRecurrentStatesDiffer(
            width: 8, rolledBackTo: 4, committedTo: 3,
            "a rollback one position too deep must not look committed")
    }

    /// Capture is opt-in: a pass that does not ask for it must not pay for it,
    /// and must not publish stashes a caller could mistake for this round's.
    func testCaptureIsAbsentUnlessRequested() throws {
        let model = try makeModel()
        let cache = try model.newCache(parameters: nil)
        let tokens = MLXArray((0 ..< 4).map { Int32($0 + 1) }).reshaped(1, 4)

        let plain = model(LMInput.Text(tokens: tokens), cache: cache, state: nil)
        XCTAssertNil(plain.state?[mtpGatedDeltaCapturesKey])

        var emitOnly = LMOutput.State()
        emitOnly[mtpEmitFlagKey] = true
        let tapped = model(LMInput.Text(tokens: tokens), cache: cache, state: emitOnly)
        XCTAssertNotNil(tapped.state, "the emit flag must still return a state")
        XCTAssertNil(tapped.state?[mtpGatedDeltaCapturesKey])
    }
}
