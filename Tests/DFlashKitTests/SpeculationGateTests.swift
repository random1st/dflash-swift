//
//  SpeculationGateTests.swift
//  DFlashKitTests
//

import XCTest

@testable import DFlashKit

final class SpeculationGateTests: XCTestCase {
    func testStaysSpeculativeWhileAcceptanceIsHigh() {
        var gate = SpeculationGate()
        for _ in 0 ..< 20 {
            gate.recordRound(accepted: 3)
            XCTAssertFalse(gate.isPlain)
        }
    }

    func testDoesNotJudgeBeforeTheWindowFills() {
        var gate = SpeculationGate()
        for _ in 0 ..< (SpeculationGate.window - 1) {
            gate.recordRound(accepted: 0)
            XCTAssertFalse(gate.isPlain, "one bad round is noise, not a verdict")
        }
        gate.recordRound(accepted: 0)
        XCTAssertTrue(gate.isPlain)
    }

    func testPlainRunEndsAndProbesAgain() {
        var gate = SpeculationGate()
        for _ in 0 ..< SpeculationGate.window { gate.recordRound(accepted: 1) }
        XCTAssertTrue(gate.isPlain)
        for _ in 0 ..< (SpeculationGate.plainRun - 1) {
            gate.recordPlainToken()
            XCTAssertTrue(gate.isPlain)
        }
        gate.recordPlainToken()
        XCTAssertFalse(gate.isPlain, "the run is over; the drafter gets another chance")

        // The probe starts from a clean window: the rounds that sent it plain must not
        // count against it again.
        for _ in 0 ..< (SpeculationGate.window - 1) {
            gate.recordRound(accepted: 0)
            XCTAssertFalse(gate.isPlain)
        }
    }

    func testNoticesProseAfterCode() {
        var gate = SpeculationGate()
        for _ in 0 ..< (10 * SpeculationGate.window) { gate.recordRound(accepted: 6) }
        XCTAssertFalse(gate.isPlain)
        for _ in 0 ..< SpeculationGate.window { gate.recordRound(accepted: 1) }
        XCTAssertTrue(gate.isPlain, "forty good rounds must not outvote the last block")
    }

    func testJudgesWholeBlocksNotEveryRound() {
        var gate = SpeculationGate()
        // A block that averages above the line, ending in a bad round, then a block that
        // starts badly: a sliding window would see four bad rounds in a row here.
        for accepted in [7, 7, 7, 0] { gate.recordRound(accepted: accepted) }
        XCTAssertFalse(gate.isPlain)
        for accepted in [0, 0, 0] { gate.recordRound(accepted: accepted) }
        XCTAssertFalse(gate.isPlain, "the block is not complete; no verdict yet")
        gate.recordRound(accepted: 7)
        XCTAssertFalse(gate.isPlain, "a block of 0 0 0 7 averages 1.75 and stays")
    }

    func testThresholdSitsAboveTheLosingPromptsAndBelowTheWinningOnes() {
        // The measured points: 1.53 accepted ran at 0.87x plain, 2.50 at 1.15x.
        XCTAssertGreaterThan(SpeculationGate.minimumAccepted, 1.53)
        XCTAssertLessThan(SpeculationGate.minimumAccepted, 2.5)
    }
}
