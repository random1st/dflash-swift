//
//  NgramTableTests.swift
//  DFlashKitTests
//

import XCTest

@testable import DFlashKit

final class NgramTableTests: XCTestCase {
    func testNewTailHasNoGuess() {
        let table = NgramTable(tokens: [1, 2, 3])
        XCTAssertNil(table.guess(), "nothing has followed [1, 2, 3] yet")
    }

    func testRepeatedTailGuessesItsFollower() {
        // "1 2 3 4" seen once; the tail is "1 2 3" again.
        let table = NgramTable(tokens: [1, 2, 3, 4, 9, 1, 2, 3])
        XCTAssertEqual(table.guess(), 4)
    }

    func testLongestMatchWinsOverShorter() {
        // "2 3" was followed by 5 earlier, but "1 2 3" was followed by 4.
        let table = NgramTable(tokens: [7, 2, 3, 5, 1, 2, 3, 4, 8, 1, 2, 3])
        XCTAssertEqual(table.guess(), 4)
    }

    func testFallsBackToTwoTokenTail() {
        let table = NgramTable(tokens: [2, 3, 5, 8, 1, 2, 3])
        XCTAssertEqual(table.guess(), 5, "[1, 2, 3] is new, [2, 3] was followed by 5")
    }

    func testLatestOccurrenceOverridesEarlier() {
        var table = NgramTable(tokens: [1, 2, 3, 4])
        for token in [1, 2, 3, 6, 1, 2, 3] { table.append(token) }
        XCTAssertEqual(table.guess(), 6)
    }

    func testFallsBackToOneTokenTail() {
        let table = NgramTable(tokens: [1, 2, 1])
        XCTAssertEqual(table.guess(), 2, "[1] was followed by 2")
    }

    func testFirstTokenHasNoGuess() {
        XCTAssertNil(NgramTable(tokens: [1]).guess())
    }
}
