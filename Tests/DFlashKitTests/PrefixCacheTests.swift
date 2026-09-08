//
//  PrefixCacheTests.swift
//  DFlashKitTests
//
//  The property a prefix cache lives or dies by: a restored snapshot plus the remaining
//  suffix must produce what a cold prefill of the whole prompt would have produced. A
//  wrong one does not crash - it answers a slightly different prompt, fluently, and the
//  only symptom is a reply that ignores the start of the conversation.
//
//  The target is hybrid, so the interesting half is the gated-DeltaNet state: it is an
//  accumulation over every token seen and cannot be trimmed, which is why entries are
//  snapshots at one fixed length rather than something rewindable.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import DFlashKit

final class PrefixCacheTests: XCTestCase {
    private let prompt = (0 ..< 20).map { ($0 * 7 + 3) % TinyHybridTarget.vocabularySize }
    private let boundary = 12

    private func tokens(_ ids: ArraySlice<Int>) -> MLXArray {
        MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
    }

    /// Logits at the last position, which is the only thing generation starts from.
    private func lastLogits(_ output: LMOutput) -> MLXArray {
        output.logits[0, -1].asType(.float32)
    }

    private func coldLogits(_ model: Qwen35TextModel) throws -> MLXArray {
        let cache = try model.newCache(parameters: nil)
        return lastLogits(model(LMInput.Text(tokens: tokens(prompt[...])), cache: cache, state: nil))
    }

    private func assertClose(
        _ a: MLXArray, _ b: MLXArray, _ label: String, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let scale = max(1, abs(b).max().item(Float.self))
        let difference = abs(a - b).max().item(Float.self)
        // Split and whole prefills run the same ops over different sequence lengths, so
        // they agree to floating-point tiling, not bit for bit. A restore that dropped or
        // duplicated state is off by orders of magnitude, not by ulps.
        XCTAssertLessThan(difference / scale, 2e-3, label, file: file, line: line)
    }

    // MARK: - Equivalence

    /// Store at the boundary, restore, forward only the suffix: the model must be in the
    /// same place as if it had seen the prompt in one pass.
    func testRestoredPrefixMatchesColdPrefill() throws {
        let model = try TinyHybridTarget.make()
        let cold = try coldLogits(model)

        let cache = PrefixCache()
        let live = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: tokens(prompt[..<boundary])), cache: live, state: nil)
        cache.store(
            tokens: Array(prompt[..<boundary]), target: live, drafter: [],
            context: MLXArray.zeros([1, 1, 1]))

        let restored = try XCTUnwrap(cache.lookup(prompt: prompt))
        XCTAssertEqual(restored.tokenCount, boundary)
        let warm = lastLogits(
            model(
                LMInput.Text(tokens: tokens(prompt[boundary...])), cache: restored.target,
                state: nil))
        assertClose(warm, cold, "restored prefix + suffix vs cold prefill")
    }

    /// The same entry has to serve a second request after the first one generated into it.
    /// MLX arrays are handles; if `copy()` handed out an alias instead of an independent
    /// cache, the first generation would quietly poison the entry - and it would only show
    /// up as a wrong answer on the second turn.
    func testEntrySurvivesGenerationIntoTheRestoredCache() throws {
        let model = try TinyHybridTarget.make()
        let cold = try coldLogits(model)

        let cache = PrefixCache()
        let live = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: tokens(prompt[..<boundary])), cache: live, state: nil)
        cache.store(
            tokens: Array(prompt[..<boundary]), target: live, drafter: [],
            context: MLXArray.zeros([1, 1, 1]))

        // Keep using the caches the entry was taken from, exactly as a generation would.
        for _ in 0 ..< 3 {
            _ = model(LMInput.Text(tokens: tokens(prompt[...])), cache: live, state: nil)
        }
        // And use one restored copy before asking for another.
        let first = try XCTUnwrap(cache.lookup(prompt: prompt))
        _ = model(
            LMInput.Text(tokens: tokens(prompt[boundary...])), cache: first.target, state: nil)

        let second = try XCTUnwrap(cache.lookup(prompt: prompt))
        let warm = lastLogits(
            model(
                LMInput.Text(tokens: tokens(prompt[boundary...])), cache: second.target,
                state: nil))
        assertClose(warm, cold, "second restore of the same entry")
    }

    /// The negative control for the assertion above: it has to be able to fail. A snapshot
    /// taken one token short of the boundary is exactly the off-by-one a slice bug would
    /// produce, and it must not look like a cold prefill.
    func testSnapshotAtTheWrongLengthDivergesFromColdPrefill() throws {
        let model = try TinyHybridTarget.make()
        let cold = try coldLogits(model)

        let cache = PrefixCache()
        let live = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: tokens(prompt[..<boundary])), cache: live, state: nil)
        // Claim the entry covers one token more than it does, so the suffix is short by one.
        cache.store(
            tokens: Array(prompt[..<(boundary + 1)]), target: live, drafter: [],
            context: MLXArray.zeros([1, 1, 1]))

        let restored = try XCTUnwrap(cache.lookup(prompt: prompt))
        let warm = lastLogits(
            model(
                LMInput.Text(tokens: tokens(prompt[(boundary + 1)...])), cache: restored.target,
                state: nil))
        let scale = max(1, abs(cold).max().item(Float.self))
        XCTAssertGreaterThan(
            abs(warm - cold).max().item(Float.self) / scale, 2e-3,
            "a prefix cache that answers the wrong prompt must not pass the equivalence test")
    }

    // MARK: - Lookup

    private func makeCache(_ prefixes: [[Int]], slots: Int = 4) throws -> PrefixCache {
        let model = try TinyHybridTarget.make()
        let cache = PrefixCache(slots: slots)
        for prefix in prefixes {
            let live = try model.newCache(parameters: nil)
            _ = model(LMInput.Text(tokens: tokens(prefix[...])), cache: live, state: nil)
            cache.store(
                tokens: prefix, target: live, drafter: [], context: MLXArray.zeros([1, 1, 1]))
        }
        return cache
    }

    func testLookupTakesTheLongestMatchingPrefix() throws {
        let cache = try makeCache([Array(prompt[..<6]), Array(prompt[..<12])])
        XCTAssertEqual(cache.lookup(prompt: prompt)?.tokenCount, 12)
    }

    func testLookupRejectsAPrefixThatDiverges() throws {
        var other = Array(prompt[..<12])
        other[5] += 1
        let cache = try makeCache([other])
        XCTAssertNil(cache.lookup(prompt: prompt))
    }

    func testLookupRejectsAnEntryAsLongAsThePrompt() throws {
        // A full-length hit would leave nothing to forward, and a restored cache carries
        // state, not logits - there would be no token to start generating from.
        let cache = try makeCache([prompt])
        XCTAssertNil(cache.lookup(prompt: prompt))
    }

    func testStoringTheSamePrefixTwiceKeepsOneEntry() throws {
        let cache = try makeCache([Array(prompt[..<12]), Array(prompt[..<12])])
        XCTAssertEqual(cache.count, 1)
    }

    // MARK: - Eviction

    func testEvictionDropsTheLeastRecentlyUsed() throws {
        let cache = try makeCache(
            [Array(prompt[..<6]), Array(prompt[..<12])], slots: 2)
        XCTAssertEqual(cache.count, 2)

        // Touch the shorter one so it is not the oldest, then overflow the slots.
        XCTAssertEqual(cache.lookup(prompt: Array(prompt[..<8]))?.tokenCount, 6)

        let model = try TinyHybridTarget.make()
        let third = [9, 9, 9, 9]
        let live = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: tokens(third[...])), cache: live, state: nil)
        cache.store(tokens: third, target: live, drafter: [], context: MLXArray.zeros([1, 1, 1]))

        XCTAssertEqual(cache.count, 2)
        // The full prompt matches both stored prefixes, so what it comes back with says
        // which one survived: the touched 6-token entry, not the longer untouched one.
        XCTAssertEqual(
            cache.lookup(prompt: prompt)?.tokenCount, 6,
            "the untouched 12-token entry should be the one gone")
    }

    func testByteLimitEvictsEvenWithSlotsToSpare() throws {
        let model = try TinyHybridTarget.make()
        let cache = PrefixCache(slots: 8, byteLimit: 1)
        for prefix in [Array(prompt[..<6]), Array(prompt[..<12])] {
            let live = try model.newCache(parameters: nil)
            _ = model(LMInput.Text(tokens: tokens(prefix[...])), cache: live, state: nil)
            cache.store(
                tokens: prefix, target: live, drafter: [], context: MLXArray.zeros([1, 1, 1]))
        }
        // One entry always stays: evicting the last one would make the limit a way to
        // disable the cache silently rather than to bound it.
        XCTAssertEqual(cache.count, 1)
        XCTAssertGreaterThan(cache.bytes, 0)
    }
}
