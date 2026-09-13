//
//  SpeculativeTreeTests.swift
//  DFlashKitTests
//
//  A tree round claims something strong: the logits it produces at row r are the
//  logits a plain forward over that row's own path would have produced, and the
//  rollback leaves the caches holding what a forward over the accepted path would
//  have left. Both claims can be wrong in ways that still generate fluent text -
//  a leaked sibling in the attention mask, a position taken from row order
//  instead of depth, a recurrence that followed the wrong branch - and all of
//  them show up only as acceptance quietly collapsing. So they are checked
//  numerically against real forwards, not by shape.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import DFlashKit

final class SpeculativeTreeTests: XCTestCase {

    // MARK: - Plan shape

    /// ```
    /// 0 ─┬─ 1 ─┬─ 3 ─── 5
    ///    │     └─ 4
    ///    └─ 2
    /// ```
    private let forkedParents = [-1, 0, 0, 1, 1, 3]

    func testDepthsFollowTheTreeAndNotRowOrder() {
        let plan = SpeculativeTreePlan(parents: forkedParents)
        XCTAssertEqual(plan.depths, [0, 1, 1, 2, 2, 3])
    }

    func testEveryLeafBecomesOneRootPath() {
        let plan = SpeculativeTreePlan(parents: forkedParents)
        XCTAssertEqual(Set(plan.paths), [[0, 1, 3, 5], [0, 1, 4], [0, 2]])
        XCTAssertEqual(plan.pathLength, 4)
        for (row, source) in plan.rowSources.enumerated() {
            let path = plan.paths[source.path]
            XCTAssertEqual(
                path[source.depth], row, "row \(row) reads back from the wrong batch slot")
        }
    }

    func testAttentionMaskShowsAncestorsAndNothingElse() {
        let plan = SpeculativeTreePlan(parents: forkedParents)
        let mask = plan.attentionMask(prefix: 2)
        XCTAssertEqual(mask.shape, [6, 8])
        let bits = mask.asType(.int32).asArray(Int32.self)
        func sees(_ row: Int, _ column: Int) -> Bool { bits[row * 8 + column] == 1 }

        for row in 0 ..< 6 {
            XCTAssertTrue(sees(row, 0) && sees(row, 1), "row \(row) must see the whole prefix")
        }
        // Row 5's ancestors are 3, 1 and the anchor; 2 and 4 are cousins it must
        // not read, and 4 precedes it in row order, which is exactly the mistake
        // a plain causal mask would make.
        XCTAssertEqual(
            (0 ..< 6).filter { sees(5, 2 + $0) }, [0, 1, 3, 5])
        XCTAssertEqual((0 ..< 6).filter { sees(2, 2 + $0) }, [0, 2])
    }

    func testPositionsComeFromDepth() {
        let plan = SpeculativeTreePlan(parents: forkedParents)
        XCTAssertEqual(
            plan.positions(offset: 10).asArray(Int32.self), [10, 11, 11, 12, 12, 13])
    }

    func testChainPlanIsTheDegenerateTree() {
        let plan = SpeculativeTreePlan.chain(rows: 4)
        XCTAssertEqual(plan.parents, [-1, 0, 1, 2])
        XCTAssertEqual(plan.paths, [[0, 1, 2, 3]])
        XCTAssertTrue(plan.isRootPath([0, 1, 2]))
        XCTAssertFalse(plan.isRootPath([0, 2]))
    }

    // MARK: - Against real forwards

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func assertClose(
        _ a: MLXArray, _ b: MLXArray, _ label: String, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, "\(label): shape", file: file, line: line)
        guard a.shape == b.shape else { return }
        // A tree row and a plain forward run the same ops over different
        // sequence lengths, so they agree to floating-point tiling. A leaked
        // sibling or a wrong position is off by orders of magnitude, not ulps.
        let scale = max(1, abs(b.asType(.float32)).max().item(Float.self))
        XCTAssertLessThan(
            maxDifference(a, b) / scale, 2e-3, "\(label): value", file: file, line: line)
    }

    private let prompt = MLXArray((0 ..< 6).map { Int32($0 + 1) }).reshaped(1, 6)
    /// Token ids per row of ``forkedParents``. Deliberately not distinct across
    /// branches: rows 2 and 4 share an id, so a mask that leaked a sibling would
    /// still look plausible token-wise.
    private let rowTokens: [Int32] = [11, 23, 37, 41, 37, 53]

    private func treeRound(
        model: Qwen35TextModel, cache: [any KVCache], plan: SpeculativeTreePlan
    ) -> LMOutput {
        var request = LMOutput.State()
        request[mtpGatedDeltaCaptureFlagKey] = true
        request[mtpTreePlanKey] = plan
        return model(
            LMInput.Text(tokens: MLXArray(rowTokens).reshaped(1, rowTokens.count)),
            cache: cache, state: request)
    }

    /// The load-bearing claim: each row's logits are its own path's logits.
    func testEveryRowScoresItsOwnPath() throws {
        let model = try TinyHybridTarget.make()
        let plan = SpeculativeTreePlan(parents: forkedParents)

        let cache = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: prompt), cache: cache, state: nil)
        let round = treeRound(model: model, cache: cache, plan: plan)

        for path in plan.paths {
            let reference = try model.newCache(parameters: nil)
            _ = model(LMInput.Text(tokens: prompt), cache: reference, state: nil)
            let tokens = MLXArray(path.map { rowTokens[$0] }).reshaped(1, path.count)
            let plain = model(LMInput.Text(tokens: tokens), cache: reference, state: nil)

            for (depth, row) in path.enumerated() {
                assertClose(
                    round.logits[0, row], plain.logits[0, depth],
                    "row \(row) at depth \(depth) of path \(path)")
            }
        }
    }

    /// A negative control: the comparison above must be able to fail. Verifying
    /// the same rows as a plain chain - which is what an ignored plan would do -
    /// has to disagree with the branch structure.
    func testChainVerificationOfTreeRowsDisagrees() throws {
        let model = try TinyHybridTarget.make()
        let plan = SpeculativeTreePlan(parents: forkedParents)

        let cache = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: prompt), cache: cache, state: nil)
        let round = treeRound(model: model, cache: cache, plan: plan)

        let chained = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: prompt), cache: chained, state: nil)
        let asChain = model(
            LMInput.Text(tokens: MLXArray(rowTokens).reshaped(1, rowTokens.count)),
            cache: chained, state: nil)

        // Row 2 is the anchor's second child: as a chain it would sit after rows
        // 0 and 1 and at position offset+2, as a tree it follows the anchor alone.
        let scale = max(1, abs(asChain.logits[0, 2].asType(.float32)).max().item(Float.self))
        XCTAssertGreaterThan(
            maxDifference(round.logits[0, 2], asChain.logits[0, 2]) / scale, 2e-3,
            "a branch row must not score the same as it would in a chain")
    }

    /// The other half: after the rollback the caches hold what a forward over
    /// the accepted path would have left, so the next round starts from the
    /// right state rather than from whichever branch ran last.
    private func assertRollbackMatchesPath(_ keepRows: [Int]) throws {
        let model = try TinyHybridTarget.make()
        let plan = SpeculativeTreePlan(parents: forkedParents)
        XCTAssertTrue(plan.isRootPath(keepRows), "the fixture must keep a real path")

        let speculative = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: prompt), cache: speculative, state: nil)
        let round = treeRound(model: model, cache: speculative, plan: plan)
        let captures = try XCTUnwrap(
            round.state?[mtpGatedDeltaCapturesKey], "capture was requested but not returned")

        rollbackGatedDeltaTree(
            cache: speculative, captures: captures, width: rowTokens.count, keepRows: keepRows)

        let committed = try model.newCache(parameters: nil)
        _ = model(LMInput.Text(tokens: prompt), cache: committed, state: nil)
        _ = model(
            LMInput.Text(
                tokens: MLXArray(keepRows.map { rowTokens[$0] }).reshaped(1, keepRows.count)),
            cache: committed, state: nil)

        for (index, caches) in zip(speculative, committed).enumerated() {
            let (rolledBack, reference) = caches
            if let rolledBack = rolledBack as? MambaCache,
                let reference = reference as? MambaCache
            {
                assertClose(
                    try XCTUnwrap(rolledBack[0]), try XCTUnwrap(reference[0]),
                    "layer \(index) conv window")
                assertClose(
                    try XCTUnwrap(rolledBack[1]), try XCTUnwrap(reference[1]),
                    "layer \(index) recurrent state")
            } else {
                XCTAssertEqual(
                    rolledBack.offset, reference.offset, "layer \(index): attention offset")
                for (slot, pair) in zip(rolledBack.state, reference.state).enumerated() {
                    assertClose(pair.0, pair.1, "layer \(index) attention slot \(slot)")
                }
            }
        }
    }

    /// A branch whose rows are scattered - 0, 1, 3, 5 - which is the case plain
    /// tail-trimming cannot express.
    func testRollbackToDeepBranchMatchesCommittedForward() throws {
        try assertRollbackMatchesPath([0, 1, 3, 5])
    }

    /// The other child of the anchor: row 2 sits after rows it must not inherit.
    func testRollbackToSecondChildMatchesCommittedForward() throws {
        try assertRollbackMatchesPath([0, 2])
    }

    func testRollbackToAnchorOnlyMatchesCommittedForward() throws {
        try assertRollbackMatchesPath([0])
    }

    // MARK: - Accept walk

    func testAcceptedPathFollowsTheTargetIntoABranch() {
        let tree = DFlashDraftTree(
            tokens: rowTokens, slots: [-1, 0, 0, 1, 1, 2],
            plan: SpeculativeTreePlan(parents: forkedParents))
        // The target continues the anchor with row 2's token and then names
        // something nobody drafted, so the walk takes the second branch and stops.
        let (rows, bonus) = tree.acceptedPath(targetTokens: [37, 99, 71, 99, 99, 99])
        XCTAssertEqual(rows, [0, 2])
        XCTAssertEqual(bonus, 71)
    }

    func testAcceptedPathStopsWhenNoChildMatches() {
        let tree = DFlashDraftTree(
            tokens: rowTokens, slots: [-1, 0, 0, 1, 1, 2],
            plan: SpeculativeTreePlan(parents: forkedParents))
        let (rows, bonus) = tree.acceptedPath(targetTokens: [99, 11, 22, 33, 44, 55])
        XCTAssertEqual(rows, [0], "no child holds 99, so only the anchor survives")
        XCTAssertEqual(bonus, 99)
    }

    func testAcceptedPathTakesTheWholeBranch() {
        let tree = DFlashDraftTree(
            tokens: rowTokens, slots: [-1, 0, 0, 1, 1, 2],
            plan: SpeculativeTreePlan(parents: forkedParents))
        let (rows, bonus) = tree.acceptedPath(targetTokens: [23, 41, 0, 53, 0, 67])
        XCTAssertEqual(rows, [0, 1, 3, 5])
        XCTAssertEqual(bonus, 67)
    }
}
