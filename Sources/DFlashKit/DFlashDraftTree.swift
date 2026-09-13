//
//  DFlashDraftTree.swift
//  DFlashKit
//
//  A tree of drafts cut out of one block's lattice.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One block's drafts arranged as a tree of verification rows.
///
/// The greedy walk keeps one token per slot, so a round dies at the first slot
/// where the selector ranked the target's token second. Measured on 1523
/// recorded lattices, that is what usually happens: when the chain is rejected,
/// the target's token is rank 2 in 29.1% of rounds and outside the drafter's
/// top-16 in only 12.5%. So the miss is mostly a ranking error inside a
/// candidate set that already contains the answer, and spending the same
/// verified rows on a tree instead of a chain recovers part of it - 3.235
/// accepted tokens per round becomes 3.634 at the identical eight rows.
///
/// Rows are emitted parent-first, which is the order a tree-verified round
/// needs: see ``SpeculativeTreePlan``.
public struct DFlashDraftTree {
    /// Token id per row; row 0 is the anchor the block continues from.
    public let tokens: [Int32]
    /// Which lattice slot each row fills; `-1` for the anchor.
    public let slots: [Int]
    /// The shape of the round these rows make up.
    public let plan: SpeculativeTreePlan

    public var rowCount: Int { tokens.count }

    /// `[1, rows]` verification input.
    public var verifyIds: MLXArray {
        MLXArray(tokens).reshaped(1, tokens.count)
    }

    /// Expand the lattice into `budget` drafted rows, best first.
    ///
    /// The frontier is ordered by the path's total log-probability, so the
    /// budget divides itself between depth and width: where the drafter is
    /// confident one branch keeps extending, and where it is not the rows go to
    /// siblings. At seven drafted rows the average shape is 1.71 1.72 1.35 0.92
    /// 0.61 0.40 0.29 rows per slot - the chain plus two or three short stubs
    /// near the front, which is where the rejections are.
    ///
    /// The scores must be normalised per slot before they are summed. They are
    /// raw logits, and a sum of raw logits grows with depth, so best-first over
    /// them always extends the deepest path and degenerates into the chain
    /// (measured: +1.3% over the chain, against +12.4% once normalised).
    public static func build(
        lattice: DFlashBlockLattice, anchorId: Int32, budget: Int
    ) -> DFlashDraftTree {
        let slotCount = lattice.scores.dim(0)
        let k = lattice.scores.dim(2)
        let logProbabilities = logSoftmax(lattice.scores, axis: -1)
        eval(logProbabilities, lattice.candidateIds)
        let logp = logProbabilities.asArray(Float.self)
        let candidates = lattice.candidateIds.asArray(Int32.self)

        func score(slot: Int, predecessor: Int, candidate: Int) -> Float {
            logp[(slot * k + predecessor) * k + candidate]
        }

        /// A path of candidate indices, one per slot from slot 0.
        struct Frontier {
            let total: Float
            let path: [Int]
        }
        // Slot 0's predecessor rows are all the anchor, so any of them will do.
        var frontier = (0 ..< k).map {
            Frontier(total: score(slot: 0, predecessor: 0, candidate: $0), path: [$0])
        }

        var tokens: [Int32] = [anchorId]
        var slots: [Int] = [-1]
        var parents: [Int] = [-1]
        // Row of every materialised path. A path extends its own prefix, and a
        // prefix always scores at least as high as its extension, so the parent
        // has already been popped by the time a child is.
        var rowOfPath: [[Int]: Int] = [[]: 0]

        while tokens.count <= budget, !frontier.isEmpty {
            // The frontier is a few hundred entries at most and is drained at
            // most fifteen times, so a scan is cheaper to read than a heap.
            var best = 0
            for (index, entry) in frontier.enumerated()
            where entry.total > frontier[best].total {
                best = index
            }
            let chosen = frontier.remove(at: best)
            let slot = chosen.path.count - 1
            let candidate = chosen.path[slot]

            let row = tokens.count
            tokens.append(candidates[slot * k + candidate])
            slots.append(slot)
            parents.append(rowOfPath[Array(chosen.path.dropLast())]!)
            rowOfPath[chosen.path] = row

            guard slot + 1 < slotCount else { continue }
            for child in 0 ..< k {
                frontier.append(
                    Frontier(
                        total: chosen.total
                            + score(slot: slot + 1, predecessor: candidate, candidate: child),
                        path: chosen.path + [child]))
            }
        }

        return DFlashDraftTree(
            tokens: tokens, slots: slots, plan: SpeculativeTreePlan(parents: parents))
    }

    /// The rows the target agreed with, anchor first.
    ///
    /// A row's logits are the target's continuation of that row's own path,
    /// because the round's attention mask let the row see nothing but its
    /// ancestors. So the walk descends from the anchor, at each step looking for
    /// a child row holding the token the target just named, and stops at the
    /// first level where no child does.
    ///
    /// - Parameter targetTokens: `[rows]` argmax of the verify pass.
    /// - Returns: the accepted path, and the target's own next token after it.
    public func acceptedPath(targetTokens: [Int32]) -> (rows: [Int], bonus: Int32) {
        precondition(
            targetTokens.count == rowCount,
            "\(targetTokens.count) target tokens for \(rowCount) rows")

        var childrenOf = [[Int]](repeating: [], count: rowCount)
        for row in 1 ..< rowCount {
            childrenOf[plan.parents[row]].append(row)
        }

        var rows = [0]
        var row = 0
        while let next = childrenOf[row].first(where: { tokens[$0] == targetTokens[row] }) {
            rows.append(next)
            row = next
        }
        return (rows, targetTokens[row])
    }
}
