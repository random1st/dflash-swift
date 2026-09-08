//
//  DFlashSpeculativeGenerator.swift
//  DFlashKit
//
//  The speculative verify loop: one DFlash block drafted per round, verified in
//  a single target forward, with the target's recurrent caches rolled back to
//  the accept point.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// What one round proposed and how much of it the target kept.
public struct DFlashRound: Sendable {
    /// Drafted tokens the target verified this round.
    public let proposed: Int
    /// How many of them matched the target's own argmax.
    public let accepted: Int

    public init(proposed: Int, accepted: Int) {
        self.proposed = proposed
        self.accepted = accepted
    }
}

public struct DFlashGenerationStatistics: Sendable {
    /// Wall time per phase, so a slow round can be attributed instead of guessed at.
    public var draftSeconds: Double = 0
    public var verifySeconds: Double = 0
    public var rollbackSeconds: Double = 0
    public var prefillSeconds: Double = 0

    /// Tokens emitted, including the bonus token of every round.
    public let tokens: Int
    public let rounds: [DFlashRound]
    public let seconds: Double

    public var roundCount: Int { rounds.count }

    /// Accepted drafts per round. Add one for tokens per target forward: every
    /// round also commits the target's own next token for free.
    public var meanAcceptedPerRound: Double {
        guard !rounds.isEmpty else { return 0 }
        return Double(rounds.reduce(0) { $0 + $1.accepted }) / Double(rounds.count)
    }
}

public enum DFlashGenerationEvent: Sendable {
    case token(Int)
    case finished(DFlashGenerationStatistics)
    case failed(String)
}

public enum DFlashGenerationError: LocalizedError {
    case emptyPrompt
    case missingTapStates

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt:
            "The prompt is empty; the drafter needs at least one anchor position"
        case .missingTapStates:
            "The target returned no tapped layer states - was the bridge's requestState used?"
        }
    }
}

/// One greedy generation driven by a DFlash 2 drafter over a hybrid Qwen3.5
/// target.
///
/// Each round costs one target forward over `[anchor] + drafts`. Positions the
/// target disagrees with are rolled back: attention caches trim their rejected
/// tail, and the gated-DeltaNet caches are rebuilt from the round's captured
/// recurrence inputs (see ``rollbackGatedDeltaRound(cache:captures:width:keep:)``),
/// which is what keeps a round at one weight sweep instead of two.
///
/// `@unchecked Sendable`: the MLX modules it holds are not `Sendable`, and all
/// access to them is serialised onto this object's own queue.
public final class DFlashSpeculativeGenerator: @unchecked Sendable {
    private let target: Qwen35TextModel
    private let drafter: DFlashDraftModel
    private let bridge: Qwen35Bridge
    private let queue = DispatchQueue(label: "com.dflashkit.generate")

    /// Drafted tokens verified per round, at most `blockSize - 1`. The backbone
    /// always drafts the full block - it was trained at that width - so a lower
    /// cap only shortens the verified suffix.
    public let cap: Int

    public init(
        target: Qwen35TextModel,
        drafter: DFlashDraftModel,
        maximumDraftTokens: Int? = nil
    ) {
        self.target = target
        self.drafter = drafter
        // The tap order comes from the drafter's own config: any other order
        // silently mismatches `fc` and produces drafts the target rejects.
        self.bridge = Qwen35Bridge(
            target: target, tapIndices: drafter.configuration.targetLayerIds)
        let blockDrafts = max(1, drafter.configuration.blockSize - 1)
        // Verifying the full block is past the point where it pays. On stock MLX kernels a
        // quantised matmul does not amortise the weight read across a handful of rows: on
        // Qwen3.8-27B a 1-row pass costs 55 ms, 2 rows 1.14x, 4 rows 1.75x and 8 rows 3.24x.
        // Acceptance barely grows over that range, so the full block spends 3.24x to collect
        // 3.29 tokens while four drafts spend 1.75x to collect 3.14 — measured end to end at
        // 27.7 tok/s against 20.0. A drafter-side kernel that amortised the read would move
        // this optimum back out to the full block.
        self.cap = min(maximumDraftTokens ?? min(4, blockDrafts), blockDrafts)
        drafter.bind(bridge)
    }

    // MARK: - Streaming

    /// Tokens as they are committed, then one `.finished` with the statistics.
    public func stream(
        prompt: [Int], maximumTokens: Int, stopTokens: Set<Int> = []
    ) -> AsyncStream<DFlashGenerationEvent> {
        let (stream, continuation) = AsyncStream.makeStream(of: DFlashGenerationEvent.self)
        queue.async { [self] in
            do {
                let statistics = try generate(
                    prompt: prompt, maximumTokens: maximumTokens, stopTokens: stopTokens
                ) { token in
                    continuation.yield(.token(token))
                }
                continuation.yield(.finished(statistics))
            } catch {
                continuation.yield(.failed(error.localizedDescription))
            }
            continuation.finish()
        }
        return stream
    }

    // MARK: - The loop

    /// Runs the loop synchronously, calling `onToken` for each committed token.
    @discardableResult
    public func generate(
        prompt: [Int],
        maximumTokens: Int,
        stopTokens: Set<Int> = [],
        onToken: (Int) -> Void = { _ in }
    ) throws -> DFlashGenerationStatistics {
        guard !prompt.isEmpty else { throw DFlashGenerationError.emptyPrompt }
        let start = Date()

        let targetCache = try target.newCache(parameters: nil)
        let draftCache = drafter.makeCache()
        let maskToken = Int32(drafter.configuration.maskTokenId)
        let blockSize = drafter.configuration.blockSize

        // --- prefill ---
        let promptTokens = MLXArray(prompt.map { Int32($0) }).reshaped(1, prompt.count)
        let prefill = target(
            LMInput.Text(tokens: promptTokens), cache: targetCache,
            state: bridge.requestState())
        guard let prefillState = prefill.state, let promptFused = bridge.fuse(prefillState) else {
            throw DFlashGenerationError.missingTapStates
        }

        var pending = argMax(prefill.logits[0, -1], axis: -1).item(Int.self)
        var emitted = [pending]
        onToken(pending)

        // The drafter's context row at position i pairs with the token at i + 1,
        // so the prompt is appended shifted by one: everything but the last row
        // goes into the cache now, and that last row rides into the first round
        // as the context for the anchor.
        var pendingContext = promptFused[0..., (prompt.count - 1)..., 0...]
        if prompt.count > 1 {
            drafter.appendContext(
                drafter.projectContext(promptFused[0..., ..<(prompt.count - 1), 0...]),
                cache: draftCache)
        }

        var rounds: [DFlashRound] = []
        var draftSeconds = 0.0
        var verifySeconds = 0.0
        var rollbackSeconds = 0.0
        let prefillSeconds = Date().timeIntervalSince(start)
        var stopped = stopTokens.contains(pending)

        while emitted.count < maximumTokens && !stopped {
            let anchor = Int32(pending)
            // The block is [anchor] + mask slots; the drafter reads its logits at
            // the mask positions, which is why `selectBlock` drops the anchor row.
            let blockIds = [anchor] + Array(repeating: maskToken, count: blockSize - 1)
            let block = MLXArray(blockIds).reshaped(1, blockSize)
            var mark = Date()
            let proposals = drafter.selectBlock(
                block, fusedTargetHidden: pendingContext, cache: draftCache,
                cap: cap, anchorId: anchor)
            // The draft is lazy until something forces it; without this the draft cost
            // would be charged to whichever phase happens to evaluate first.
            eval(proposals)
            draftSeconds += Date().timeIntervalSince(mark)

            mark = Date()
            let verifyIds = concatenated([MLXArray([anchor]), proposals]).reshaped(1, cap + 1)
            var verifyState = bridge.requestState()
            verifyState[mtpGatedDeltaCaptureFlagKey] = true
            let verified = target(
                LMInput.Text(tokens: verifyIds), cache: targetCache, state: verifyState)

            let outcome = Self.acceptedPrefix(
                targetLogits: verified.logits[0], proposals: proposals)
            verifySeconds += Date().timeIntervalSince(mark)
            mark = Date()
            let accepted = outcome.accepted
            rounds.append(DFlashRound(proposed: cap, accepted: accepted))

            // Roll the target back to [anchor + accepted]: the bonus token is the
            // target's own next prediction, so it is committed without being in
            // this round's cache.
            rollbackGatedDeltaRound(
                cache: targetCache,
                captures: verified.state?[mtpGatedDeltaCapturesKey] ?? [],
                width: cap + 1,
                keep: accepted + 1)
            rollbackSeconds += Date().timeIntervalSince(mark)

            guard let verifiedState = verified.state,
                let verifiedFused = bridge.fuse(verifiedState)
            else {
                throw DFlashGenerationError.missingTapStates
            }
            // Context for the next round: the rows of the positions that stayed.
            pendingContext = verifiedFused[0..., ..<(accepted + 1), 0...]

            let committed =
                outcome.proposals[0 ..< accepted] + [outcome.targetTokens[accepted]]
            for token in committed {
                emitted.append(token)
                onToken(token)
                pending = token
                if stopTokens.contains(token) {
                    stopped = true
                    break
                }
                if emitted.count >= maximumTokens { break }
            }
        }

        return DFlashGenerationStatistics(
            draftSeconds: draftSeconds, verifySeconds: verifySeconds,
            rollbackSeconds: rollbackSeconds, prefillSeconds: prefillSeconds,
            tokens: emitted.count, rounds: rounds, seconds: Date().timeIntervalSince(start))
    }

    // MARK: - Accept test

    struct AcceptOutcome {
        /// Drafts the target agreed with, counted from the front.
        let accepted: Int
        /// The target's argmax at every verified position; index `accepted` is
        /// the bonus token.
        let targetTokens: [Int]
        let proposals: [Int]
    }

    /// Greedy accept length for one round.
    ///
    /// The target's argmax at position `i` is its own continuation of
    /// `[anchor] + drafts[0 ..< i]`, so it must be compared against the draft at
    /// position `i` - the token the drafter proposed for the next slot. The
    /// accepted prefix is the leading run of agreements: `cumprod` of the match
    /// vector is 1 up to the first disagreement and 0 after it, so its sum is
    /// that run's length, computed in the graph rather than by walking the
    /// tokens on the CPU.
    ///
    /// - Parameters:
    ///   - targetLogits: `[1 + drafts, V]` - the verify pass's logits, batch row
    ///     already indexed.
    ///   - proposals: `[drafts]` drafted token ids.
    static func acceptedPrefix(targetLogits: MLXArray, proposals: MLXArray) -> AcceptOutcome {
        let draftCount = proposals.dim(0)
        precondition(
            targetLogits.dim(0) == draftCount + 1,
            "verify logits cover \(targetLogits.dim(0)) positions for \(draftCount) drafts")

        let targetTokens = argMax(targetLogits, axis: -1).asType(.int32)
        let match = (proposals .== targetTokens[..<draftCount]).asType(.int32)
        let acceptedArray = cumprod(match, axis: 0).sum()

        // One sync per round: everything above is still one graph.
        eval(acceptedArray, targetTokens, proposals)
        return AcceptOutcome(
            accepted: acceptedArray.item(Int.self),
            targetTokens: targetTokens.asArray(Int32.self).map(Int.init),
            proposals: proposals.asArray(Int32.self).map(Int.init))
    }
}
