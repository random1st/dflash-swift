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
    /// Whether those rows formed a tree rather than a single chain.
    public let tree: Bool

    public init(proposed: Int, accepted: Int, tree: Bool = false) {
        self.proposed = proposed
        self.accepted = accepted
        self.tree = tree
    }
}

public struct DFlashGenerationStatistics: Sendable {
    /// Wall time per phase, so a slow round can be attributed instead of guessed at.
    public var draftSeconds: Double = 0
    public var verifySeconds: Double = 0
    public var rollbackSeconds: Double = 0
    public var prefillSeconds: Double = 0
    /// Time in single-token forwards, taken while the gate had drafting switched off.
    public var plainSeconds: Double = 0
    /// Tokens committed without a draft. See ``SpeculationGate`` for when that happens.
    public var plainTokens: Int = 0
    /// Plain steps that carried an n-gram guess, and how many of those guesses the
    /// target accepted. See ``NgramTable``.
    public var ngramGuesses: Int = 0
    public var ngramHits: Int = 0

    /// Prompt tokens that came from a prefix-cache hit instead of being prefilled.
    public var reusedPromptTokens: Int = 0

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

    /// How many rounds took a tree. Reported so the shape that actually ran can be
    /// read off a benchmark instead of inferred from tok/s, which drifts here.
    public var treeRounds: Int { rounds.filter(\.tree).count }
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

    /// Whether the round width follows the drafter's measured hit rate.
    ///
    /// A fixed cap assumes the drafter matches the target. It often does not: z-lab's
    /// drafter for Qwen3.6-35B-A3B lands 7.08 accepted tokens per round on the stock
    /// weights and 3.74 on an abliterated variant of them, and every drafted token beyond
    /// what gets accepted is a verified row paid for and thrown away. On a MoE those rows
    /// are not free - a wider block selects a wider union of experts, which is real weight
    /// traffic - so the cost of guessing high is paid twice.
    ///
    /// Measured on the abliterated model, three interleaved rounds: cap 7 gives 124.8
    /// tok/s, cap 9 gives 117.8, cap 15 gives 94.0. The best cap sits about three above
    /// the accepted mean, which is where the extra rows still have a chance of paying and
    /// have not yet started buying experts nobody reads.
    public let adaptiveWidth: Bool

    /// How far past the running accepted mean to keep drafting, before the result is
    /// snapped to a tile boundary.
    private static let widthHeadroom = 3

    /// The width to draft next, given how many tokens are being accepted.
    ///
    /// Two rules, both measured. Draft about three past the accepted mean, because that is
    /// where the extra rows still have a chance of being kept. Then round the verify pass
    /// up to a whole MMA tile: the kernel computes all eight rows of a tile whether or not
    /// they were asked for, so a width of seven rows costs what eight costs and returns
    /// less. The sweep shows it plainly - on the abliterated MoE, caps of 3, 5, 7 (four,
    /// six and eight rows) beat 4, 6, 9 around them, and cap 7 wins outright.
    ///
    /// Reproduces every optimum measured here: 3.74 accepted picks cap 7 (best of the
    /// sweep at 124.8 tok/s), 4.10 picks 7 (best), 7.08 picks 15 (best).
    static func width(forAccepted mean: Double, blockDrafts: Int) -> Int {
        let wanted = Int(mean.rounded()) + widthHeadroom
        let rows = max(wanted + 1, SmallMQuantizedMatmul.rowsPerTile)
        let snapped = ((rows + SmallMQuantizedMatmul.rowsPerTile - 1)
            / SmallMQuantizedMatmul.rowsPerTile) * SmallMQuantizedMatmul.rowsPerTile
        return max(4, min(snapped - 1, blockDrafts))
    }

    /// How many of the target's projections run on ``SmallMQuantizedMatmul``.
    /// Zero means the verify pass is on stock kernels and the cap is the narrow one.
    public let acceleratedLayers: Int

    /// Whether a round's verified rows form a tree instead of a single chain.
    ///
    /// Same rows, same weight sweep - only their shape changes, and the committed
    /// text is the same greedy text either way. See ``DFlashDraftTree`` for what
    /// the tree buys. Off by default because whether it buys anything depends on
    /// the target's weights, not on the code: with the same DFlash2 drafter the
    /// tree takes 10% fewer target forwards against stock Qwen3.8-27B (461 to 415
    /// over eight prompts) and 13% *more* against the abliterated variant of those
    /// weights (125 to 141 over four), which is the model actually served. Where
    /// the chain already accepts 3.5 or more of its 7 slots, cutting the trunk to
    /// pay for branches loses. A per-reply policy that tried both shapes and kept
    /// the better one was measured and discarded: interleaving the shapes within
    /// one reply biases both estimates, and it came out below the fixed chain on
    /// the abliterated weights and below the fixed tree on stock.
    public let treeSpeculation: Bool

    /// Whether plain steps verify an n-gram guess alongside the pending token.
    ///
    /// The plain phase is where the drafter has given up, and on the abliterated 27B it
    /// is a third of the tokens and nearly half the time, every one of them a single-row
    /// forward at the memory floor. A guess from ``NgramTable`` rides in the same forward
    /// as a second row, and the committed text stays the greedy text, because the guess
    /// goes through the same accept test as a drafted block.
    ///
    /// Off by default because it was measured and lost. ABBA on five prompts against
    /// Qwen3.8-27B-Uncensored: the text is identical, the plain phase is slower on every
    /// prompt. Hit rates were 4, 10, 30, 15 and 7 percent, and a miss costs more than
    /// the 4.8 ms second row predicts - 7 to 25 ms once the accept sync and the rollback
    /// of 48 recurrent layers are paid - so break-even sits at 15 to 35 percent. The
    /// plain phase is plain because the text there is unpredictable; a lookup table is a
    /// weaker predictor than the drafter that already gave up on it.
    public let ngramLookup: Bool

    /// Prompt prefixes kept hot between calls, or nil to prefill every prompt cold.
    public let prefixCache: PrefixCache?

    /// How far below the end of the prompt a snapshot is taken.
    ///
    /// Snapshots cannot be trimmed after the fact (see ``PrefixCache``), so the boundary
    /// has to be chosen before the prefill runs, and it has to be one the *next* prompt
    /// will also contain. The tail of a prompt is the least stable part of it: a chat
    /// template renders the trailing generation prompt one way while the turn is open and
    /// another once it is closed, so a snapshot taken at the exact end of turn N is not a
    /// prefix of turn N+1 and would never be hit. Four tokens covers the Qwen templates'
    /// tails, and leaving them out costs one extra short forward.
    public static let snapshotSlack = 4

    /// Prompts shorter than this are prefilled cold: their prefill is already brief, and
    /// an entry that is mostly boilerplate would push out one that carries a real context.
    public static let minimumCachedPrompt = 64

    /// - Parameter useSmallMKernel: swap the target's eligible quantised projections
    ///   for the small-M kernel. It is what makes a full-width block affordable, and it
    ///   is why the default cap is the whole block. Only 6-to-8-row forwards take the
    ///   new path, so single-token decode and prefill through the same model stay bit
    ///   for bit what they were; pass `false` to leave the model untouched.
    public init(
        target: Qwen35TextModel,
        drafter: DFlashDraftModel,
        maximumDraftTokens: Int? = nil,
        useSmallMKernel: Bool = true,
        prefixCache: PrefixCache? = nil,
        treeSpeculation: Bool = false,
        ngramLookup: Bool = false
    ) {
        self.target = target
        self.drafter = drafter
        self.prefixCache = prefixCache
        self.treeSpeculation = treeSpeculation
        self.ngramLookup = ngramLookup
        // The tap order comes from the drafter's own config: any other order
        // silently mismatches `fc` and produces drafts the target rejects.
        self.bridge = Qwen35Bridge(
            target: target, tapIndices: drafter.configuration.targetLayerIds)
        self.acceleratedLayers = useSmallMKernel ? enableSmallMQuantizedMatmul(in: target) : 0

        let blockDrafts = max(1, drafter.configuration.blockSize - 1)
        // How wide a block pays for itself is decided entirely by whether the verify pass
        // re-reads the weights per row. On stock kernels it does: a target forward on
        // Qwen3.8-27B costs 52 ms at 1 row, 1.05x at 2, 1.63x at 4 and 3.07x at 8, so a
        // full block spends 3.07x to collect ~3.7 accepted tokens and speculation gives
        // most of its winnings back — 21.6 tok/s at cap 7 against 29.6 at cap 4.
        //
        // With the small-M kernel the same curve reads 1.00 / 1.06 / 1.64 / 1.51x: from 6
        // rows up the weight read is paid once and width is nearly free, so the optimum
        // moves back out to the whole block — measured 44.5 tok/s at cap 7, against 21.6
        // for the same cap on stock kernels. Hence: kernel on, draft the full block; kernel
        // off, stay at four.
        let defaultCap = acceleratedLayers > 0 ? blockDrafts : min(4, blockDrafts)
        self.cap = min(maximumDraftTokens ?? defaultCap, blockDrafts)
        // An explicit cap is an instruction, not a starting point: the bench pins it to
        // measure one width, and a caller who names a number means that number.
        self.adaptiveWidth = maximumDraftTokens == nil
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

        let maskToken = Int32(drafter.configuration.maskTokenId)
        let blockSize = drafter.configuration.blockSize

        // --- prefill ---
        var targetCache: [any KVCache]
        var draftCache: [BaseKVCache]
        // The fused row of the last token already in the caches. Context rows lag their
        // token by one position, so this one belongs to no cache yet: it is the context
        // for whatever token comes next.
        var carried: MLXArray?
        var prefilled = 0
        var reused = 0

        if let restored = prefixCache?.lookup(prompt: prompt) {
            targetCache = restored.target
            draftCache = restored.drafter
            carried = restored.context
            prefilled = restored.tokenCount
            reused = restored.tokenCount
        } else {
            targetCache = try target.newCache(parameters: nil)
            draftCache = drafter.makeCache()
            // Snapshot first, then finish the prompt: an entry cannot be trimmed down to a
            // boundary afterwards, so the prefill is split at the boundary instead.
            if let prefixCache, prompt.count >= Self.minimumCachedPrompt {
                let boundary = prompt.count - Self.snapshotSlack
                let head = Array(prompt[..<boundary])
                carried = try advance(
                    tokens: head, targetCache: targetCache, draftCache: draftCache,
                    carried: nil
                ).context
                prefilled = boundary
                prefixCache.store(
                    tokens: head, target: targetCache, drafter: draftCache, context: carried!)
            }
        }

        let tail = Array(prompt[prefilled...])
        let prefill = try advance(
            tokens: tail, targetCache: targetCache, draftCache: draftCache, carried: carried)
        var pendingContext = prefill.context

        var pending = argMax(prefill.logits[0, -1], axis: -1).item(Int.self)
        var emitted = [pending]
        onToken(pending)
        var stopped = stopTokens.contains(pending)

        // Everything the reply has said so far, prompt included, for the plain steps'
        // guesses. Kept in step by `hand`, which is the only place tokens are committed.
        var table = NgramTable(tokens: prompt)
        table.append(pending)

        /// Commits verified tokens in order until a stop token or the budget cuts the
        /// list short; returns how many were handed out.
        func hand(_ committed: [Int]) -> Int {
            var handed = 0
            for token in committed {
                emitted.append(token)
                handed += 1
                onToken(token)
                table.append(token)
                pending = token
                if stopTokens.contains(token) {
                    stopped = true
                    break
                }
                if emitted.count >= maximumTokens { break }
            }
            return handed
        }

        // What the caches hold, which is not what was emitted: a round commits its anchor
        // and its accepted drafts, while the bonus token it emits only enters the cache as
        // the next round's anchor. Tracked so the state at the end of the reply can be
        // remembered under exactly the tokens that produced it.
        var cached = prompt

        var rounds: [DFlashRound] = []
        var draftSeconds = 0.0
        var verifySeconds = 0.0
        var rollbackSeconds = 0.0
        let prefillSeconds = Date().timeIntervalSince(start)

        // Start at one tile and grow, rather than starting at the whole block. A drafter
        // that matches its target climbs back to the full block within a few rounds, and
        // one that does not never pays for the wide rounds it would have spent learning
        // that. It also keeps the first generation from compiling a second kernel variant
        // for a width it abandons.
        var roundCap =
            adaptiveWidth ? min(SmallMQuantizedMatmul.rowsPerTile - 1, cap) : cap
        var acceptedTotal = 0

        // Whether to draft at all is the same kind of question as how wide: measured
        // per reply, not assumed. A pinned cap is a bench measuring one width and gets
        // that width every round.
        var gate = SpeculationGate()
        var plainSeconds = 0.0
        var plainTokens = 0
        var ngramGuesses = 0
        var ngramHits = 0

        while emitted.count < maximumTokens && !stopped {
            if adaptiveWidth && gate.isPlain {
                let mark = Date()
                if ngramLookup, let guess = table.guess() {
                    // The guess rides as a second row of the same forward and takes the
                    // same accept test as a drafted block, so the committed text is still
                    // the greedy text; a miss costs the extra row, a hit saves a forward.
                    // Two rows stay on the stock kernels - the small-M path starts at six.
                    let anchor = pending
                    var verifyState = bridge.requestState()
                    verifyState[mtpGatedDeltaCaptureFlagKey] = true
                    let verifyIds = MLXArray([Int32(anchor), Int32(guess)]).reshaped(1, 2)
                    let verified = target(
                        LMInput.Text(tokens: verifyIds), cache: targetCache, state: verifyState)
                    let outcome = Self.acceptedPrefix(
                        targetLogits: verified.logits[0], proposals: MLXArray([Int32(guess)]))
                    guard let verifiedState = verified.state,
                        let verifiedFused = bridge.fuse(verifiedState)
                    else {
                        throw DFlashGenerationError.missingTapStates
                    }
                    ngramGuesses += 1
                    ngramHits += outcome.accepted

                    let handed = hand(
                        outcome.proposals[0 ..< outcome.accepted]
                            + [outcome.targetTokens[outcome.accepted]])
                    let keep = min(handed, outcome.accepted) + 1
                    rollbackGatedDeltaRound(
                        cache: targetCache,
                        captures: verified.state?[mtpGatedDeltaCapturesKey] ?? [],
                        width: 2, keep: keep)
                    // The rows carried in are not in the drafter's cache yet - nothing ran
                    // the drafter this step - so they stay in front of the new ones. Dropping
                    // them leaves the target's text unchanged and the drafter's context one
                    // row short per step, which showed up as fewer accepted drafts on the
                    // same reply once drafting resumed.
                    pendingContext = concatenated(
                        [pendingContext, verifiedFused[0..., ..<keep, 0...]], axis: 1)
                    cached.append(anchor)
                    if keep > 1 { cached.append(guess) }

                    plainSeconds += Date().timeIntervalSince(mark)
                    plainTokens += handed
                    for _ in 0 ..< handed { gate.recordPlainToken() }
                    continue
                }

                // One token per forward, greedy: the same output the verify loop would
                // commit, without paying for a block the target is about to reject. The
                // drafter's context cache still has to follow along - `advance` appends
                // the rows the verified positions completed - so that when the gate lets
                // drafting resume, the drafter sees the whole reply and not a hole.
                let step = try advance(
                    tokens: [pending], targetCache: targetCache, draftCache: draftCache,
                    carried: pendingContext)
                let next = argMax(step.logits[0, -1], axis: -1).item(Int.self)
                plainSeconds += Date().timeIntervalSince(mark)
                plainTokens += 1

                cached.append(pending)
                pendingContext = step.context
                _ = hand([next])
                gate.recordPlainToken()
                continue
            }

            let anchor = Int32(pending)
            // The block is [anchor] + mask slots; the drafter reads its logits at
            // the mask positions, which is why `selectBlock` drops the anchor row.
            let blockIds = [anchor] + Array(repeating: maskToken, count: blockSize - 1)
            let block = MLXArray(blockIds).reshaped(1, blockSize)
            var mark = Date()
            // A tree spends the same verified rows on several branches instead of one
            // chain; `tree` is nil when that is switched off or when the checkpoint has
            // no selector and therefore no lattice to branch over.
            let tree =
                treeSpeculation
                ? drafter.draftLattice(
                    block, fusedTargetHidden: pendingContext, cache: draftCache,
                    cap: roundCap, anchorId: anchor
                ).map {
                    DFlashDraftTree.build(lattice: $0, anchorId: anchor, budget: roundCap)
                } : nil
            let proposals =
                tree == nil
                ? drafter.selectBlock(
                    block, fusedTargetHidden: pendingContext, cache: draftCache,
                    cap: roundCap, anchorId: anchor)
                : nil
            // The draft is lazy until something forces it; without this the draft cost
            // would be charged to whichever phase happens to evaluate first.
            if let proposals { eval(proposals) }
            draftSeconds += Date().timeIntervalSince(mark)

            mark = Date()
            let width = tree?.rowCount ?? (roundCap + 1)
            var verifyState = bridge.requestState()
            verifyState[mtpGatedDeltaCaptureFlagKey] = true
            if let tree { verifyState[mtpTreePlanKey] = tree.plan }
            let verifyIds =
                tree?.verifyIds
                ?? concatenated([MLXArray([anchor]), proposals!]).reshaped(1, roundCap + 1)
            let verified = target(
                LMInput.Text(tokens: verifyIds), cache: targetCache, state: verifyState)

            // Rows the round keeps if everything it proposed is handed out, anchor
            // first. A chain keeps a prefix; a tree keeps the branch the target walked.
            let keptRows: [Int]
            let committed: [Int]
            if let tree {
                let targetTokens = argMax(verified.logits[0], axis: -1).asType(.int32)
                eval(targetTokens)
                let (rows, bonus) = tree.acceptedPath(
                    targetTokens: targetTokens.asArray(Int32.self))
                keptRows = rows
                committed = rows.dropFirst().map { Int(tree.tokens[$0]) } + [Int(bonus)]
            } else {
                let outcome = Self.acceptedPrefix(
                    targetLogits: verified.logits[0], proposals: proposals!)
                keptRows = Array(0 ... outcome.accepted)
                committed =
                    outcome.proposals[0 ..< outcome.accepted] + [outcome.targetTokens[outcome.accepted]]
            }
            verifySeconds += Date().timeIntervalSince(mark)
            mark = Date()
            let accepted = keptRows.count - 1
            rounds.append(
                DFlashRound(proposed: width - 1, accepted: accepted, tree: tree != nil))

            guard let verifiedState = verified.state,
                let verifiedFused = bridge.fuse(verifiedState)
            else {
                throw DFlashGenerationError.missingTapStates
            }

            let handed = hand(committed)

            // Roll the target back to the tokens that were actually handed out, which is
            // usually [anchor + accepted] - the bonus token is the target's own next
            // prediction and is committed without being in this round's cache. A round cut
            // short by a stop token or by the budget keeps less: leaving the unhanded tail
            // in the caches would make their state describe a reply nobody was shown, and
            // the snapshot taken at the end of this generation would then be filed under
            // tokens no later prompt contains.
            mark = Date()
            let keep = min(handed, accepted) + 1
            let keepRows = Array(keptRows[..<keep])
            let captures = verified.state?[mtpGatedDeltaCapturesKey] ?? []
            if tree != nil {
                rollbackGatedDeltaTree(
                    cache: targetCache, captures: captures, width: width, keepRows: keepRows)
            } else {
                rollbackGatedDeltaRound(
                    cache: targetCache, captures: captures, width: width, keep: keep)
            }
            rollbackSeconds += Date().timeIntervalSince(mark)

            // Context for the next round: the rows of the positions that stayed.
            pendingContext =
                tree == nil
                ? verifiedFused[0..., ..<keep, 0...]
                : take(verifiedFused, MLXArray(keepRows.map { Int32($0) }), axis: 1)
            cached.append(Int(anchor))
            cached.append(contentsOf: committed[0 ..< (keep - 1)])

            // Next round's width follows the hit rate this drafter is actually achieving
            // on these weights, which is the thing a fixed cap cannot know in advance.
            acceptedTotal += accepted
            if adaptiveWidth {
                let mean = Double(acceptedTotal) / Double(rounds.count)
                roundCap = Self.width(forAccepted: mean, blockDrafts: cap)
                gate.recordRound(accepted: accepted)
            }
        }

        // The reply is the expensive half of the next turn's prompt, and the caches are
        // holding it right now. Remembering it here is what turns "the first turn is slow"
        // into "only the first turn is slow": measured, the next turn's prompt does contain
        // this one plus the whole reply.
        //
        // `cached` is exactly what the caches hold, and every token in it was handed to the
        // caller, so it is a prefix of any transcript that continues this reply.
        if let prefixCache, cached.count > prompt.count {
            prefixCache.store(
                tokens: cached, target: targetCache, drafter: draftCache,
                context: pendingContext[0..., (pendingContext.dim(1) - 1)..., 0...])
        }

        return DFlashGenerationStatistics(
            draftSeconds: draftSeconds, verifySeconds: verifySeconds,
            rollbackSeconds: rollbackSeconds, prefillSeconds: prefillSeconds,
            plainSeconds: plainSeconds, plainTokens: plainTokens,
            ngramGuesses: ngramGuesses, ngramHits: ngramHits,
            reusedPromptTokens: reused,
            tokens: emitted.count, rounds: rounds, seconds: Date().timeIntervalSince(start))
    }

    /// Forwards `tokens` through the target, appends the drafter contexts they complete,
    /// and hands back the row that has to be carried into whatever comes next.
    ///
    /// The shift is the whole subtlety: the fused row at position `i` is the drafter's
    /// context for the token at `i + 1`. So every row but the last is appendable now -
    /// preceded by `carried`, the row left over from the tokens already in the caches -
    /// and the last row is returned to be carried in turn.
    private func advance(
        tokens: [Int], targetCache: [any KVCache], draftCache: [BaseKVCache],
        carried: MLXArray?
    ) throws -> (context: MLXArray, logits: MLXArray) {
        let ids = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)
        let output = target(
            LMInput.Text(tokens: ids), cache: targetCache, state: bridge.requestState())
        guard let state = output.state, let fused = bridge.fuse(state) else {
            throw DFlashGenerationError.missingTapStates
        }

        var rows = fused[0..., ..<(tokens.count - 1), 0...]
        if let carried {
            rows = concatenated([carried, rows], axis: 1)
        }
        if rows.dim(1) > 0 {
            drafter.appendContext(drafter.projectContext(rows), cache: draftCache)
        }
        return (fused[0..., (tokens.count - 1)..., 0...], output.logits)
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
