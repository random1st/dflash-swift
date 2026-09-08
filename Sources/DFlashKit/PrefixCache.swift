//
//  PrefixCache.swift
//  DFlashKit
//

import Foundation
import MLX
import MLXLMCommon

/// Keeps the cache state of recent prompts so a conversation does not re-prefill what it
/// already computed.
///
/// A multi-turn prompt is almost the previous prompt plus the reply plus the new turn, so
/// nearly all of it was computed last time. That is the difference a user actually feels:
/// prefill is a full weight sweep per position, and on a long system prompt it dwarfs the
/// generation that follows.
///
/// **Snapshots, not trimming.** A plain attention cache can be rewound to any earlier
/// position, but this target is hybrid: its gated-delta layers carry recurrent state
/// accumulated over every token seen, with no way to unwind to an arbitrary point. So an
/// entry is reusable only at exactly the length it was taken at. It is exact for the same
/// reason the speculative rollback is: the state after k tokens is a function of the first
/// k tokens alone, so a snapshot there is precisely what a cold prefill of that prefix
/// would have produced.
///
/// Because reuse is all-or-nothing at one length, *where* the snapshot sits decides whether
/// it is ever hit - see ``DFlashSpeculativeGenerator``, which splits its prefill to place
/// one below the prompt's unstable tail.
///
/// Trading memory for time is the whole point: an entry costs the KV of its prefix and
/// buys back the prefill of every future prompt that starts with it.
public final class PrefixCache: @unchecked Sendable {

    /// One remembered prefix.
    private struct Entry {
        let tokens: [Int]
        let target: [any KVCache]
        let drafter: [BaseKVCache]
        /// The drafter context row for the last token of `tokens`, which the next round
        /// needs and which no cache holds: context rows lag their token by one position.
        let context: MLXArray
        var lastUsed: Date
        let bytes: Int
    }

    /// A hit: caches already copied, ready to be generated into.
    public struct Restored {
        /// How many leading prompt tokens are already in these caches.
        public let tokenCount: Int
        public let target: [any KVCache]
        public let drafter: [BaseKVCache]
        public let context: MLXArray
    }

    private let lock = NSLock()
    private var entries: [Entry] = []

    /// How many prefixes to keep and how much memory they may hold between them.
    public let slots: Int
    public let byteLimit: Int

    /// - Parameters:
    ///   - slots: how many conversations stay hot.
    ///   - byteLimit: ceiling across all entries; the least recently used one goes first.
    ///     Generous by default - the caller has already decided that a resident cache is
    ///     worth more than the memory it sits on, and an evicted entry costs a full
    ///     prefill to rebuild.
    public init(slots: Int = 4, byteLimit: Int = 12 << 30) {
        self.slots = max(1, slots)
        self.byteLimit = byteLimit
    }

    public var count: Int {
        lock.withLock { entries.count }
    }

    /// Bytes held by the snapshots, as MLX reports them.
    public var bytes: Int {
        lock.withLock { entries.reduce(0) { $0 + $1.bytes } }
    }

    /// The longest remembered prefix of `prompt`, copied so the caller may generate into it.
    ///
    /// A hit is returned only if at least one prompt token is left to forward: the loop
    /// needs logits to start from, and a restored cache carries state, not logits.
    public func lookup(prompt: [Int]) -> Restored? {
        lock.withLock {
            var best: Int? = nil
            for (index, entry) in entries.enumerated() {
                let length = entry.tokens.count
                guard length < prompt.count else { continue }
                guard length > (best.map { entries[$0].tokens.count } ?? 0) else { continue }
                if Array(prompt[..<length]) == entry.tokens {
                    best = index
                }
            }
            guard let index = best else { return nil }
            entries[index].lastUsed = Date()
            let entry = entries[index]
            // Copies, because the caller is about to generate into these and the entry has
            // to survive for the next request.
            return Restored(
                tokenCount: entry.tokens.count,
                target: entry.target.map { $0.copy() },
                drafter: entry.drafter.compactMap { $0.copy() as? BaseKVCache },
                context: entry.context)
        }
    }

    /// Remembers the caches as they stand after `tokens` have been forwarded.
    ///
    /// The caches are copied, so the caller may keep generating into its own.
    public func store(
        tokens: [Int], target: [any KVCache], drafter: [BaseKVCache], context: MLXArray
    ) {
        guard !tokens.isEmpty else { return }
        let targetCopy = target.map { $0.copy() }
        let drafterCopy = drafter.compactMap { $0.copy() as? BaseKVCache }
        let contextCopy = context.asType(context.dtype)
        // A snapshot is only a snapshot once it is materialised; left lazy it would hold a
        // graph rooted in caches that keep moving.
        let arrays =
            targetCopy.flatMap { $0.state } + drafterCopy.flatMap { $0.state } + [contextCopy]
        eval(arrays)

        let bytes = arrays.reduce(0) { $0 + $1.nbytes }

        lock.withLock {
            entries.removeAll { $0.tokens == tokens }
            entries.append(
                Entry(
                    tokens: tokens, target: targetCopy, drafter: drafterCopy,
                    context: contextCopy, lastUsed: Date(), bytes: bytes))
            evict()
        }
    }

    public func clear() {
        lock.withLock { entries.removeAll() }
    }

    /// Least recently used first. Called under `lock`.
    private func evict() {
        while entries.count > slots
            || (entries.reduce(0) { $0 + $1.bytes } > byteLimit && entries.count > 1)
        {
            guard
                let oldest = entries.indices.min(by: {
                    entries[$0].lastUsed < entries[$1].lastUsed
                })
            else { return }
            entries.remove(at: oldest)
        }
    }
}
