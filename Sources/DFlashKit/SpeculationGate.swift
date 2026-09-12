//
//  SpeculationGate.swift
//  DFlashKit
//
//  Decides, round by round, whether drafting is still paying for itself.
//

import Foundation

/// Speculation is a bet that the drafter's block is cheaper than the tokens it returns,
/// and on prose the bet loses. Measured on the abliterated Qwen3.6-35B-A3B with the
/// stock z-lab drafter, decode only, against plain greedy at 86 tok/s: a short story
/// accepts 0.96 drafts per round and runs at 0.68x plain; a paragraph on latency, 1.53
/// and 0.87x; a two-sentence definition, 2.50 and 1.15x; a Python function, 3.07 and
/// 1.41x. A round at one tile of width costs about 2.9 single-token forwards on the MoE
/// whatever it accepts, so it pays only from about 1.9 accepted drafts per round.
///
/// Nothing about the prompt says in advance which side of that line a reply falls on -
/// the same model writes prose and code in one answer - so the decision is made from
/// the acceptance actually observed, and revisited: after a plain run the drafter gets
/// another chance, so a reply that turns from explanation into code climbs back.
///
/// The gate is deliberately blind to timing. Wall time on a laptop swings with thermal
/// state and whatever else is running; the accepted count is a property of the text.
struct SpeculationGate {
    /// Accepted drafts per round below which a round costs more than it returns.
    /// Break-even measured at about 1.9; the margin keeps a 2.5 reply speculating
    /// through a bad stretch rather than flapping.
    static let minimumAccepted = 1.75

    /// Rounds observed before the mean is trusted. Acceptance is bursty - the story
    /// histogram is mostly zeros and ones with an occasional five - so a single round
    /// says little, and four rounds is already 2 to 20 tokens of evidence. Longer
    /// windows were simulated on recorded sequences and did not help: they react later
    /// on prose and still trip on code, because a bad stretch of code looks like prose
    /// at any window length short enough to matter.
    static let window = 4

    /// Tokens decoded plainly before the drafter is tried again. Each probe spends
    /// `window` rounds at roughly 2.9 plain steps apiece, so on pure prose the probe
    /// costs about a third of the run it follows; longer runs would make the return
    /// to code slower.
    static let plainRun = 32

    private var recent: [Int] = []
    private var plainRemaining = 0

    init() {}

    /// Whether the next token should be decoded without drafting.
    var isPlain: Bool { plainRemaining > 0 }

    /// Records a speculative round's outcome and decides whether to keep drafting.
    mutating func recordRound(accepted: Int) {
        recent.append(accepted)
        guard recent.count == Self.window else { return }
        // Judged in whole blocks of rounds, not on a window slid one round at a time.
        // Acceptance on code is bursty enough that a sliding window trips on a bad
        // stretch every few rounds - simulated on recorded sequences, sliding cost the
        // semver reply 10% against 2% for blocks, for the same gain on the story. And
        // the window is not a running mean since the start: a reply that opened with a
        // long code block still notices when it turns into prose.
        let mean = Double(recent.reduce(0, +)) / Double(recent.count)
        recent.removeAll()
        if mean < Self.minimumAccepted {
            plainRemaining = Self.plainRun
        }
    }

    /// Records one plainly decoded token; the run ends when the count runs out.
    mutating func recordPlainToken() {
        plainRemaining = max(0, plainRemaining - 1)
    }
}
