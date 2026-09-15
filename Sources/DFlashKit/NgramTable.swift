//
//  NgramTable.swift
//  DFlashKit
//
//  Prompt-lookup drafting for the steps the gate decodes without the drafter.
//

import Foundation

/// Guesses the next token from the text so far, for the plain phase of a reply.
///
/// When ``SpeculationGate`` switches drafting off, every token costs a full
/// single-row forward - 45.8 ms on Qwen3.8-27B, which is the memory floor - and
/// on the abliterated 27B that phase is a third of the tokens and nearly half the
/// time. A second row in the same forward costs 4.8 ms, so verifying one guessed
/// token alongside the pending one pays for itself from about a 10% hit rate.
///
/// The guess is the token that followed the most recent earlier occurrence of the
/// current tail - the last three tokens, or the last two when the three have not
/// been seen before. Replies repeat themselves: identifiers, closing brackets,
/// phrases lifted from the prompt. No model runs; the table is a dictionary kept
/// in step with the tokens as they are committed. mlx-lm's prompt-lookup on the
/// same hybrid architecture (issue #1497) landed 44% of one-token guesses; here,
/// on the steps the gate actually hands over, it landed 4 to 30 percent and lost -
/// see `DFlashSpeculativeGenerator.ngramLookup` for the numbers.
struct NgramTable {
    /// Longest tail tried first: a three-token match is rarely a coincidence.
    static let longestKey = 3
    /// Shortest tail worth a guess. Two- and three-token tails alone fire on almost
    /// nothing - 2 guesses over 64 plain steps of a 200-token reply - because a short
    /// reply barely repeats itself at that length; a one-token tail is loose, but with
    /// the break-even at a 10% hit rate loose is affordable, and it is what gives the
    /// table something to say on most steps.
    static let shortestKey = 1

    private var tokens: [Int] = []
    /// The token that followed the latest occurrence of each tail.
    private var follower: [[Int]: Int] = [:]

    init(tokens: [Int] = []) {
        for token in tokens { append(token) }
    }

    /// Records one committed token, in order.
    mutating func append(_ token: Int) {
        let end = tokens.count
        tokens.append(token)
        for length in Self.shortestKey ... Self.longestKey where end >= length {
            follower[Array(tokens[(end - length) ..< end])] = token
        }
    }

    /// The token expected after the current tail, or nil when the tail is new.
    ///
    /// The tail's own occurrence has no follower yet, so a hit is always an earlier
    /// occurrence - the latest one, since later appends overwrite.
    func guess() -> Int? {
        for length in stride(from: Self.longestKey, through: Self.shortestKey, by: -1)
        where tokens.count >= length {
            if let token = follower[Array(tokens.suffix(length))] {
                return token
            }
        }
        return nil
    }
}
