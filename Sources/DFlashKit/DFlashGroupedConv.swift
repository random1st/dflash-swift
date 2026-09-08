//
//  DFlashGroupedConv.swift
//  DFlashKit
//
//  Port of the DFlash 2 grouped dynamic convolution. See NOTICE.
//

import Foundation
import MLX
import MLXNN

/// A two-tap dynamic depthwise convolution wrapping one sublayer of a decoder
/// layer, applied across the positions of a draft block.
///
/// Both coefficient sets - the one for the sublayer's input and the one for its
/// output - come from a single projection of that input, which is why the type
/// is used as a pair of calls (`prepare` then `finish`) rather than as a plain
/// layer. Each coefficient is a learned per-channel base plus a content-adaptive
/// correction shared by `groupSize` channels.
///
/// It runs on the block rows only; the context rows never see it. That matters
/// for correctness, not just cost: the zero-padded tap shift is exactly the
/// block-boundary mask the reference applies, because block position 0 - the
/// verified anchor - has no in-block predecessor, and position 1 reads the
/// anchor's representation.
final class DFlashGroupedConv: Module {
    let taps: Int
    let groupSize: Int
    let groupCount: Int

    /// `[side, tap, channel]`, where side 0 convolves the input and side 1 the
    /// output. Identity at initialisation (tap 0 = 1) - the layout training
    /// exports.
    @ModuleInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, taps: Int, groupSize: Int) {
        precondition(
            hiddenSize % groupSize == 0,
            "conv_group_size \(groupSize) must divide hidden_size \(hiddenSize)")
        self.taps = taps
        self.groupSize = groupSize
        self.groupCount = hiddenSize / groupSize

        _baseKernel.wrappedValue = concatenated(
            [
                MLXArray.ones([2, 1, hiddenSize]),
                MLXArray.zeros([2, taps - 1, hiddenSize]),
            ], axis: 1)
        _kernelProjection.wrappedValue = Linear(
            hiddenSize, 2 * taps * groupCount, bias: false)
    }

    /// - Parameters:
    ///   - x: `[B, L, H]`
    ///   - delta: `[B, L, taps, groupCount]` content-adaptive corrections
    ///   - side: 0 for the input convolution, 1 for the output one
    private func convolve(_ x: MLXArray, delta: MLXArray, side: Int) -> MLXArray {
        let (B, L, H) = (x.dim(0), x.dim(1), x.dim(2))
        let grouped = x.reshaped(B, L, groupCount, groupSize)
        // The base kernel is per channel, so it unfolds to the full
        // [groups, groupSize] grid; the adaptive correction is per group and
        // broadcasts across the channels inside each group.
        let coefficients =
            baseKernel[side].reshaped(1, 1, taps, groupCount, groupSize)
            + delta.expandedDimensions(axis: -1)

        var out = coefficients[0..., 0..., 0] * grouped
        for tap in 1 ..< taps {
            // Shift by `tap` positions with zero padding, so a position never
            // reads across the start of the block.
            let shifted = padded(
                grouped[0..., ..<(L - tap)],
                widths: [IntOrPair(0), IntOrPair((tap, 0)), IntOrPair(0), IntOrPair(0)])
            out = out + coefficients[0..., 0..., tap] * shifted
        }
        return out.reshaped(B, L, H)
    }

    /// Convolves the sublayer input and returns it together with the
    /// coefficients `finish` needs for the sublayer output.
    func prepare(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let projected = kernelProjection(x)
        let coefficients = projected.reshaped(
            x.dim(0), x.dim(1), 2, taps, groupCount)
        let input = coefficients[0..., 0..., 0]
        let output = coefficients[0..., 0..., 1]
        return (convolve(x, delta: input, side: 0), output)
    }

    /// Convolves the sublayer output with the coefficients from `prepare`.
    func finish(_ y: MLXArray, delta: MLXArray) -> MLXArray {
        convolve(y, delta: delta, side: 1)
    }
}
