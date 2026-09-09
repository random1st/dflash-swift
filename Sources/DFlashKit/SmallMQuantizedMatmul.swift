//
//  SmallMQuantizedMatmul.swift
//  DFlashKit
//
//  Port of mlx-dspark's `small_m_qmm`, whose kernel is avlp12's `qmm_mma4`. See NOTICE.
//

import Foundation
import MLX
import MLXNN

/// A quantised matmul for the handful of rows a speculative verify pass has.
///
/// `quantizedMatmul` grows nearly linearly in M for M in 2...8: the weight read is
/// re-paid per row until the GEMM tiling takes over around M=13. Measured here on
/// Qwen3.8-27B-4bit a target forward costs 55 ms at 1 row, 1.14x at 2, 1.75x at 4 and
/// 3.24x at 8 - so verifying a block of 8 spends 3.24x to collect 3.29 tokens and
/// speculation nets nothing. The weights are read once; the cost should be flat.
///
/// This kernel makes it flat. An 8x8 `simdgroup_matrix` MMA tile covers M <= 8 exactly,
/// so every quantised weight group is read and dequantised ONCE and reused by all rows;
/// K is split across the 8 simdgroups of a threadgroup so each serial loop is short.
///
/// It is a supplement, never a replacement: below M=5 `quantizedMatmul`'s GEMV path is
/// already at roofline and wins. Anything outside the eligibility window falls through
/// to the stock op unchanged, bit for bit.
enum SmallMQuantizedMatmul {

    /// Measured crossover on a dependent chain: M=5 is 1.03-1.07x (noise, and measured
    /// net-negative in-model upstream), M=6 is the first clear win. A constant of
    /// (chip x MLX version) - revisit after an MLX upgrade.
    static let minimumRows = 6
    /// Two MMA tiles. Eight was one tile, which was enough while every drafter here used
    /// an eight-token block - but the MoE drafter drafts sixteen, so its verify pass was
    /// exactly one row too wide to qualify and ran entirely on the stock op.
    static let maximumRows = 16

    /// Rows one MMA tile covers. A pass narrower than this pays for one tile, not two.
    static let rowsPerTile = 8
    /// Below this the grid is too few threadgroups to occupy the GPU.
    static let minimumOutputFeatures = 4096
    /// The K-loop strides 64 values per simdgroup over a K/8 slice, so K must be a
    /// multiple of 512. Upstream gates on 128, which would mis-handle K in
    /// {128, 256, 384} mod 512; every real shape here is a multiple of 512 anyway.
    static let inputFeatureMultiple = 512
    /// The group size the kernel's dequant loop hard-codes (one scale/bias per 64).
    static let requiredGroupSize = 64

    private static let threadgroupSize = 256  // 8 simdgroups

    // The dequant unpack is the only part that differs by quantisation width: 4-bit
    // packs 8 values per uint32 (16 values = 2 uints per lane-slice), 8-bit packs 4
    // (16 values = 4 uints). Split-K, staging, MMA and reduction are identical.
    private static let unpack: [Int: String] = [
        4: """
                        const device uint* wr = w + (size_t)n * (K / 8) + (ka >> 3) + kq * 2;
                        uint p0 = wr[0], p1 = wr[1];
                        for (int t = 0; t < 8; ++t)
                            bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)((float)((p0 >> (4 * t)) & 15u) * s + bb);
                        for (int t = 0; t < 8; ++t)
                            bt[(kq * 16 + 8 + t) * 8 + j] = (bfloat16_t)((float)((p1 >> (4 * t)) & 15u) * s + bb);
            """,
        8: """
                        const device uint* wr = w + (size_t)n * (K / 4) + (ka >> 2) + kq * 4;
                        for (int u = 0; u < 4; ++u) {
                            uint p = wr[u];
                            for (int t = 0; t < 4; ++t)
                                bt[(kq * 16 + u * 4 + t) * 8 + j] =
                                    (bfloat16_t)((float)((p >> (8 * t)) & 255u) * s + bb);
                        }
            """,
    ]

    /// The kernel body, in a one-tile and a two-tile form.
    ///
    /// Both read and dequantise the weights exactly once; the second tile only adds another
    /// `simdgroup_multiply_accumulate` against the same `B`. That is the whole point - the
    /// weight traffic is what decode is bound by, so sixteen rows cost what eight do plus
    /// the arithmetic nobody notices.
    private static func source(tiles: Int) -> String {
        let secondTileMMA =
            tiles == 2
            ? """
                            simdgroup_load(A1, x + (size_t)8 * K + ka + kt * 8, K);
                            simdgroup_multiply_accumulate(C1, A1, B, C1);
                """
            : ""
        let secondTileDeclare =
            tiles == 2 ? "simdgroup_matrix<float, 8, 8> C1 = simdgroup_matrix<float, 8, 8>(0);" : ""
        let secondTileStore =
            tiles == 2 ? "simdgroup_store(C1, red + 512 + sg * 64, 8);" : ""
        let secondTileLoad = tiles == 2 ? "simdgroup_matrix<bfloat16_t, 8, 8> A1;" : ""

        return """
            const int K = KD, N = ND, M = MD;
            const int KPS = KD / 8;                 // K-span per simdgroup (split-K)

            uint tid  = thread_position_in_threadgroup.x;
            uint tgid = threadgroup_position_in_grid.x;
            uint sg   = tid >> 5;
            uint lane = tid & 31;

            int n0 = (int)tgid * 8;                 // one threadgroup -> 8 output columns

            // x is read straight from device into the MMA (bf16 in, fp32 accumulate): no
            // staging keeps threadgroup memory small and, more importantly, the critical
            // path short.
            threadgroup bfloat16_t bs[8 * 512];     // per-simdgroup 64k x 8n dequant stage
            threadgroup float red[8 * 64 * \(tiles)];   // cross-simdgroup reduction

            simdgroup_matrix<float, 8, 8> C0 = simdgroup_matrix<float, 8, 8>(0);
            \(secondTileDeclare)
            threadgroup bfloat16_t* bt = bs + sg * 512;

            // Split-K: the 8 simdgroups each walk 1/8 of K in short serial loops. (A prior
            // revision walked all of K per threadgroup and put ~160 barrier pairs on the
            // critical path.)
            int kbeg = (int)sg * KPS;
            for (int kk = 0; kk < KPS; kk += 64) {
                int ka = kbeg + kk;
                int j  = (int)(lane & 7);
                int kq = (int)(lane >> 3);
                int n  = n0 + j;
                if (n < N) {
                    // dequantize per quantization group (64 values), not per MMA tile (8):
                    // one scale/bias load serves the whole group and barriers drop 8x.
                    int g = ka >> 6;
                    float s  = (float)sc[(size_t)n * (K / 64) + g];
                    float bb = (float)bi[(size_t)n * (K / 64) + g];
        __UNPACK__
                } else {
                    for (int t = 0; t < 16; ++t) bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)0;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_matrix<bfloat16_t, 8, 8> A0, B;
                \(secondTileLoad)
                for (int kt = 0; kt < 8; ++kt) {
                    simdgroup_load(B, bt + kt * 64, 8);
                    simdgroup_load(A0, x + ka + kt * 8, K);   // x rows 0..7
                    simdgroup_multiply_accumulate(C0, A0, B, C0);
        \(secondTileMMA)
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            simdgroup_store(C0, red + sg * 64, 8);
            \(secondTileStore)
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // sum the 8 split-K partials and write out
            for (int i = (int)tid; i < 64 * \(tiles); i += 256) {
                int t = i >> 6;                     // which MMA tile
                int r = i & 63;                     // position inside it
                int m = t * 8 + (r >> 3);
                int j = r & 7;
                int n = n0 + j;
                if (m < M && n < N) {
                    float v = 0.0f;
                    for (int q = 0; q < 8; ++q) v += red[t * 512 + q * 64 + r];
                    out[(size_t)m * N + n] = (bfloat16_t)v;
                }
            }
        """
    }

    /// JIT compilation is per kernel object, so each (width, tile count) pair is built once
    /// and held: two quantisation widths by one or two tiles.
    private struct Variant: Hashable {
        let bits: Int
        let tiles: Int
    }

    private static let kernels: [Variant: MLX.MLXFast.MLXFastKernel] = {
        var built: [Variant: MLX.MLXFast.MLXFastKernel] = [:]
        for (bits, body) in unpack {
            for tiles in 1 ... 2 {
                built[Variant(bits: bits, tiles: tiles)] = MLX.MLXFast.metalKernel(
                    name: "dflash_qmm_mma_\(bits)_\(tiles)",
                    inputNames: ["x", "w", "sc", "bi"],
                    outputNames: ["out"],
                    source: source(tiles: tiles)
                        .replacingOccurrences(of: "__UNPACK__", with: body))
            }
        }
        return built
    }()

    /// Can this layer's shape and quantisation format run on the kernel at all?
    ///
    /// This is the static half of the test - it does not depend on the row count, so it
    /// is answered once when the model is swapped rather than per forward.
    static func isEligible(_ layer: QuantizedLinear) -> Bool {
        guard layer.mode == .affine, layer.biases != nil else { return false }
        guard unpack[layer.bits] != nil, layer.groupSize == requiredGroupSize else {
            return false
        }
        let (outputFeatures, inputFeatures) = layer.shape
        return outputFeatures >= minimumOutputFeatures
            && inputFeatures % inputFeatureMultiple == 0
    }

    /// Tiles needed to cover `rows`: one up to eight, two beyond. A narrow pass must not
    /// pay for a tile it does not fill.
    static func tiles(for rows: Int) -> Int {
        rows <= rowsPerTile ? 1 : 2
    }

    /// `x[tiles * 8, K] @ dequant(w)[K, N]`, keeping the first `rows` result rows.
    static func apply(
        x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        rows: Int, outputFeatures: Int, inputFeatures: Int, bits: Int
    ) -> MLXArray {
        guard let kernel = kernels[Variant(bits: bits, tiles: tiles(for: rows))] else {
            fatalError("no small-M quantised matmul kernel for \(bits)-bit weights")
        }
        return kernel(
            [x, weight, scales, biases],
            template: [("KD", inputFeatures), ("ND", outputFeatures), ("MD", rows)],
            grid: (((outputFeatures + 7) / 8) * threadgroupSize, 1, 1),
            threadGroup: (threadgroupSize, 1, 1),
            outputShapes: [[rows, outputFeatures]],
            outputDTypes: [.bfloat16])[0]
    }
}

/// A ``QuantizedLinear`` that routes verify-width forwards through
/// ``SmallMQuantizedMatmul`` and everything else through the stock op.
///
/// The MLX reference monkey-patches `nn.QuantizedLinear.__call__`; Swift has no such
/// hook, so the layer is subclassed and swapped into the module tree instead - which is
/// also narrower, since only layers that passed ``SmallMQuantizedMatmul/isEligible(_:)``
/// are ever replaced and the M=1 decode path pays one integer comparison.
public final class SmallMQuantizedLinear: QuantizedLinear {

    /// Whether this layer's shape and format are ones the kernel can handle at all,
    /// settled once here rather than re-derived per forward. ``enableSmallMQuantizedMatmul(in:)``
    /// only ever wraps eligible layers, but the initialiser is public and a caller who
    /// wraps an ineligible one must still get the stock result, not a wrong one.
    private let kernelEligible: Bool

    /// Wraps an existing layer, sharing its arrays - no weights are copied.
    public init(_ other: QuantizedLinear) {
        self.kernelEligible = SmallMQuantizedMatmul.isEligible(other)
        super.init(
            weight: other.weight, bias: other.bias,
            scales: other.scales, biases: other.biases,
            groupSize: other.groupSize, bits: other.bits, mode: other.mode)
        // The array-taking initialiser skips the freeze that the quantising one applies;
        // a swapped layer must stay as frozen as the layer it replaced.
        freeze(recursive: false)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let rows = shape.dropLast().reduce(1, *)
        // bfloat16 only: the kernel's MMA fragments and its output are bf16, so any other
        // input dtype would silently change this layer's output type.
        guard kernelEligible,
            rows >= SmallMQuantizedMatmul.minimumRows,
            rows <= SmallMQuantizedMatmul.maximumRows,
            x.dtype == .bfloat16,
            let biases
        else {
            return super.callAsFunction(x)
        }

        let inputFeatures = shape[shape.count - 1]
        let outputFeatures = weight.dim(0)
        var flat = x.reshaped(rows, inputFeatures)
        let padded =
            SmallMQuantizedMatmul.tiles(for: rows) * SmallMQuantizedMatmul.rowsPerTile
        if rows < padded {
            // The kernel reads whole MMA tiles from device; the padding rows are computed
            // and then dropped by the `m < M` guard on the store.
            flat = concatenated(
                [
                    flat,
                    MLXArray.zeros([padded - rows, inputFeatures], dtype: .bfloat16),
                ], axis: 0)
        }

        var y = SmallMQuantizedMatmul.apply(
            x: flat, weight: weight, scales: scales, biases: biases,
            rows: rows, outputFeatures: outputFeatures, inputFeatures: inputFeatures,
            bits: bits)
        y = y.reshaped(shape.dropLast() + [outputFeatures])
        if let bias {
            y = y + bias
        }
        return y
    }
}

/// Swaps every eligible ``QuantizedLinear`` in `model` for a ``SmallMQuantizedLinear``.
///
/// Opt-in by design: it changes the numerics of the swapped layers by a bf16 ulp or two
/// (the kernel accumulates in a different order), which is fine under a verify loop that
/// re-checks every token but is not something a plain `load` should decide for a caller.
///
/// - Returns: how many layers were swapped, so a caller can tell "nothing was eligible"
///   from "the kernel is on".
@discardableResult
public func enableSmallMQuantizedMatmul(in model: Module) -> Int {
    let updates = model
        .leafModules()
        .flattened()
        .compactMap { (path, module) -> (String, Module)? in
            guard !(module is SmallMQuantizedLinear),
                let quantized = module as? QuantizedLinear,
                SmallMQuantizedMatmul.isEligible(quantized)
            else { return nil }
            return (path, SmallMQuantizedLinear(quantized))
        }
    model.update(modules: ModuleChildren.unflattened(updates))
    return updates.count
}
