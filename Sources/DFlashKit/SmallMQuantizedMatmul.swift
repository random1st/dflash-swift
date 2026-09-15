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
/// so every quantised weight group is read ONCE and reused by all rows; K is split across
/// the 8 simdgroups of a threadgroup so each serial loop is short.
///
/// The weights are never dequantised. Each lane keeps its column's 64-value group in
/// registers and feeds the RAW quantised integers into the MMA as bf16 - which is exact,
/// since a 4- or 8-bit integer fits bf16's 8 significant bits - accumulating
/// `T = sum(q * x)` per group. The affine map is applied once per group on the fragment:
/// `C += s * T + b * sum(x)`, with `sum(x)` taken from the B fragment as it goes past.
/// For 4-bit the integer is built by OR-ing the nibble into the bf16 bit pattern of 128
/// (`0x4300 | q` = 128 + q), so the bias term becomes `b - 128 s`; one OR instead of a
/// convert, multiply and add per weight. No threadgroup staging, no barriers in the K loop.
/// Measured on an isolated chain of the 27B's six projection shapes (M=8, vs the M=1
/// weight-streaming floor of 43.6 ms): stock 151 ms, the earlier dequant-to-threadgroup
/// version 90-94 ms, this 55.5 ms. The MMA work itself is ~34 ms at this GPU's bf16 peak,
/// so at eight rows the kernel is close to compute-bound, not memory-bound; that is why
/// the register form had to drop every instruction it could.
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
    /// The group size the kernel's K loop hard-codes (one scale/bias per 64).
    static let requiredGroupSize = 64

    private static let threadgroupSize = 256  // 8 simdgroups

    /// Output columns per threadgroup. Four MMA column tiles share one load of the x
    /// fragment. Measured on the chain: 16 columns 60.6 ms (x traffic re-paid), 32 columns
    /// 55.5 ms, 48 columns 77.7 ms and 64 columns 99.6 ms (the per-lane weight registers
    /// spill to the stack). Thirty-two is the knee.
    private static let columnTiles = 4
    private static let columnsPerThreadgroup = 32

    /// How a quantisation width lays its 64-value group out in memory and how a lane turns
    /// its two packed values into MMA operands. Everything else is shared.
    private struct Width {
        /// 16-byte words per column per group: 4-bit packs 64 nibbles into 2, 8-bit 64
        /// bytes into 4. A lane loads them all - contiguous, one `uint4` per instruction.
        let uint4PerGroup: Int
        /// The weight row stride and the group offset, in `uint`s.
        let rowStride: String
        let groupOffset: String
        /// Bit position of the lane's first value inside its `uint`.
        let firstShift: String
        let shiftStep: Int
        /// The `uint` holding values `kt * 8 + fn` and `fn + 1` of column tile `c`, given
        /// the words `q<c>0 ... q<c>k` (each a `uint4`).
        let word: (_ c: Int, _ kt: Int) -> String
        /// The value's bf16 encoding as an MMA operand, exact for the full range.
        let element: (_ shift: String) -> String
        /// What that encoding adds to the true value (0 or 128), folded into the bias.
        let offset: Int
    }

    private static let widths: [Int: Width] = [
        // A nibble OR-ed into bf16 128's bit pattern is exactly 128 + q. The lane's
        // values are at nibbles fn, fn+1 of uint kt.
        4: Width(
            uint4PerGroup: 2, rowStride: "(K / 8)", groupOffset: "(ka >> 3)",
            firstShift: "4u * (uint)fn", shiftStep: 4,
            word: { c, kt in "q\(c)\(kt / 4)[\(kt % 4)]" },
            element: { "as_type<bfloat16_t>((ushort)(0x4300u | ((p >> \($0)) & 15u)))" },
            offset: 128),
        // A byte converts to bf16 exactly (8 significant bits). Values kt*8+fn, +1 sit in
        // bytes fn&3, fn&3 + 1 of uint 2*kt + (fn >> 2); the two candidate words are
        // constant-indexed registers and the pick is one select.
        8: Width(
            uint4PerGroup: 4, rowStride: "(K / 4)", groupOffset: "(ka >> 2)",
            firstShift: "8u * (uint)(fn & 3)", shiftStep: 8,
            word: { c, kt in
                "((fn >> 2) ? q\(c)\((2 * kt + 1) / 4)[\((2 * kt + 1) % 4)] : q\(c)\((2 * kt) / 4)[\((2 * kt) % 4)])"
            },
            element: { "(bfloat16_t)(float)((p >> \($0)) & 255u)" },
            offset: 0),
    ]

    /// The kernel body for one quantisation width and one or two row tiles.
    ///
    /// The weight tile is the MMA's A operand (rows = output columns, cols = k) and x is
    /// loaded transposed as B, so the accumulator is `C^T[n][m]`; that is what lets each
    /// lane's A fragment be the two adjacent packed values it already holds. A second row
    /// tile is another transposed load of x rows 8..15 and another MMA against the same A -
    /// the weight traffic is what decode is bound by, so sixteen rows cost what eight do
    /// plus arithmetic.
    ///
    /// The kt loop is unrolled in the generator rather than by pragma so every register
    /// index is a literal: an indexed register array spills to the stack, and that alone
    /// was 88 ms against 62 ms on the chain.
    private static func source(rowTiles: Int, width: Width) -> String {
        let cs = 0 ..< columnTiles
        let rs = 0 ..< rowTiles
        let ws = 0 ..< width.uint4PerGroup
        func lines(_ parts: [String]) -> String { parts.joined(separator: "\n") }

        let accumulators = lines(cs.flatMap { c in
            rs.map { r in "    simdgroup_matrix<float, 8, 8> C\(c)_\(r) = simdgroup_matrix<float, 8, 8>(0);" }
        })
        let groupRegisters = lines(cs.map { c in
            "    uint4 " + ws.map { "q\(c)\($0)" }.joined(separator: ", ") + "; float s\(c), bb\(c);"
        })
        let groupLoads = lines(cs.map { c in
            """
                    {
                        int n = n0 + \(c) * 8 + fm;
                        if (n < N) {
                            const device uint4* wr = reinterpret_cast<const device uint4*>(
                                w + (size_t)n * \(width.rowStride) + \(width.groupOffset));
            \(lines(ws.map { "                q\(c)\($0) = wr[\($0)];" }))
                            s\(c)  = (float)sc[(size_t)n * (K / 64) + (ka >> 6)];
                            bb\(c) = (float)bi[(size_t)n * (K / 64) + (ka >> 6)];
                        } else {
            \(lines(ws.map { "                q\(c)\($0) = uint4(0);" }))
                            s\(c) = 0.0f; bb\(c) = 0.0f;
                        }
                    }
            """
        })
        let groupAccumulators = lines(cs.flatMap { c in
            rs.map { r in "        simdgroup_matrix<float, 8, 8> T\(c)_\(r) = simdgroup_matrix<float, 8, 8>(0);" }
        })
        let xSums = lines(rs.map { r in "        float xa0_\(r) = 0.0f, xa1_\(r) = 0.0f;" })
        let bDeclare = rs.map { "B\($0)" }.joined(separator: ", ")

        // One MMA step: 8 values of k for every row tile and column tile.
        let steps = lines((0 ..< 8).map { kt in
            let bLoads = lines(rs.map { r in
                """
                        simdgroup_load(B\(r), x + (size_t)\(r) * 8 * K + ka + \(kt * 8), K, ulong2(0, 0), true);
                        {
                            thread vec<bfloat16_t, 2>& be = reinterpret_cast<thread vec<bfloat16_t, 2>&>(B\(r).thread_elements());
                            xa0_\(r) += (float)be[0];
                            xa1_\(r) += (float)be[1];
                        }
                """
            })
            let mmas = lines(cs.map { c in
                """
                        {
                            uint p = \(width.word(c, kt));
                            thread vec<bfloat16_t, 2>& e = reinterpret_cast<thread vec<bfloat16_t, 2>&>(A.thread_elements());
                            e[0] = \(width.element("sh0"));
                            e[1] = \(width.element("sh1"));
                \(lines(rs.map { r in "            simdgroup_multiply_accumulate(T\(c)_\(r), A, B\(r), T\(c)_\(r));" }))
                        }
                """
            })
            return bLoads + "\n" + mmas
        })
        let xSumReduce = lines(rs.map { r in
            """
                    xa0_\(r) += simd_shuffle_xor(xa0_\(r), 2); xa0_\(r) += simd_shuffle_xor(xa0_\(r), 4); xa0_\(r) += simd_shuffle_xor(xa0_\(r), 16);
                    xa1_\(r) += simd_shuffle_xor(xa1_\(r), 2); xa1_\(r) += simd_shuffle_xor(xa1_\(r), 4); xa1_\(r) += simd_shuffle_xor(xa1_\(r), 16);
            """
        })
        let applyAffine = lines(cs.flatMap { c in
            rs.map { r in
                """
                        {
                            thread vec<float, 2>& ce = reinterpret_cast<thread vec<float, 2>&>(C\(c)_\(r).thread_elements());
                            thread vec<float, 2>& te = reinterpret_cast<thread vec<float, 2>&>(T\(c)_\(r).thread_elements());
                            float b2 = bb\(c) - \(width.offset).0f * s\(c);
                            ce[0] += s\(c) * te[0] + b2 * xa0_\(r);
                            ce[1] += s\(c) * te[1] + b2 * xa1_\(r);
                        }
                """
            }
        })
        let stores = lines(cs.flatMap { c in
            rs.map { r in
                "    simdgroup_store(C\(c)_\(r), red + ((sg * \(columnTiles) + \(c)) * \(rowTiles) + \(r)) * 64, 8);"
            }
        })

        return """
            const int K = KD, N = ND, M = MD;
            const int KPS = KD / 8;                 // K-span per simdgroup (split-K)

            uint tid  = thread_position_in_threadgroup.x;
            uint tgid = threadgroup_position_in_grid.x;
            uint sg   = tid >> 5;
            uint lane = tid & 31;
            int n0 = (int)tgid * \(columnsPerThreadgroup);

            // The lane's position in an 8x8 fragment (MLX steel's layout): row fm, columns
            // fn and fn + 1. For A that is weight column n0 + 8c + fm at k = kt * 8 + fn.
            int qid = (int)lane / 4;
            int fm = (qid & 4) + (((int)lane / 2) % 4);
            int fn = (qid & 2) * 2 + ((int)lane % 2) * 2;
            uint sh0 = \(width.firstShift), sh1 = sh0 + \(width.shiftStep)u;

            threadgroup float red[8 * \(columnTiles) * \(rowTiles) * 64];   // cross-simdgroup reduction
        \(accumulators)
        \(groupRegisters)

            // Split-K: the 8 simdgroups each walk 1/8 of K in short serial loops, one
            // 64-value quantisation group per iteration, and never synchronise inside it.
            int kbeg = (int)sg * KPS;
            for (int kk = 0; kk < KPS; kk += 64) {
                int ka = kbeg + kk;
        \(groupLoads)
        \(groupAccumulators)
        \(xSums)
                simdgroup_matrix<bfloat16_t, 8, 8> A, \(bDeclare);
        \(steps)
                // sum(x) over the group for the lane's two m columns: lanes sharing fn
                // differ in bits 1, 2 and 4 of the lane index.
        \(xSumReduce)
        \(applyAffine)
            }

        \(stores)
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // sum the 8 split-K partials and write out; the fragments are C^T, so
            // position r inside a tile is (column r >> 3, row r & 7).
            for (int i = (int)tid; i < \(columnTiles * rowTiles) * 64; i += \(threadgroupSize)) {
                int t = i >> 6;
                int c = t / \(rowTiles);
                int rt = t % \(rowTiles);
                int r = i & 63;
                int m = rt * 8 + (r & 7);
                int n = n0 + c * 8 + (r >> 3);
                if (m < M && n < N) {
                    float v = 0.0f;
                    for (int q = 0; q < 8; ++q) v += red[((q * \(columnTiles) + c) * \(rowTiles) + rt) * 64 + r];
                    out[(size_t)m * N + n] = (bfloat16_t)v;
                }
            }
        """
    }

    /// JIT compilation is per kernel object, so each (width, tile count) pair is built once
    /// and held: two quantisation widths by one or two row tiles.
    private struct Variant: Hashable {
        let bits: Int
        let tiles: Int
    }

    private static let kernels: [Variant: MLX.MLXFast.MLXFastKernel] = {
        var built: [Variant: MLX.MLXFast.MLXFastKernel] = [:]
        for (bits, width) in widths {
            for tiles in 1 ... 2 {
                built[Variant(bits: bits, tiles: tiles)] = MLX.MLXFast.metalKernel(
                    name: "dflash_qmm_fold_\(bits)_\(tiles)",
                    inputNames: ["x", "w", "sc", "bi"],
                    outputNames: ["out"],
                    source: source(rowTiles: tiles, width: width))
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
        guard widths[layer.bits] != nil, layer.groupSize == requiredGroupSize else {
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
            grid: (
                ((outputFeatures + columnsPerThreadgroup - 1) / columnsPerThreadgroup)
                    * threadgroupSize, 1, 1
            ),
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
