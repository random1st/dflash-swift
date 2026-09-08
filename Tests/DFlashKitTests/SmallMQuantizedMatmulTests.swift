//
//  SmallMQuantizedMatmulTests.swift
//  DFlashKitTests
//
//  The small-M kernel replaces a stock op inside the verify pass, so the thing worth
//  pinning is numerical: a kernel that indexes its packed weights wrongly still returns
//  a plausible tensor, and the damage would show up much later as an acceptance rate
//  that quietly sagged - never as an error.
//
//  It is NOT bit-identical to `quantizedMatmul`: it accumulates fp32 partials in a
//  different order (split-K across 8 simdgroups, one dequantised group reused by all
//  rows). So these tests assert closeness, at the tolerance the MLX reference uses:
//  max|kernel - stock| <= 0.02 * max(max|stock|, 1). bfloat16 carries an 8-bit
//  significand, so one ulp is ~2^-8 = 0.004 relative; the reference measured this kernel
//  at 1-2 ulps (~0.007 relative) on real weights, and 0.02 is ~5 ulps - headroom over
//  reordering noise, far below anything a mis-indexed unpack would produce.
//
//  `testKernelOutputIsNotBitIdentical` is the guard that keeps the above honest: if the
//  swap silently fell through to the stock op, every closeness test would pass for the
//  wrong reason.
//

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import DFlashKit

final class SmallMQuantizedMatmulTests: XCTestCase {
    /// The kernel's smallest legal shape: N at the 4096 floor, K a multiple of 512.
    private let outputFeatures = 4096
    private let inputFeatures = 1024
    private let tolerance: Float = 0.02

    /// A module with one replaceable child, so the swap can be exercised through the
    /// real `update(modules:)` path rather than by constructing the subclass directly.
    private final class Holder: Module {
        @ModuleInfo var projection: Linear
        init(_ layer: Linear) {
            self._projection.wrappedValue = layer
            super.init()
        }
    }

    private func makeLayer(
        bits: Int, outputFeatures: Int? = nil, inputFeatures: Int? = nil, groupSize: Int = 64
    ) -> QuantizedLinear {
        let weight = MLXRandom.normal(
            [outputFeatures ?? self.outputFeatures, inputFeatures ?? self.inputFeatures],
            scale: 0.05, key: MLXRandom.key(7)
        ).asType(.bfloat16)
        return QuantizedLinear(weight: weight, bias: nil, groupSize: groupSize, bits: bits)
    }

    private func input(rows: Int, inputFeatures: Int? = nil) -> MLXArray {
        MLXRandom.normal(
            [rows, inputFeatures ?? self.inputFeatures], scale: 0.1, key: MLXRandom.key(11)
        ).asType(.bfloat16)
    }

    // MARK: - Numerics

    private func assertAgrees(bits: Int, rows: Int, file: StaticString = #filePath, line: UInt = #line) {
        let stock = makeLayer(bits: bits)
        let fast = SmallMQuantizedLinear(stock)
        let x = input(rows: rows)

        let reference = stock(x).asType(.float32)
        let kernel = fast(x).asType(.float32)
        XCTAssertEqual(kernel.shape, [rows, outputFeatures], file: file, line: line)

        let difference = MLX.abs(reference - kernel).max().item(Float.self)
        let magnitude = MLX.abs(reference).max().item(Float.self)
        XCTAssertLessThanOrEqual(
            difference, tolerance * Swift.max(magnitude, 1),
            "\(bits)-bit, M=\(rows): max|kernel - stock| = \(difference) at max|stock| = \(magnitude)",
            file: file, line: line)
    }

    func test4BitMatchesQuantizedMatmul() {
        for rows in SmallMQuantizedMatmul.minimumRows ... SmallMQuantizedMatmul.maximumRows {
            assertAgrees(bits: 4, rows: rows)
        }
    }

    func test8BitMatchesQuantizedMatmul() {
        for rows in SmallMQuantizedMatmul.minimumRows ... SmallMQuantizedMatmul.maximumRows {
            assertAgrees(bits: 8, rows: rows)
        }
    }

    func testKernelOutputIsNotBitIdentical() {
        // Proof that the kernel actually ran: the reordered accumulation has to move at
        // least one of the 8 x 4096 outputs by an ulp. Without this, a dispatch bug that
        // silently fell back to `quantizedMatmul` would pass every test above.
        let stock = makeLayer(bits: 4)
        let fast = SmallMQuantizedLinear(stock)
        let x = input(rows: 8)
        XCTAssertTrue((stock(x) .!= fast(x)).any().item(Bool.self))
    }

    // MARK: - Eligibility

    func testEligibleShapeIsSwapped() {
        let model = Holder(makeLayer(bits: 4))
        XCTAssertEqual(enableSmallMQuantizedMatmul(in: model), 1)
        XCTAssertTrue(model.projection is SmallMQuantizedLinear)
    }

    func testSwapIsIdempotent() {
        let model = Holder(makeLayer(bits: 4))
        XCTAssertEqual(enableSmallMQuantizedMatmul(in: model), 1)
        // Re-running must not wrap the wrapper: the count is what a caller reports.
        XCTAssertEqual(enableSmallMQuantizedMatmul(in: model), 0)
    }

    func testIneligibleShapesAreLeftAlone() {
        // Each of these violates exactly one bound, so a loosened gate names itself.
        let cases: [(String, QuantizedLinear)] = [
            ("N below 4096", makeLayer(bits: 4, outputFeatures: 2048)),
            ("K not a multiple of 512", makeLayer(bits: 4, inputFeatures: 1024 + 64)),
            ("group size 32", makeLayer(bits: 4, groupSize: 32)),
            ("unsupported bit width", makeLayer(bits: 3)),
        ]
        for (reason, layer) in cases {
            XCTAssertFalse(SmallMQuantizedMatmul.isEligible(layer), reason)
            let model = Holder(layer)
            XCTAssertEqual(enableSmallMQuantizedMatmul(in: model), 0, reason)
            XCTAssertFalse(model.projection is SmallMQuantizedLinear, reason)
        }
    }

    // MARK: - Negative controls

    func testIneligibleShapeReturnsTheStockResult() {
        // The negative control for the whole dispatch: an ineligible shape must come
        // back bit-identical, so a kernel that ran where it should not cannot pass.
        let stock = makeLayer(bits: 4, outputFeatures: 2048)
        let fast = SmallMQuantizedLinear(stock)
        let x = input(rows: 8)
        XCTAssertTrue((stock(x) .== fast(x)).all().item(Bool.self))
    }

    func testRowCountsOutsideTheWindowReturnTheStockResult() {
        // Below M=6 the stock GEMV path wins and above M=8 the single MMA tile cannot
        // hold the rows, so both must fall through unchanged - bit for bit.
        let stock = makeLayer(bits: 4)
        let fast = SmallMQuantizedLinear(stock)
        for rows in [1, 3, 5, 9, 16] {
            let x = input(rows: rows)
            XCTAssertTrue((stock(x) .== fast(x)).all().item(Bool.self), "M=\(rows)")
        }
    }

    func testFloat16InputFallsThroughToStock() {
        // The kernel's fragments and its output are bf16, so a float16 forward has to fall
        // through. What it must keep is whatever the stock op does - which for float16
        // against bfloat16 weights is a float32 result, not a float16 one; asserting the
        // input's dtype here would be asserting the wrong contract.
        let stock = makeLayer(bits: 4)
        let fast = SmallMQuantizedLinear(stock)
        let x = input(rows: 8).asType(.float16)
        let reference = stock(x)
        let y = fast(x)
        XCTAssertEqual(y.dtype, reference.dtype)
        XCTAssertTrue((reference .== y).all().item(Bool.self))
    }
}
