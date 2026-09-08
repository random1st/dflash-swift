// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "dflash-swift",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(name: "DFlashKit", targets: ["DFlashKit"]),
        .executable(name: "dflash-bench", targets: ["dflash-bench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.6")),
        // Apple's package plus two patches this drafter needs: hidden states from a
        // chosen ladder of target layers, and capture/rollback of a speculative round's
        // gated-delta recurrence. Both are proposed upstream; when they land this becomes
        // an ordinary versioned dependency on ml-explore/mlx-swift-lm.
        .package(url: "https://github.com/random1st/mlx-swift-lm", branch: "dflash-multilayer-tap"),
        // Only the bench executable needs the Hub client and tokenizers; DFlashKit
        // itself stays free of them so an embedder can bring its own.
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "DFlashKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "dflash-bench",
            dependencies: [
                "DFlashKit",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "DFlashKitTests",
            dependencies: ["DFlashKit"],
            resources: [.copy("Golden")]
        ),
    ]
)
