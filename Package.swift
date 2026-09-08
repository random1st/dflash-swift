// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "dflash-swift",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(name: "DFlashKit", targets: ["DFlashKit"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.6")),
        // Local checkout: DFlash consumes hidden states from several target layers, which
        // needed a small patch to the Qwen3.5 backbone (branch dflash-multilayer-tap).
        // Point this back at the upstream URL once that patch is merged.
        .package(path: "../mlx-swift-lm"),
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
        .testTarget(
            name: "DFlashKitTests",
            dependencies: ["DFlashKit"],
            resources: [.copy("Golden")]
        ),
    ]
)
