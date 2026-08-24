// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Flux2Swift",
    platforms: [.macOS(.v15)],
    products: [
        // Libraries
        .library(name: "FluxTextEncoders", targets: ["FluxTextEncoders"]),
        .library(name: "Flux2Core", targets: ["Flux2Core"]),
        .library(name: "Flux2Chains", targets: ["Flux2Chains"]),
        // Opt-in: link this to enrich prompts / score checkpoints with Gemma 4 E2B
        // instead of the bundled Qwen3.5. Nothing else pulls it in.
        .library(name: "FluxGemma4VLM", targets: ["FluxGemma4VLM"]),
        // CLI Tools
        .executable(name: "FluxEncodersCLI", targets: ["FluxEncodersCLI"]),
        .executable(name: "Flux2CLI", targets: ["Flux2CLI"]),
        // Main Application
        .executable(name: "Flux2App", targets: ["Flux2App"]),
    ],
    dependencies: [
        // Pinned exactly: mlx-swift introduces breaking API changes even in patch
        // releases (e.g. AdamW TupleState -> AdamState in 0.31.4). Bump deliberately.
        // 0.31.6 verified against the full Metal test suite + a generation run;
        // 0.31.5+ requires a Swift 6.3 toolchain (Xcode 26+).
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        .package(url: "https://github.com/jpsim/Yams", from: "6.0.0"),
        .package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.4.0"),
        // Optional VLM provider (target `FluxGemma4VLM` only): Gemma 4 E2B as an
        // alternative to the bundled Qwen3.5 for prompt enrichment / scoring.
        // Its mlx-swift range is 0.31.4..<0.32, compatible with the exact pin above;
        // it brings mlx-swift-lm in, which is why it stays out of the base targets.
        .package(url: "https://github.com/VincentGourbin/gemma-4-swift-mlx", from: "1.5.0"),
    ],
    targets: [
        // MARK: - Libraries
        .target(
            name: "FluxTextEncoders",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
            ]
        ),
        .target(
            name: "Flux2Core",
            dependencies: [
                "FluxTextEncoders",  // Internal dependency
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
            ]
        ),
        .target(
            name: "Flux2Chains",
            dependencies: [
                "Flux2Core",
                "FluxTextEncoders",  // Qwen3.5 VLM service for opt-in prompt enrichment
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "FluxGemma4VLM",
            dependencies: [
                "FluxTextEncoders",  // FluxVLMProvider protocol + registry
                .product(name: "Gemma4Swift", package: "gemma-4-swift-mlx"),
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        // MARK: - CLI Tools
        .executableTarget(
            name: "FluxEncodersCLI",
            dependencies: [
                "FluxTextEncoders",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "Flux2CLI",
            dependencies: [
                "Flux2Core",
                "Flux2Chains",
                "FluxGemma4VLM",  // --vlm-provider gemma4
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Yams", package: "Yams"),
            ]
        ),
        // MARK: - Main Application
        .executableTarget(
            name: "Flux2App",
            dependencies: ["FluxTextEncoders", "Flux2Core"]
        ),
        // MARK: - Tests
        .testTarget(
            name: "FluxTextEncodersTests",
            dependencies: ["FluxTextEncoders"]
        ),
        .testTarget(
            name: "FluxGemma4VLMTests",
            dependencies: ["FluxGemma4VLM", "FluxTextEncoders"]
        ),
        .testTarget(
            name: "Flux2CoreTests",
            dependencies: ["Flux2Core"]
        ),
        .testTarget(
            name: "Flux2ChainsTests",
            dependencies: ["Flux2Chains", "Flux2Core"]
        ),
    ]
)
