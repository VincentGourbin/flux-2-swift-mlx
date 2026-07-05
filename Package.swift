// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "Flux2Swift",
    platforms: [.macOS(.v26), .iOS(.v26)],
    products: [
        // Libraries
        .library(name: "FluxTextEncoders", targets: ["FluxTextEncoders"]),
        .library(name: "Flux2Core", targets: ["Flux2Core"]),
        // Main Application
        .executable(name: "Flux2App", targets: ["Flux2App"]),
    ],
    dependencies: [
        // Pinned to exactly 0.31.3. mlx-swift 0.31.4 carries upstream #410
        // (evalLock-during-toString deadlock / EINVAL fatal) that breaks
        // generation. A floor (.upToNextMajor) is not enough — SPM would still
        // resolve 0.31.4 as the highest in range — so pin exactly until an
        // upstream release fixes #410. ResumableAdamW is kept on the 0.31.3
        // (TupleState / no biasCorrection) API to match.
        .package(url: "https://github.com/ml-explore/mlx-swift", .exact("0.31.3")),
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMajor(from: "1.7.1")),
        // swift-tokenizers 0.7.1 carries upstream 0.6.3's "Fixes for Xcode build
        // with artifact bundle", which resolves the UniFFI module-map/linker
        // blocker that previously froze us at 0.5.x. 0.6.0+ also makes the
        // Tokenizer protocol typed-throwing (throws(TokenizerError)) and
        // relabels the encode/decode/tokenize convenience overloads.
        .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers", .upToNextMinor(from: "0.7.1")),
        .package(url: "https://github.com/intrusive-memory/SwiftTuberia.git", .upToNextMajor(from: "0.7.8")),
        .package(url: "https://github.com/intrusive-memory/SwiftAcervo", .upToNextMajor(from: "0.23.0")),
        .package(url: "https://github.com/marcprux/universal", .upToNextMajor(from: "5.3.0")),
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
                .product(name: "Tokenizers", package: "swift-tokenizers"),
                .product(name: "SwiftAcervo", package: "SwiftAcervo"),
            ]
        ),
        .target(
            name: "Flux2Core",
            dependencies: [
                "FluxTextEncoders",  // Internal dependency
                .product(name: "Tuberia", package: "SwiftTuberia"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "SwiftAcervo", package: "SwiftAcervo"),
            ]
        ),
        // MARK: - Main Application
        .executableTarget(
            name: "Flux2App",
            dependencies: ["FluxTextEncoders", "Flux2Core"]
        ),
        // MARK: - Test Helpers (regular target so test targets can depend on it)
        .target(
            name: "TestHelpers",
            dependencies: ["Flux2Core", "FluxTextEncoders"],
            path: "Tests/TestHelpers"
        ),
        // MARK: - Tests
        .testTarget(
            name: "FluxTextEncodersTests",
            dependencies: ["FluxTextEncoders", "TestHelpers"]
        ),
        .testTarget(
            name: "Flux2CoreTests",
            dependencies: ["Flux2Core", "TestHelpers"]
        ),
        .testTarget(
            name: "Flux2GPUTests",
            dependencies: ["Flux2Core", "FluxTextEncoders", "TestHelpers"],
            path: "Tests/Flux2GPUTests"
        ),
    ]
)
