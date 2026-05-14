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
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMajor(from: "0.31.3")),
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMajor(from: "1.7.1")),
        // Pinned to 0.5.x. swift-tokenizers 0.6.0 switched to a UniFFI-based
        // Rust artifactbundle that has known Xcode module-map / compile
        // issues (the 0.6.2 tag ships an explicit "Temporary fix for Xcode
        // builds" commit, 37f999a, the maintainer flagged as a possible
        // Xcode bug). Wait for a stable 0.6.x release without these Xcode
        // compile issues before bumping past 0.5.x.
        .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers", .upToNextMinor(from: "0.5.0")),
        .package(url: "https://github.com/intrusive-memory/SwiftTuberia.git", .upToNextMajor(from: "0.7.0")),
        .package(url: "https://github.com/intrusive-memory/SwiftAcervo", .upToNextMajor(from: "0.13.0")),
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
