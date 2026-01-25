// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Flux2Swift",
    platforms: [.macOS(.v14)],
    products: [
        // Libraries
        .library(name: "FluxTextEncoders", targets: ["FluxTextEncoders"]),
        .library(name: "Flux2Core", targets: ["Flux2Core"]),
        // Executables - Text Encoders
        .executable(name: "FluxEncodersCLI", targets: ["FluxEncodersCLI"]),
        .executable(name: "FluxEncodersApp", targets: ["FluxEncodersApp"]),
        // Executables - Image Generation
        .executable(name: "Flux2CLI", targets: ["Flux2CLI"]),
        .executable(name: "Flux2App", targets: ["Flux2App"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.2"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
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
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        // MARK: - Text Encoder Executables
        .executableTarget(
            name: "FluxEncodersCLI",
            dependencies: [
                "FluxTextEncoders",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "FluxEncodersApp",
            dependencies: ["FluxTextEncoders"]
        ),
        // MARK: - Image Generation Executables
        .executableTarget(
            name: "Flux2CLI",
            dependencies: [
                "Flux2Core",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "Flux2App",
            dependencies: ["Flux2Core"]
        ),
        // MARK: - Tests
        .testTarget(
            name: "FluxTextEncodersTests",
            dependencies: ["FluxTextEncoders"]
        ),
        .testTarget(
            name: "Flux2CoreTests",
            dependencies: ["Flux2Core"]
        ),
    ]
)
