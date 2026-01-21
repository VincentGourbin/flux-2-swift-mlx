// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Flux2Swift",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "Flux2Core", targets: ["Flux2Core"]),
        .executable(name: "Flux2CLI", targets: ["Flux2CLI"]),
        .executable(name: "Flux2App", targets: ["Flux2App"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.2"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // Flux2 Text Encoders (MistralCore + FluxTextEncoders) depuis GitHub
        .package(url: "https://github.com/VincentGourbin/flux2-text-encoders-swift-mlx", branch: "main"),
    ],
    targets: [
        .target(
            name: "Flux2Core",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "FluxTextEncoders", package: "flux2-text-encoders-swift-mlx"),
            ]
        ),
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
        .testTarget(
            name: "Flux2CoreTests",
            dependencies: ["Flux2Core"]
        ),
    ]
)
