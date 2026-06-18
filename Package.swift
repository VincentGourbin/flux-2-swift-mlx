// swift-tools-version: 6.2

import Foundation
import PackageDescription

// In CI we always pin to released remotes. Locally, prefer a sibling checkout
// at ../<name> if present so in-flight changes can be exercised end-to-end
// without publishing a release. Falls back to the remote pin if the sibling
// directory is missing, so fresh clones still build.
//
// When this manifest is evaluated as a transitive dependency inside Xcode's
// `SourcePackages/checkouts/` or SwiftPM's `.build/checkouts/`, every other
// dependency lives as a sibling in the same directory. Treating those as
// in-development local paths produces conflicting package identities, so we
// must skip the sibling shortcut in that context.
let manifestDir = (#filePath as NSString).deletingLastPathComponent
let isSPMCheckout =
  manifestDir.contains("/SourcePackages/checkouts/")
  || manifestDir.contains("/.build/checkouts/")
let isCI = ProcessInfo.processInfo.environment["CI"] == "true"
let useLocalSiblings = !isCI && !isSPMCheckout

func sibling(_ name: String, remote: String, from version: Version) -> Package.Dependency {
  let localPath = "../\(name)"
  if useLocalSiblings && FileManager.default.fileExists(atPath: localPath) {
    return .package(path: localPath)
  }
  return .package(url: remote, .upToNextMajor(from: version))
}

/// Same sibling-priority pattern as ``sibling(_:remote:from:)`` but pins to a
/// remote branch when no local sibling exists. Use only when a temporary
/// pre-release dependency on a feature branch is required; switch back to the
/// version-pinned ``sibling(_:remote:from:)`` once the upstream tags a release.
func sibling(_ name: String, remote: String, branch: String) -> Package.Dependency {
  let localPath = "../\(name)"
  if useLocalSiblings && FileManager.default.fileExists(atPath: localPath) {
    return .package(path: localPath)
  }
  return .package(url: remote, branch: branch)
}

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
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMajor(from: "0.31.4")),
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMajor(from: "1.7.1")),
        // swift-tokenizers 0.7.1 carries upstream 0.6.3's "Fixes for Xcode build
        // with artifact bundle", which resolves the UniFFI module-map/linker
        // blocker that previously froze us at 0.5.x. 0.6.0+ also makes the
        // Tokenizer protocol typed-throwing (throws(TokenizerError)) and
        // relabels the encode/decode/tokenize convenience overloads.
        .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers", .upToNextMinor(from: "0.7.1")),
        sibling(
          "SwiftTuberia",
          remote: "https://github.com/intrusive-memory/SwiftTuberia.git",
          from: "0.7.5"),
        sibling("SwiftAcervo", remote: "https://github.com/intrusive-memory/SwiftAcervo", from: "0.19.2"),
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
