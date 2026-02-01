// AspectRatioBucket.swift - Aspect ratio bucketing for multi-resolution training
// Copyright 2025 Vincent Gourbin

import Foundation

/// A resolution bucket defined by width and height
public struct ResolutionBucket: Hashable, Sendable {
    public let width: Int
    public let height: Int

    public var aspectRatio: Float {
        Float(width) / Float(height)
    }

    public var totalPixels: Int {
        width * height
    }

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    public var description: String {
        "\(width)x\(height)"
    }
}

/// Manages aspect ratio bucketing for multi-resolution training
public final class AspectRatioBucketManager: @unchecked Sendable {

    /// Standard aspect ratios to support (width:height)
    /// These cover common photo and artistic formats
    public static let standardAspectRatios: [(Float, String)] = [
        (1.0, "1:1"),       // Square
        (4.0/3.0, "4:3"),   // Standard photo
        (3.0/4.0, "3:4"),   // Portrait photo
        (16.0/9.0, "16:9"), // Widescreen
        (9.0/16.0, "9:16"), // Vertical video
        (3.0/2.0, "3:2"),   // Classic 35mm
        (2.0/3.0, "2:3"),   // Portrait 35mm
        (21.0/9.0, "21:9"), // Ultrawide
        (9.0/21.0, "9:21"), // Ultra-tall
    ]

    /// Available base resolutions
    public let baseResolutions: [Int]

    /// Generated buckets (resolution -> bucket)
    public private(set) var buckets: [ResolutionBucket] = []

    /// Bucket assignments (filename -> bucket)
    private var assignments: [String: ResolutionBucket] = [:]

    /// Samples grouped by bucket
    private var bucketedSamples: [ResolutionBucket: [(filename: String, caption: String)]] = [:]

    /// Initialize bucket manager with available resolutions
    /// - Parameter resolutions: Base resolutions to use (e.g., [512, 768, 1024])
    public init(resolutions: [Int]) {
        self.baseResolutions = resolutions.sorted()
        generateBuckets()
    }

    /// Generate all possible buckets from base resolutions and aspect ratios
    private func generateBuckets() {
        var generatedBuckets: Set<ResolutionBucket> = []

        for baseRes in baseResolutions {
            for (aspectRatio, _) in Self.standardAspectRatios {
                // Calculate dimensions maintaining approximate pixel count
                let targetPixels = baseRes * baseRes
                let height = Int(sqrt(Float(targetPixels) / aspectRatio))
                let width = Int(Float(height) * aspectRatio)

                // Round to nearest multiple of 64 (required for VAE)
                let roundedWidth = ((width + 32) / 64) * 64
                let roundedHeight = ((height + 32) / 64) * 64

                let bucket = ResolutionBucket(width: roundedWidth, height: roundedHeight)
                generatedBuckets.insert(bucket)
            }
        }

        buckets = Array(generatedBuckets).sorted { $0.totalPixels < $1.totalPixels }

        Flux2Debug.log("[Bucketing] Generated \(buckets.count) resolution buckets:")
        for bucket in buckets.prefix(10) {
            Flux2Debug.log("  - \(bucket.description) (AR: \(String(format: "%.2f", bucket.aspectRatio)))")
        }
        if buckets.count > 10 {
            Flux2Debug.log("  ... and \(buckets.count - 10) more")
        }
    }

    /// Find the best matching bucket for an image
    /// - Parameters:
    ///   - width: Original image width
    ///   - height: Original image height
    /// - Returns: Best matching resolution bucket
    public func findBestBucket(width: Int, height: Int) -> ResolutionBucket {
        let imageAspectRatio = Float(width) / Float(height)

        // Find bucket with closest aspect ratio that doesn't exceed max resolution
        var bestBucket = buckets.first!
        var bestScore = Float.infinity

        for bucket in buckets {
            // Aspect ratio difference (weighted heavily)
            let arDiff = abs(bucket.aspectRatio - imageAspectRatio)

            // Prefer buckets that don't require too much upscaling
            let scaleFactor = max(Float(bucket.width) / Float(width),
                                  Float(bucket.height) / Float(height))
            let upscalePenalty = scaleFactor > 1.5 ? (scaleFactor - 1.5) * 0.5 : 0

            let score = arDiff + upscalePenalty

            if score < bestScore {
                bestScore = score
                bestBucket = bucket
            }
        }

        return bestBucket
    }

    /// Assign a sample to its best bucket
    /// - Parameters:
    ///   - filename: Image filename
    ///   - caption: Image caption
    ///   - originalWidth: Original image width
    ///   - originalHeight: Original image height
    public func assignSample(
        filename: String,
        caption: String,
        originalWidth: Int,
        originalHeight: Int
    ) {
        let bucket = findBestBucket(width: originalWidth, height: originalHeight)
        assignments[filename] = bucket

        if bucketedSamples[bucket] == nil {
            bucketedSamples[bucket] = []
        }
        bucketedSamples[bucket]!.append((filename: filename, caption: caption))
    }

    /// Get the assigned bucket for a filename
    public func getBucket(for filename: String) -> ResolutionBucket? {
        assignments[filename]
    }

    /// Get all samples in a specific bucket
    public func getSamples(in bucket: ResolutionBucket) -> [(filename: String, caption: String)] {
        bucketedSamples[bucket] ?? []
    }

    /// Get all non-empty buckets
    public func getNonEmptyBuckets() -> [ResolutionBucket] {
        buckets.filter { (bucketedSamples[$0]?.count ?? 0) > 0 }
    }

    /// Get bucket statistics
    public var statistics: String {
        let nonEmpty = getNonEmptyBuckets()
        var stats = "Bucket Statistics:\n"
        stats += "  Total buckets: \(buckets.count)\n"
        stats += "  Non-empty buckets: \(nonEmpty.count)\n"

        for bucket in nonEmpty {
            let count = bucketedSamples[bucket]?.count ?? 0
            stats += "  - \(bucket.description): \(count) samples\n"
        }

        return stats
    }

    /// Clear all assignments
    public func clear() {
        assignments.removeAll()
        bucketedSamples.removeAll()
    }
}
