// TrainingDataset.swift - Training dataset loading and batching
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// A single training sample
public struct TrainingSample: @unchecked Sendable {
    /// Image filename (for debugging/logging)
    public let filename: String
    
    /// Image data as MLXArray [H, W, C] in range [0, 1]
    public let image: MLXArray
    
    /// Caption text
    public let caption: String
    
    /// Original image size before resizing
    public let originalSize: (width: Int, height: Int)
}

/// A batch of training samples
public struct TrainingBatch: @unchecked Sendable {
    /// Batch of images [B, H, W, C]
    public let images: MLXArray

    /// Captions for each image in batch
    public let captions: [String]

    /// Filenames for logging
    public let filenames: [String]

    /// Target resolution for this batch (all images same size when bucketing)
    public let resolution: ResolutionBucket?

    /// Batch size
    public var count: Int { captions.count }

    /// Image width
    public var width: Int {
        resolution?.width ?? images.shape[2]
    }

    /// Image height
    public var height: Int {
        resolution?.height ?? images.shape[1]
    }
}

/// Training dataset with image loading and batching
public final class TrainingDataset: @unchecked Sendable {

    /// Dataset configuration
    public let config: LoRATrainingConfig

    /// Loaded samples (filename -> caption)
    private var samples: [(filename: String, caption: String)] = []

    /// Dataset path
    private let datasetPath: URL

    /// Current epoch
    private(set) var currentEpoch: Int = 0

    /// Current index within epoch
    private var currentIndex: Int = 0

    /// Sample order (for shuffling)
    private var sampleOrder: [Int] = []

    /// Bucket manager for multi-resolution training
    private var bucketManager: AspectRatioBucketManager?

    /// Current bucket index (for bucketed iteration)
    private var currentBucketIndex: Int = 0

    /// Buckets with samples (for bucketed iteration)
    private var activeBuckets: [ResolutionBucket] = []

    /// Current index within current bucket
    private var currentBucketSampleIndex: Int = 0

    /// Initialize training dataset
    /// - Parameter config: Training configuration
    public init(config: LoRATrainingConfig) throws {
        self.config = config
        self.datasetPath = config.datasetPath

        // Parse captions
        let parser = CaptionParser(triggerWord: config.triggerWord)
        self.samples = try parser.parseDataset(
            at: datasetPath,
            extension: config.captionExtension
        )

        guard !samples.isEmpty else {
            throw TrainingDatasetError.emptyDataset
        }

        Flux2Debug.log("[TrainingDataset] Loaded \(samples.count) samples")

        // Initialize bucketing if enabled
        if config.enableBucketing {
            try initializeBucketing()
        }

        // Initialize sample order
        resetOrder()
    }

    /// Initialize aspect ratio bucketing
    private func initializeBucketing() throws {
        bucketManager = AspectRatioBucketManager(resolutions: config.bucketResolutions)

        Flux2Debug.log("[TrainingDataset] Analyzing images for bucketing...")

        for sample in samples {
            let imagePath = datasetPath.appendingPathComponent(sample.filename)

            // Get image dimensions without fully loading the image
            guard let dimensions = getImageDimensions(at: imagePath) else {
                Flux2Debug.log("[TrainingDataset] Warning: Could not get dimensions for \(sample.filename)")
                // Fallback to square bucket
                bucketManager?.assignSample(
                    filename: sample.filename,
                    caption: sample.caption,
                    originalWidth: config.imageSize,
                    originalHeight: config.imageSize
                )
                continue
            }

            bucketManager?.assignSample(
                filename: sample.filename,
                caption: sample.caption,
                originalWidth: dimensions.width,
                originalHeight: dimensions.height
            )
        }

        activeBuckets = bucketManager?.getNonEmptyBuckets() ?? []
        Flux2Debug.log("[TrainingDataset] Bucketing complete:")
        Flux2Debug.log(bucketManager?.statistics ?? "No statistics available")
    }

    /// Get image dimensions without loading full image
    private func getImageDimensions(at url: URL) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            return nil
        }

        let options: [CFString: Any] = [kCGImageSourceShouldCache: false]
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, options as CFDictionary) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            return nil
        }

        return (width: width, height: height)
    }
    
    // MARK: - Dataset Info
    
    /// Number of samples in dataset
    public var count: Int { samples.count }
    
    /// Number of batches per epoch
    public var batchesPerEpoch: Int {
        (samples.count + config.batchSize - 1) / config.batchSize
    }
    
    /// Total number of training steps
    public var totalSteps: Int {
        if let maxSteps = config.maxSteps {
            return maxSteps
        }
        return batchesPerEpoch * config.epochs
    }
    
    // MARK: - Iteration
    
    /// Reset sample order (optionally shuffle)
    private func resetOrder() {
        sampleOrder = Array(0..<samples.count)
        if config.shuffleDataset {
            sampleOrder.shuffle()
        }
        currentIndex = 0
    }
    
    /// Start a new epoch
    public func startEpoch() {
        currentEpoch += 1

        if config.enableBucketing {
            // Reset bucketing iteration
            currentBucketIndex = 0
            currentBucketSampleIndex = 0
            // Shuffle buckets and samples within buckets
            if config.shuffleDataset {
                activeBuckets.shuffle()
                // Note: samples within buckets are already in the bucket manager
            }
        } else {
            resetOrder()
        }

        Flux2Debug.log("[TrainingDataset] Starting epoch \(currentEpoch)")
    }

    /// Get next batch of samples
    /// - Returns: Training batch, or nil if epoch is complete
    public func nextBatch() throws -> TrainingBatch? {
        if config.enableBucketing {
            return try nextBucketedBatch()
        } else {
            return try nextUnbucketedBatch()
        }
    }

    /// Get next batch without bucketing (original behavior)
    private func nextUnbucketedBatch() throws -> TrainingBatch? {
        guard currentIndex < samples.count else {
            return nil
        }

        let endIndex = Swift.min(currentIndex + config.batchSize, samples.count)
        let batchIndices = sampleOrder[currentIndex..<endIndex]
        currentIndex = endIndex

        var images: [MLXArray] = []
        var captions: [String] = []
        var filenames: [String] = []

        for idx in batchIndices {
            let sample = samples[idx]
            let imagePath = datasetPath.appendingPathComponent(sample.filename)

            // Load and preprocess image
            let image = try loadImage(at: imagePath, targetWidth: config.imageSize, targetHeight: config.imageSize)

            images.append(image)
            captions.append(sample.caption)
            filenames.append(sample.filename)
        }

        // Stack images into batch
        let batchedImages = MLX.stacked(images, axis: 0)

        return TrainingBatch(
            images: batchedImages,
            captions: captions,
            filenames: filenames,
            resolution: nil
        )
    }

    /// Get next batch with bucketing (samples grouped by resolution)
    private func nextBucketedBatch() throws -> TrainingBatch? {
        guard let bucketManager = bucketManager else {
            return try nextUnbucketedBatch()
        }

        // Find next non-empty bucket with remaining samples
        while currentBucketIndex < activeBuckets.count {
            let bucket = activeBuckets[currentBucketIndex]
            let bucketSamples = bucketManager.getSamples(in: bucket)

            if currentBucketSampleIndex < bucketSamples.count {
                // Get batch from current bucket
                let endIndex = Swift.min(currentBucketSampleIndex + config.batchSize, bucketSamples.count)
                let batchSamples = Array(bucketSamples[currentBucketSampleIndex..<endIndex])
                currentBucketSampleIndex = endIndex

                var images: [MLXArray] = []
                var captions: [String] = []
                var filenames: [String] = []

                for sample in batchSamples {
                    let imagePath = datasetPath.appendingPathComponent(sample.filename)

                    // Load with bucket's target resolution
                    let image = try loadImage(at: imagePath, targetWidth: bucket.width, targetHeight: bucket.height)

                    images.append(image)
                    captions.append(sample.caption)
                    filenames.append(sample.filename)
                }

                // Stack images into batch
                let batchedImages = MLX.stacked(images, axis: 0)

                return TrainingBatch(
                    images: batchedImages,
                    captions: captions,
                    filenames: filenames,
                    resolution: bucket
                )
            }

            // Move to next bucket
            currentBucketIndex += 1
            currentBucketSampleIndex = 0
        }

        return nil  // Epoch complete
    }
    
    /// Get a specific sample
    public func getSample(at index: Int) throws -> TrainingSample {
        guard index >= 0 && index < samples.count else {
            throw TrainingDatasetError.indexOutOfBounds(index)
        }

        let sample = samples[index]
        let imagePath = datasetPath.appendingPathComponent(sample.filename)

        // Determine target size based on bucketing
        let targetWidth: Int
        let targetHeight: Int

        if config.enableBucketing, let bucket = bucketManager?.getBucket(for: sample.filename) {
            targetWidth = bucket.width
            targetHeight = bucket.height
        } else {
            targetWidth = config.imageSize
            targetHeight = config.imageSize
        }

        let image = try loadImage(at: imagePath, targetWidth: targetWidth, targetHeight: targetHeight)

        return TrainingSample(
            filename: sample.filename,
            image: image,
            caption: sample.caption,
            originalSize: (targetWidth, targetHeight)
        )
    }

    // MARK: - Image Loading

    /// Load and preprocess an image
    /// - Parameters:
    ///   - url: Path to the image file
    ///   - targetWidth: Target width for resizing
    ///   - targetHeight: Target height for resizing
    /// - Returns: MLXArray of shape [H, W, C] in range [0, 1]
    private func loadImage(at url: URL, targetWidth: Int, targetHeight: Int) throws -> MLXArray {
        // Load image using ImageIO
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw TrainingDatasetError.failedToLoadImage(url)
        }

        guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw TrainingDatasetError.failedToLoadImage(url)
        }

        // Get original dimensions
        let originalWidth = cgImage.width
        let originalHeight = cgImage.height

        // Create bitmap context for resizing and conversion
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = targetWidth * 4

        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw TrainingDatasetError.failedToCreateContext
        }

        // Calculate crop/resize to fit target size (center crop)
        let scale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat

        let imageAspect = CGFloat(originalWidth) / CGFloat(originalHeight)
        let targetAspect = CGFloat(targetWidth) / CGFloat(targetHeight)

        if imageAspect > targetAspect {
            // Image is wider than target - scale by height, crop width
            scale = CGFloat(targetHeight) / CGFloat(originalHeight)
            offsetX = (CGFloat(originalWidth) * scale - CGFloat(targetWidth)) / 2
            offsetY = 0
        } else {
            // Image is taller than target - scale by width, crop height
            scale = CGFloat(targetWidth) / CGFloat(originalWidth)
            offsetX = 0
            offsetY = (CGFloat(originalHeight) * scale - CGFloat(targetHeight)) / 2
        }

        let drawRect = CGRect(
            x: -offsetX,
            y: -offsetY,
            width: CGFloat(originalWidth) * scale,
            height: CGFloat(originalHeight) * scale
        )
        
        context.interpolationQuality = .high
        context.draw(cgImage, in: drawRect)

        // Get pixel data
        guard let pixelData = context.data else {
            throw TrainingDatasetError.failedToGetPixelData
        }

        // Convert to MLXArray [H, W, C] normalized to [0, 1]
        let totalPixels = targetWidth * targetHeight
        let pixels = pixelData.bindMemory(to: UInt8.self, capacity: totalPixels * 4)
        var floatPixels: [Float] = []
        floatPixels.reserveCapacity(totalPixels * 3)

        for i in 0..<totalPixels {
            let r = Float(pixels[i * 4]) / 255.0
            let g = Float(pixels[i * 4 + 1]) / 255.0
            let b = Float(pixels[i * 4 + 2]) / 255.0
            floatPixels.append(r)
            floatPixels.append(g)
            floatPixels.append(b)
        }

        let array = MLXArray(floatPixels, [targetHeight, targetWidth, 3])

        return array
    }
    
    // MARK: - Validation
    
    /// Validate the dataset
    public func validate() -> DatasetValidationResult {
        let parser = CaptionParser(triggerWord: config.triggerWord)
        return parser.validateDataset(at: datasetPath, extension: config.captionExtension)
    }
    
    /// Get sample statistics
    public func getStatistics() -> DatasetStatistics {
        let captionLengths = samples.map { $0.caption.count }
        
        return DatasetStatistics(
            totalSamples: samples.count,
            minCaptionLength: captionLengths.min() ?? 0,
            maxCaptionLength: captionLengths.max() ?? 0,
            avgCaptionLength: captionLengths.isEmpty ? 0 : captionLengths.reduce(0, +) / captionLengths.count
        )
    }
}

// MARK: - Supporting Types

/// Dataset statistics
public struct DatasetStatistics: Sendable {
    public let totalSamples: Int
    public let minCaptionLength: Int
    public let maxCaptionLength: Int
    public let avgCaptionLength: Int
    
    public var summary: String {
        """
        Dataset Statistics:
          Total samples: \(totalSamples)
          Caption length: min=\(minCaptionLength), max=\(maxCaptionLength), avg=\(avgCaptionLength)
        """
    }
}

/// Dataset errors
public enum TrainingDatasetError: Error, LocalizedError {
    case emptyDataset
    case indexOutOfBounds(Int)
    case failedToLoadImage(URL)
    case failedToCreateContext
    case failedToGetPixelData
    case invalidImageFormat(URL)
    
    public var errorDescription: String? {
        switch self {
        case .emptyDataset:
            return "Dataset contains no valid samples"
        case .indexOutOfBounds(let index):
            return "Sample index \(index) out of bounds"
        case .failedToLoadImage(let url):
            return "Failed to load image: \(url.lastPathComponent)"
        case .failedToCreateContext:
            return "Failed to create image processing context"
        case .failedToGetPixelData:
            return "Failed to extract pixel data from image"
        case .invalidImageFormat(let url):
            return "Invalid image format: \(url.lastPathComponent)"
        }
    }
}

// MARK: - Iterator Protocol

extension TrainingDataset: Sequence {
    public struct Iterator: IteratorProtocol {
        private let dataset: TrainingDataset
        private var index: Int = 0
        
        init(dataset: TrainingDataset) {
            self.dataset = dataset
        }
        
        public mutating func next() -> TrainingSample? {
            guard index < dataset.count else { return nil }
            let sample = try? dataset.getSample(at: index)
            index += 1
            return sample
        }
    }
    
    public func makeIterator() -> Iterator {
        Iterator(dataset: self)
    }
}
