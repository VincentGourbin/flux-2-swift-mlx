// LatentCache.swift - Pre-compute and cache VAE latents for training
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Cached latent entry
public struct CachedLatent: @unchecked Sendable {
    /// Original filename
    public let filename: String
    
    /// Caption text
    public let caption: String
    
    /// Latent representation [H/8, W/8, C]
    public let latent: MLXArray
    
    /// Image size used for encoding
    public let imageSize: Int
}

/// Cache for pre-encoded VAE latents
///
/// Pre-encoding images with VAE and caching the latents provides:
/// - ~50% memory savings during training (no need to keep VAE in memory)
/// - Faster training (no VAE encoding per step)
/// - Consistent latents across epochs
public final class LatentCache: @unchecked Sendable {
    
    /// Cache directory path
    public let cacheDirectory: URL
    
    /// Configuration
    public let config: LoRATrainingConfig
    
    /// In-memory cache (optional)
    private var memoryCache: [String: MLXArray] = [:]
    
    /// Whether to keep latents in memory
    public let useMemoryCache: Bool
    
    /// Initialize latent cache
    /// - Parameters:
    ///   - config: Training configuration
    ///   - cacheDirectory: Directory to store cached latents
    ///   - useMemoryCache: Whether to also cache in memory
    public init(
        config: LoRATrainingConfig,
        cacheDirectory: URL? = nil,
        useMemoryCache: Bool = true
    ) {
        self.config = config
        self.useMemoryCache = useMemoryCache
        
        // Default cache directory next to dataset
        if let dir = cacheDirectory {
            self.cacheDirectory = dir
        } else {
            self.cacheDirectory = config.datasetPath
                .appendingPathComponent(".latent_cache")
        }
        
        // Create cache directory if needed
        try? FileManager.default.createDirectory(
            at: self.cacheDirectory,
            withIntermediateDirectories: true
        )
    }
    
    // MARK: - Cache Management
    
    /// Check if latent is cached
    public func isCached(filename: String) -> Bool {
        if useMemoryCache && memoryCache[filename] != nil {
            return true
        }
        
        let cacheFile = cacheFilePath(for: filename)
        return FileManager.default.fileExists(atPath: cacheFile.path)
    }
    
    /// Get cache file path for a filename (without resolution suffix)
    private func cacheFilePath(for filename: String) -> URL {
        let baseName = (filename as NSString).deletingPathExtension
        return cacheDirectory.appendingPathComponent("\(baseName)_latent.safetensors")
    }

    /// Get cache file path including resolution for bucketed caching
    private func cacheFilePath(for filename: String, width: Int, height: Int) -> URL {
        let baseName = (filename as NSString).deletingPathExtension
        return cacheDirectory.appendingPathComponent("\(baseName)_\(width)x\(height)_latent.safetensors")
    }

    /// Memory cache key including resolution
    private func memoryCacheKey(filename: String, width: Int? = nil, height: Int? = nil) -> String {
        if let w = width, let h = height {
            return "\(filename)_\(w)x\(h)"
        }
        return filename
    }

    /// Get cached latent (simple version for non-bucketed caching)
    public func getLatent(for filename: String) throws -> MLXArray? {
        return try getLatent(for: filename, width: config.imageSize, height: config.imageSize)
    }

    /// Get cached latent with specific resolution
    public func getLatent(for filename: String, width: Int, height: Int) throws -> MLXArray? {
        let cacheKey = memoryCacheKey(filename: filename, width: width, height: height)

        // Check memory cache first
        if useMemoryCache, let latent = memoryCache[cacheKey] {
            return latent
        }

        // Load from disk (try resolution-specific first, then fallback to generic)
        var cacheFile = cacheFilePath(for: filename, width: width, height: height)
        if !FileManager.default.fileExists(atPath: cacheFile.path) {
            // Fallback to non-resolution-specific cache (for backwards compatibility)
            cacheFile = cacheFilePath(for: filename)
            guard FileManager.default.fileExists(atPath: cacheFile.path) else {
                return nil
            }
        }

        let weights = try loadArrays(url: cacheFile)
        guard let latent = weights["latent"] else {
            return nil
        }

        // Optionally store in memory
        if useMemoryCache {
            memoryCache[cacheKey] = latent
        }

        return latent
    }

    /// Save latent to cache (simple version)
    public func saveLatent(_ latent: MLXArray, for filename: String) throws {
        try saveLatent(latent, for: filename, width: config.imageSize, height: config.imageSize)
    }

    /// Save latent to cache with specific resolution
    public func saveLatent(_ latent: MLXArray, for filename: String, width: Int, height: Int) throws {
        let cacheKey = memoryCacheKey(filename: filename, width: width, height: height)

        // Save to memory
        if useMemoryCache {
            memoryCache[cacheKey] = latent
        }

        // Save to disk with resolution in filename
        let cacheFile = cacheFilePath(for: filename, width: width, height: height)
        try save(arrays: ["latent": latent], url: cacheFile)
    }
    
    /// Clear all cached latents
    public func clearCache() throws {
        memoryCache.removeAll()
        
        let contents = try FileManager.default.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: nil
        )
        
        for file in contents where file.pathExtension == "safetensors" {
            try FileManager.default.removeItem(at: file)
        }
        
        Flux2Debug.log("[LatentCache] Cleared all cached latents")
    }
    
    /// Clear memory cache only (keep disk cache)
    public func clearMemoryCache() {
        memoryCache.removeAll()
        Flux2Debug.log("[LatentCache] Cleared memory cache")
    }
    
    // MARK: - Pre-encoding
    
    /// Batch size for VAE encoding (balances GPU efficiency vs memory)
    private static let encodingBatchSize = 4
    
    /// Pre-encode all images in dataset and cache latents (optimized version)
    /// - Parameters:
    ///   - dataset: Training dataset
    ///   - vae: VAE encoder
    ///   - progressCallback: Called with (current, total) progress
    /// - Returns: Number of latents cached
    @discardableResult
    public func preEncodeDataset(
        _ dataset: TrainingDataset,
        vae: AutoencoderKLFlux2,
        progressCallback: ((Int, Int) -> Void)? = nil
    ) async throws -> Int {
        let total = dataset.count
        
        Flux2Debug.log("[LatentCache] Pre-encoding \(total) images...")
        
        // OPTIMIZATION 1: Pre-load list of cached files for O(1) lookup
        let cachedFilenames = loadCachedFilenameSet()
        
        // OPTIMIZATION 2: Collect uncached samples for batch encoding
        var uncachedSamples: [(index: Int, sample: TrainingSample)] = []
        var cachedCount = 0
        
        for (index, sample) in dataset.enumerated() {
            let baseName = (sample.filename as NSString).deletingPathExtension
            if cachedFilenames.contains(baseName) {
                cachedCount += 1
                progressCallback?(index + 1, total)
            } else {
                uncachedSamples.append((index, sample))
            }
        }
        
        Flux2Debug.log("[LatentCache] Found \(cachedCount) already cached, \(uncachedSamples.count) to encode")
        
        // OPTIMIZATION 3: Batch encode uncached images
        let batchSize = Self.encodingBatchSize
        for batchStart in stride(from: 0, to: uncachedSamples.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, uncachedSamples.count)
            let batch = Array(uncachedSamples[batchStart..<batchEnd])
            
            // Stack images into batch [B, H, W, C]
            var batchImages: [MLXArray] = []
            var batchFilenames: [String] = []
            
            for (_, sample) in batch {
                // Normalize to [-1, 1]
                let normalizedImage = sample.image * 2.0 - 1.0
                batchImages.append(normalizedImage)
                batchFilenames.append(sample.filename)
            }
            
            // Stack and transpose to NCHW [B, C, H, W]
            let stackedImages = MLX.stacked(batchImages, axis: 0)
            let nchwBatch = stackedImages.transposed(0, 3, 1, 2)
            
            // Encode entire batch at once
            let latentsBatch = vae.encode(nchwBatch)
            
            // Ensure computation is done
            eval(latentsBatch)
            
            // Save each latent to cache
            for (i, filename) in batchFilenames.enumerated() {
                let latent = latentsBatch[i]
                try saveLatent(latent, for: filename)
            }
            
            // Update progress for each item in batch
            for (origIndex, _) in batch {
                progressCallback?(origIndex + 1, total)
            }
            
            // Clear GPU memory after each batch
            MLX.Memory.clearCache()
        }
        
        let totalEncoded = cachedCount + uncachedSamples.count
        Flux2Debug.log("[LatentCache] Pre-encoded \(totalEncoded) latents (batched \(uncachedSamples.count) new)")
        
        return totalEncoded
    }
    
    /// Load set of cached filenames for O(1) lookup
    private func loadCachedFilenameSet() -> Set<String> {
        var cached = Set<String>()
        
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: nil
        ) else {
            return cached
        }
        
        for file in files where file.pathExtension == "safetensors" {
            // Extract original filename from cache filename (e.g., "image_latent.safetensors" -> "image")
            let cacheName = file.deletingPathExtension().lastPathComponent
            if cacheName.hasSuffix("_latent") {
                let originalName = String(cacheName.dropLast(7)) // Remove "_latent"
                cached.insert(originalName)
            }
        }
        
        return cached
    }
    
    /// Get latent for a batch (loading from cache or encoding)
    /// - Parameters:
    ///   - batch: Training batch
    ///   - vae: VAE encoder (used if not cached)
    /// - Returns: Batched latents [B, C, H/8, W/8]
    public func getLatents(
        for batch: TrainingBatch,
        vae: AutoencoderKLFlux2?
    ) throws -> MLXArray {
        var latents: [MLXArray] = []

        // Get resolution from batch (bucketed) or default (non-bucketed)
        let width = batch.resolution?.width ?? config.imageSize
        let height = batch.resolution?.height ?? config.imageSize

        for (i, filename) in batch.filenames.enumerated() {
            if let cached = try getLatent(for: filename, width: width, height: height) {
                latents.append(cached)
            } else if let vae = vae {
                // Encode on the fly
                let image = batch.images[i]
                let normalizedImage = image * 2.0 - 1.0
                let batchedImage = normalizedImage.expandedDimensions(axis: 0)
                let nchwImage = batchedImage.transposed(0, 3, 1, 2)  // NHWC -> NCHW
                let latent = vae.encode(nchwImage).squeezed(axis: 0)

                // Cache for next time with resolution
                try saveLatent(latent, for: filename, width: width, height: height)
                latents.append(latent)
            } else {
                throw LatentCacheError.latentNotCached(filename)
            }
        }

        return MLX.stacked(latents, axis: 0)
    }
    
    // MARK: - Statistics
    
    /// Get cache statistics
    public func getStatistics() -> CacheStatistics {
        let fileManager = FileManager.default
        
        var diskCount = 0
        var diskSize: Int64 = 0
        
        if let files = try? fileManager.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: [.fileSizeKey]
        ) {
            for file in files where file.pathExtension == "safetensors" {
                diskCount += 1
                if let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    diskSize += Int64(size)
                }
            }
        }
        
        return CacheStatistics(
            memoryCacheCount: memoryCache.count,
            diskCacheCount: diskCount,
            diskCacheSizeMB: Float(diskSize) / (1024 * 1024)
        )
    }
}

// MARK: - Supporting Types

/// Cache statistics
public struct CacheStatistics: Sendable {
    public let memoryCacheCount: Int
    public let diskCacheCount: Int
    public let diskCacheSizeMB: Float
    
    public var summary: String {
        """
        Latent Cache Statistics:
          Memory cache: \(memoryCacheCount) entries
          Disk cache: \(diskCacheCount) files (\(String(format: "%.1f", diskCacheSizeMB)) MB)
        """
    }
}

/// Cache errors
public enum LatentCacheError: Error, LocalizedError {
    case latentNotCached(String)
    case failedToSave(String)
    case failedToLoad(String)
    
    public var errorDescription: String? {
        switch self {
        case .latentNotCached(let filename):
            return "Latent not cached for: \(filename)"
        case .failedToSave(let filename):
            return "Failed to save latent for: \(filename)"
        case .failedToLoad(let filename):
            return "Failed to load latent for: \(filename)"
        }
    }
}

// MARK: - Text Embedding Cache

/// Cache for pre-computed text embeddings
public final class TextEmbeddingCache: @unchecked Sendable {
    
    /// Cache directory path
    public let cacheDirectory: URL
    
    /// In-memory cache
    private var memoryCache: [String: (pooled: MLXArray, hidden: MLXArray)] = [:]
    
    /// Initialize text embedding cache
    public init(cacheDirectory: URL) {
        self.cacheDirectory = cacheDirectory
        
        // Create cache directory
        try? FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
    }
    
    /// Check if embedding is cached
    public func isCached(caption: String) -> Bool {
        let key = cacheKey(for: caption)
        if memoryCache[key] != nil { return true }
        
        let cacheFile = cacheDirectory.appendingPathComponent("\(key).safetensors")
        return FileManager.default.fileExists(atPath: cacheFile.path)
    }
    
    /// Get cache key for caption
    private func cacheKey(for caption: String) -> String {
        // Use hash of caption as key
        let hash = caption.hashValue
        return String(format: "emb_%016llx", UInt64(bitPattern: Int64(hash)))
    }
    
    /// Get cached embeddings
    public func getEmbeddings(for caption: String) throws -> (pooled: MLXArray, hidden: MLXArray)? {
        let key = cacheKey(for: caption)
        
        // Check memory cache
        if let cached = memoryCache[key] {
            return cached
        }
        
        // Load from disk
        let cacheFile = cacheDirectory.appendingPathComponent("\(key).safetensors")
        guard FileManager.default.fileExists(atPath: cacheFile.path) else {
            return nil
        }
        
        let weights = try loadArrays(url: cacheFile)
        guard let pooled = weights["pooled"],
              let hidden = weights["hidden"] else {
            return nil
        }
        
        let result = (pooled: pooled, hidden: hidden)
        memoryCache[key] = result
        return result
    }
    
    /// Save embeddings to cache
    public func saveEmbeddings(
        pooled: MLXArray,
        hidden: MLXArray,
        for caption: String
    ) throws {
        let key = cacheKey(for: caption)
        memoryCache[key] = (pooled: pooled, hidden: hidden)
        
        let cacheFile = cacheDirectory.appendingPathComponent("\(key).safetensors")
        try save(arrays: ["pooled": pooled, "hidden": hidden], url: cacheFile)
    }
    
    /// Clear cache
    public func clearCache() throws {
        memoryCache.removeAll()
        
        let contents = try FileManager.default.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: nil
        )
        
        for file in contents where file.pathExtension == "safetensors" {
            try FileManager.default.removeItem(at: file)
        }
    }
}
