// ProfilingEvent.swift - Event types for pipeline profiling
// Copyright 2025 Vincent Gourbin

import Foundation

/// Category of a profiling event (maps to visual lanes in Chrome Trace)
public enum ProfilingCategory: String, Codable, Sendable {
    case textEncoderLoad = "text_encoder_load"
    case textEncoding = "text_encoding"
    case textEncoderUnload = "text_encoder_unload"
    case vlmInterpretation = "vlm_interpretation"
    case transformerLoad = "transformer_load"
    case vaeLoad = "vae_load"
    case denoisingLoop = "denoising_loop"
    case denoisingStep = "denoising_step"
    case vaeDecode = "vae_decode"
    case postProcess = "post_process"
    case evalSync = "eval_sync"
    case memoryOp = "memory_op"
    case custom = "custom"

    /// Thread ID for Chrome Trace lane grouping
    public var threadId: Int {
        switch self {
        case .textEncoderLoad, .textEncoding, .textEncoderUnload, .vlmInterpretation:
            return 1
        case .transformerLoad, .denoisingLoop, .denoisingStep:
            return 2
        case .vaeLoad, .vaeDecode:
            return 3
        case .postProcess:
            return 4
        case .memoryOp:
            return 5
        case .evalSync:
            return 6
        case .custom:
            return 7
        }
    }

    /// Human-readable thread name for Chrome Trace
    public var threadName: String {
        switch self {
        case .textEncoderLoad, .textEncoding, .textEncoderUnload, .vlmInterpretation:
            return "Text Encoding"
        case .transformerLoad, .denoisingLoop, .denoisingStep:
            return "Transformer"
        case .vaeLoad, .vaeDecode:
            return "VAE"
        case .postProcess:
            return "Post-processing"
        case .memoryOp:
            return "Memory"
        case .evalSync:
            return "eval() Syncs"
        case .custom:
            return "Other"
        }
    }
}

/// Phase type matching Chrome Trace Event Format
public enum ProfilingPhase: String, Codable, Sendable {
    case begin = "B"
    case end = "E"
    case complete = "X"
    case instant = "i"
    case counter = "C"
    case metadata = "M"
}

/// A single profiling event with timing and optional memory snapshot
public struct ProfilingEvent: Sendable, Codable {
    public let name: String
    public let category: ProfilingCategory
    public let phase: ProfilingPhase
    public let timestampUs: UInt64
    public let durationUs: UInt64?
    public let threadId: Int

    // Memory snapshot (optional, for phases that track memory)
    public let mlxActiveBytes: Int?
    public let mlxCacheBytes: Int?
    public let mlxPeakBytes: Int?
    public let processFootprintBytes: Int64?

    // Denoising step metadata
    public let stepIndex: Int?
    public let totalSteps: Int?

    public init(
        name: String,
        category: ProfilingCategory,
        phase: ProfilingPhase,
        timestampUs: UInt64,
        durationUs: UInt64? = nil,
        threadId: Int? = nil,
        mlxActiveBytes: Int? = nil,
        mlxCacheBytes: Int? = nil,
        mlxPeakBytes: Int? = nil,
        processFootprintBytes: Int64? = nil,
        stepIndex: Int? = nil,
        totalSteps: Int? = nil
    ) {
        self.name = name
        self.category = category
        self.phase = phase
        self.timestampUs = timestampUs
        self.durationUs = durationUs
        self.threadId = threadId ?? category.threadId
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.mlxPeakBytes = mlxPeakBytes
        self.processFootprintBytes = processFootprintBytes
        self.stepIndex = stepIndex
        self.totalSteps = totalSteps
    }
}

/// Memory timeline entry for counter events
public struct MemoryTimelineEntry: Sendable, Codable {
    public let timestampUs: UInt64
    public let context: String
    public let mlxActiveMB: Double
    public let mlxCacheMB: Double
    public let mlxPeakMB: Double
    public let processFootprintMB: Double

    public init(timestampUs: UInt64, context: String, mlxActiveMB: Double, mlxCacheMB: Double, mlxPeakMB: Double, processFootprintMB: Double) {
        self.timestampUs = timestampUs
        self.context = context
        self.mlxActiveMB = mlxActiveMB
        self.mlxCacheMB = mlxCacheMB
        self.mlxPeakMB = mlxPeakMB
        self.processFootprintMB = processFootprintMB
    }
}
