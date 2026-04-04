// ProfilingSession.swift - Central profiling session coordinator
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX

/// Collects profiling events during a generation run
public final class ProfilingSession: @unchecked Sendable {
    public let sessionId: String
    public let startTime: Date
    public let config: ProfilingConfig

    // Device metadata
    public let deviceArchitecture: String
    public let systemRAMGB: Int
    public var modelVariant: String = ""
    public var quantization: String = ""
    public var imageSize: String = ""
    public var steps: Int = 0

    private var events: [ProfilingEvent] = []
    private var memoryTimeline: [MemoryTimelineEntry] = []
    private let lock = NSLock()
    private let sessionStartTime: CFAbsoluteTime

    public init(config: ProfilingConfig = .singleRun) {
        self.sessionId = UUID().uuidString
        self.startTime = Date()
        self.config = config
        self.sessionStartTime = CFAbsoluteTimeGetCurrent()
        self.deviceArchitecture = GPU.deviceInfo().architecture
        self.systemRAMGB = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
    }

    // MARK: - Timestamps

    /// Microseconds elapsed since session start
    private func currentTimestampUs() -> UInt64 {
        UInt64((CFAbsoluteTimeGetCurrent() - sessionStartTime) * 1_000_000)
    }

    // MARK: - Event Recording

    /// Record a phase begin event
    public func beginPhase(_ name: String, category: ProfilingCategory) {
        let ts = currentTimestampUs()
        let snapshot = config.trackMemory ? takeMemorySnapshot() : nil

        lock.lock()
        events.append(ProfilingEvent(
            name: name,
            category: category,
            phase: .begin,
            timestampUs: ts,
            mlxActiveBytes: snapshot?.mlxActive,
            mlxCacheBytes: snapshot?.mlxCache,
            mlxPeakBytes: snapshot?.mlxPeak,
            processFootprintBytes: snapshot?.processFootprint
        ))

        if config.trackMemory, let snap = snapshot {
            memoryTimeline.append(MemoryTimelineEntry(
                timestampUs: ts,
                context: "begin:\(name)",
                mlxActiveMB: Double(snap.mlxActive) / 1_048_576,
                mlxCacheMB: Double(snap.mlxCache) / 1_048_576,
                mlxPeakMB: Double(snap.mlxPeak) / 1_048_576,
                processFootprintMB: Double(snap.processFootprint) / 1_048_576
            ))
        }
        lock.unlock()
    }

    /// Record a phase end event
    public func endPhase(_ name: String, category: ProfilingCategory) {
        let ts = currentTimestampUs()
        let snapshot = config.trackMemory ? takeMemorySnapshot() : nil

        lock.lock()
        events.append(ProfilingEvent(
            name: name,
            category: category,
            phase: .end,
            timestampUs: ts,
            mlxActiveBytes: snapshot?.mlxActive,
            mlxCacheBytes: snapshot?.mlxCache,
            mlxPeakBytes: snapshot?.mlxPeak,
            processFootprintBytes: snapshot?.processFootprint
        ))

        if config.trackMemory, let snap = snapshot {
            memoryTimeline.append(MemoryTimelineEntry(
                timestampUs: ts,
                context: "end:\(name)",
                mlxActiveMB: Double(snap.mlxActive) / 1_048_576,
                mlxCacheMB: Double(snap.mlxCache) / 1_048_576,
                mlxPeakMB: Double(snap.mlxPeak) / 1_048_576,
                processFootprintMB: Double(snap.processFootprint) / 1_048_576
            ))
        }
        lock.unlock()
    }

    /// Record a complete event (with known duration)
    public func recordComplete(_ name: String, category: ProfilingCategory, durationUs: UInt64) {
        let ts = currentTimestampUs()
        let startTs = ts >= durationUs ? ts - durationUs : 0
        lock.lock()
        events.append(ProfilingEvent(
            name: name,
            category: category,
            phase: .complete,
            timestampUs: startTs,
            durationUs: durationUs
        ))
        lock.unlock()
    }

    /// Record a denoising step
    public func recordDenoisingStep(index: Int, total: Int, durationUs: UInt64) {
        let ts = currentTimestampUs()
        let startTs = ts >= durationUs ? ts - durationUs : 0
        let snapshot = config.trackPerStepMemory ? takeMemorySnapshot() : nil

        lock.lock()
        events.append(ProfilingEvent(
            name: "Step \(index)/\(total)",
            category: .denoisingStep,
            phase: .complete,
            timestampUs: startTs,
            durationUs: durationUs,
            mlxActiveBytes: snapshot?.mlxActive,
            mlxCacheBytes: snapshot?.mlxCache,
            mlxPeakBytes: snapshot?.mlxPeak,
            processFootprintBytes: snapshot?.processFootprint,
            stepIndex: index,
            totalSteps: total
        ))

        if config.trackPerStepMemory, let snap = snapshot {
            memoryTimeline.append(MemoryTimelineEntry(
                timestampUs: ts,
                context: "step:\(index)/\(total)",
                mlxActiveMB: Double(snap.mlxActive) / 1_048_576,
                mlxCacheMB: Double(snap.mlxCache) / 1_048_576,
                mlxPeakMB: Double(snap.mlxPeak) / 1_048_576,
                processFootprintMB: Double(snap.processFootprint) / 1_048_576
            ))
        }
        lock.unlock()
    }

    /// Record a memory snapshot at an arbitrary point
    public func recordMemorySnapshot(context: String) {
        let ts = currentTimestampUs()
        let snap = takeMemorySnapshot()

        lock.lock()
        memoryTimeline.append(MemoryTimelineEntry(
            timestampUs: ts,
            context: context,
            mlxActiveMB: Double(snap.mlxActive) / 1_048_576,
            mlxCacheMB: Double(snap.mlxCache) / 1_048_576,
            mlxPeakMB: Double(snap.mlxPeak) / 1_048_576,
            processFootprintMB: Double(snap.processFootprint) / 1_048_576
        ))
        lock.unlock()
    }

    // MARK: - Data Access

    /// Get all collected events
    public func getEvents() -> [ProfilingEvent] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }

    /// Get memory timeline
    public func getMemoryTimeline() -> [MemoryTimelineEntry] {
        lock.lock()
        defer { lock.unlock() }
        return memoryTimeline
    }

    /// Total session duration in seconds
    public var elapsedSeconds: TimeInterval {
        CFAbsoluteTimeGetCurrent() - sessionStartTime
    }

    // MARK: - Memory Snapshot (private)

    private struct RawMemorySnapshot {
        let mlxActive: Int
        let mlxCache: Int
        let mlxPeak: Int
        let processFootprint: Int64
    }

    private func takeMemorySnapshot() -> RawMemorySnapshot {
        RawMemorySnapshot(
            mlxActive: Memory.activeMemory,
            mlxCache: Memory.cacheMemory,
            mlxPeak: Memory.peakMemory,
            processFootprint: Self.getProcessFootprint()
        )
    }

    private static func getProcessFootprint() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        return result == KERN_SUCCESS ? Int64(info.phys_footprint) : 0
    }

    // MARK: - Report Generation

    /// Generate human-readable summary report
    public func generateReport() -> String {
        let events = getEvents()
        let timeline = getMemoryTimeline()

        // Build paired begin/end phases
        var phases: [(name: String, durationMs: Double)] = []
        var stepDurations: [Double] = []

        // Collect complete events (steps) and compute B/E pairs
        var beginTimestamps: [String: UInt64] = [:]
        for event in events {
            switch event.phase {
            case .begin:
                beginTimestamps[event.name] = event.timestampUs
            case .end:
                if let beginTs = beginTimestamps[event.name] {
                    let durationMs = Double(event.timestampUs - beginTs) / 1000.0
                    phases.append((name: event.name, durationMs: durationMs))
                    beginTimestamps.removeValue(forKey: event.name)
                }
            case .complete:
                if event.category == .denoisingStep, let dur = event.durationUs {
                    stepDurations.append(Double(dur) / 1000.0)
                }
            default:
                break
            }
        }

        let totalMs = phases.reduce(0.0) { $0 + $1.durationMs }

        var report = """

        \u{256D}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256E}
        \u{2502}              FLUX.2 PROFILING REPORT                      \u{2502}
        \u{251C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2524}

        """

        // Config
        report += "  Model: \(modelVariant)  Quant: \(quantization)\n"
        report += "  Image: \(imageSize)  Steps: \(steps)\n"
        report += "  Device: \(deviceArchitecture)  RAM: \(systemRAMGB)GB\n"
        report += "\n"

        // Phase timings
        report += "  PHASE TIMINGS\n"
        report += "  \(String(repeating: "\u{2500}", count: 58))\n"

        for phase in phases {
            let pct = totalMs > 0 ? (phase.durationMs / totalMs) * 100 : 0
            let barLen = min(20, Int(pct / 5))
            let bar = String(repeating: "\u{2588}", count: barLen)
            let name = phase.name.padding(toLength: 28, withPad: " ", startingAt: 0)
            report += "  \(name) \(formatMs(phase.durationMs))  \(String(format: "%5.1f", pct))% \(bar)\n"
        }

        report += "  \(String(repeating: "\u{2500}", count: 58))\n"
        report += "  \("TOTAL".padding(toLength: 28, withPad: " ", startingAt: 0)) \(formatMs(totalMs))  100.0%\n"

        // Step statistics
        if !stepDurations.isEmpty {
            let avgMs = stepDurations.reduce(0, +) / Double(stepDurations.count)
            let minMs = stepDurations.min() ?? 0
            let maxMs = stepDurations.max() ?? 0
            let variance = stepDurations.map { ($0 - avgMs) * ($0 - avgMs) }.reduce(0, +) / Double(stepDurations.count)
            let stdMs = sqrt(variance)

            report += "\n  DENOISING STEP STATISTICS\n"
            report += "  \(String(repeating: "\u{2500}", count: 58))\n"
            report += "  Steps: \(stepDurations.count)\n"
            report += "  Average: \(formatMs(avgMs))  Std: \(formatMs(stdMs))\n"
            report += "  Min: \(formatMs(minMs))  Max: \(formatMs(maxMs))\n"

            // Estimates
            report += "\n  Estimated for different step counts:\n"
            for count in [10, 20, 28, 50] {
                report += "    \(String(format: "%2d", count)) steps: \(formatMs(avgMs * Double(count)))\n"
            }
        }

        // Memory summary
        if !timeline.isEmpty {
            let peakActive = timeline.map(\.mlxActiveMB).max() ?? 0
            let peakProcess = timeline.map(\.processFootprintMB).max() ?? 0

            report += "\n  MEMORY\n"
            report += "  \(String(repeating: "\u{2500}", count: 58))\n"
            report += "  Peak MLX Active: \(String(format: "%.1f", peakActive)) MB\n"
            report += "  Peak Process: \(String(format: "%.1f", peakProcess)) MB\n"

            // Show memory at key points
            let keyPoints = timeline.filter { entry in
                entry.context.hasPrefix("begin:") || entry.context.hasPrefix("end:")
            }
            if !keyPoints.isEmpty {
                report += "\n  Memory Timeline:\n"
                for entry in keyPoints {
                    let label = entry.context.padding(toLength: 35, withPad: " ", startingAt: 0)
                    report += "    \(label) MLX: \(String(format: "%7.1f", entry.mlxActiveMB)) MB\n"
                }
            }
        }

        report += "\n\u{2570}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256F}\n"

        return report
    }

    private func formatMs(_ ms: Double) -> String {
        if ms < 1000 {
            return String(format: "%7.1fms", ms)
        } else if ms < 60000 {
            return String(format: "%7.2fs ", ms / 1000)
        } else {
            let min = Int(ms / 60000)
            let sec = (ms - Double(min) * 60000) / 1000
            return String(format: "%dm %04.1fs", min, sec)
        }
    }
}

// MARK: - Category Inference

extension ProfilingSession {
    /// Infer category from existing Flux2Profiler phase names
    public static func inferCategory(_ phaseName: String) -> ProfilingCategory {
        if phaseName.contains("Load Text") { return .textEncoderLoad }
        if phaseName.contains("VLM") { return .vlmInterpretation }
        if phaseName.contains("Text Encoding") || phaseName.contains("2. Text") { return .textEncoding }
        if phaseName.contains("Unload Text") { return .textEncoderUnload }
        if phaseName.contains("Load Transformer") || phaseName.contains("4. Load") { return .transformerLoad }
        if phaseName.contains("Load VAE") || phaseName.contains("5. Load") { return .vaeLoad }
        if phaseName.contains("Denoising") { return .denoisingLoop }
        if phaseName.contains("VAE Decode") || phaseName.contains("7. VAE") { return .vaeDecode }
        if phaseName.contains("Post") || phaseName.contains("8. Post") { return .postProcess }
        return .custom
    }
}
