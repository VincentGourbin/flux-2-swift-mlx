// Flux2Profiler.swift - Performance Profiling for Flux.2
// Copyright 2025 Vincent Gourbin

import Foundation

// MARK: - String Extension for Padding

private extension String {
    /// Pad string on the left to reach target length
    func leftPadding(toLength length: Int, withPad character: String) -> String {
        let stringLength = self.count
        if stringLength >= length {
            return self
        }
        return String(repeating: character, count: length - stringLength) + self
    }
}

/// Timing entry for a profiled operation
public struct TimingEntry: Sendable {
    public let name: String
    public let duration: TimeInterval
    public let startTime: Date
    public let endTime: Date

    public var durationMs: Double {
        duration * 1000
    }

    public var durationFormatted: String {
        if duration < 1 {
            return String(format: "%.1fms", durationMs)
        } else if duration < 60 {
            return String(format: "%.2fs", duration)
        } else {
            let minutes = Int(duration / 60)
            let seconds = duration.truncatingRemainder(dividingBy: 60)
            return String(format: "%dm %.1fs", minutes, seconds)
        }
    }
}

/// Performance profiler for Flux.2 pipeline (thread-safe with actor)
public actor Flux2ProfilerActor {
    private var timings: [TimingEntry] = []
    private var activeTimers: [String: Date] = [:]
    private var stepTimes: [TimeInterval] = []

    public init() {}

    /// Start timing an operation
    public func start(_ name: String) {
        activeTimers[name] = Date()
    }

    /// End timing an operation and record it
    public func end(_ name: String) {
        let endTime = Date()
        guard let startTime = activeTimers[name] else { return }
        activeTimers.removeValue(forKey: name)

        let entry = TimingEntry(
            name: name,
            duration: endTime.timeIntervalSince(startTime),
            startTime: startTime,
            endTime: endTime
        )
        timings.append(entry)
    }

    /// Record a timing entry directly
    public func record(_ name: String, duration: TimeInterval) {
        let now = Date()
        let entry = TimingEntry(
            name: name,
            duration: duration,
            startTime: now.addingTimeInterval(-duration),
            endTime: now
        )
        timings.append(entry)
    }

    /// Record a denoising step time
    public func recordStep(duration: TimeInterval) {
        stepTimes.append(duration)
    }

    /// Clear all recorded timings
    public func reset() {
        timings.removeAll()
        activeTimers.removeAll()
        stepTimes.removeAll()
    }

    /// Get all recorded timings
    public func getTimings() -> [TimingEntry] {
        return timings
    }

    /// Get step times
    public func getStepTimes() -> [TimeInterval] {
        return stepTimes
    }
}

/// Synchronous wrapper for profiler (for non-async contexts)
public class Flux2Profiler: @unchecked Sendable {
    public static let shared = Flux2Profiler()

    public var isEnabled: Bool = false

    /// Optional profiling session for rich data collection (Chrome Trace, memory timeline)
    public var activeSession: ProfilingSession? = nil

    private var timings: [TimingEntry] = []
    private var stepTimes: [TimeInterval] = []
    private var stepCount: Int = 0
    private var totalSteps: Int = 0
    private var activeTimers: [String: Date] = [:]
    private let queue = DispatchQueue(label: "com.flux2.profiler", attributes: .concurrent)

    private init() {}

    /// Enable profiling
    public func enable() {
        isEnabled = true
        reset()
    }

    /// Disable profiling
    public func disable() {
        isEnabled = false
    }

    /// Set the total step count for denoising (used by session for step labeling)
    public func setTotalSteps(_ total: Int) {
        queue.async(flags: .barrier) {
            self.totalSteps = total
            self.stepCount = 0
        }
    }

    /// Start timing an operation
    public func start(_ name: String) {
        guard isEnabled else { return }
        let category = ProfilingSession.inferCategory(name)
        activeSession?.beginPhase(name, category: category)
        queue.async(flags: .barrier) {
            self.activeTimers[name] = Date()
        }
    }

    /// End timing an operation and record it
    public func end(_ name: String) {
        guard isEnabled else { return }
        let endTime = Date()
        let category = ProfilingSession.inferCategory(name)
        activeSession?.endPhase(name, category: category)
        queue.async(flags: .barrier) {
            guard let startTime = self.activeTimers[name] else { return }
            self.activeTimers.removeValue(forKey: name)

            let entry = TimingEntry(
                name: name,
                duration: endTime.timeIntervalSince(startTime),
                startTime: startTime,
                endTime: endTime
            )
            self.timings.append(entry)
        }
    }

    /// Record a timing entry directly (synchronous, for use in loops)
    public func record(_ name: String, duration: TimeInterval) {
        guard isEnabled else { return }
        let now = Date()
        queue.async(flags: .barrier) {
            let entry = TimingEntry(
                name: name,
                duration: duration,
                startTime: now.addingTimeInterval(-duration),
                endTime: now
            )
            self.timings.append(entry)
        }
    }

    /// Record a denoising step time
    public func recordStep(duration: TimeInterval) {
        guard isEnabled else { return }
        queue.async(flags: .barrier) {
            self.stepTimes.append(duration)
            self.stepCount += 1
            let currentStep = self.stepCount
            let total = self.totalSteps
            self.activeSession?.recordDenoisingStep(
                index: currentStep,
                total: total,
                durationUs: UInt64(duration * 1_000_000)
            )
        }
    }

    /// Measure a synchronous operation
    @discardableResult
    public func measure<T>(_ name: String, _ operation: () throws -> T) rethrows -> T {
        guard isEnabled else { return try operation() }

        let startTime = Date()
        let result = try operation()
        let endTime = Date()

        queue.async(flags: .barrier) {
            let entry = TimingEntry(
                name: name,
                duration: endTime.timeIntervalSince(startTime),
                startTime: startTime,
                endTime: endTime
            )
            self.timings.append(entry)
        }

        return result
    }

    /// Clear all recorded timings
    public func reset() {
        queue.async(flags: .barrier) {
            self.timings.removeAll()
            self.activeTimers.removeAll()
            self.stepTimes.removeAll()
            self.stepCount = 0
            self.totalSteps = 0
        }
    }

    /// Get all recorded timings (synchronous read)
    public func getTimings() -> [TimingEntry] {
        var result: [TimingEntry] = []
        queue.sync {
            result = self.timings
        }
        return result
    }

    /// Get step times (synchronous read)
    public func getStepTimes() -> [TimeInterval] {
        var result: [TimeInterval] = []
        queue.sync {
            result = self.stepTimes
        }
        return result
    }

    /// Generate a formatted report
    public func generateReport() -> String {
        var currentTimings: [TimingEntry] = []
        var currentStepTimes: [TimeInterval] = []

        queue.sync {
            currentTimings = self.timings
            currentStepTimes = self.stepTimes
        }

        guard !currentTimings.isEmpty else {
            return "No timing data recorded."
        }

        var report = """

        ╔══════════════════════════════════════════════════════════════╗
        ║                  FLUX.2 PERFORMANCE REPORT                   ║
        ╠══════════════════════════════════════════════════════════════╣

        """

        // Calculate total time
        let totalTime = currentTimings.map(\.duration).reduce(0, +)

        // Main phases
        report += "📊 PHASE TIMINGS:\n"
        report += "────────────────────────────────────────────────────────────────\n"

        for entry in currentTimings {
            let percentage = totalTime > 0 ? (entry.duration / totalTime) * 100 : 0
            let bar = String(repeating: "█", count: min(20, Int(percentage / 5)))
            let namePadded = entry.name.padding(toLength: 30, withPad: " ", startingAt: 0)
            let durationPadded = entry.durationFormatted.leftPadding(toLength: 10, withPad: " ")
            report += "  \(namePadded) \(durationPadded)  \(String(format: "%5.1f", percentage))% \(bar)\n"
        }

        report += "────────────────────────────────────────────────────────────────\n"
        let totalPadded = "TOTAL".padding(toLength: 30, withPad: " ", startingAt: 0)
        let totalDurPadded = formatDuration(totalTime).leftPadding(toLength: 10, withPad: " ")
        report += "  \(totalPadded) \(totalDurPadded)  100.0%\n"

        // Step statistics
        if !currentStepTimes.isEmpty {
            let total = currentStepTimes.reduce(0, +)
            let average = total / Double(currentStepTimes.count)
            let minStep = currentStepTimes.min() ?? 0
            let maxStep = currentStepTimes.max() ?? 0

            report += "\n📈 DENOISING STEP STATISTICS:\n"
            report += "────────────────────────────────────────────────────────────────\n"
            report += "  Steps:              \(currentStepTimes.count)\n"
            report += "  Total denoising:    \(formatDuration(total))\n"
            report += "  Average per step:   \(formatDuration(average))\n"
            report += "  Fastest step:       \(formatDuration(minStep))\n"
            report += "  Slowest step:       \(formatDuration(maxStep))\n"

            // Estimate for different step counts
            report += "\n  📐 Estimated times for different step counts:\n"
            for stepCount in [10, 20, 28, 50] {
                let estimated = average * Double(stepCount)
                report += "     \(String(format: "%2d", stepCount)) steps: \(formatDuration(estimated))\n"
            }
        }

        // Performance insights
        report += "\n💡 INSIGHTS:\n"
        report += "────────────────────────────────────────────────────────────────\n"

        // Find slowest phase
        if let slowest = currentTimings.max(by: { $0.duration < $1.duration }) {
            let percentage = totalTime > 0 ? (slowest.duration / totalTime) * 100 : 0
            report += "  Bottleneck: \(slowest.name) (\(String(format: "%.1f", percentage))% of total)\n"
        }

        // Transformer vs overhead
        if let transformerEntry = currentTimings.first(where: { $0.name.contains("Denoising") }) {
            let overhead = totalTime - transformerEntry.duration
            report += "  Overhead (non-denoising): \(formatDuration(overhead))\n"
        }

        report += "\n╚══════════════════════════════════════════════════════════════╝\n"

        return report
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        if duration < 1 {
            return String(format: "%.1fms", duration * 1000)
        } else if duration < 60 {
            return String(format: "%.2fs", duration)
        } else {
            let minutes = Int(duration / 60)
            let seconds = duration.truncatingRemainder(dividingBy: 60)
            return String(format: "%dm %.1fs", minutes, seconds)
        }
    }
}
