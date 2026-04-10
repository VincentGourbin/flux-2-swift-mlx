// Flux2Profiler.swift - Performance Profiling for Flux.2
// Copyright 2025 Vincent Gourbin

// MARK: - Profiler (re-exported from MLXProfiler package)

@_exported import MLXProfiler

/// Backward-compatible alias for Flux2Profiler
public typealias Flux2Profiler = MLXProfiler

// MARK: - Flux.2 Report Generation

extension MLXProfiler {
    /// Generate a Flux.2 performance report from recorded timings.
    ///
    /// If an active ProfilingSession exists, delegates to its full report.
    /// Otherwise, generates a simple report from the profiler's timing data.
    public func generateReport() -> String {
        if let session = activeSession {
            return session.generateReport()
        }

        // Simple report from timing data
        let timings = getTimings()
        let stepTimes = getStepTimes()

        guard !timings.isEmpty else { return "" }

        let totalDuration = timings.reduce(0.0) { $0 + $1.duration }

        var lines: [String] = []
        lines.append("")
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                  FLUX.2 PERFORMANCE REPORT                   ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append("📊 PHASE TIMINGS:")
        lines.append("────────────────────────────────────────────────────────────────")

        for timing in timings {
            let pct = totalDuration > 0 ? (timing.duration / totalDuration * 100) : 0
            let bars = String(repeating: "█", count: max(0, Int(pct / 5)))
            let durationStr: String
            if timing.duration < 0.1 {
                durationStr = String(format: "%5.1fms", timing.duration * 1000)
            } else {
                durationStr = String(format: "%5.2fs ", timing.duration)
            }
            lines.append("  \(timing.name.padding(toLength: 30, withPad: " ", startingAt: 0)) \(durationStr) \(String(format: "%5.1f%%", pct)) \(bars)")
        }

        lines.append("────────────────────────────────────────────────────────────────")
        let totalStr = String(format: "%.2fs", totalDuration)
        lines.append("  TOTAL\(String(repeating: " ", count: 30)) \(totalStr)  100.0%")

        if !stepTimes.isEmpty {
            lines.append("")
            lines.append("📈 DENOISING STEP STATISTICS:")
            lines.append("────────────────────────────────────────────────────────────────")
            let totalDenoising = stepTimes.reduce(0, +)
            let avgStep = totalDenoising / Double(stepTimes.count)
            let minStep = stepTimes.min() ?? 0
            let maxStep = stepTimes.max() ?? 0
            lines.append("  Steps:              \(stepTimes.count)")
            lines.append("  Total denoising:    \(String(format: "%.2f", totalDenoising))s")
            lines.append("  Average per step:   \(String(format: "%.2f", avgStep))s")
            lines.append("  Fastest step:       \(String(format: "%.2f", minStep))s")
            lines.append("  Slowest step:       \(String(format: "%.2f", maxStep))s")

            lines.append("")
            lines.append("  📐 Estimated times for different step counts:")
            for n in [10, 20, 28, 50] {
                let est = avgStep * Double(n)
                if est >= 60 {
                    lines.append("     \(n) steps: \(String(format: "%.0f", est / 60))m \(String(format: "%.1f", est.truncatingRemainder(dividingBy: 60)))s")
                } else {
                    lines.append("     \(n) steps: \(String(format: "%.2f", est))s")
                }
            }
        }

        // Insights
        if let bottleneck = timings.max(by: { $0.duration < $1.duration }) {
            let pct = totalDuration > 0 ? (bottleneck.duration / totalDuration * 100) : 0
            let overhead = totalDuration - (stepTimes.reduce(0, +))
            lines.append("")
            lines.append("💡 INSIGHTS:")
            lines.append("────────────────────────────────────────────────────────────────")
            lines.append("  Bottleneck: \(bottleneck.name) (\(String(format: "%.1f", pct))% of total)")
            lines.append("  Overhead (non-denoising): \(String(format: "%.2f", overhead))s")
        }

        lines.append("")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")

        return lines.joined(separator: "\n")
    }
}
