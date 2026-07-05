/**
 * StatsViews.swift
 * Live and final generation stats bars plus the profiling Profile Details view
 * (and its byte-formatting helpers). Extracted from ContentView.swift.
 */

import SwiftUI
import AppKit
import FluxTextEncoders
import Flux2Core
import MLX

// MARK: - Live Stats Bar View (during generation)

struct LiveStatsBarView: View {
    let tokenCount: Int

    var body: some View {
        HStack(spacing: 16) {
            ProgressView()
                .scaleEffect(0.7)

            Image(systemName: "text.cursor")
                .foregroundStyle(.blue)

            Text("Generating (\(tokenCount) tokens)...")
                .foregroundStyle(.blue)
                .fontWeight(.medium)

            Spacer()
        }
        .font(.caption)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Stats Bar View (final)

struct StatsBarView: View {
    let stats: GenerationStats
    let profileSummary: ProfileSummary?
    @State private var showProfileDetails = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 20) {
                Label("\(stats.tokenCount) tokens", systemImage: "number")
                Label(String(format: "%.1fs", stats.duration), systemImage: "clock")
                Label(String(format: "%.1f tok/s", stats.tokensPerSecond), systemImage: "speedometer")

                Spacer()

                if profileSummary != nil {
                    Button(action: { showProfileDetails.toggle() }) {
                        Label(showProfileDetails ? "Hide Profile" : "Show Profile",
                              systemImage: showProfileDetails ? "chevron.up" : "chevron.down")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.blue)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal)
            .padding(.vertical, 8)

            if showProfileDetails, let summary = profileSummary {
                ProfileDetailsView(summary: summary)
            }
        }
        .background(.ultraThinMaterial)
    }
}

// MARK: - Profile Details View

struct ProfileDetailsView: View {
    let summary: ProfileSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()

            // Device info
            HStack {
                Text("Device:")
                    .foregroundStyle(.secondary)
                Text(summary.deviceInfo.architecture)
                    .fontWeight(.medium)
                Spacer()
                Text("RAM: \(formatBytesUI(summary.deviceInfo.memorySize))")
                    .foregroundStyle(.secondary)
            }
            .font(.caption)

            Divider()

            // Steps table header
            HStack {
                Text("Step")
                    .frame(width: 140, alignment: .leading)
                Text("Time")
                    .frame(width: 70, alignment: .trailing)
                Text("MLX \u{0394}")
                    .frame(width: 80, alignment: .trailing)
                Text("Process \u{0394}")
                    .frame(width: 80, alignment: .trailing)
            }
            .font(.caption2.bold())
            .foregroundStyle(.secondary)

            // Steps
            ForEach(Array(summary.steps.enumerated()), id: \.offset) { _, step in
                HStack {
                    Text(step.name)
                        .frame(width: 140, alignment: .leading)
                        .lineLimit(1)
                    Text(String(format: "%.3fs", step.duration))
                        .frame(width: 70, alignment: .trailing)
                    Text(formatDeltaBytesUI(step.endMemory.mlxActive - step.startMemory.mlxActive))
                        .frame(width: 80, alignment: .trailing)
                        .foregroundStyle(step.endMemory.mlxActive > step.startMemory.mlxActive ? .orange : .green)
                    Text(formatDeltaBytesUI(Int(step.endMemory.processFootprint - step.startMemory.processFootprint)))
                        .frame(width: 80, alignment: .trailing)
                        .foregroundStyle(step.endMemory.processFootprint > step.startMemory.processFootprint ? .orange : .green)
                }
                .font(.caption)
            }

            Divider()

            // Totals
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Peak")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(summary.peakMemoryUsed))
                        .fontWeight(.medium)
                        .foregroundStyle(.orange)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Active")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(summary.finalSnapshot.mlxActive))
                        .fontWeight(.medium)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Cache")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(summary.finalSnapshot.mlxCache))
                        .fontWeight(.medium)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("Process")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(Int(summary.finalSnapshot.processFootprint)))
                        .fontWeight(.medium)
                        .foregroundStyle(.blue)
                }

                Spacer()
            }
            .font(.caption)
        }
        .padding(.horizontal)
        .padding(.bottom, 8)
    }
}

// Helper functions for formatting
private func formatBytesUI(_ bytes: Int) -> String {
    let absBytes = abs(bytes)
    if absBytes >= 1024 * 1024 * 1024 {
        return String(format: "%.2f GB", Double(bytes) / (1024 * 1024 * 1024))
    } else if absBytes >= 1024 * 1024 {
        return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
    } else if absBytes >= 1024 {
        return String(format: "%.1f KB", Double(bytes) / 1024)
    }
    return "\(bytes) B"
}

private func formatDeltaBytesUI(_ bytes: Int) -> String {
    let sign = bytes >= 0 ? "+" : ""
    return sign + formatBytesUI(bytes)
}

