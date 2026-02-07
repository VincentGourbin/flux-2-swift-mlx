#!/usr/bin/env swift
// fix_learning_curve.swift - Regenerate learning curve SVG with correct scaling
//
// Usage: swift fix_learning_curve.swift <input.svg> [output.svg]
//
// This script fixes learning curve SVGs where the Y-axis scale was computed
// from smoothed data only, causing raw data points to go outside the plot area.

import Foundation

// MARK: - Helpers

func extractCoordinates(from pathData: String) -> [(x: Float, y: Float)] {
    var coords: [(x: Float, y: Float)] = []
    let parts = pathData.components(separatedBy: CharacterSet(charactersIn: "ML "))
        .filter { !$0.isEmpty }

    var i = 0
    while i < parts.count - 1 {
        if let x = Float(parts[i]), let y = Float(parts[i + 1]) {
            coords.append((x, y))
        }
        i += 2
    }
    return coords
}

func generateAxisLabels(min: Float, max: Float, count: Int) -> [Float] {
    let step = (max - min) / Float(count - 1)
    return (0..<count).map { min + Float($0) * step }
}

// MARK: - Main

guard CommandLine.arguments.count >= 2 else {
    print("Usage: swift fix_learning_curve.swift <input.svg> [output.svg]")
    exit(1)
}

let inputPath = CommandLine.arguments[1]
let outputPath = CommandLine.arguments.count >= 3
    ? CommandLine.arguments[2]
    : inputPath.replacingOccurrences(of: ".svg", with: "_fixed.svg")

guard let svgContent = try? String(contentsOfFile: inputPath, encoding: .utf8) else {
    print("Error: Cannot read \(inputPath)")
    exit(1)
}

// Extract path data using regex
let rawPathPattern = #"class="raw-line"[^/]*d="([^"]+)""#
let smoothedPathPattern = #"class="smooth-line"[^/]*d="([^"]+)""#

// Alternative patterns (d comes before class)
let rawPathPattern2 = #"d="([^"]+)"[^/]*class="raw-line""#
let smoothedPathPattern2 = #"d="([^"]+)"[^/]*class="smooth-line""#

func extractPath(_ content: String, patterns: [String]) -> String? {
    for pattern in patterns {
        if let regex = try? NSRegularExpression(pattern: pattern),
           let match = regex.firstMatch(in: content, range: NSRange(content.startIndex..., in: content)),
           let range = Range(match.range(at: 1), in: content) {
            return String(content[range])
        }
    }
    return nil
}

guard let rawPathData = extractPath(svgContent, patterns: [rawPathPattern, rawPathPattern2]) else {
    print("Error: Cannot find raw-line path in SVG")
    exit(1)
}

guard let smoothedPathData = extractPath(svgContent, patterns: [smoothedPathPattern, smoothedPathPattern2]) else {
    print("Error: Cannot find smooth-line path in SVG")
    exit(1)
}

// SVG dimensions (standard)
let width: Float = 800
let height: Float = 400
let padding: Float = 60
let plotWidth = width - 2 * padding
let plotHeight = height - 2 * padding

// Extract old scale from SVG axis labels
let axisLabelPattern = #"text-anchor="end"[^>]*>(\d+\.\d+)</text>"#
var oldLossLabels: [Float] = []
if let regex = try? NSRegularExpression(pattern: axisLabelPattern) {
    let matches = regex.matches(in: svgContent, range: NSRange(svgContent.startIndex..., in: svgContent))
    for match in matches {
        if let range = Range(match.range(at: 1), in: svgContent),
           let value = Float(svgContent[range]) {
            oldLossLabels.append(value)
        }
    }
}

guard oldLossLabels.count >= 2 else {
    print("Error: Cannot extract axis labels from SVG")
    exit(1)
}

let oldPaddedMinLoss = oldLossLabels.min()!
let oldPaddedMaxLoss = oldLossLabels.max()!
let oldPaddedLossRange = oldPaddedMaxLoss - oldPaddedMinLoss

// Extract step range from SVG
let stepLabelPattern = #"text-anchor="middle"[^>]*>(\d+)</text>"#
var stepLabelsFound: [Float] = []
if let regex = try? NSRegularExpression(pattern: stepLabelPattern) {
    let matches = regex.matches(in: svgContent, range: NSRange(svgContent.startIndex..., in: svgContent))
    for match in matches {
        if let range = Range(match.range(at: 1), in: svgContent),
           let value = Float(svgContent[range]) {
            stepLabelsFound.append(value)
        }
    }
}

let minStep = stepLabelsFound.min() ?? 1
let maxStep = stepLabelsFound.max() ?? 200
let stepRange = max(maxStep - minStep, 1)

print("Detected scale: loss [\(oldPaddedMinLoss) - \(oldPaddedMaxLoss)], steps [\(minStep) - \(maxStep)]")

// Convert SVG coordinates back to data
func yToLoss(_ y: Float) -> Float {
    let normalized = 1 - (y - padding) / plotHeight
    return normalized * oldPaddedLossRange + oldPaddedMinLoss
}

func xToStep(_ x: Float) -> Float {
    return ((x - padding) / plotWidth) * stepRange + minStep
}

let rawCoords = extractCoordinates(from: rawPathData)
let smoothedCoords = extractCoordinates(from: smoothedPathData)

var rawLossData: [(step: Int, loss: Float)] = []
for coord in rawCoords {
    let step = Int(xToStep(coord.x).rounded())
    let loss = yToLoss(coord.y)
    rawLossData.append((step, loss))
}

var smoothedLossData: [(step: Int, loss: Float)] = []
for coord in smoothedCoords {
    let step = Int(xToStep(coord.x).rounded())
    let loss = yToLoss(coord.y)
    smoothedLossData.append((step, loss))
}

print("Recovered \(rawLossData.count) raw data points")

// Compute NEW scaling based on RAW data (the fix)
let rawLosses = rawLossData.map { $0.loss }
let newMinLoss = rawLosses.min()!
let newMaxLoss = rawLosses.max()!
let newLossRange = newMaxLoss - newMinLoss

let newPaddedMinLoss = newMinLoss - newLossRange * 0.1
let newPaddedMaxLoss = newMaxLoss + newLossRange * 0.1
let newPaddedLossRange = newPaddedMaxLoss - newPaddedMinLoss

print("New scale: loss [\(String(format: "%.3f", newPaddedMinLoss)) - \(String(format: "%.3f", newPaddedMaxLoss))]")

// Convert data to NEW SVG coordinates
func toSVG(step: Int, loss: Float) -> (x: Float, y: Float) {
    let x = padding + (Float(step) - minStep) / stepRange * plotWidth
    let y = padding + (1 - (loss - newPaddedMinLoss) / newPaddedLossRange) * plotHeight
    return (x, y)
}

// Build new SVG paths
var newRawPath = ""
for (i, point) in rawLossData.enumerated() {
    let (x, y) = toSVG(step: point.step, loss: point.loss)
    newRawPath += i == 0 ? "M \(x) \(y)" : " L \(x) \(y)"
}

var newSmoothedPath = ""
for (i, point) in smoothedLossData.enumerated() {
    let (x, y) = toSVG(step: point.step, loss: point.loss)
    newSmoothedPath += i == 0 ? "M \(x) \(y)" : " L \(x) \(y)"
}

// Generate new axis labels
let newStepLabels = generateAxisLabels(min: minStep, max: maxStep, count: 5)
let newLossLabels = generateAxisLabels(min: newPaddedMinLoss, max: newPaddedMaxLoss, count: 5)

// Build new SVG
var svg = """
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 \(width) \(height)" width="\(Int(width))" height="\(Int(height))">
  <style>
    .title { font: bold 16px sans-serif; }
    .label { font: 12px sans-serif; }
    .axis-label { font: 10px sans-serif; fill: #666; }
    .grid { stroke: #e0e0e0; stroke-width: 1; }
    .raw-line { fill: none; stroke: #ccc; stroke-width: 1; }
    .smooth-line { fill: none; stroke: #2196F3; stroke-width: 2; }
    .current-loss { font: bold 14px sans-serif; fill: #2196F3; }
  </style>

  <!-- Background -->
  <rect width="\(width)" height="\(height)" fill="white"/>

  <!-- Title -->
  <text x="\(width/2)" y="25" text-anchor="middle" class="title">Learning Curve</text>

  <!-- Grid lines -->
"""

for label in newLossLabels {
    let (_, y) = toSVG(step: Int(minStep), loss: label)
    svg += """
      <line x1="\(padding)" y1="\(y)" x2="\(width - padding)" y2="\(y)" class="grid"/>
      <text x="\(padding - 5)" y="\(y + 4)" text-anchor="end" class="axis-label">\(String(format: "%.3f", label))</text>
    """
}

for label in newStepLabels {
    let (x, _) = toSVG(step: Int(label), loss: newPaddedMinLoss)
    svg += """
      <line x1="\(x)" y1="\(padding)" x2="\(x)" y2="\(height - padding)" class="grid"/>
      <text x="\(x)" y="\(height - padding + 15)" text-anchor="middle" class="axis-label">\(Int(label))</text>
    """
}

svg += """
  <text x="\(width/2)" y="\(height - 10)" text-anchor="middle" class="label">Step</text>
  <text x="15" y="\(height/2)" text-anchor="middle" class="label" transform="rotate(-90, 15, \(height/2))">Loss</text>
"""

svg += """
  <path d="\(newRawPath)" class="raw-line"/>
  <path d="\(newSmoothedPath)" class="smooth-line"/>
"""

if let last = smoothedLossData.last {
    let (x, y) = toSVG(step: last.step, loss: last.loss)
    svg += """
      <circle cx="\(x)" cy="\(y)" r="4" fill="#2196F3"/>
      <text x="\(x + 10)" y="\(y + 5)" class="current-loss">\(String(format: "%.4f", last.loss))</text>
    """
}

let currentLoss = rawLossData.last?.loss ?? 0
let avgLoss = smoothedLossData.last?.loss ?? 0
svg += """
  <rect x="\(width - 150)" y="40" width="140" height="60" fill="white" stroke="#ddd" rx="4"/>
  <text x="\(width - 145)" y="58" class="axis-label">Step: \(rawLossData.last?.step ?? 0)</text>
  <text x="\(width - 145)" y="73" class="axis-label">Loss: \(String(format: "%.4f", currentLoss))</text>
  <text x="\(width - 145)" y="88" class="axis-label">Smoothed: \(String(format: "%.4f", avgLoss))</text>
</svg>
"""

do {
    try svg.write(toFile: outputPath, atomically: true, encoding: .utf8)
    print("Fixed SVG written to: \(outputPath)")
} catch {
    print("Error writing output: \(error)")
    exit(1)
}
