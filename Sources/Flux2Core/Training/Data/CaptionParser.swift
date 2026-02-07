// CaptionParser.swift - Parse training captions from txt/jsonl files
// Copyright 2025 Vincent Gourbin

import Foundation

/// Parsed caption with metadata
public struct ParsedCaption: Sendable {
    /// The image filename (relative to dataset)
    public let filename: String
    
    /// The caption text
    public let caption: String
    
    /// Optional additional metadata
    public let metadata: [String: String]?
}

/// Parser for training dataset captions
public struct CaptionParser: Sendable {
    
    /// Trigger word to replace [trigger] placeholder
    public let triggerWord: String?
    
    /// Initialize parser
    /// - Parameter triggerWord: Word to replace [trigger] placeholder in captions
    public init(triggerWord: String? = nil) {
        self.triggerWord = triggerWord
    }
    
    // MARK: - Parse Individual Caption
    
    /// Parse caption from a .txt file
    /// - Parameter url: URL to the caption file
    /// - Returns: Parsed caption text
    public func parseTextFile(_ url: URL) throws -> String {
        let rawCaption = try String(contentsOf: url, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return processCaption(rawCaption)
    }
    
    /// Parse caption from a single JSONL line
    /// - Parameter line: JSON line string
    /// - Returns: Parsed caption entry
    public func parseJSONLine(_ line: String) throws -> ParsedCaption {
        guard let data = line.data(using: .utf8) else {
            throw CaptionParserError.invalidEncoding
        }
        
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let json = json else {
            throw CaptionParserError.invalidJSON(line)
        }
        
        guard let filename = json["file_name"] as? String else {
            throw CaptionParserError.missingField("file_name")
        }
        
        // Try different caption field names (prompt, caption, text)
        let rawCaption: String
        if let prompt = json["prompt"] as? String {
            rawCaption = prompt
        } else if let caption = json["caption"] as? String {
            rawCaption = caption
        } else if let text = json["text"] as? String {
            rawCaption = text
        } else {
            throw CaptionParserError.missingField("prompt/caption/text")
        }
        
        // Extract metadata (everything except filename and caption)
        var metadata: [String: String] = [:]
        for (key, value) in json {
            if key != "file_name" && key != "prompt" && key != "caption" && key != "text" {
                if let strValue = value as? String {
                    metadata[key] = strValue
                } else if let numValue = value as? NSNumber {
                    metadata[key] = numValue.stringValue
                }
            }
        }
        
        return ParsedCaption(
            filename: filename,
            caption: processCaption(rawCaption),
            metadata: metadata.isEmpty ? nil : metadata
        )
    }
    
    // MARK: - Parse Full Dataset
    
    /// Parse all captions from a dataset directory
    /// - Parameters:
    ///   - datasetPath: Path to the dataset directory
    ///   - extension: Caption file extension ("txt" or "jsonl")
    /// - Returns: Array of image filename to caption pairs
    public func parseDataset(at datasetPath: URL, extension ext: String) throws -> [(filename: String, caption: String)] {
        switch ext.lowercased() {
        case "txt":
            return try parseTextDataset(at: datasetPath)
        case "jsonl", "json":
            return try parseJSONLDataset(at: datasetPath)
        default:
            throw CaptionParserError.unsupportedFormat(ext)
        }
    }
    
    /// Parse dataset with .txt captions (one per image)
    private func parseTextDataset(at datasetPath: URL) throws -> [(filename: String, caption: String)] {
        let fileManager = FileManager.default
        
        // Get all files in directory
        let files = try fileManager.contentsOfDirectory(
            at: datasetPath,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        
        // Find image files and their corresponding caption files
        let imageExtensions = Set(["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
        var results: [(String, String)] = []
        
        for file in files {
            let ext = file.pathExtension.lowercased()
            guard imageExtensions.contains(ext) else { continue }
            
            // Look for corresponding .txt file
            let captionFile = file.deletingPathExtension().appendingPathExtension("txt")
            
            guard fileManager.fileExists(atPath: captionFile.path) else {
                Flux2Debug.log("[CaptionParser] Warning: No caption for \(file.lastPathComponent)")
                continue
            }
            
            let caption = try parseTextFile(captionFile)
            results.append((file.lastPathComponent, caption))
        }
        
        return results
    }
    
    /// Parse dataset with metadata.jsonl file
    private func parseJSONLDataset(at datasetPath: URL) throws -> [(filename: String, caption: String)] {
        let metadataFile = datasetPath.appendingPathComponent("metadata.jsonl")
        
        guard FileManager.default.fileExists(atPath: metadataFile.path) else {
            throw CaptionParserError.metadataFileNotFound(metadataFile)
        }
        
        let content = try String(contentsOf: metadataFile, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        
        var results: [(String, String)] = []
        
        for (index, line) in lines.enumerated() {
            do {
                let parsed = try parseJSONLine(line)
                results.append((parsed.filename, parsed.caption))
            } catch {
                Flux2Debug.log("[CaptionParser] Warning: Failed to parse line \(index + 1): \(error)")
            }
        }
        
        return results
    }
    
    // MARK: - Caption Processing
    
    /// Process caption text (replace trigger word, etc.)
    private func processCaption(_ caption: String) -> String {
        var processed = caption
        
        // Replace [trigger] placeholder with trigger word
        if let trigger = triggerWord {
            processed = processed.replacingOccurrences(of: "[trigger]", with: trigger)
            processed = processed.replacingOccurrences(of: "[TRIGGER]", with: trigger)
            processed = processed.replacingOccurrences(of: "{trigger}", with: trigger)
        }
        
        // Clean up whitespace
        processed = processed
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        
        return processed
    }
    
    // MARK: - Validation
    
    /// Validate dataset structure
    /// - Parameters:
    ///   - datasetPath: Path to dataset
    ///   - extension: Caption file extension
    /// - Returns: Validation result with warnings and errors
    public func validateDataset(at datasetPath: URL, extension ext: String) -> DatasetValidationResult {
        let fileManager = FileManager.default
        
        var errors: [String] = []
        var warnings: [String] = []
        var imageCount = 0
        var captionCount = 0
        
        // Check dataset directory exists
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: datasetPath.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            errors.append("Dataset directory not found: \(datasetPath.path)")
            return DatasetValidationResult(imageCount: 0, errors: errors, warnings: warnings)
        }
        
        do {
            let files = try fileManager.contentsOfDirectory(
                at: datasetPath,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            
            let imageExtensions = Set(["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
            
            if ext.lowercased() == "txt" {
                // Check for paired txt files
                for file in files {
                    let fileExt = file.pathExtension.lowercased()
                    if imageExtensions.contains(fileExt) {
                        imageCount += 1
                        let captionFile = file.deletingPathExtension().appendingPathExtension("txt")
                        if fileManager.fileExists(atPath: captionFile.path) {
                            captionCount += 1
                        } else {
                            warnings.append("Missing caption: \(file.lastPathComponent)")
                        }
                    }
                }
            } else {
                // Check for metadata.jsonl
                let metadataFile = datasetPath.appendingPathComponent("metadata.jsonl")
                if fileManager.fileExists(atPath: metadataFile.path) {
                    captionCount = try parseJSONLDataset(at: datasetPath).count
                } else {
                    errors.append("metadata.jsonl not found")
                }
                
                // Count images
                for file in files {
                    if imageExtensions.contains(file.pathExtension.lowercased()) {
                        imageCount += 1
                    }
                }
            }
            
            // Minimum image requirement
            if imageCount < 5 {
                warnings.append("Very small dataset (\(imageCount) images). Recommend at least 10-15 images.")
            }
            
            // Check for mismatched counts
            if imageCount != captionCount {
                warnings.append("\(imageCount) images but \(captionCount) captions")
            }
            
        } catch {
            errors.append("Failed to read dataset directory: \(error.localizedDescription)")
        }
        
        return DatasetValidationResult(
            imageCount: imageCount,
            errors: errors,
            warnings: warnings
        )
    }
}

// MARK: - Supporting Types

/// Result of dataset validation
public struct DatasetValidationResult: Sendable {
    public let imageCount: Int
    public let errors: [String]
    public let warnings: [String]
    
    public var isValid: Bool {
        errors.isEmpty && imageCount > 0
    }
    
    public var summary: String {
        var lines: [String] = []
        lines.append("Dataset Validation:")
        lines.append("  Images: \(imageCount)")
        
        if !errors.isEmpty {
            lines.append("  Errors:")
            for error in errors {
                lines.append("    - \(error)")
            }
        }
        
        if !warnings.isEmpty {
            lines.append("  Warnings:")
            for warning in warnings {
                lines.append("    - \(warning)")
            }
        }
        
        if isValid {
            lines.append("  Status: Valid")
        } else {
            lines.append("  Status: Invalid")
        }
        
        return lines.joined(separator: "\n")
    }
}

/// Errors during caption parsing
public enum CaptionParserError: Error, LocalizedError {
    case invalidEncoding
    case invalidJSON(String)
    case missingField(String)
    case unsupportedFormat(String)
    case metadataFileNotFound(URL)
    case captionFileNotFound(URL)
    
    public var errorDescription: String? {
        switch self {
        case .invalidEncoding:
            return "Invalid text encoding (expected UTF-8)"
        case .invalidJSON(let line):
            return "Invalid JSON line: \(line.prefix(50))..."
        case .missingField(let field):
            return "Missing required field: \(field)"
        case .unsupportedFormat(let format):
            return "Unsupported caption format: \(format)"
        case .metadataFileNotFound(let url):
            return "metadata.jsonl not found at: \(url.path)"
        case .captionFileNotFound(let url):
            return "Caption file not found: \(url.path)"
        }
    }
}
