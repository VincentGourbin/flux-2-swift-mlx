// ImageCleanup.swift — public entry point for SCUNet-based image cleanup.
//
// Standalone Flux2Core type (the agreed seam): the routing layer calls
// `ImageCleanup.scunet(_:)` on a source CGImage, then sizes with ImageScaler /
// ScalingPolicy. The SCUNet model + 72 MB weights are bundled (Bundle.module) and
// loaded once on first use.
//
// SCUNet runs at 1:1 (it's a denoiser / JPEG-artifact remover, not a scaler) and
// requires spatial dims that are multiples of 64; inputs are edge-padded to satisfy
// that and cropped back. Images larger than `wholeMaxDim` are processed in 512-px
// tiles with feathered overlap blending to bound memory and avoid seams.

import Foundation
import CoreGraphics
import MLX

public enum ImageCleanupError: LocalizedError {
    case weightsMissing
    case weightsIncomplete(matched: Int, missing: Int)
    case pixelBufferFailed
    case outputFailed

    public var errorDescription: String? {
        switch self {
        case .weightsMissing:
            return "SCUNet weights resource not found in the Flux2Core bundle."
        case .weightsIncomplete(let matched, let missing):
            return "SCUNet weights incomplete (matched \(matched), missing \(missing))."
        case .pixelBufferFailed:
            return "Failed to read pixels from the input image."
        case .outputFailed:
            return "Failed to build the cleaned output image."
        }
    }
}

public enum ImageCleanup {
    /// Largest dimension processed whole; larger images are tiled.
    private static let wholeMaxDim = 640
    private static let tile = 512
    private static let overlap = 64

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cachedModel: SCUNetModel?

    /// Preload the model + weights (e.g. off the main thread) so the first cleanup is fast.
    @discardableResult
    public static func warmUp() throws -> Bool {
        _ = try sharedModel()
        return true
    }

    private static func sharedModel() throws -> SCUNetModel {
        lock.lock()
        defer { lock.unlock() }
        if let model = cachedModel { return model }
        guard let url = Bundle.module.url(forResource: "scunet_color_real_psnr_mlx", withExtension: "safetensors") else {
            throw ImageCleanupError.weightsMissing
        }
        let model = SCUNetModel()
        let (matched, missing, _) = try model.loadWeights(fromSafetensors: url)
        guard matched > 0, missing.isEmpty else {
            throw ImageCleanupError.weightsIncomplete(matched: matched, missing: missing.count)
        }
        cachedModel = model
        return model
    }

    /// Clean a single image with SCUNet. Output matches the input resolution (1:1).
    public static func scunet(_ image: CGImage) throws -> CGImage {
        let model = try sharedModel()
        let width = image.width
        let height = image.height
        let input = try preprocess(image)

        let output: MLXArray
        if max(width, height) <= wholeMaxDim {
            output = runWhole(model, input)
        } else {
            output = runTiled(model, input, width: width, height: height)
        }
        guard let result = postprocess(output) else { throw ImageCleanupError.outputFailed }
        return result
    }

    // MARK: - Core passes

    /// Pad H and W up to a multiple of 64 by edge replication (SCUNet's ReplicationPad2d).
    private static func padTo64(_ x: MLXArray) -> (padded: MLXArray, h: Int, w: Int) {
        let h = x.shape[1]
        let w = x.shape[2]
        let padH = (64 - h % 64) % 64
        let padW = (64 - w % 64) % 64
        if padH == 0 && padW == 0 { return (x, h, w) }
        let padded = padded(x, widths: [[0, 0], [0, padH], [0, padW], [0, 0]], mode: .edge)
        return (padded, h, w)
    }

    private static func runWhole(_ model: SCUNetModel, _ input: MLXArray) -> MLXArray {
        let (padded, h, w) = padTo64(input)
        var y = model(padded)
        y = y[0..., 0..<h, 0..<w, 0...]
        y = clip(y, min: 0, max: 1)
        eval(y)
        return y
    }

    private static func runTiled(_ model: SCUNetModel, _ input: MLXArray, width: Int, height: Int) -> MLXArray {
        let xs = tileStarts(total: width)
        let ys = tileStarts(total: height)
        var outAccum = [Float](repeating: 0, count: height * width * 3)
        var weightAccum = [Float](repeating: 0, count: height * width)

        for y0 in ys {
            let th = min(tile, height - y0)
            let wy = featherRamp(th)
            for x0 in xs {
                let tw = min(tile, width - x0)
                let wx = featherRamp(tw)

                let tileIn = input[0..., y0..<(y0 + th), x0..<(x0 + tw), 0...]
                let (padded, _, _) = padTo64(tileIn)
                var tileOut = model(padded)
                tileOut = tileOut[0..., 0..<th, 0..<tw, 0...]
                tileOut = clip(tileOut, min: 0, max: 1)
                eval(tileOut)
                let flat = tileOut.asArray(Float.self)

                for j in 0..<th {
                    let gy = y0 + j
                    for i in 0..<tw {
                        let weight = wx[i] * wy[j]
                        let g = gy * width + (x0 + i)
                        let s = (j * tw + i) * 3
                        outAccum[g * 3 + 0] += flat[s + 0] * weight
                        outAccum[g * 3 + 1] += flat[s + 1] * weight
                        outAccum[g * 3 + 2] += flat[s + 2] * weight
                        weightAccum[g] += weight
                    }
                }
            }
        }

        var outFloat = [Float](repeating: 0, count: height * width * 3)
        for p in 0..<(height * width) {
            let inv = 1.0 / max(weightAccum[p], 1e-6)
            outFloat[p * 3 + 0] = outAccum[p * 3 + 0] * inv
            outFloat[p * 3 + 1] = outAccum[p * 3 + 1] * inv
            outFloat[p * 3 + 2] = outAccum[p * 3 + 2] * inv
        }
        return MLXArray(outFloat).reshaped([1, height, width, 3])
    }

    /// Tile start offsets covering [0, total); the last tile is flush with the far edge.
    private static func tileStarts(total: Int) -> [Int] {
        if total <= tile { return [0] }
        let hop = tile - overlap
        var starts: [Int] = []
        var p = 0
        while p + tile < total {
            starts.append(p)
            p += hop
        }
        starts.append(total - tile)
        return starts
    }

    /// Linear taper over `overlap` px at each edge; 1.0 in the interior. Division by the
    /// accumulated weight makes single-coverage (edge) pixels independent of the taper.
    private static func featherRamp(_ length: Int) -> [Float] {
        var r = [Float](repeating: 1, count: length)
        let o = Float(overlap)
        for i in 0..<length {
            let d = Float(min(i, length - 1 - i)) + 0.5
            r[i] = Swift.min(d / o, 1)
        }
        return r
    }

    // MARK: - CGImage bridge (RGB float NHWC in [0,1])

    private static func preprocess(_ image: CGImage) throws -> MLXArray {
        let w = image.width
        let h = image.height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil, width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: w * 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else {
            throw ImageCleanupError.pixelBufferFailed
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = context.data else { throw ImageCleanupError.pixelBufferFailed }
        let px = data.assumingMemoryBound(to: UInt8.self)

        var f = [Float](repeating: 0, count: h * w * 3)
        for y in 0..<h {
            for x in 0..<w {
                let o = (y * w + x) * 4
                let d = (y * w + x) * 3
                f[d + 0] = Float(px[o + 0]) / 255
                f[d + 1] = Float(px[o + 1]) / 255
                f[d + 2] = Float(px[o + 2]) / 255
            }
        }
        return MLXArray(f).reshaped([1, h, w, 3])
    }

    private static func postprocess(_ tensor: MLXArray) -> CGImage? {
        let image = clip(tensor.squeezed(axis: 0), min: 0, max: 1)   // (h,w,3)
        let u8 = (image * 255 + 0.5).asType(.uint8)
        eval(u8)
        let h = u8.shape[0]
        let w = u8.shape[1]
        let bytes = u8.asArray(UInt8.self)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(bytes) as CFData) else {
            return nil
        }
        return CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: w * 3,
            space: colorSpace, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
        )
    }
}
