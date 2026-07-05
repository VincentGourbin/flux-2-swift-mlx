import SwiftUI

#if canImport(AppKit)
import AppKit
#endif

/// Classic black/white dashed "marching ants" outline (Photoshop-style).
struct MarchingAntsPath: View {
    let path: Path
    var lineWidth: CGFloat = 1.5
    var dash: [CGFloat] = [6, 6]

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 12.0)) { timeline in
            let period = dash.reduce(0, +)
            let phase = CGFloat(
                timeline.date.timeIntervalSinceReferenceDate
                    .truncatingRemainder(dividingBy: 1)
            ) * period

            ZStack {
                path.stroke(
                    Color.black,
                    style: StrokeStyle(lineWidth: lineWidth, lineCap: .butt, dash: dash, dashPhase: phase)
                )
                path.stroke(
                    Color.white,
                    style: StrokeStyle(lineWidth: lineWidth, lineCap: .butt, dash: dash, dashPhase: phase + dash[0])
                )
            }
        }
        .allowsHitTesting(false)
    }
}

struct MarchingAntsRect: View {
    let rect: CGRect

    var body: some View {
        MarchingAntsPath(path: Path { $0.addRect(rect) })
    }
}

extension View {
    func marchingAntsBorder(in rect: CGRect) -> some View {
        overlay {
            MarchingAntsRect(rect: rect)
        }
    }
}

/// Transient add/subtract hint shown at the cursor's lower-right while ⇧ / ⌥ are held.
enum SelectionModifierHint: Equatable {
    case add
    case subtract

    #if canImport(AppKit)
    static func from(flags: NSEvent.ModifierFlags) -> SelectionModifierHint? {
        if flags.contains(.option) { return .subtract }
        if flags.contains(.shift) { return .add }
        return nil
    }
    #endif
}

struct SelectionModifierCornerBadge: View {
    let hint: SelectionModifierHint

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 2)
                .fill(Color.white)
            RoundedRectangle(cornerRadius: 2)
                .stroke(Color.black, lineWidth: 1)
            Image(systemName: hint == .add ? "plus" : "minus")
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.black)
        }
        .frame(width: 14, height: 14)
        .allowsHitTesting(false)
    }
}
