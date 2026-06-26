/**
 * NonAutofocusTextField.swift
 * Text field that does not take first responder on appear (macOS).
 */

import SwiftUI

#if canImport(AppKit)
import AppKit

/// Accepts focus only after the user clicks — avoids stealing focus from the preview on launch.
struct NonAutofocusTextField: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String = ""
    var width: CGFloat = 60

    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text)
    }

    func makeNSView(context: Context) -> NonAutofocusNSTextField {
        let field = NonAutofocusNSTextField()
        field.delegate = context.coordinator
        field.isBezeled = true
        field.bezelStyle = .roundedBezel
        field.stringValue = text
        field.placeholderString = placeholder
        field.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
        context.coordinator.textField = field
        return field
    }

    func updateNSView(_ nsView: NonAutofocusNSTextField, context: Context) {
        context.coordinator.text = $text
        if nsView.stringValue != text {
            nsView.stringValue = text
        }
        nsView.preferredMaxLayoutWidth = width
    }

    final class Coordinator: NSObject, NSTextFieldDelegate {
        var text: Binding<String>
        weak var textField: NonAutofocusNSTextField?

        init(text: Binding<String>) {
            self.text = text
        }

        func controlTextDidChange(_ obj: Notification) {
            guard let field = obj.object as? NSTextField else { return }
            text.wrappedValue = field.stringValue
        }
    }
}

final class NonAutofocusNSTextField: NSTextField {
    private var userClicked = false

    override func mouseDown(with event: NSEvent) {
        userClicked = true
        super.mouseDown(with: event)
    }

    override func becomeFirstResponder() -> Bool {
        guard userClicked else { return false }
        return super.becomeFirstResponder()
    }
}
#endif
