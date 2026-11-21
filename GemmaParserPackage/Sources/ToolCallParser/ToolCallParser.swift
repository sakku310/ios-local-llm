// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Streaming parser that detects tool call blocks in the model's token stream.
struct ToolCallParser {
    enum Event {
        case text(String)
        case toolCall(id: String, name: String, argsJSON: String)
    }

    private let openMarker: String
    private let closeMarker: String
    private let gemmaOpenMarker: String = "<|tool_call>"
    private let gemmaCloseMarker: String = "<tool_call|>"
    private var buffer: String = ""
    private var isInsideToolCall: Bool = false

    init(openMarker: String = "<tool_call>", closeMarker: String = "</tool_call>") {
        self.openMarker = openMarker
        self.closeMarker = closeMarker
    }

    mutating func consume(_ delta: String) -> [Event] {
        buffer.append(delta)
        return drain(isFinal: false)
    }

    /// Emit any pending buffered content as final events.
    ///
    /// Required at end of stream — without it, text held back to wait for a
    /// possible marker match is silently lost. An unclosed `<tool_call>` block
    /// at EOS is dropped (malformed JSON is not useful to surface as text).
    /// Exception: newline-terminated formats (e.g. Mistral's `[TOOL_CALLS]`)
    /// have no trailing close token, so the buffered content is parsed on EOS.
    mutating func flush() -> [Event] {
        drain(isFinal: true)
    }

    private mutating func drain(isFinal: Bool) -> [Event] {
        var events: [Event] = []
        while true {
            // Determine candidates based on current state
            let candidates: [(String, Bool)] // (marker, isOpen)
            if isInsideToolCall {
                candidates = [(closeMarker, false), (gemmaCloseMarker, false)]
            } else {
                candidates = [(openMarker, true), (gemmaOpenMarker, true)]
            }
            // Find earliest occurrence among candidates
            var earliestRange: Range<String.Index>? = nil
            var matchedMarker: String? = nil
            for (marker, _) in candidates {
                if let range = buffer.range(of: marker) {
                    if earliestRange == nil || range.lowerBound < earliestRange!.lowerBound {
                        earliestRange = range
                        matchedMarker = marker
                    }
                }
            }
            guard let range = earliestRange else { break }
            // Process up to the marker
            processMarker(at: range, events: &events)
            // Toggle state (entering or exiting a tool call block)
            isInsideToolCall.toggle()
        }
        processRemainder(of: &events, isFinal: isFinal)
        return events
    }

    // Updated to accept any marker; behavior unchanged
    private mutating func processMarker(at range: Range<String.Index>, events: inout [Event]) {
        let before = String(buffer[buffer.startIndex..<range.lowerBound])
        if isInsideToolCall {
            events.append(contentsOf: parseToolCalls(from: before))
        } else if !before.isEmpty {
            events.append(.text(before))
        }
        buffer = String(buffer[range.upperBound...])
    }

    private mutating func processRemainder(of events: inout [Event], isFinal: Bool) {
        if isInsideToolCall {
            guard isFinal else { return }
            // Newline-terminated formats (e.g. Mistral) have no dedicated close
            // token — the block ends at EOS. Try to parse what we have.
            // For tag‑pair formats (including Gemma) an unclosed block is considered
            // malformed. Emit the raw buffer as plain text so the user sees the
            // incomplete marker rather than silently dropping it.
            if closeMarker == "\n" {
                events.append(contentsOf: parseToolCalls(from: buffer))
            } else {
                // Emit as text – this matches the EOS‑incomplete test expectation.
                if !buffer.isEmpty { events.append(.text(buffer)) }
            }
            buffer = ""
        } else {
            // Determine safe index considering both possible open markers
            let safeIndex: String.Index
            if isFinal {
                safeIndex = buffer.endIndex
            } else {
                let safe1 = lastSafeIndex(for: openMarker)
                let safe2 = lastSafeIndex(for: gemmaOpenMarker)
                safeIndex = safe1 < safe2 ? safe1 : safe2
            }
            if safeIndex > buffer.startIndex {
                let toEmit = String(buffer[buffer.startIndex..<safeIndex])
                if !toEmit.isEmpty { events.append(.text(toEmit)) }
                buffer = String(buffer[safeIndex...])
            }
        }
    }

    // MARK: - Mixed A/B/Gemma format handling
    /// Parses the content of a `<tool_call>` block. It detects:
    ///   * Gemma 4 format: `<|tool_call>call:NAME{args}<tool_call|>`
    ///   * B‑format: `<function=NAME>QUERY</function>`
    ///   * A‑format: JSON object `{...}` or array `[{...}]`
    /// The detection preserves the original order of occurrences.
    private func parseToolCalls(from json: String) -> [Event] {
        let trimmed = json.trimmingCharacters(in: .whitespacesAndNewlines)
        var events: [Event] = []
        var consumedRanges: [NSRange] = []

        // ---- Gemma 4 format ----
        let gemmaPattern = #"<\|tool_call>call:([^\{\s]+)\{([^}]*)\}<tool_call\|>"#
        if let gemmaRegex = try? NSRegularExpression(pattern: gemmaPattern, options: []) {
            let range = NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)
            gemmaRegex.enumerateMatches(in: trimmed, options: [], range: range) { match, _, _ in
                guard let match = match,
                      let nameRange = Range(match.range(at: 1), in: trimmed),
                      let argsRange = Range(match.range(at: 2), in: trimmed) else { return }
                let rawName = String(trimmed[nameRange])
                let name = (rawName == "web_search" || rawName == "websearch") ? "web_search" : rawName
                guard name == "web_search" else { return }
                let argsBody = String(trimmed[argsRange])
                // Minimal JSON‑compatible wrapper – keeps the original
                // semantics (Int, Double, Bool, String) while satisfying
                // `Encodable`.  No array / object / null handling is added
                // because the parser never produces those types.
                enum JSONValue: Encodable {
                    case int(Int)
                    case double(Double)
                    case bool(Bool)
                    case string(String)

                    func encode(to encoder: Encoder) throws {
                        var container = encoder.singleValueContainer()
                        switch self {
                        case .int(let v):   try container.encode(v)
                        case .double(let v):try container.encode(v)
                        case .bool(let v):  try container.encode(v)
                        case .string(let v):try container.encode(v)
                        }
                    }
                }
                var argsDict: [String: JSONValue] = [:]

                // Quoted string arguments: key:<|"|>value<|"|>
                let kvPattern = #"(\w+):<\|\"\|>(.*?)<\|\"\|>"#
                if let kvRegex = try? NSRegularExpression(pattern: kvPattern, options: []) {
                    let kvRange = NSRange(argsBody.startIndex..<argsBody.endIndex, in: argsBody)
                    kvRegex.enumerateMatches(in: argsBody, options: [], range: kvRange) { kvMatch, _, _ in
                        guard let kvMatch = kvMatch,
                              let keyRange = Range(kvMatch.range(at: 1), in: argsBody),
                              let valRange = Range(kvMatch.range(at: 2), in: argsBody) else { return }
                        let key = String(argsBody[keyRange])
                        let rawVal = String(argsBody[valRange])
                        if let intVal = Int(rawVal) {
                            argsDict[key] = .int(intVal)
                        } else if let doubleVal = Double(rawVal) {
                            argsDict[key] = .double(doubleVal)
                        } else if rawVal.lowercased() == "true" {
                            argsDict[key] = .bool(true)
                        } else if rawVal.lowercased() == "false" {
                            argsDict[key] = .bool(false)
                        } else {
                            argsDict[key] = .string(rawVal)
                        }
                    }
                }
                // Unquoted arguments (e.g., max_results:3)
                let otherPattern = #"(\w+):([^,]+)"#
                if let otherRegex = try? NSRegularExpression(pattern: otherPattern, options: []) {
                    let otherRange = NSRange(argsBody.startIndex..<argsBody.endIndex, in: argsBody)
                    otherRegex.enumerateMatches(in: argsBody, options: [], range: otherRange) { otherMatch, _, _ in
                        guard let otherMatch = otherMatch,
                              let keyRange = Range(otherMatch.range(at: 1), in: argsBody),
                              let valRange = Range(otherMatch.range(at: 2), in: argsBody) else { return }
                        let key = String(argsBody[keyRange])
                        if argsDict[key] != nil { return }
                        let rawVal = String(argsBody[valRange])
                        if let intVal = Int(rawVal) {
                            argsDict[key] = .int(intVal)
                        } else if let doubleVal = Double(rawVal) {
                            argsDict[key] = .double(doubleVal)
                        } else if rawVal.lowercased() == "true" {
                            argsDict[key] = .bool(true)
                        } else if rawVal.lowercased() == "false" {
                            argsDict[key] = .bool(false)
                        } else {
                            argsDict[key] = .string(rawVal)
                        }
                    }
                }
                if argsDict.isEmpty && !argsBody.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    argsDict["query"] = argsBody
                }
                if let argsData = try? JSONEncoder().encode(argsDict),
                   let argsJSON = String(data: argsData, encoding: .utf8) {
                    events.append(.toolCall(id: UUID().uuidString, name: name, argsJSON: argsJSON))
                }
                consumedRanges.append(match.range(at: 0))
            }
        }

        // ---- B‑format and JSON handling (existing) ----
        let pattern = #"(\u003cfunction=([^\u003e]+)\u003e([\s\S]*?)\u003c/function\u003e)|([\{\[][^\{\}\[\]]*[\}\]])"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return events + parseJSONOnly(trimmed)
        }
        let range = NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)
        regex.enumerateMatches(in: trimmed, options: [], range: range) { match, _, _ in
            guard let match = match else { return }
            // B‑format match?
            if let _ = Range(match.range(at: 1), in: trimmed) {
                if let nameRange = Range(match.range(at: 2), in: trimmed),
                   let queryRange = Range(match.range(at: 3), in: trimmed) {
                    let name = String(trimmed[nameRange])
                    guard name == "websearch" else { return }
                    let rawQuery = String(trimmed[queryRange])
                    let trimmedQuery = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmedQuery.isEmpty else { return }
                    if rawQuery.contains("<function=") { return }
                    let argsDict = ["query": rawQuery]
                    if let argsData = try? JSONEncoder().encode(argsDict),
                       let argsJSON = String(data: argsData, encoding: .utf8) {
                        events.append(.toolCall(id: UUID().uuidString, name: name, argsJSON: argsJSON))
                    }
                    consumedRanges.append(match.range(at: 1))
                }
                return
            }
            // JSON match (object or array)
            if let jsonRange = Range(match.range(at: 4), in: trimmed) {
                let jsonSnippet = String(trimmed[jsonRange])
                events.append(contentsOf: parseJSONOnly(jsonSnippet))
                consumedRanges.append(match.range(at: 4))
            }
        }

        // Add any leftover text preserving order
        var cursor = trimmed.startIndex
        for r in consumedRanges.sorted(by: { $0.location < $1.location }) {
            let nsStart = trimmed.index(trimmed.startIndex, offsetBy: r.location)
            if cursor < nsStart {
                let segment = String(trimmed[cursor..<nsStart])
                if !segment.isEmpty { events.append(.text(segment)) }
            }
            cursor = trimmed.index(trimmed.startIndex, offsetBy: r.location + r.length)
        }
        if cursor < trimmed.endIndex {
            let tail = String(trimmed[cursor..<trimmed.endIndex])
            if !tail.isEmpty { events.append(.text(tail)) }
        }
        return events
    }

    // Helper that parses pure JSON (object or array) without B‑format handling.
    private func parseJSONOnly(_ snippet: String) -> [Event] {
        guard let data = snippet.data(using: .utf8) else { return [] }
        if snippet.hasPrefix("[") {
            if let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                return array.compactMap { makeToolCallEvent(from: $0) }
            }
        } else if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return makeToolCallEvent(from: obj).map { [$0] } ?? []
        }
        return []
    }

    // MARK: - B‑format handling (legacy wrapper for external calls)
    // Kept for backward compatibility – not used by the new mixed parser.
    private func parseBFormat(_ text: String) -> [Event]? {
        let pattern = #"<function=([^\>]+)>([\s\S]*?)</function>"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        var events: [Event] = []
        regex.enumerateMatches(in: text, options: []) { match, _, _ in
            guard let match = match,
                  let nameRange = Range(match.range(at: 1), in: text),
                  let queryRange = Range(match.range(at: 2), in: text) else { return }
            let name = String(text[nameRange])
            guard name == "websearch" else { return }
            let rawQuery = String(text[queryRange])
            let trimmedQuery = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedQuery.isEmpty else { return }
            if rawQuery.contains("<function=") { return }
            let argsDict = ["query": rawQuery]
            guard let argsData = try? JSONEncoder().encode(argsDict),
                  let argsJSON = String(data: argsData, encoding: .utf8) else { return }
            events.append(.toolCall(id: UUID().uuidString, name: name, argsJSON: argsJSON))
        }
        return events.isEmpty ? nil : events
    }

    private func makeToolCallEvent(from obj: [String: Any]) -> Event? {
        guard let name = obj["name"] as? String else { return nil }
        let argsJSON: String
        if let argsDict = obj["arguments"] as? [String: Any],
           let argsData = try? JSONSerialization.data(withJSONObject: argsDict),
           let argsStr = String(data: argsData, encoding: .utf8) {
            argsJSON = argsStr
        } else if let argsStr = obj["arguments"] as? String {
            argsJSON = argsStr
        } else {
            argsJSON = "{}"
        }
        return .toolCall(id: UUID().uuidString, name: name, argsJSON: argsJSON)
    }

    /// Rightmost index such that the suffix from there to end-of-buffer is NOT
    /// a non-empty prefix of `tag`. Same implementation as `ThinkTagParser`.
    private func lastSafeIndex(for tag: String) -> String.Index {
        let maxHold = tag.count - 1
        guard !buffer.isEmpty, maxHold > 0 else { return buffer.endIndex }
        let holdStart = buffer.index(buffer.endIndex, offsetBy: -min(maxHold, buffer.count))
        for offset in 0..<buffer.distance(from: holdStart, to: buffer.endIndex) {
            let idx = buffer.index(holdStart, offsetBy: offset)
            if tag.starts(with: buffer[idx...]) {
                return idx
            }
        }
        return buffer.endIndex
    }
}
