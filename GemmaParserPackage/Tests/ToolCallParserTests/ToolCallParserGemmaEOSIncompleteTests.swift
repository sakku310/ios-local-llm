//
// Gemma‑4 EOS‑incomplete streaming test (edge case)
//
import XCTest
@testable import ToolCallParser

final class ToolCallParserGemmaEOSIncompleteTests: XCTestCase {
    private func parseChunks(_ chunks: [String]) -> [ToolCallParser.Event] {
        var parser = ToolCallParser()
        var events: [ToolCallParser.Event] = []
        for chunk in chunks {
            events.append(contentsOf: parser.consume(chunk))
        }
        // Flush at EOS – the incomplete block should be emitted as plain text, not a toolCall
        events.append(contentsOf: parser.flush())
        return events
    }
    private func extractToolCall(_ event: ToolCallParser.Event) -> (name: String, argsJSON: String)? {
        if case let .toolCall(_, name, argsJSON) = event { return (name, argsJSON) }
        return nil
    }

    func testIncompleteGemmaBlockAtEOS() {
        // Provide only the opening marker and partial content, no closing marker
        let chunks = ["Hello ", "<|tool_call>call:websearch{query:<|\"|>Partial"]
        let events = parseChunks(chunks)
        // Expect two events: the leading text and the remainder as text (no toolCall)
        XCTAssertEqual(events.count, 2)
        guard case .text(let prefix) = events[0] else { XCTFail("First event not text"); return }
        XCTAssertEqual(prefix, "Hello ")
        guard case .text(let suffix) = events[1] else { XCTFail("Second event not text"); return }
        XCTAssertTrue(suffix.contains("<|tool_call>call:websearch{query:<|\"|>Partial"))
    }
}
