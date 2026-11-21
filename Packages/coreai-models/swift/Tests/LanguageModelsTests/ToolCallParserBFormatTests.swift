//
// B‑format ToolCallParser tests
//
import XCTest
@testable import CoreAILanguageModels

final class ToolCallParserBFormatTests: XCTestCase {
    // Helper to run parser on a full string (single consume + flush)
    private func parse(_ text: String) -> [ToolCallParser.Event] {
        var parser = ToolCallParser()
        var events = parser.consume(text)
        events.append(contentsOf: parser.flush())
        return events
    }

    // Helper to extract toolCall event components
    private func extractToolCall(_ event: ToolCallParser.Event) -> (name: String, argsJSON: String)? {
        if case let .toolCall(_, name, argsJSON) = event {
            return (name, argsJSON)
        }
        return nil
    }

    // 1. Single delta B‑format
    func testBFormatSingleDelta() {
        let input = "<tool_call><function=websearch>Apple iPadOS</function></tool_call>"
        let events = parse(input)
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertEqual(tc.name, "websearch")
        // argsJSON should be {"query":"Apple iPadOS"}
        let expected = "{\"query\":\"Apple iPadOS\"}"
        XCTAssertEqual(tc.argsJSON, expected)
    }

    // 2. Multiple delta (split across two consumes)
    func testBFormatMultipleDelta() {
        var parser = ToolCallParser()
        var events: [ToolCallParser.Event] = []
        events.append(contentsOf: parser.consume("<tool_call><function=websearch>Apple "))
        events.append(contentsOf: parser.consume("iPadOS</function></tool_call>"))
        events.append(contentsOf: parser.flush())
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertEqual(tc.name, "websearch")
        XCTAssertEqual(tc.argsJSON, "{\"query\":\"Apple iPadOS\"}")
    }

    // 3. Mixed text and B‑format
    func testMixedTextAndBFormat() {
        let input = "Hello <tool_call><function=websearch>Apple</function></tool_call> world"
        let events = parse(input)
        XCTAssertEqual(events.count, 3)
        XCTAssertEqual(events[0], .text("Hello "))
        guard let tc = extractToolCall(events[1]) else { XCTFail("Not a toolCall"); return }
        XCTAssertEqual(tc.name, "websearch")
        XCTAssertEqual(tc.argsJSON, "{\"query\":\"Apple\"}")
        XCTAssertEqual(events[2], .text(" world"))
    }

    // 4. A‑format compatibility (JSON)
    func testAFormatCompatibility() {
        let json = "<tool_call>{\"name\":\"websearch\",\"arguments\":{\"query\":\"Apple iPadOS\"}}</tool_call>"
        let events = parse(json)
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertEqual(tc.name, "websearch")
        XCTAssertEqual(tc.argsJSON, "{\"query\":\"Apple iPadOS\"}")
    }

    // 5. Multiple B‑format calls in one block
    func testMultipleToolCalls() {
        let input = "<tool_call><function=websearch>Apple</function><function=websearch>iPadOS</function></tool_call>"
        let events = parse(input)
        XCTAssertEqual(events.count, 2)
        guard let tc1 = extractToolCall(events[0]), let tc2 = extractToolCall(events[1]) else { XCTFail("Missing toolCall"); return }
        XCTAssertEqual(tc1.argsJSON, "{\"query\":\"Apple\"}")
        XCTAssertEqual(tc2.argsJSON, "{\"query\":\"iPadOS\"}")
    }

    // 6. Missing closing </tool_call> (should emit no toolCall)
    func testMissingCloseTag() {
        let input = "<tool_call><function=websearch>Apple"
        let events = parse(input)
        XCTAssertTrue(events.isEmpty, "No events should be emitted for unclosed block")
    }

    // 7. Empty query
    func testEmptyQuery() {
        let input = "<tool_call><function=websearch></function></tool_call>"
        let events = parse(input)
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertEqual(tc.argsJSON, "{\"query\":\"\"}")
    }

    // 8. Invalid function name (ignored)
    func testInvalidFunctionName() {
        let input = "<tool_call><function=unknown>test</function></tool_call>"
        let events = parse(input)
        XCTAssertTrue(events.isEmpty, "Unknown function should be ignored")
    }

    // 9. Nested tags (ignored)
    func testNestedTags() {
        let input = "<tool_call><function=websearch>Apple <function=websearch>iPad</function></function></tool_call>"
        let events = parse(input)
        XCTAssertTrue(events.isEmpty, "Nested function tags should be ignored")
    }

    // 10. Special characters escaping
    func testArgumentsEscaping() {
        let input = "<tool_call><function=websearch>Quote \" and \\ backslash</function></tool_call>"
        let events = parse(input)
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        // JSON string should have escaped characters
        let expected = "{\"query\":\"Quote \\\" and \\\\ backslash\"}"
        XCTAssertEqual(tc.argsJSON, expected)
    }
}
