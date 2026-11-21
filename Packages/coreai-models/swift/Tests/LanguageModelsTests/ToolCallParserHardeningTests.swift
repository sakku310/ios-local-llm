//
// Hardening tests for mixed A/B formats and edge cases.
//
import XCTest
@testable import CoreAILanguageModels

final class ToolCallParserHardeningTests: XCTestCase {
    // Helper to parse a whole string (single consume + flush)
    private func parse(_ text: String) -> [ToolCallParser.Event] {
        var parser = ToolCallParser()
        var events = parser.consume(text)
        events.append(contentsOf: parser.flush())
        return events
    }

    private func extractToolCall(_ event: ToolCallParser.Event) -> (name: String, argsJSON: String)? {
        if case let .toolCall(_, name, argsJSON) = event {
            return (name, argsJSON)
        }
        return nil
    }

    // 1. A+B mixed in same block – A follows B
    func testMixedABFormat_BThenA() {
        let input = "<tool_call><function=websearch>Apple</function>{\"name\":\"websearch\",\"arguments\":{\"query\":\"iPad\"}}</tool_call>"
        let events = parse(input)
        XCTAssertEqual(events.count, 2)
        guard let tc1 = extractToolCall(events[0]), let tc2 = extractToolCall(events[1]) else { XCTFail("Missing toolCalls"); return }
        XCTAssertEqual(tc1.argsJSON, "{\"query\":\"Apple\"}")
        XCTAssertEqual(tc2.argsJSON, "{\"query\":\"iPad\"}")
    }

    // 2. A+B mixed – A precedes B
    func testMixedABFormat_AThenB() {
        let input = "<tool_call>{\"name\":\"websearch\",\"arguments\":{\"query\":\"Apple\"}}<function=websearch>iPad</function></tool_call>"
        let events = parse(input)
        XCTAssertEqual(events.count, 2)
        guard let tc1 = extractToolCall(events[0]), let tc2 = extractToolCall(events[1]) else { XCTFail("Missing toolCalls"); return }
        XCTAssertEqual(tc1.argsJSON, "{\"query\":\"Apple\"}")
        XCTAssertEqual(tc2.argsJSON, "{\"query\":\"iPad\"}")
    }

    // 3. Query containing </function>
    func testQueryContainsCloseTag() {
        let input = "<tool_call><function=websearch>close </function> inside</function></tool_call>"
        // Expected: the parser stops at the first </function> -> query = "close "
        let events = parse(input)
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("No toolCall"); return }
        XCTAssertEqual(tc.argsJSON, "{\"query\":\"close \"}")
    }

    // 4. Nested <function> tags – should be ignored (no toolCall)
    func testNestedFunctionTags() {
        let input = "<tool_call><function=websearch>outer <function=websearch>inner</function> end</function></tool_call>"
        let events = parse(input)
        XCTAssertTrue(events.isEmpty, "Nested tags must not produce a toolCall")
    }

    // 5. Unclosed </tool_call> – no toolCall emitted
    func testUnclosedToolCallTag() {
        let input = "<tool_call><function=websearch>Apple"
        let events = parse(input)
        XCTAssertTrue(events.isEmpty)
    }

    // 6. Whitespace‑only query – should be ignored (no toolCall)
    func testWhitespaceOnlyQuery() {
        let input = "<tool_call><function=websearch>   \n  </function></tool_call>"
        let events = parse(input)
        XCTAssertTrue(events.isEmpty)
    }

    // 7. Multiple B‑format calls – order preserved
    func testMultipleBCallsOrder() {
        let input = "<tool_call><function=websearch>first</function><function=websearch>second</function></tool_call>"
        let events = parse(input)
        XCTAssertEqual(events.count, 2)
        guard let tc1 = extractToolCall(events[0]), let tc2 = extractToolCall(events[1]) else { XCTFail("Missing calls"); return }
        XCTAssertTrue(tc1.argsJSON.contains("first"))
        XCTAssertTrue(tc2.argsJSON.contains("second"))
    }

    // 8. Mixed normal text and B‑format
    func testTextAroundBFormat() {
        let input = "Start <tool_call><function=websearch>Apple</function></tool_call> End"
        let events = parse(input)
        XCTAssertEqual(events.count, 3)
        XCTAssertEqual(events[0], .text("Start "))
        guard let tc = extractToolCall(events[1]) else { XCTFail("Missing toolCall"); return }
        XCTAssertEqual(tc.argsJSON, "{\"query\":\"Apple\"}")
        XCTAssertEqual(events[2], .text(" End"))
    }

    // 9. JSON compatibility – ensure JSON can be decoded again
    func testJSONReDecode() {
        let input = "<tool_call>{\"name\":\"websearch\",\"arguments\":{\"query\":\"Apple\"}}</tool_call>"
        let events = parse(input)
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Missing"); return }
        // Ensure the JSON is valid by round‑tripping through JSONDecoder
        let data = tc.argsJSON.data(using: .utf8)!
        let dict = try! JSONSerialization.jsonObject(with: data) as! [String: String]
        XCTAssertEqual(dict["query"], "Apple")
    }

    // 10. Split across deltas – B‑format broken into two consumes
    func testSplitDeltaBFormat() {
        var parser = ToolCallParser()
        var events: [ToolCallParser.Event] = []
        events.append(contentsOf: parser.consume("<tool_call><function=websearch>A"))
        events.append(contentsOf: parser.consume("pple</function></tool_call>"))
        events.append(contentsOf: parser.flush())
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Missing"); return }
        XCTAssertEqual(tc.argsJSON, "{\"query\":\"Apple\"}")
    }
}
