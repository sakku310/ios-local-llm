//
// ToolCallParser name normalization tests for Gemma format
//
import XCTest
@testable import CoreAILanguageModels

final class ToolCallParserWebSearchNameNormalizationTests: XCTestCase {
    // Helper to feed chunks and flush
    private func parseChunks(_ chunks: [String]) -> [ToolCallParser.Event] {
        var parser = ToolCallParser()
        var events: [ToolCallParser.Event] = []
        for chunk in chunks {
            events.append(contentsOf: parser.consume(chunk))
        }
        events.append(contentsOf: parser.flush())
        return events
    }

    private func extractToolCall(_ event: ToolCallParser.Event) -> (name: String, argsJSON: String)? {
        if case let .toolCall(_, name, argsJSON) = event {
            return (name, argsJSON)
        }
        return nil
    }

    // 1. websearch -> web_search canonicalization
    func testWebsearchCanonicalizesToWebSearch() {
        let input = "<|tool_call>call:websearch{query:<|\"|>test<|\"|>,max_results:2}<tool_call|>"
        let events = parseChunks([input])
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertEqual(tc.name, "web_search")
        XCTAssertTrue(tc.argsJSON.contains("\"query\":\"test\""))
        XCTAssertTrue(tc.argsJSON.contains("\"max_results\":2"))
    }

    // 2. web_search -> web_search (already correct)
    func testWebSearchUnderscorePassesThrough() {
        let input = "<|tool_call>call:web_search{query:<|\"|>example<|\"|>}<tool_call|>"
        let events = parseChunks([input])
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertEqual(tc.name, "web_search")
        XCTAssertTrue(tc.argsJSON.contains("\"query\":\"example\""))
    }

    // 3. unknown function should not produce a toolCall
    func testUnknownFunctionProducesNoToolCall() {
        let input = "<|tool_call>call:foobar{query:<|\"|>data<|\"|>}<tool_call|>"
        let events = parseChunks([input])
        // Should be a single text event (the raw string) because guard rejects
        XCTAssertEqual(events.count, 1)
        if case .toolCall = events[0] { XCTFail("Unexpected toolCall") }
    }
}
