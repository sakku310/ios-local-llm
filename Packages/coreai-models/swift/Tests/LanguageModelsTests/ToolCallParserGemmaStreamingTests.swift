//
// Gemma‑4 streaming ToolCallParser tests
//
import XCTest
@testable import CoreAILanguageModels

final class ToolCallParserGemmaStreamingTests: XCTestCase {
    // Helper to feed a series of chunks to the parser and then flush.
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

    // 1. Full Gemma block in a single chunk.
    func testFullGemmaSingleChunk() {
        let input = "<|tool_call>call:websearch{query:<|\"|>NVIDIA latest news<|\"|>,max_results:3}<tool_call|>"
        let events = parseChunks([input])
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertEqual(tc.name, "websearch")
        XCTAssertTrue(tc.argsJSON.contains("\"query\":\"NVIDIA latest news\""))
        XCTAssertTrue(tc.argsJSON.contains("\"max_results\":3"))
    }

    // 2. Start marker split across two chunks.
    func testStartMarkerSplitTwoChunks() {
        let chunks = ["<|to", "ol_call>call:websearch{query:<|\"|>Apple<|\"|>}<tool_call|>"]
        let events = parseChunks(chunks)
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertEqual(tc.name, "websearch")
        XCTAssertTrue(tc.argsJSON.contains("Apple"))
    }

    // 3. End marker split across three chunks.
    func testEndMarkerSplitThreeChunks() {
        let chunks = ["<|tool_call>call:websearch{query:<|\"|>Swift<|\"|>}", "<tool_", "call|>"]
        let events = parseChunks(chunks)
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertTrue(tc.argsJSON.contains("Swift"))
    }

    // 4. Function name split across chunks.
    func testNameSplit() {
        let chunks = ["<|tool_call>call:w", "ebsearch{query:<|\"|>Gemma<|\"|>}<tool_call|>"]
        let events = parseChunks(chunks)
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertEqual(tc.name, "websearch")
    }

    // 5. Arguments split (including string delimiter split).
    func testArgsSplit() {
        let chunks = [
            "<|tool_call>call:websearch{query:<|",
            "\"|>OpenAI",
            "<|\"|>,max_results:5}<tool_call|>"
        ]
        let events = parseChunks(chunks)
        XCTAssertEqual(events.count, 1)
        guard let tc = extractToolCall(events[0]) else { XCTFail("Not a toolCall"); return }
        XCTAssertTrue(tc.argsJSON.contains("OpenAI"))
        XCTAssertTrue(tc.argsJSON.contains("max_results\":5"))
    }

    // 6. Mixed text surrounding a Gemma block.
    func testMixedTextAndGemma() {
        let chunks = ["Hello ", "world! ", "<|tool_call>call:websearch{query:<|\"|>testing<|\"|>}<tool_call|>", " goodbye"]
        let events = parseChunks(chunks)
        XCTAssertEqual(events.count, 3)
        guard case .text(let prefix) = events[0] else { XCTFail("First event not text"); return }
        XCTAssertEqual(prefix, "Hello world! ")
        guard let tc = extractToolCall(events[1]) else { XCTFail("Second event not toolCall"); return }
        XCTAssertTrue(tc.argsJSON.contains("testing"))
        guard case .text(let suffix) = events[2] else { XCTFail("Third event not text"); return }
        XCTAssertEqual(suffix, " goodbye")
    }

    // 7. Multiple consecutive Gemma calls.
    func testMultipleConsecutiveGemmaCalls() {
        let input = "<|tool_call>call:websearch{query:<|\"|>A<|\"|>}<tool_call|><|tool_call>call:websearch{query:<|\"|>B<|\"|>}<tool_call|>"
        let events = parseChunks([input])
        XCTAssertEqual(events.count, 2)
        guard let tc1 = extractToolCall(events[0]), let tc2 = extractToolCall(events[1]) else { XCTFail("Missing toolCalls"); return }
        XCTAssertTrue(tc1.argsJSON.contains("A"))
        XCTAssertTrue(tc2.argsJSON.contains("B"))
    }
}
