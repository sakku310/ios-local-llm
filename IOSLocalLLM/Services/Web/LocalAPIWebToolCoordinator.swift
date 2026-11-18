import Foundation

// MARK: - LocalAPIWebToolCoordinator
// Bridges OnDeviceLAS's hardened WebSearchService + WebPageFetchService into
// LocalAPIServer's OpenAI-compatible tool-calling pipeline and optional direct
// `/v1/web/search` and `/v1/web/fetch` helper endpoints.

public actor LocalAPIWebToolCoordinator {
    public static let shared = LocalAPIWebToolCoordinator()

    public static let webSearchToolName = "web_search"
    public static let webFetchToolName = "web_fetch"
    public static let maxAutoToolRounds = 3
    public static let maxCharsPerFetchedPage = 3_600

    public struct ToolExecutionResult: Sendable {
        public let toolCallID: String
        public let toolName: String
        public let contentJSON: String
        public let citations: [WebSourceCitation]
        public let isError: Bool
    }

    public struct DirectSearchResponse: Sendable {
        public let query: String
        public let provider: String
        public let results: [[String: Any]]
        public let citations: [WebSourceCitation]
    }

    private let validator: URLSafetyValidator
    private let limiter: WebRateLimiter
    private let extractor: WebContentExtractor
    private let sanitizer: WebContentSanitizer
    private let searchService: WebSearchService
    private let fetchService: WebPageFetchService

    public init(
        validator: URLSafetyValidator? = nil,
        limiter: WebRateLimiter? = nil,
        extractor: WebContentExtractor = WebContentExtractor(),
        sanitizer: WebContentSanitizer = WebContentSanitizer(),
        transport: (any WebHTTPTransport)? = nil
    ) {
        let resolvedValidator = validator ?? URLSafetyValidator()
        let resolvedLimiter = limiter ?? WebRateLimiter(minIntervalPerHost: 1.2, defaultBudget: 24)
        self.validator = resolvedValidator
        self.limiter = resolvedLimiter
        self.extractor = extractor
        self.sanitizer = sanitizer
        self.searchService = WebSearchService(
            validator: resolvedValidator,
            limiter: resolvedLimiter,
            sanitizer: sanitizer,
            transport: transport
        )
        self.fetchService = WebPageFetchService(
            validator: resolvedValidator,
            limiter: resolvedLimiter,
            extractor: extractor,
            sanitizer: sanitizer,
            transport: transport
        )
    }

    // MARK: - Canonical OpenAI Function Tool Schemas

    public static let webSearchParametersJSON =
        #"{"properties":{"fetch_top_pages":{"description":"Fetch and extract readable body text from the top search results (default true).","type":"boolean"},"max_results":{"description":"Maximum number of search results to return (1 to 5).","type":"integer"},"query":{"description":"Concise public web search query. Never include API keys, private IPs, or local file paths.","type":"string"}},"required":["query"],"type":"object"}"#

    public static let webFetchParametersJSON =
        #"{"properties":{"url":{"description":"Public http:// or https:// webpage URL to fetch and extract.","type":"string"}},"required":["url"],"type":"object"}"#

    public nonisolated static func builtInToolDefinitions() -> [(name: String, description: String, parametersJSON: String)] {
        [
            (
                name: webSearchToolName,
                description: "Search the public web for current facts, documentation, or news. Returns numbered results with title, URL, snippet, and extracted text. Cite sources inline as [n](url).",
                parametersJSON: webSearchParametersJSON
            ),
            (
                name: webFetchToolName,
                description: "Fetch and extract the clean main text of a public HTTP/HTTPS URL (scripts/navigation removed, private IPs blocked). Cite the URL inline as [n](url).",
                parametersJSON: webFetchParametersJSON
            )
        ]
    }

    public nonisolated static func isBuiltInWebToolName(_ name: String) -> Bool {
        name == webSearchToolName || name == webFetchToolName
    }

    // MARK: - Tool Execution

    public func resetTurnBudget(maxNetworkOps: Int = 12) async {
        await limiter.resetRequestBudget(maxNetworkOps)
    }

    public func executeToolCall(
        id: String,
        name: String,
        argumentsJSON: String,
        provider: WebSearchProvider = .freeAuto,
        nextCitationIndex: Int = 1,
        visitedURLs: inout Set<String>
    ) async -> ToolExecutionResult {
        let args: [String: Any] = {
            guard let data = argumentsJSON.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return [:]
            }
            return obj
        }()

        switch name {
        case Self.webSearchToolName:
            return await executeSearchTool(
                id: id,
                args: args,
                provider: provider,
                nextCitationIndex: nextCitationIndex,
                visitedURLs: &visitedURLs
            )
        case Self.webFetchToolName:
            return await executeFetchTool(
                id: id,
                args: args,
                nextCitationIndex: nextCitationIndex,
                visitedURLs: &visitedURLs
            )
        default:
            let errJSON = Self.encodeJSON([
                "tool": name,
                "status": "error",
                "error": "Unsupported server web tool '\(name)'."
            ])
            return ToolExecutionResult(
                toolCallID: id,
                toolName: name,
                contentJSON: errJSON,
                citations: [],
                isError: true
            )
        }
    }

    private func executeSearchTool(
        id: String,
        args: [String: Any],
        provider: WebSearchProvider,
        nextCitationIndex: Int,
        visitedURLs: inout Set<String>
    ) async -> ToolExecutionResult {
        guard let rawQuery = (args["query"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !rawQuery.isEmpty else {
            return errorToolResult(id: id, name: Self.webSearchToolName, message: "Missing required 'query' string.")
        }
        let requestedMax = (args["max_results"] as? NSNumber)?.intValue ?? 4
        let maxResults = max(1, min(requestedMax, 5))
        let fetchTopPages = (args["fetch_top_pages"] as? Bool) ?? true

        var settings = WebToolSettings()
        settings.provider = provider
        settings.maxSearchResults = maxResults

        do {
            let results = try await searchService.search(query: rawQuery, settings: settings)
            var citations: [WebSourceCitation] = []
            var serializedRows: [[String: Any]] = []
            let iso = ISO8601DateFormatter()
            var currentIndex = max(1, nextCitationIndex)
            var fetchedCount = 0

            for item in results {
                let urlKey = item.url.absoluteString
                var extractedSnippet = item.snippet
                var pageBody = item.rawContent ?? ""
                var siteName = item.url.host ?? "web"
                var fetchedAt = Date()

                if fetchTopPages,
                   pageBody.isEmpty,
                   fetchedCount < 2,
                   visitedURLs.insert(urlKey).inserted {
                    fetchedCount += 1
                    if let page = try? await fetchService.fetch(item.url, isDirectUserURL: false),
                       page.error == nil,
                       !page.extractedText.isEmpty {
                        pageBody = String(page.extractedText.prefix(Self.maxCharsPerFetchedPage))
                        if extractedSnippet.isEmpty {
                            extractedSnippet = page.excerpt
                        }
                        siteName = page.siteName
                        fetchedAt = page.fetchedAt
                    }
                }

                let citation = WebSourceCitation(
                    index: currentIndex,
                    title: item.title,
                    url: item.url,
                    siteName: siteName,
                    fetchedAt: fetchedAt
                )
                citations.append(citation)

                var row: [String: Any] = [
                    "citation_index": currentIndex,
                    "citation_tag": "[\(currentIndex)](\(item.url.absoluteString))",
                    "title": item.title,
                    "url": item.url.absoluteString,
                    "site_name": siteName,
                    "snippet": extractedSnippet,
                    "fetched_at": iso.string(from: fetchedAt)
                ]
                if !pageBody.isEmpty {
                    row["content"] = String(pageBody.prefix(Self.maxCharsPerFetchedPage))
                }
                serializedRows.append(row)
                currentIndex += 1
            }

            let payloadJSON = Self.encodeJSON([
                "tool": Self.webSearchToolName,
                "status": "ok",
                "query": rawQuery,
                "provider": provider.rawValue,
                "citation_instruction": "Ground your answer in these results and cite sources inline as [n](url).",
                "results": serializedRows
            ])
            return ToolExecutionResult(
                toolCallID: id,
                toolName: Self.webSearchToolName,
                contentJSON: payloadJSON,
                citations: citations,
                isError: false
            )
        } catch {
            return errorToolResult(
                id: id,
                name: Self.webSearchToolName,
                message: error.localizedDescription
            )
        }
    }

    private func executeFetchTool(
        id: String,
        args: [String: Any],
        nextCitationIndex: Int,
        visitedURLs: inout Set<String>
    ) async -> ToolExecutionResult {
        guard let rawURL = (args["url"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
              let url = URL(string: rawURL) else {
            return errorToolResult(id: id, name: Self.webFetchToolName, message: "Invalid or missing 'url' parameter.")
        }

        let canonicalKey = url.absoluteString
        if !visitedURLs.insert(canonicalKey).inserted {
            return errorToolResult(
                id: id,
                name: Self.webFetchToolName,
                message: "URL '\(canonicalKey)' was already fetched in this turn; reuse the existing content instead of re-fetching."
            )
        }

        do {
            let page = try await fetchService.fetch(url, isDirectUserURL: true)
            let iso = ISO8601DateFormatter()
            let citation = WebSourceCitation(
                index: max(1, nextCitationIndex),
                title: page.title.isEmpty ? (page.finalURL.host ?? canonicalKey) : page.title,
                url: page.finalURL,
                siteName: page.siteName,
                fetchedAt: page.fetchedAt
            )
            let truncatedContent = String(page.extractedText.prefix(Self.maxCharsPerFetchedPage))
            let payloadJSON = Self.encodeJSON([
                "tool": Self.webFetchToolName,
                "status": "ok",
                "citation_index": citation.index,
                "citation_tag": "[\(citation.index)](\(page.finalURL.absoluteString))",
                "title": citation.title,
                "url": page.finalURL.absoluteString,
                "requested_url": url.absoluteString,
                "site_name": page.siteName,
                "word_count": page.wordCount,
                "excerpt": page.excerpt,
                "content": truncatedContent,
                "fetched_at": iso.string(from: page.fetchedAt),
                "citation_instruction": "Cite facts from this page inline as [\(citation.index)](\(page.finalURL.absoluteString))."
            ])
            return ToolExecutionResult(
                toolCallID: id,
                toolName: Self.webFetchToolName,
                contentJSON: payloadJSON,
                citations: [citation],
                isError: false
            )
        } catch {
            return errorToolResult(
                id: id,
                name: Self.webFetchToolName,
                message: error.localizedDescription
            )
        }
    }

    // MARK: - Direct Authenticated Helper Endpoints (/v1/web/search, /v1/web/fetch)

    public func directSearch(
        query: String,
        maxResults: Int = 5,
        fetchTopPages: Bool = false,
        provider: WebSearchProvider = .freeAuto
    ) async throws -> [String: Any] {
        await limiter.resetRequestBudget(12)
        var settings = WebToolSettings()
        settings.provider = provider
        settings.maxSearchResults = max(1, min(maxResults, 10))
        let results = try await searchService.search(query: query, settings: settings)
        let iso = ISO8601DateFormatter()
        var rows: [[String: Any]] = []
        var fetchedCount = 0

        for (offset, r) in results.enumerated() {
            var content: String? = r.rawContent
            var fetchedAt = Date()
            var siteName = r.url.host ?? "web"
            if fetchTopPages, content == nil, fetchedCount < 2 {
                fetchedCount += 1
                if let page = try? await fetchService.fetch(r.url, isDirectUserURL: false),
                   page.error == nil,
                   !page.extractedText.isEmpty {
                    content = String(page.extractedText.prefix(Self.maxCharsPerFetchedPage))
                    fetchedAt = page.fetchedAt
                    siteName = page.siteName
                }
            }
            var item: [String: Any] = [
                "index": offset + 1,
                "title": r.title,
                "url": r.url.absoluteString,
                "site_name": siteName,
                "snippet": r.snippet,
                "provider": r.provider.rawValue,
                "fetched_at": iso.string(from: fetchedAt)
            ]
            if let content {
                item["content"] = content
            }
            rows.append(item)
        }

        return [
            "object": "web.search",
            "query": try WebQueryPrivacyFilter.sanitize(query).query,
            "provider": provider.rawValue,
            "results": rows
        ]
    }

    public func directFetch(url: URL) async throws -> [String: Any] {
        await limiter.resetRequestBudget(6)
        let page = try await fetchService.fetch(url, isDirectUserURL: true)
        let iso = ISO8601DateFormatter()
        return [
            "object": "web.page",
            "requested_url": page.url.absoluteString,
            "url": page.finalURL.absoluteString,
            "title": page.title,
            "site_name": page.siteName,
            "status_code": page.statusCode,
            "content_type": page.contentType,
            "word_count": page.wordCount,
            "excerpt": page.excerpt,
            "content": String(page.extractedText.prefix(Self.maxCharsPerFetchedPage * 2)),
            "fetched_at": iso.string(from: page.fetchedAt)
        ]
    }

    // MARK: - Citation Formatting Helpers

    public nonisolated static func serializedCitations(_ citations: [WebSourceCitation]) -> [[String: Any]] {
        let iso = ISO8601DateFormatter()
        return citations.map { c in
            [
                "index": c.index,
                "title": c.title,
                "url": c.url.absoluteString,
                "site_name": c.siteName,
                "fetched_at": iso.string(from: c.fetchedAt)
            ]
        }
    }

    public nonisolated static func openAIAnnotations(_ citations: [WebSourceCitation]) -> [[String: Any]] {
        citations.map { c in
            [
                "type": "url_citation",
                "url_citation": [
                    "title": c.title,
                    "url": c.url.absoluteString
                ]
            ]
        }
    }

    /// Ensures that when server-executed Web tools supplied grounded sources,
    /// the final assistant text always includes clickable source URLs even if
    /// a compact local model only wrote `[1]` without repeating the full URL.
    public nonisolated static func ensureCitationsRendered(
        in text: String,
        citations: [WebSourceCitation]
    ) -> (fullText: String, appendedSuffix: String) {
        guard !citations.isEmpty else { return (text, "") }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return (text, "") }

        let missingURLs = citations.filter { !trimmed.contains($0.url.absoluteString) }
        guard !missingURLs.isEmpty else { return (text, "") }

        let lines = citations.map { c in
            "- [\(c.index)] [\(c.title)](\(c.url.absoluteString))"
        }
        let suffix = "\n\nSources:\n" + lines.joined(separator: "\n")
        return (text + suffix, suffix)
    }

    private func errorToolResult(id: String, name: String, message: String) -> ToolExecutionResult {
        let payloadJSON = Self.encodeJSON([
            "tool": name,
            "status": "error",
            "error": message
        ])
        return ToolExecutionResult(
            toolCallID: id,
            toolName: name,
            contentJSON: payloadJSON,
            citations: [],
            isError: true
        )
    }

    private nonisolated static func encodeJSON(_ object: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let str = String(data: data, encoding: .utf8) else {
            return #"{"status":"error","error":"serialization failed"}"#
        }
        return str
    }
}
