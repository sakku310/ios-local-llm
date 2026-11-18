import Foundation

// MARK: - WebSearchService
// Provider-aware search facade with:
//   • Outbound query privacy filtering (blocks API keys, local paths, private IPs)
//   • Bounded 5-minute TTL query cache + per-host rate limiting
//   • Compliant keyless free providers (`.freeAuto`, `.wikimedia`, `.duckduckgoInstant`, `.duckduckgoHTML`)
//   • Strict anti-bot non-bypass rule (never solves/bypasses CAPTCHA or HTTP 202/403 challenges)
//   • SSRF validation of all returned result URLs

public actor WebSearchService {

    private struct CacheEntry {
        let results: [WebSearchResult]
        let expiresAt: Date
    }

    private static let maxSearchResponseBytes = 1_572_864 // 1.5 MB
    private static let maxSearchRedirects = 3

    private let validator: URLSafetyValidator
    private let limiter: WebRateLimiter
    private let sanitizer: WebContentSanitizer
    private let cacheTTL: TimeInterval
    private let transport: any WebHTTPTransport
    private var cache: [String: CacheEntry] = [:]

    public init(
        validator: URLSafetyValidator,
        limiter: WebRateLimiter = WebRateLimiter(minIntervalPerHost: 1.5, defaultBudget: 24),
        sanitizer: WebContentSanitizer = WebContentSanitizer(),
        cacheTTL: TimeInterval = 300,
        transport: (any WebHTTPTransport)? = nil
    ) {
        self.validator = validator
        self.limiter = limiter
        self.sanitizer = sanitizer
        self.cacheTTL = max(0, cacheTTL)
        self.transport = transport ?? URLSessionStreamingTransport(validator: validator)
    }

    public func search(query: String, settings: WebToolSettings) async throws -> [WebSearchResult] {
        if settings.provider == .directURLOnly {
            return []
        }

        let sanitizedQuery = try WebQueryPrivacyFilter.sanitize(query).query
        let maxCount = max(1, min(settings.maxSearchResults, 10))
        let cacheKey = "\(settings.provider.rawValue)|\(maxCount)|\(sanitizedQuery.lowercased())"
        let now = Date()
        if cacheTTL > 0, let hit = cache[cacheKey], hit.expiresAt > now {
            return hit.results
        }

        let rawResults: [WebSearchResult]
        switch settings.provider {
        case .directURLOnly:
            return []
        case .freeAuto:
            rawResults = try await searchFreeAuto(query: sanitizedQuery, max: maxCount)
        case .wikimedia:
            rawResults = try await searchWikimedia(query: sanitizedQuery, max: maxCount)
        case .duckduckgoInstant:
            rawResults = try await searchDuckDuckGoInstant(query: sanitizedQuery, max: maxCount)
        case .duckduckgoHTML:
            rawResults = try await searchDuckDuckGoHTML(query: sanitizedQuery, max: maxCount)
        case .brave:
            rawResults = try await searchBrave(query: sanitizedQuery, max: maxCount)
        case .tavily:
            rawResults = try await searchTavily(query: sanitizedQuery, max: maxCount)
        case .exa:
            rawResults = try await searchExa(query: sanitizedQuery, max: maxCount)
        case .searxng:
            guard let endpoint = settings.searxngEndpoint else {
                throw WebToolError.providerEndpointMissing
            }
            rawResults = try await searchSearXNG(endpoint: endpoint, query: sanitizedQuery, max: maxCount)
        case .custom:
            guard let endpoint = settings.customSearchEndpoint else {
                throw WebToolError.providerEndpointMissing
            }
            rawResults = try await searchCustom(endpoint: endpoint, query: sanitizedQuery, max: maxCount)
        }

        let vetted = Self.vetResults(rawResults, max: maxCount, sanitizer: sanitizer)
        if cacheTTL > 0, !vetted.isEmpty {
            if cache.count >= 48 {
                let expired = cache.filter { $0.value.expiresAt <= now }.map(\.key)
                for key in expired { cache.removeValue(forKey: key) }
                if cache.count >= 48, let oldest = cache.min(by: { $0.value.expiresAt < $1.value.expiresAt })?.key {
                    cache.removeValue(forKey: oldest)
                }
            }
            cache[cacheKey] = CacheEntry(results: vetted, expiresAt: now.addingTimeInterval(cacheTTL))
        }
        return vetted
    }

    // MARK: - Free Auto Cascade (Keyless, ToS-aware, Zero CAPTCHA Bypass)

    private func searchFreeAuto(query: String, max: Int) async throws -> [WebSearchResult] {
        var combined: [WebSearchResult] = []
        var firstError: Error?

        do {
            let ddgHTML = try await searchDuckDuckGoHTML(query: query, max: max)
            combined.append(contentsOf: ddgHTML)
        } catch {
            // Never retry or bypass anti-bot/CAPTCHA challenges; fall through to
            // official free JSON APIs (DuckDuckGo Instant Answer + Wikimedia).
            firstError = error
        }

        if combined.count < max {
            if let instant = try? await searchDuckDuckGoInstant(query: query, max: max) {
                combined.append(contentsOf: instant)
            }
        }
        if combined.count < max {
            do {
                let wiki = try await searchWikimedia(query: query, max: max)
                combined.append(contentsOf: wiki)
            } catch {
                if firstError == nil { firstError = error }
            }
        }

        if combined.isEmpty {
            if let webErr = firstError as? WebToolError {
                switch webErr {
                case .timeout, .offline, .rateLimited:
                    throw webErr
                default:
                    break
                }
            }
            throw WebToolError.noResults
        }
        return combined
    }

    // MARK: - Wikimedia / Wikipedia Action API (Official Free Keyless JSON API, CC BY-SA 4.0)

    private func searchWikimedia(query: String, max: Int) async throws -> [WebSearchResult] {
        let host = Self.containsJapaneseCharacters(query) ? "ja.wikipedia.org" : "en.wikipedia.org"
        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = host
        comps.path = "/w/api.php"
        comps.queryItems = [
            URLQueryItem(name: "action", value: "query"),
            URLQueryItem(name: "list", value: "search"),
            URLQueryItem(name: "srsearch", value: query),
            URLQueryItem(name: "srlimit", value: String(max)),
            URLQueryItem(name: "utf8", value: "1"),
            URLQueryItem(name: "format", value: "json")
        ]
        guard let url = comps.url else {
            throw WebToolError.providerEndpointMissing
        }
        var req = URLRequest(url: url)
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        let payload = try await performSearchRequest(req, host: host)
        guard payload.statusCode == 200 else {
            throw WebToolError.searchParsingFailed
        }
        let out = Self.parseWikimediaJSONResults(payload.data, wikiHost: host, max: max)
        if out.isEmpty { throw WebToolError.noResults }
        return out
    }

    // MARK: - DuckDuckGo Instant Answer API (Official Free Keyless JSON API)

    private func searchDuckDuckGoInstant(query: String, max: Int) async throws -> [WebSearchResult] {
        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "api.duckduckgo.com"
        comps.path = "/"
        comps.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "format", value: "json"),
            URLQueryItem(name: "no_html", value: "1"),
            URLQueryItem(name: "skip_disambig", value: "1")
        ]
        guard let url = comps.url else {
            throw WebToolError.providerEndpointMissing
        }
        var req = URLRequest(url: url)
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        let payload = try await performSearchRequest(req, host: "api.duckduckgo.com")
        guard payload.statusCode == 200 else {
            throw WebToolError.searchParsingFailed
        }
        let out = Self.parseDuckDuckGoInstantJSONResults(payload.data, max: max)
        if out.isEmpty { throw WebToolError.noResults }
        return out
    }

    // MARK: - DuckDuckGo HTML (Best-effort keyless HTML; strict non-bypass on challenge)

    private func searchDuckDuckGoHTML(query: String, max: Int) async throws -> [WebSearchResult] {
        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "html.duckduckgo.com"
        comps.path = "/html/"
        comps.queryItems = [URLQueryItem(name: "q", value: query)]
        guard let url = comps.url else {
            throw WebToolError.providerEndpointMissing
        }
        var req = URLRequest(url: url)
        req.setValue("text/html", forHTTPHeaderField: "Accept")
        let payload = try await performSearchRequest(req, host: "html.duckduckgo.com")
        if payload.statusCode == 202 || payload.statusCode == 403 {
            throw WebToolError.antiBotChallenge(provider: "DuckDuckGo HTML")
        }
        guard payload.statusCode == 200,
              let html = String(data: payload.data, encoding: .utf8) else {
            throw WebToolError.searchParsingFailed
        }
        let lowerHTML = html.lowercased()
        if lowerHTML.contains("anomaly-modal")
            || lowerHTML.contains("bots use duckduckgo too")
            || lowerHTML.contains("g-recaptcha")
            || lowerHTML.contains("hcaptcha") {
            throw WebToolError.antiBotChallenge(provider: "DuckDuckGo HTML")
        }
        let out = Self.parseDuckDuckGoHTMLResults(html, max: max)
        if out.isEmpty { throw WebToolError.noResults }
        return out
    }

    // MARK: - Brave

    private func searchBrave(query: String, max: Int) async throws -> [WebSearchResult] {
        guard let key = KeychainStore.get(account: "brave.apiKey") else {
            throw WebToolError.providerAuthFailed
        }
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        guard let url = URL(string: "https://api.search.brave.com/res/v1/web/search?q=\(encoded)&count=\(max)") else {
            throw WebToolError.providerEndpointMissing
        }
        var req = URLRequest(url: url)
        req.setValue(key, forHTTPHeaderField: "X-Subscription-Token")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        let payload = try await performSearchRequest(req, host: "api.search.brave.com")
        if payload.statusCode == 401 || payload.statusCode == 403 { throw WebToolError.providerAuthFailed }
        guard payload.statusCode == 200 else { throw WebToolError.searchParsingFailed }

        struct BraveResponse: Decodable {
            struct Web: Decodable {
                struct Result: Decodable {
                    let title: String
                    let url: String
                    let description: String?
                }
                let results: [Result]?
            }
            let web: Web?
        }
        guard let decoded = try? JSONDecoder().decode(BraveResponse.self, from: payload.data),
              let results = decoded.web?.results else {
            throw WebToolError.searchParsingFailed
        }
        return results.prefix(max).compactMap { r in
            guard let u = URL(string: r.url) else { return nil }
            return WebSearchResult(
                title: r.title,
                url: u,
                snippet: r.description ?? "",
                provider: .brave
            )
        }
    }

    // MARK: - Tavily

    private func searchTavily(query: String, max: Int) async throws -> [WebSearchResult] {
        guard let key = KeychainStore.get(account: "tavily.apiKey") else {
            throw WebToolError.providerAuthFailed
        }
        guard let url = URL(string: "https://api.tavily.com/search") else {
            throw WebToolError.providerEndpointMissing
        }
        let body: [String: Any] = [
            "api_key": key,
            "query": query,
            "max_results": max,
            "include_raw_content": true
        ]
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        let payload = try await performSearchRequest(req, host: "api.tavily.com")
        if payload.statusCode == 401 || payload.statusCode == 403 { throw WebToolError.providerAuthFailed }
        guard payload.statusCode == 200 else { throw WebToolError.searchParsingFailed }

        struct TavilyResponse: Decodable {
            struct Result: Decodable {
                let title: String
                let url: String
                let content: String?
                let raw_content: String?
            }
            let results: [Result]
        }
        guard let decoded = try? JSONDecoder().decode(TavilyResponse.self, from: payload.data) else {
            throw WebToolError.searchParsingFailed
        }
        return decoded.results.prefix(max).compactMap { r in
            guard let u = URL(string: r.url) else { return nil }
            return WebSearchResult(
                title: r.title,
                url: u,
                snippet: r.content ?? "",
                provider: .tavily,
                rawContent: r.raw_content
            )
        }
    }

    // MARK: - Exa

    private func searchExa(query: String, max: Int) async throws -> [WebSearchResult] {
        guard let key = KeychainStore.get(account: "exa.apiKey") else {
            throw WebToolError.providerAuthFailed
        }
        guard let url = URL(string: "https://api.exa.ai/search") else {
            throw WebToolError.providerEndpointMissing
        }
        let body: [String: Any] = [
            "query": query,
            "numResults": max,
            "contents": ["text": true]
        ]
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue(key, forHTTPHeaderField: "x-api-key")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        let payload = try await performSearchRequest(req, host: "api.exa.ai")
        if payload.statusCode == 401 || payload.statusCode == 403 { throw WebToolError.providerAuthFailed }
        guard payload.statusCode == 200 else { throw WebToolError.searchParsingFailed }

        struct ExaResponse: Decodable {
            struct Result: Decodable {
                let title: String?
                let url: String
                let text: String?
            }
            let results: [Result]
        }
        guard let decoded = try? JSONDecoder().decode(ExaResponse.self, from: payload.data) else {
            throw WebToolError.searchParsingFailed
        }
        return decoded.results.prefix(max).compactMap { r in
            guard let u = URL(string: r.url) else { return nil }
            return WebSearchResult(
                title: r.title ?? u.host ?? "",
                url: u,
                snippet: String((r.text ?? "").prefix(280)),
                provider: .exa,
                rawContent: r.text
            )
        }
    }

    // MARK: - SearXNG (user-hosted)

    private func searchSearXNG(endpoint: URL, query: String, max: Int) async throws -> [WebSearchResult] {
        let verdict = await validator.validateStable(endpoint)
        if !verdict.isSafe { throw WebToolError.blockedByURLValidator(reason: verdict.reason) }
        guard var comps = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
            throw WebToolError.providerEndpointMissing
        }
        let existingPath = comps.path
        comps.path = existingPath.hasSuffix("/search") ? existingPath : (existingPath + "/search")
        comps.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "format", value: "json")
        ]
        guard let url = comps.url else { throw WebToolError.providerEndpointMissing }
        let req = URLRequest(url: url)
        let payload = try await performSearchRequest(req, host: endpoint.host ?? "searxng")
        guard payload.statusCode == 200 else {
            throw WebToolError.searchParsingFailed
        }
        struct SXResponse: Decodable {
            struct Result: Decodable { let title: String?; let url: String; let content: String? }
            let results: [Result]
        }
        guard let decoded = try? JSONDecoder().decode(SXResponse.self, from: payload.data) else {
            throw WebToolError.searchParsingFailed
        }
        return decoded.results.prefix(max).compactMap { r in
            guard let u = URL(string: r.url) else { return nil }
            return WebSearchResult(
                title: r.title ?? u.host ?? "",
                url: u,
                snippet: r.content ?? "",
                provider: .searxng
            )
        }
    }

    // MARK: - Custom

    private func searchCustom(endpoint: URL, query: String, max: Int) async throws -> [WebSearchResult] {
        let verdict = await validator.validateStable(endpoint)
        if !verdict.isSafe { throw WebToolError.blockedByURLValidator(reason: verdict.reason) }
        var comps = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)
        var items = comps?.queryItems ?? []
        items.append(URLQueryItem(name: "q", value: query))
        comps?.queryItems = items
        guard let url = comps?.url else { throw WebToolError.providerEndpointMissing }
        let req = URLRequest(url: url)
        let payload = try await performSearchRequest(req, host: endpoint.host ?? "custom")
        guard payload.statusCode == 200 else {
            throw WebToolError.searchParsingFailed
        }
        struct CustomResponse: Decodable {
            struct Result: Decodable { let title: String?; let url: String; let snippet: String? }
            let results: [Result]
        }
        guard let decoded = try? JSONDecoder().decode(CustomResponse.self, from: payload.data) else {
            throw WebToolError.searchParsingFailed
        }
        return decoded.results.prefix(max).compactMap { r in
            guard let u = URL(string: r.url) else { return nil }
            return WebSearchResult(
                title: r.title ?? u.host ?? "",
                url: u,
                snippet: r.snippet ?? "",
                provider: .custom
            )
        }
    }

    // MARK: - Transport & Vetting Helpers

    private func performSearchRequest(
        _ request: URLRequest,
        host: String
    ) async throws -> WebHTTPResponsePayload {
        let allowed = await limiter.consumeBudgetAndWait(forHost: host.lowercased())
        if !allowed {
            throw WebToolError.rateLimited(retryAfter: nil)
        }
        do {
            let payload = try await transport.send(
                request,
                maxBodyBytes: Self.maxSearchResponseBytes,
                maxRedirects: Self.maxSearchRedirects
            )
            if payload.statusCode == 429 {
                let retry = payload.header("Retry-After").flatMap { TimeInterval($0) }
                throw WebToolError.rateLimited(retryAfter: retry)
            }
            return payload
        } catch let webErr as WebToolError {
            throw webErr
        } catch let urlErr as URLError {
            if urlErr.code == .timedOut {
                throw WebToolError.timeout
            }
            if urlErr.code == .notConnectedToInternet || urlErr.code == .networkConnectionLost {
                throw WebToolError.offline
            }
            throw WebToolError.searchParsingFailed
        }
    }

    private static func vetResults(
        _ raw: [WebSearchResult],
        max: Int,
        sanitizer: WebContentSanitizer
    ) -> [WebSearchResult] {
        var seenURLs = Set<String>()
        var vetted: [WebSearchResult] = []
        for item in raw {
            guard vetted.count < max else { break }
            guard let scheme = item.url.scheme?.lowercased(),
                  scheme == "http" || scheme == "https",
                  item.url.user == nil,
                  item.url.password == nil,
                  let host = item.url.host?.lowercased(),
                  !host.isEmpty else {
                continue
            }
            if host == "localhost"
                || host.hasSuffix(".localhost")
                || host.hasSuffix(".local")
                || host.hasSuffix(".internal") {
                continue
            }
            if (host.contains(":") || host.allSatisfy({ $0.isNumber || $0 == "." })),
               URLSafetyValidator.isPrivateOrReserved(host) {
                continue
            }
            let key = item.url.absoluteString
            guard seenURLs.insert(key).inserted else { continue }
            let cleanTitle = sanitizer.sanitize(item.title).sanitized
            let cleanSnippet = sanitizer.sanitize(item.snippet).sanitized
            vetted.append(WebSearchResult(
                title: cleanTitle.isEmpty ? host : cleanTitle,
                url: item.url,
                snippet: cleanSnippet,
                provider: item.provider,
                rawContent: item.rawContent.map { sanitizer.sanitize($0).sanitized }
            ))
        }
        return vetted
    }

    private static func containsJapaneseCharacters(_ text: String) -> Bool {
        for scalar in text.unicodeScalars {
            let v = scalar.value
            // Hiragana, Katakana, CJK Unified Ideographs
            if (0x3040...0x30FF).contains(v) || (0x4E00...0x9FFF).contains(v) {
                return true
            }
        }
        return false
    }

    // MARK: - Deterministic Parsers (Unit-Testable Offline)

    static func parseWikimediaJSONResults(
        _ data: Data,
        wikiHost: String = "en.wikipedia.org",
        max: Int
    ) -> [WebSearchResult] {
        struct WikiEnvelope: Decodable {
            struct Query: Decodable {
                struct Item: Decodable {
                    let title: String
                    let snippet: String?
                }
                let search: [Item]?
            }
            let query: Query?
        }
        guard let decoded = try? JSONDecoder().decode(WikiEnvelope.self, from: data),
              let items = decoded.query?.search else {
            return []
        }
        var results: [WebSearchResult] = []
        for item in items.prefix(max) {
            let slug = item.title
                .replacingOccurrences(of: " ", with: "_")
                .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? item.title
            guard let url = URL(string: "https://\(wikiHost)/wiki/\(slug)") else { continue }
            let snippet = decodeEntities(stripTags(item.snippet ?? ""))
                .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            results.append(WebSearchResult(
                title: item.title,
                url: url,
                snippet: snippet,
                provider: .wikimedia
            ))
        }
        return results
    }

    static func parseDuckDuckGoInstantJSONResults(
        _ data: Data,
        max: Int
    ) -> [WebSearchResult] {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return []
        }
        var results: [WebSearchResult] = []
        if let abstractURLString = json["AbstractURL"] as? String,
           !abstractURLString.isEmpty,
           let abstractURL = URL(string: abstractURLString),
           let abstractText = json["AbstractText"] as? String,
           !abstractText.isEmpty {
            let heading = (json["Heading"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
            results.append(WebSearchResult(
                title: (heading?.isEmpty == false ? heading! : abstractURL.host ?? "DuckDuckGo Instant Answer"),
                url: abstractURL,
                snippet: abstractText,
                provider: .duckduckgoInstant
            ))
        }

        func collectTopics(_ array: [[String: Any]]) {
            for topic in array {
                guard results.count < max else { return }
                if let nested = topic["Topics"] as? [[String: Any]] {
                    collectTopics(nested)
                    continue
                }
                guard let firstURL = topic["FirstURL"] as? String,
                      let url = URL(string: firstURL),
                      let text = topic["Text"] as? String,
                      !text.isEmpty else {
                    continue
                }
                let title = text.components(separatedBy: " - ").first ?? (url.host ?? "Result")
                results.append(WebSearchResult(
                    title: String(title.prefix(120)),
                    url: url,
                    snippet: text,
                    provider: .duckduckgoInstant
                ))
            }
        }

        if let related = json["RelatedTopics"] as? [[String: Any]] {
            collectTopics(related)
        }
        return Array(results.prefix(max))
    }

    static func parseDuckDuckGoHTMLResults(_ html: String, max: Int) -> [WebSearchResult] {
        let titleHrefPat = #"<a\b(?=[^>]*class=["'][^"']*\bresult__a\b[^"']*["'])(?=[^>]*href=["']([^"']+)["'])[^>]*>([\s\S]*?)</a>"#
        let snippetPat = #"<a[^>]*class=["'][^"']*\bresult__snippet\b[^"']*["'][^>]*>([\s\S]*?)</a>"#

        let titles = matches(html, pattern: titleHrefPat)
        let snippets = matches(html, pattern: snippetPat)
        var out: [WebSearchResult] = []
        for (i, t) in titles.enumerated() where i < max {
            guard t.count >= 3,
                  let u = normalizedDuckDuckGoResultURL(t[1]) else { continue }
            let title = decodeEntities(stripTags(t[2]))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let snippet = i < snippets.count
                ? decodeEntities(stripTags(snippets[i][1]))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                : ""
            out.append(WebSearchResult(title: title, url: u, snippet: snippet, provider: .duckduckgoHTML))
        }
        return out
    }

    private static func normalizedDuckDuckGoResultURL(_ hrefRaw: String) -> URL? {
        let decodedHref = decodeEntities(hrefRaw)
        let absolute: String
        if decodedHref.hasPrefix("//") {
            absolute = "https:" + decodedHref
        } else if decodedHref.hasPrefix("/") {
            absolute = "https://duckduckgo.com" + decodedHref
        } else {
            absolute = decodedHref
        }
        guard let url = URL(string: absolute) else { return nil }
        let path = url.path
        guard url.host?.lowercased().contains("duckduckgo.com") == true,
              (path == "/l" || path.hasPrefix("/l/")),
              let comps = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let uddg = comps.queryItems?.first(where: { $0.name == "uddg" })?.value,
              let unwrapped = URL(string: uddg.removingPercentEncoding ?? uddg) else {
            return url
        }
        return unwrapped
    }

    private static func matches(_ s: String, pattern: String) -> [[String]] {
        guard let re = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]) else {
            return []
        }
        let ns = s as NSString
        let ms = re.matches(in: s, range: NSRange(location: 0, length: ns.length))
        return ms.map { m in
            (0..<m.numberOfRanges).map { i -> String in
                let r = m.range(at: i)
                return r.location == NSNotFound ? "" : ns.substring(with: r)
            }
        }
    }

    private static func stripTags(_ s: String) -> String {
        s.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
    }

    private static func decodeEntities(_ s: String) -> String {
        var out = s
        let map: [(String, String)] = [
            ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
            ("&quot;", "\""), ("&#39;", "'"), ("&apos;", "'"),
            ("&nbsp;", " ")
        ]
        for (k, v) in map {
            out = out.replacingOccurrences(of: k, with: v)
        }
        return out
    }
}
