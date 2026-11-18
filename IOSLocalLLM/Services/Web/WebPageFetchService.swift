import Foundation
import Network
import os

// MARK: - Injectable HTTP Transport (enables 100% offline deterministic unit tests)

public struct WebHTTPResponsePayload: Sendable {
    public let data: Data
    public let statusCode: Int
    public let headers: [String: String]
    public let finalURL: URL

    public init(
        data: Data,
        statusCode: Int,
        headers: [String: String] = [:],
        finalURL: URL
    ) {
        self.data = data
        self.statusCode = statusCode
        self.headers = headers
        self.finalURL = finalURL
    }

    public func header(_ name: String) -> String? {
        let lower = name.lowercased()
        return headers.first { $0.key.lowercased() == lower }?.value
    }
}

public protocol WebHTTPTransport: Sendable {
    func send(
        _ request: URLRequest,
        maxBodyBytes: Int,
        maxRedirects: Int
    ) async throws -> WebHTTPResponsePayload
}

// MARK: - WebPageFetchService
// Fetches a single public URL safely. Hard ceilings:
//   • 2 MB body cap enforced INCREMENTALLY while streaming (never buffers oversized payloads)
//   • 5 redirects max, EVERY hop validated via `URLSafetyValidator.validateStable`
//   • 15 s request timeout / 20 s resource timeout
//   • Accept allowlist: html / text / json / markdown / xhtml / xml
//   • Bounded 5-minute TTL cache + per-host rate limiter to prevent duplicate hammering

public actor WebPageFetchService {
    public static let defaultMaxBodyBytes = 2 * 1_024 * 1_024
    public static let defaultMaxRedirects = 5

    private struct CacheEntry {
        let page: WebFetchedPage
        let expiresAt: Date
    }

    private let validator: URLSafetyValidator
    private let limiter: WebRateLimiter
    private let extractor: WebContentExtractor
    private let sanitizer: WebContentSanitizer
    private let maxBodyBytes: Int
    private let maxRedirects: Int
    private let cacheTTL: TimeInterval
    private let transport: any WebHTTPTransport
    private var cache: [String: CacheEntry] = [:]

    public init(
        validator: URLSafetyValidator,
        limiter: WebRateLimiter,
        extractor: WebContentExtractor,
        sanitizer: WebContentSanitizer = WebContentSanitizer(),
        maxBodyBytes: Int = WebPageFetchService.defaultMaxBodyBytes,
        maxRedirects: Int = WebPageFetchService.defaultMaxRedirects,
        cacheTTL: TimeInterval = 300,
        transport: (any WebHTTPTransport)? = nil
    ) {
        self.validator = validator
        self.limiter = limiter
        self.extractor = extractor
        self.sanitizer = sanitizer
        self.maxBodyBytes = max(1_024, maxBodyBytes)
        self.maxRedirects = max(0, maxRedirects)
        self.cacheTTL = max(0, cacheTTL)
        self.transport = transport ?? URLSessionStreamingTransport(validator: validator)
    }

    /// Fetches, validates, extracts, and sanitizes `url`.
    public func fetch(_ url: URL, isDirectUserURL: Bool = false) async throws -> WebFetchedPage {
        _ = isDirectUserURL
        let cacheKey = url.absoluteString
        let now = Date()
        if cacheTTL > 0, let hit = cache[cacheKey], hit.expiresAt > now {
            return hit.page
        }

        // 1. SSRF / scheme / host / DNS-rebinding gate BEFORE issuing request.
        let verdict = await validator.validateStable(url)
        if !verdict.isSafe {
            throw WebToolError.blockedByURLValidator(reason: verdict.reason)
        }

        // 2. Rate-limit per host to prevent abusive bursts.
        let host = url.host?.lowercased() ?? ""
        let allowed = await limiter.consumeBudgetAndWait(forHost: host)
        if !allowed { throw WebToolError.rateLimited(retryAfter: nil) }

        // 3. Issue streaming byte-capped request with redirect-by-redirect SSRF validation.
        var req = URLRequest(url: url)
        req.httpMethod = "GET"

        let payload: WebHTTPResponsePayload
        do {
            payload = try await transport.send(
                req,
                maxBodyBytes: maxBodyBytes,
                maxRedirects: maxRedirects
            )
        } catch let webErr as WebToolError {
            throw webErr
        } catch let urlErr as URLError {
            if urlErr.code == .timedOut {
                throw WebToolError.timeout
            }
            if urlErr.code == .notConnectedToInternet || urlErr.code == .networkConnectionLost {
                throw WebToolError.offline
            }
            if urlErr.code == .httpTooManyRedirects {
                throw WebToolError.tooManyRedirects(max: maxRedirects)
            }
            throw WebToolError.parsingFailed(reason: urlErr.localizedDescription)
        }

        // 4. Status / content-type gating.
        if payload.statusCode == 429 {
            let retry = payload.header("Retry-After").flatMap { TimeInterval($0) }
            throw WebToolError.rateLimited(retryAfter: retry)
        }
        if payload.statusCode == 202 || payload.statusCode == 403 {
            let preview = String(decoding: payload.data.prefix(2_048), as: UTF8.self).lowercased()
            if preview.contains("captcha")
                || preview.contains("cf-turnstile")
                || preview.contains("verify you are human") {
                throw WebToolError.antiBotChallenge(provider: host)
            }
        }
        if !(200...299).contains(payload.statusCode) {
            throw WebToolError.parsingFailed(reason: "status \(payload.statusCode)")
        }

        let ctRaw = payload.header("Content-Type") ?? ""
        let ct = ctRaw.lowercased()
        let allowedTypes = [
            "text/html", "text/plain", "application/json",
            "text/markdown", "application/xhtml+xml", "text/xml"
        ]
        if !allowedTypes.contains(where: { ct.hasPrefix($0) }) {
            throw WebToolError.unsupportedContentType(ctRaw.isEmpty ? "unknown" : ctRaw)
        }

        // 5. Final belt-and-suspenders size & final-URL SSRF check.
        if payload.data.count > maxBodyBytes {
            throw WebToolError.tooLarge(bytes: payload.data.count)
        }

        let finalURL = payload.finalURL
        let finalVerdict = await validator.validateStable(finalURL)
        if !finalVerdict.isSafe {
            throw WebToolError.blockedByURLValidator(reason: "final URL blocked: \(finalVerdict.reason)")
        }

        // 6. Extract readable body & sanitize untrusted text.
        let extracted = await extractor.extract(
            data: payload.data,
            contentType: ct,
            requestURL: url,
            finalURL: finalURL,
            statusCode: payload.statusCode
        )
        let cleanBody = sanitizer.sanitize(extracted.extractedText).sanitized
        let cleanTitle = sanitizer.sanitize(extracted.title).sanitized
        let cleanExcerpt = String(cleanBody.prefix(280))
        let page = WebFetchedPage(
            url: extracted.url,
            finalURL: extracted.finalURL,
            title: cleanTitle,
            siteName: extracted.siteName,
            fetchedAt: extracted.fetchedAt,
            statusCode: extracted.statusCode,
            extractedText: cleanBody,
            excerpt: cleanExcerpt,
            wordCount: extracted.wordCount,
            contentType: extracted.contentType,
            language: extracted.language,
            error: extracted.error
        )

        if cacheTTL > 0 {
            if cache.count >= 48 {
                let expiredKeys = cache.filter { $0.value.expiresAt <= now }.map(\.key)
                for key in expiredKeys { cache.removeValue(forKey: key) }
                if cache.count >= 48, let oldestKey = cache.min(by: { $0.value.expiresAt < $1.value.expiresAt })?.key {
                    cache.removeValue(forKey: oldestKey)
                }
            }
            cache[cacheKey] = CacheEntry(page: page, expiresAt: now.addingTimeInterval(cacheTTL))
        }
        return page
    }
}

// MARK: - URLSession Streaming Transport + RedirectGuard

public final class URLSessionStreamingTransport: WebHTTPTransport, @unchecked Sendable {
    private let validator: URLSafetyValidator
    private let session: URLSession
    private let redirectGuard: RedirectGuard

    public init(
        validator: URLSafetyValidator,
        requestTimeout: TimeInterval = 15,
        resourceTimeout: TimeInterval = 20,
        userAgent: String = "OnDeviceLAS/1.0 (iPadOS Local LLM Web Tool; +https://github.com/Mesutcydev/ios-local-llm)"
    ) {
        self.validator = validator
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = requestTimeout
        cfg.timeoutIntervalForResource = resourceTimeout
        cfg.httpAdditionalHeaders = [
            "User-Agent": userAgent,
            "Accept": "text/html, text/plain, application/json, text/markdown, application/xhtml+xml, text/xml"
        ]
        cfg.httpMaximumConnectionsPerHost = 2
        cfg.waitsForConnectivity = false

        let guardDelegate = RedirectGuard(validator: validator)
        self.redirectGuard = guardDelegate
        self.session = URLSession(configuration: cfg, delegate: guardDelegate, delegateQueue: nil)
    }

    /// Eliminates DNS-rebinding TOCTOU on plain `http://` requests (where TLS
    /// certificate/SNI verification does not exist) by pinning the socket URL
    /// directly to the validated public IP while preserving the original `Host`
    /// header. `https://` requests keep their hostname for strict TLS SNI +
    /// X.509 certificate verification and are additionally checked post-connect
    /// via `URLSessionTaskMetrics.transactionMetrics[].remoteAddress`.
    public static func pinnedPlainHTTPRequest(
        for request: URLRequest,
        resolvedIPs: [String]
    ) -> (request: URLRequest, logicalURL: URL?) {
        guard let url = request.url,
              url.scheme?.lowercased() == "http",
              let originalHost = url.host,
              !originalHost.isEmpty,
              let pinnedIP = resolvedIPs.first(where: { !URLSafetyValidator.isPrivateOrReserved($0) }),
              var comps = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            return (request, request.url)
        }
        // If already an IP literal, no rewrite needed.
        if originalHost == pinnedIP || originalHost == "[\(pinnedIP)]" {
            return (request, url)
        }
        comps.percentEncodedHost = pinnedIP.contains(":") ? "[\(pinnedIP)]" : pinnedIP
        guard let pinnedURL = comps.url else {
            return (request, url)
        }
        var updated = request
        updated.url = pinnedURL
        let hostHeader = url.port.map { "\(originalHost):\($0)" } ?? originalHost
        updated.setValue(hostHeader, forHTTPHeaderField: "Host")
        return (updated, url)
    }

    public func send(
        _ request: URLRequest,
        maxBodyBytes: Int,
        maxRedirects: Int
    ) async throws -> WebHTTPResponsePayload {
        var wireRequest = request
        wireRequest.timeoutInterval = min(request.timeoutInterval > 0 ? request.timeoutInterval : 15, 20)
        var logicalInitialURL = wireRequest.url
        if let reqURL = wireRequest.url, reqURL.scheme?.lowercased() == "http" {
            let verdict = await validator.validateStable(reqURL)
            if !verdict.isSafe {
                throw WebToolError.blockedByURLValidator(reason: verdict.reason)
            }
            let pinned = Self.pinnedPlainHTTPRequest(for: wireRequest, resolvedIPs: verdict.resolvedIPs)
            wireRequest = pinned.request
            logicalInitialURL = pinned.logicalURL
        }

        let urlKey = wireRequest.url?.absoluteString ?? ""
        redirectGuard.configureMaxRedirects(maxRedirects, forOriginalURL: urlKey)

        do {
            let (bytes, response) = try await session.bytes(for: wireRequest)
            let taskID = bytes.task.taskIdentifier
            defer { _ = redirectGuard.clearTask(taskID, originalURL: urlKey) }

            if let recordedError = redirectGuard.consumeFailure(taskID: taskID, originalURL: urlKey) {
                bytes.task.cancel()
                throw recordedError
            }
            guard let http = response as? HTTPURLResponse else {
                throw WebToolError.parsingFailed(reason: "non-HTTP response")
            }
            if http.expectedContentLength > Int64(maxBodyBytes) {
                bytes.task.cancel()
                throw WebToolError.tooLarge(bytes: Int(clamping: http.expectedContentLength))
            }

            var accumulated = Data()
            accumulated.reserveCapacity(min(maxBodyBytes, max(4_096, Int(max(0, http.expectedContentLength)))))
            for try await byte in bytes {
                accumulated.append(byte)
                if accumulated.count > maxBodyBytes {
                    bytes.task.cancel()
                    throw WebToolError.tooLarge(bytes: accumulated.count)
                }
            }

            if let postMetricsError = redirectGuard.consumeFailure(taskID: taskID, originalURL: urlKey) {
                throw postMetricsError
            }

            var headers: [String: String] = [:]
            for (k, v) in http.allHeaderFields {
                if let ks = k as? String, let vs = v as? String {
                    headers[ks] = vs
                }
            }
            let resolvedFinalURL: URL = {
                if let httpURL = http.url, httpURL != wireRequest.url {
                    return httpURL
                }
                return logicalInitialURL ?? http.url ?? wireRequest.url ?? URL(string: "https://invalid.local")!
            }()
            return WebHTTPResponsePayload(
                data: accumulated,
                statusCode: http.statusCode,
                headers: headers,
                finalURL: resolvedFinalURL
            )
        } catch {
            if let recordedError = redirectGuard.consumeFailure(taskID: nil, originalURL: urlKey) {
                throw recordedError
            }
            throw error
        }
    }
}

private final class RedirectGuard: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private struct TaskState {
        var count: Int = 0
        var maxRedirects: Int
        var failure: WebToolError?
    }

    private let validator: URLSafetyValidator
    private let lock = OSAllocatedUnfairLock(initialState: (
        byTaskID: [Int: TaskState](),
        maxByURL: [String: Int](),
        recentFailureByURL: [String: WebToolError]()
    ))

    init(validator: URLSafetyValidator) {
        self.validator = validator
    }

    func configureMaxRedirects(_ maxRedirects: Int, forOriginalURL urlKey: String) {
        guard !urlKey.isEmpty else { return }
        lock.withLock { state in
            state.maxByURL[urlKey] = maxRedirects
        }
    }

    func consumeFailure(taskID: Int?, originalURL urlKey: String) -> WebToolError? {
        lock.withLock { state in
            if let taskID, let err = state.byTaskID[taskID]?.failure {
                state.byTaskID[taskID]?.failure = nil
                state.recentFailureByURL.removeValue(forKey: urlKey)
                return err
            }
            return state.recentFailureByURL.removeValue(forKey: urlKey)
        }
    }

    func clearTask(_ taskID: Int, originalURL urlKey: String) {
        lock.withLock { state in
            state.byTaskID.removeValue(forKey: taskID)
            state.maxByURL.removeValue(forKey: urlKey)
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        _ = session
        _ = response
        let taskID = task.taskIdentifier
        let origURLKey = task.originalRequest?.url?.absoluteString ?? ""
        guard let nextURL = request.url else {
            recordFailure(
                taskID: taskID,
                originalURL: origURLKey,
                error: .blockedByURLValidator(reason: "redirect missing URL")
            )
            task.cancel()
            completionHandler(nil)
            return
        }

        let exceededMax: Int? = lock.withLock { state in
            let configuredMax = state.maxByURL[origURLKey] ?? WebPageFetchService.defaultMaxRedirects
            var entry = state.byTaskID[taskID] ?? TaskState(count: 0, maxRedirects: configuredMax, failure: nil)
            entry.count += 1
            if entry.count > entry.maxRedirects {
                let err = WebToolError.tooManyRedirects(max: entry.maxRedirects)
                entry.failure = err
                state.byTaskID[taskID] = entry
                if !origURLKey.isEmpty { state.recentFailureByURL[origURLKey] = err }
                return entry.maxRedirects
            }
            state.byTaskID[taskID] = entry
            return nil
        }

        if exceededMax != nil {
            task.cancel()
            completionHandler(nil)
            return
        }

        Task { [validator] in
            let verdict = await validator.validateStable(nextURL)
            if !verdict.isSafe {
                self.recordFailure(
                    taskID: taskID,
                    originalURL: origURLKey,
                    error: .blockedByURLValidator(reason: "redirect target blocked: \(verdict.reason)")
                )
                task.cancel()
                completionHandler(nil)
                return
            }
            let forwarded = URLSessionStreamingTransport.pinnedPlainHTTPRequest(
                for: request,
                resolvedIPs: verdict.resolvedIPs
            ).request
            completionHandler(forwarded)
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didFinishCollecting metrics: URLSessionTaskMetrics
    ) {
        _ = session
        let taskID = task.taskIdentifier
        let origURLKey = task.originalRequest?.url?.absoluteString ?? ""
        for tx in metrics.transactionMetrics {
            if let remoteIP = tx.remoteAddress,
               !remoteIP.isEmpty,
               URLSafetyValidator.isPrivateOrReserved(remoteIP) {
                recordFailure(
                    taskID: taskID,
                    originalURL: origURLKey,
                    error: .blockedByURLValidator(
                        reason: "connected socket peer IP is private or reserved"
                    )
                )
                task.cancel()
                return
            }
        }
    }

    private func recordFailure(taskID: Int, originalURL urlKey: String, error: WebToolError) {
        lock.withLock { state in
            let configuredMax = state.maxByURL[urlKey] ?? WebPageFetchService.defaultMaxRedirects
            var entry = state.byTaskID[taskID] ?? TaskState(count: 0, maxRedirects: configuredMax, failure: nil)
            if entry.failure == nil {
                entry.failure = error
                state.byTaskID[taskID] = entry
            }
            if !urlKey.isEmpty {
                state.recentFailureByURL[urlKey] = error
            }
        }
    }
}
