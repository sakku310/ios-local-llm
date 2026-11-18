import Foundation

// MARK: - WebContentSanitizer
// Defends the local LLM from prompt-injection text and control-token smuggling
// living inside fetched pages, and prevents sensitive local data (API keys,
// private IPs, local file paths) from leaking into external search queries.

public struct WebContentSanitizer: Sendable {

    public init() {}

    public struct Outcome: Sendable {
        public let sanitized: String
        /// True when at least one redaction fired. Caller logs this to the
        /// activity log so the user knows a page was suspicious.
        public let didRedact: Bool
    }

    public func sanitize(_ input: String) -> Outcome {
        var s = input
        var didRedact = false

        // 0. Neutralize chat-template & tool-call control tokens anywhere in
        //    the extracted text (ChatML / Qwen3.5 / Gemma 4 / Gemma 2-3 / Llama).
        let inlineControlTokens: [String] = [
            #"<\|im_start\|>"#,
            #"<\|im_end\|>"#,
            #"<\|endoftext\|>"#,
            #"<\|turn>"#,
            #"<turn\|>"#,
            #"<\|tool>"#,
            #"<tool\|>"#,
            #"<\|tool_call>"#,
            #"<tool_call\|>"#,
            #"<\|tool_response>"#,
            #"<tool_response\|>"#,
            #"<\|channel>"#,
            #"<channel\|>"#,
            #"<\|think\|>"#,
            #"<\|"\|>"#,
            #"</?tool_call\b[^>]*>"#,
            #"</?tool_response\b[^>]*>"#,
            #"<start_of_turn>"#,
            #"<end_of_turn>"#,
            #"<\|(?:begin|end)_of_text\|>"#,
            #"<\|start_header_id\|>"#,
            #"<\|end_header_id\|>"#,
            #"<\|eot_id\|>"#
        ]
        for pat in inlineControlTokens {
            if s.range(of: pat, options: [.regularExpression, .caseInsensitive]) != nil {
                didRedact = true
                s = s.replacingOccurrences(
                    of: pat,
                    with: "`[control-token neutralized]`",
                    options: [.regularExpression, .caseInsensitive]
                )
            }
        }

        // 1. Role-marker imitations at line starts. Wrap in backticks so
        //    citation context is preserved but the LLM doesn't read them as
        //    role tokens.
        let rolePatterns: [String] = [
            "system:", "assistant:", "user:", "developer:",
            "<\\|im_start\\|>", "<\\|im_end\\|>",
            "\\[INST\\]", "\\[/INST\\]",
            "</s>", "<<SYS>>", "<</SYS>>"
        ]
        for pat in rolePatterns {
            let regex = "(?im)^[\\s>]*\(pat)"
            if s.range(of: regex, options: .regularExpression) != nil {
                didRedact = true
                s = s.replacingOccurrences(
                    of: regex,
                    with: "`[role-marker neutralized]`",
                    options: .regularExpression
                )
            }
        }

        // 2. "Ignore previous" style override attempts.
        let overridePatterns: [String] = [
            "(?i)ignore\\s+(?:(?:all|previous|above|prior)\\s+)*(?:instructions|rules|prompts|directives|context)",
            "(?i)disregard\\s+(?:the\\s+)?(?:system|previous|prior|above)",
            "(?i)forget\\s+(?:everything|all\\s+previous|the\\s+above)",
            "(?i)you\\s+are\\s+now\\s+(?:dan|jailbroken|in\\s+developer\\s+mode)",
            "(?i)pretend\\s+(?:you\\s+are|to\\s+be)\\s+(?:dan|an?\\s+ai\\s+without)",
            "(?i)reveal\\s+(?:your\\s+)?(?:system\\s+prompt|hidden\\s+prompt|initial\\s+instructions)"
        ]
        for pat in overridePatterns {
            if s.range(of: pat, options: .regularExpression) != nil {
                didRedact = true
                s = s.replacingOccurrences(
                    of: pat,
                    with: "[redacted: possible injection attempt]",
                    options: .regularExpression
                )
            }
        }

        // 3. Relabel instruction-fence blocks (```instructions / ```system).
        let fences = ["instructions", "system", "prompt", "ai", "assistant"]
        for fence in fences {
            let pat = "```\\s*\(fence)\\b"
            if s.range(of: pat, options: .regularExpression) != nil {
                didRedact = true
                s = s.replacingOccurrences(
                    of: pat, with: "```text",
                    options: .regularExpression
                )
            }
        }

        // 4. Long uppercase runs — typical of "URGENT INSTRUCTION TO AI"
        //    attacks. Anything > 40 consecutive uppercase chars (with spaces)
        //    gets downcased.
        let upperPat = "(?:[A-Z][A-Z\\s]{40,})"
        if let regex = try? NSRegularExpression(pattern: upperPat) {
            let ns = s as NSString
            let matches = regex.matches(in: s, range: NSRange(location: 0, length: ns.length))
            if !matches.isEmpty {
                didRedact = true
                for m in matches.reversed() {
                    let range = m.range
                    let original = ns.substring(with: range)
                    let lowered = original.lowercased()
                    s = (s as NSString).replacingCharacters(in: range, with: lowered)
                }
            }
        }

        // 5. Cap URL spam — keep first 30 absolute URLs.
        let urlPat = "https?://[^\\s)\\]\"']+"
        if let regex = try? NSRegularExpression(pattern: urlPat) {
            let ns = s as NSString
            let matches = regex.matches(in: s, range: NSRange(location: 0, length: ns.length))
            if matches.count > 30 {
                for m in matches.dropFirst(30).reversed() {
                    let urlText = ns.substring(with: m.range)
                    let host = URL(string: urlText)?.host ?? "[url]"
                    s = (s as NSString).replacingCharacters(in: m.range, with: "[\(host)]")
                }
                didRedact = true
            }
        }

        return Outcome(sanitized: s, didRedact: didRedact)
    }
}

// MARK: - Outbound Search Query Privacy Filter

/// Prevents API keys (`odl_...`, `sk-...`, `Bearer ...`), local filesystem
/// paths (`/Users/...`, `/var/mobile/...`, `file://...`), and private/internal
/// IP addresses from being exfiltrated via outbound Web search queries.
public enum WebQueryPrivacyFilter {
    public struct SanitizedQuery: Sendable, Equatable {
        public let query: String
        public let didRedact: Bool
    }

    public static func sanitize(_ rawQuery: String) throws -> SanitizedQuery {
        var cleaned = rawQuery
            .replacingOccurrences(of: "\r\n", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !cleaned.isEmpty else {
            throw WebToolError.sensitiveQueryBlocked(reason: "query is empty")
        }

        var redacted = false
        let secretPatterns: [(pattern: String, reason: String)] = [
            // OnDeviceLAS local API keys & common bearer tokens
            (#"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"#, "Authorization bearer token"),
            (#"\bodl_[A-Za-z0-9_\-]{8,}\b"#, "local API key"),
            (#"\bsk-[A-Za-z0-9_\-]{12,}\b"#, "API secret key"),
            (#"\bgh[pousr]_[A-Za-z0-9]{16,}\b"#, "GitHub token"),
            (#"\bhf_[A-Za-z0-9]{16,}\b"#, "Hugging Face token"),
            (#"\bAIza[0-9A-Za-z_\-]{20,}\b"#, "cloud API key"),
            (#"(?i)\bx-api-key\s*[:=]\s*\S+"#, "X-API-Key header"),
            // Local file URLs & iOS/macOS/Linux private paths
            (#"(?i)\bfile://[^\s]+"#, "file:// URL"),
            (#"(?:^|\s)(?:/var/mobile/|/private/var/|/Users/|/home/|/etc/passwd|/etc/shadow)[^\s]*"#, "local filesystem path")
        ]

        for entry in secretPatterns {
            if cleaned.range(of: entry.pattern, options: .regularExpression) != nil {
                redacted = true
                cleaned = cleaned.replacingOccurrences(
                    of: entry.pattern,
                    with: " ",
                    options: .regularExpression
                )
            }
        }

        // Redact private / loopback / link-local IPv4 literals embedded in queries
        let ipv4Pattern = #"\b(?:\d{1,3}\.){3}\d{1,3}\b"#
        if let regex = try? NSRegularExpression(pattern: ipv4Pattern) {
            let ns = cleaned as NSString
            let matches = regex.matches(in: cleaned, range: NSRange(location: 0, length: ns.length))
            for m in matches.reversed() {
                let candidate = ns.substring(with: m.range)
                if URLSafetyValidator.isPrivateOrReserved(candidate) {
                    redacted = true
                    cleaned = (cleaned as NSString).replacingCharacters(in: m.range, with: " ")
                }
            }
        }

        cleaned = cleaned
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Cap outbound search query length to 240 characters so entire local
        // documents or conversation dumps cannot be leaked in a query string.
        if cleaned.count > 240 {
            cleaned = String(cleaned.prefix(240)).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        guard !cleaned.isEmpty else {
            throw WebToolError.sensitiveQueryBlocked(
                reason: "query contained only sensitive tokens, private IPs, or local file paths"
            )
        }
        return SanitizedQuery(query: cleaned, didRedact: redacted)
    }
}
