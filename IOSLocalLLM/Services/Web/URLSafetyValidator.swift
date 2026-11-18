import Darwin
import Foundation
import Network

// MARK: - URLSafetyValidator
// Blocks SSRF, localhost, link-local, CGNAT, ULA, NAT64-private, and reserved
// network targets BEFORE any URLSession request leaves the device. Re-validates
// every redirect hop because a public URL can 302 to `http://10.0.0.1/admin`
// and we must not follow it.
//
// This file is security-critical. Do not relax the blocklist without review.

public actor URLSafetyValidator {
    public typealias DNSResolver = @Sendable (String) async -> [String]

    private let resolver: DNSResolver

    public init(resolver: DNSResolver? = nil) {
        self.resolver = resolver ?? { host in
            await Self.resolve(host: host)
        }
    }

    public struct Verdict: Sendable, Equatable {
        public let isSafe: Bool
        public let reason: String
        public let resolvedIPs: [String]
    }

    /// Validate `url`. Returns `.isSafe == false` with reason when blocked.
    /// Always succeeds (does not throw) so call sites can branch.
    public func validate(_ url: URL) async -> Verdict {
        // 1. Scheme: strictly HTTP / HTTPS only (rejects file://, ftp://, data:, javascript:, etc.)
        guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            return .init(isSafe: false, reason: "scheme must be http(s)", resolvedIPs: [])
        }

        // 2. Reject embedded credentials (http://user:pass@host)
        if url.user != nil || url.password != nil {
            return .init(isSafe: false, reason: "embedded URL credentials are not allowed", resolvedIPs: [])
        }

        // 3. Host present
        guard let rawHost = url.host, !rawHost.isEmpty else {
            return .init(isSafe: false, reason: "missing host", resolvedIPs: [])
        }
        let host = Self.stripIPv6Brackets(rawHost)
        let lowerHost = host.lowercased()

        // 4. Obvious blocked hostnames before DNS
        if lowerHost == "localhost"
            || lowerHost.hasSuffix(".localhost")
            || lowerHost.hasSuffix(".local")
            || lowerHost.hasSuffix(".internal")
            || lowerHost.hasSuffix(".home.arpa")
            || lowerHost.hasSuffix(".localdomain") {
            return .init(isSafe: false, reason: "private hostname '\(lowerHost)'", resolvedIPs: [])
        }
        if [
            "169.254.169.254",
            "100.100.100.200",
            "metadata.google.internal",
            "metadata.aws.internal",
            "instance-data"
        ].contains(lowerHost) {
            return .init(isSafe: false, reason: "cloud-metadata host", resolvedIPs: [])
        }

        // 5. If host is already an IP literal (including non-standard / zoned forms),
        //    validate it directly BEFORE touching DNS.
        if Self.looksLikeIPLiteral(lowerHost) {
            if Self.isPrivateOrReserved(lowerHost) {
                return .init(
                    isSafe: false,
                    reason: "literal private/reserved IP \(lowerHost)",
                    resolvedIPs: [lowerHost]
                )
            }
        }

        // 6. Resolve via DNS and check EVERY returned IP
        let ips = await resolver(host)
        if ips.isEmpty {
            return .init(isSafe: false, reason: "DNS resolution failed", resolvedIPs: [])
        }
        for ip in ips {
            if Self.isPrivateOrReserved(ip) {
                return .init(
                    isSafe: false,
                    reason: "resolves to a blocked private or reserved IP address",
                    resolvedIPs: []
                )
            }
        }
        return .init(isSafe: true, reason: "ok", resolvedIPs: ips)
    }

    /// Two back-to-back validations with matching resolved IPs. Mitigates
    /// DNS rebinding between the pre-fetch check and the actual request.
    public func validateStable(_ url: URL) async -> Verdict {
        let first = await validate(url)
        guard first.isSafe else { return first }
        let second = await validate(url)
        guard second.isSafe else { return second }
        if Set(first.resolvedIPs) != Set(second.resolvedIPs) {
            return .init(
                isSafe: false,
                reason: "DNS resolution changed between checks",
                resolvedIPs: second.resolvedIPs
            )
        }
        return first
    }

    /// Convenience for validating a redirect chain in one call.
    public func validateChain(_ urls: [URL]) async -> Verdict {
        for u in urls {
            let v = await validateStable(u)
            if !v.isSafe { return v }
        }
        return .init(isSafe: true, reason: "ok", resolvedIPs: [])
    }

    // MARK: - DNS

    private static func resolve(host: String) async -> [String] {
        await withCheckedContinuation { (cont: CheckedContinuation<[String], Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                var hints = addrinfo()
                hints.ai_family = AF_UNSPEC
                hints.ai_socktype = SOCK_STREAM
                var info: UnsafeMutablePointer<addrinfo>? = nil
                let status = getaddrinfo(host, nil, &hints, &info)
                guard status == 0, let head = info else {
                    cont.resume(returning: [])
                    return
                }
                defer { freeaddrinfo(head) }
                var out: [String] = []
                var ptr: UnsafeMutablePointer<addrinfo>? = head
                while let cur = ptr?.pointee {
                    var buf = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                    if getnameinfo(cur.ai_addr, cur.ai_addrlen,
                                   &buf, socklen_t(buf.count),
                                   nil, 0, NI_NUMERICHOST) == 0 {
                        out.append(String(cString: buf))
                    }
                    ptr = cur.ai_next
                }
                cont.resume(returning: out)
            }
        }
    }

    // MARK: - Private-IP table (byte-exact via inet_pton)

    /// True if `ip` is loopback, private, link-local, CGNAT, ULA, NAT64-wrapped
    /// private IPv4, multicast, documentation, or malformed/unparseable.
    public static func isPrivateOrReserved(_ ip: String) -> Bool {
        let trimmed = stripIPv6Brackets(ip.trimmingCharacters(in: .whitespacesAndNewlines))
        guard !trimmed.isEmpty else { return true }

        // Any IPv6 zone identifier (e.g. fe80::1%en0 or %25en0) indicates a
        // scoped local interface address and must be blocked immediately.
        if trimmed.contains("%") { return true }

        if trimmed.contains(":") {
            return isPrivateIPv6(trimmed)
        }
        return isPrivateIPv4(trimmed)
    }

    private static func stripIPv6Brackets(_ host: String) -> String {
        if host.hasPrefix("["), host.hasSuffix("]"), host.count >= 2 {
            return String(host.dropFirst().dropLast())
        }
        return host
    }

    private static func looksLikeIPLiteral(_ host: String) -> Bool {
        if host.contains(":") { return true }
        let chars = CharacterSet(charactersIn: "0123456789.xXabcdefABCDEF")
        guard host.unicodeScalars.allSatisfy({ chars.contains($0) }) else { return false }
        // Pure digits (decimal integer IP like 2130706433) or hex/octal/dotted numeric host
        if host.allSatisfy(\.isNumber) { return true }
        if host.hasPrefix("0x") || host.contains(".0x") { return true }
        let parts = host.split(separator: ".", omittingEmptySubsequences: false)
        return parts.allSatisfy { part in
            !part.isEmpty && part.allSatisfy(\.isNumber)
        }
    }

    private static func isPrivateIPv4(_ ip: String) -> Bool {
        // Reject octal/hex/short-form IPv4 literals (`0177.0.0.1`, `127.1`, `2130706433`)
        // because Darwin `inet_pton(AF_INET, ...)` can accept leading zeros as decimal
        // and non-standard forms must fail closed (`return true`).
        let parts = ip.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return true }
        for part in parts {
            guard !part.isEmpty, part.allSatisfy(\.isNumber) else { return true }
            if part.count > 1, part.first == "0" { return true } // leading zero / octal syntax
            guard let val = UInt8(part) else { return true } // must fit 0...255
            _ = val
        }

        var addr = in_addr()
        let parsed = ip.withCString { inet_pton(AF_INET, $0, &addr) }
        guard parsed == 1 else { return true }
        let bytes = withUnsafeBytes(of: &addr.s_addr) { Array($0) }
        guard bytes.count == 4 else { return true }
        return isPrivateIPv4Bytes(bytes[0], bytes[1], bytes[2], bytes[3])
    }

    private static func isPrivateIPv4Bytes(
        _ a: UInt8,
        _ b: UInt8,
        _ c: UInt8,
        _ d: UInt8
    ) -> Bool {
        _ = d
        switch a {
        case 0:                                     return true   // 0.0.0.0/8 ("this" network)
        case 10:                                    return true   // 10.0.0.0/8 RFC1918
        case 100 where (64...127).contains(b):      return true   // 100.64.0.0/10 CGNAT
        case 127:                                   return true   // 127.0.0.0/8 loopback
        case 169 where b == 254:                    return true   // 169.254.0.0/16 link-local & cloud metadata
        case 172 where (16...31).contains(b):       return true   // 172.16.0.0/12 RFC1918
        case 192 where b == 0 && (c == 0 || c == 2):return true   // 192.0.0.0/24 IETF & 192.0.2.0/24 TEST-NET-1
        case 192 where b == 88 && c == 99:          return true   // 192.88.99.0/24 6to4 relay anycast
        case 192 where b == 168:                    return true   // 192.168.0.0/16 RFC1918
        case 198 where (b == 18 || b == 19):        return true   // 198.18.0.0/15 benchmarking
        case 198 where b == 51 && c == 100:         return true   // 198.51.100.0/24 TEST-NET-2
        case 203 where b == 0 && c == 113:          return true   // 203.0.113.0/24 TEST-NET-3
        case 224...239:                             return true   // 224.0.0.0/4 multicast
        case 240...255:                             return true   // 240.0.0.0/4 reserved & 255.255.255.255 broadcast
        default:                                    return false
        }
    }

    private static func isPrivateIPv6(_ ip: String) -> Bool {
        var addr = in6_addr()
        let parsed = ip.withCString { inet_pton(AF_INET6, $0, &addr) }
        guard parsed == 1 else { return true }
        let b = withUnsafeBytes(of: &addr) { Array($0) }
        guard b.count == 16 else { return true }

        // :: (unspecified) and ::1 (loopback)
        if b.dropLast().allSatisfy({ $0 == 0 }) && (b[15] == 0 || b[15] == 1) {
            return true
        }
        // IPv4-mapped IPv6 ::ffff:0:0/96 (both dotted ::ffff:127.0.0.1 and hex ::ffff:7f00:1)
        if b[0..<10].allSatisfy({ $0 == 0 }) && b[10] == 0xff && b[11] == 0xff {
            return isPrivateIPv4Bytes(b[12], b[13], b[14], b[15])
        }
        // IPv4-compatible IPv6 ::w.x.y.z (deprecated; block all 0:0:0:0:0:0::/96)
        if b[0..<12].allSatisfy({ $0 == 0 }) {
            return true
        }
        // Well-known NAT64 64:ff9b::/96 and local-use NAT64 64:ff9b:1::/48
        if b[0] == 0x00 && b[1] == 0x64 && b[2] == 0xff && b[3] == 0x9b {
            if b[4..<12].allSatisfy({ $0 == 0 }) {
                return isPrivateIPv4Bytes(b[12], b[13], b[14], b[15])
            }
            if b[4] == 0x00 && b[5] == 0x01 {
                return true
            }
        }
        // Discard-only prefix 100::/64 (RFC 6666)
        if b[0] == 0x01 && b[1] == 0x00 && b[2..<8].allSatisfy({ $0 == 0 }) {
            return true
        }
        // 6to4 2002::/16 — embeds IPv4 in bytes 2..5
        if b[0] == 0x20 && b[1] == 0x02 {
            return isPrivateIPv4Bytes(b[2], b[3], b[4], b[5])
        }
        // Teredo 2001:0000::/32 and documentation 2001:db8::/32
        if b[0] == 0x20 && b[1] == 0x01 {
            if b[2] == 0x00 && b[3] == 0x00 { return true }
            if b[2] == 0x0d && b[3] == 0xb8 { return true }
        }
        // Unique Local fc00::/7 (fc00..fdff)
        if (b[0] & 0xfe) == 0xfc { return true }
        // Link-local fe80::/10 (fe80..febf) and deprecated site-local fec0::/10
        if b[0] == 0xfe && (b[1] & 0xc0) == 0x80 { return true }
        if b[0] == 0xfe && (b[1] & 0xc0) == 0xc0 { return true }
        // Multicast ff00::/8
        if b[0] == 0xff { return true }

        return false
    }
}
