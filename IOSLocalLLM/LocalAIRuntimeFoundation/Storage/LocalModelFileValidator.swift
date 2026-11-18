import Foundation

// MARK: - LocalModelFileValidator

enum LocalModelFileValidator {
    struct ValidationResult: Equatable {
        var isSupported: Bool
        var failures: [String]

        static let supported = ValidationResult(isSupported: true, failures: [])
    }

    struct GGUFHeaderMetadata: Equatable, Sendable {
        let version: UInt32
        let tensorCount: UInt64
        let metadataKVCount: UInt64
        let architecture: String?
        let modelName: String?
        let contextLength: Int?
        let chatTemplate: String?
        let splitIndex: Int?
        let splitCount: Int?
    }

    enum GGUFValidationError: LocalizedError, Equatable, Sendable {
        case fileNotFound(String)
        case emptyFile(String)
        case unsupportedExtension(String)
        case invalidMagic(String)
        case truncatedHeader(String)
        case unsupportedVersion(UInt32)
        case emptyTensorTable
        case standaloneProjector(String)
        case unsupportedArchitecture(String)
        case incompleteSplitShards(missing: [String])

        var errorDescription: String? {
            switch self {
            case .fileNotFound(let name):
                return "GGUF file '\(name)' does not exist."
            case .emptyFile(let name):
                return "GGUF file '\(name)' is empty (0 bytes)."
            case .unsupportedExtension(let ext):
                return "Unsupported model file extension '.\(ext)'. OnDeviceLAS requires a '.gguf' weight file."
            case .invalidMagic(let name):
                return "File '\(name)' does not have a valid GGUF header magic."
            case .truncatedHeader(let name):
                return "GGUF file '\(name)' has a truncated or corrupted header."
            case .unsupportedVersion(let version):
                return "Unsupported GGUF format version \(version). Only GGUF v2 and v3 are supported by the embedded llama.cpp runtime."
            case .emptyTensorTable:
                return "GGUF file contains 0 tensors and cannot be loaded for inference."
            case .standaloneProjector(let name):
                return "'\(name)' is a vision projector (mmproj/CLIP), not a standalone language model. Pair it with its matching text GGUF."
            case .unsupportedArchitecture(let arch):
                return "GGUF architecture '\(arch)' is not a supported causal text/chat LLM."
            case .incompleteSplitShards(let missing):
                return "Split GGUF model is incomplete. Missing shard(s): \(missing.joined(separator: ", "))."
            }
        }
    }

    private static let unsupportedGGUFArchitectures: Set<String> = [
        "whisper", "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2",
        "t5encoder", "flux", "sd1", "sd2", "sd3", "sdxl", "wan", "bark", "encodec"
    ]

    static func validateModelRoot(_ dir: URL) -> ValidationResult {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir),
              isDir.boolValue else {
            return .init(isSupported: false, failures: ["Directory does not exist."])
        }

        if let llmURL = ggufLLM(in: dir) {
            do {
                _ = try validateGGUFForAssistant(llmURL)
                return .supported
            } catch {
                return .init(isSupported: false, failures: [error.localizedDescription])
            }
        }

        guard let files = recursiveFiles(in: dir) else {
            return .init(isSupported: false, failures: ["Could not enumerate model directory."])
        }

        let ggufCandidates = files.filter { $0.pathExtension.lowercased() == "gguf" }
        if !ggufCandidates.isEmpty {
            let preferred = ggufCandidates.first { !$0.lastPathComponent.lowercased().hasPrefix("mmproj") }
                ?? ggufCandidates[0]
            do {
                _ = try validateGGUFForAssistant(preferred)
                return .supported
            } catch {
                return .init(isSupported: false, failures: [error.localizedDescription])
            }
        }

        let names = Set(files.map(\.lastPathComponent))
        let hasConfig = names.contains("config.json") || names.contains("model_index.json")
        let hasTokenizer = names.contains("tokenizer.json")
            || names.contains("tokenizer_config.json")
            || names.contains("vocab.json")
        let weightFiles = files.filter { isRecognizedWeightFile($0.lastPathComponent) }
        guard hasConfig || hasTokenizer else {
            return .init(isSupported: false, failures: ["Missing config.json or tokenizer companion file."])
        }
        guard !weightFiles.isEmpty else {
            return .init(isSupported: false, failures: ["Missing model weights."])
        }
        let emptyWeights = weightFiles.filter { !isNonEmptyFile($0) }
        guard emptyWeights.isEmpty else {
            return .init(
                isSupported: false,
                failures: emptyWeights.map { "\($0.lastPathComponent) is empty." }
            )
        }
        if let shardFailure = validateSafetensorShards(in: dir, files: files) {
            return .init(isSupported: false, failures: [shardFailure])
        }
        return .supported
    }

    /// True when `dir` holds a model the MLX/safetensors **text** loader can
    /// actually load. Stricter than `validateModelRoot` on purpose: the
    /// swift-transformers `Hub.loadConfig` path hard-requires `tokenizer.json`
    /// and throws `configurationMissing("tokenizer.json")` without it — even
    /// when `config.json`, `tokenizer_config.json`, and weights are all present.
    /// `validateModelRoot` deliberately accepts a `tokenizer_config.json` /
    /// `vocab.json` companion as "has a tokenizer", so a snapshot missing only
    /// `tokenizer.json` passed as usable and was then loaded as a LOCAL
    /// directory (giving the loader no chance to fetch the missing file) — the
    /// "Qwen3-4B failed to load: Required configuration file missing:
    /// tokenizer.json" regression. GGUF models embed their tokenizer, so
    /// they're exempt. Callers that resolve a *pre-staged* MLX text model must
    /// use this, not the generic validator, so a tokenizer-less dir is rejected
    /// and the loader falls back to the repo-id config (HubApi then re-fetches
    /// only the missing file).
    static func isUsableMLXTextModelDirectory(_ dir: URL) -> Bool {
        guard validateModelRoot(dir).isSupported else { return false }
        guard let files = recursiveFiles(in: dir) else { return false }
        let names = Set(files.map { $0.lastPathComponent.lowercased() })
        if names.contains(where: { $0.hasSuffix(".gguf") }) { return true }  // GGUF embeds its tokenizer
        return names.contains("tokenizer.json")
    }

    static func hasCompleteGGUFVLMPair(in dir: URL) -> Bool {
        ggufLLM(in: dir) != nil && ggufProjector(in: dir) != nil
    }

    /// First valid LLM-weights GGUF (any `*.gguf` not prefixed `mmproj`)
    /// anywhere under `dir`, or nil. Single source of truth for the GGUF
    /// VLM load path: the readiness probe (`hasCompleteGGUFVLMPair`) and the
    /// loader (`LlamaCppVLMService.resolveLLMPath`) BOTH route through this, so
    /// a repo whose weights sit in a subfolder — or end in uppercase `.GGUF` —
    /// can never pass the probe yet fail to resolve at load.
    static func ggufLLM(in dir: URL) -> URL? {
        firstValidGGUF(in: dir) { !$0.hasPrefix("mmproj") }
    }

    /// First valid vision-projector GGUF (`mmproj-*.gguf`) under `dir`, or nil.
    static func ggufProjector(in dir: URL) -> URL? {
        firstValidGGUF(in: dir) { $0.hasPrefix("mmproj") }
    }

    /// Shared recursive, case-insensitive GGUF finder. `nameMatches` receives
    /// the lowercased filename.
    private static func firstValidGGUF(in dir: URL, where nameMatches: (String) -> Bool) -> URL? {
        guard let files = recursiveFiles(in: dir) else { return nil }
        return files.first { url in
            url.pathExtension.lowercased() == "gguf"
                && nameMatches(url.lastPathComponent.lowercased())
                && isValidGGUFFile(url)
        }
    }

    static func hasValidGGUFTextModel(in dir: URL) -> Bool {
        guard let files = recursiveFiles(in: dir) else { return false }
        let ggufs = files.filter { url in
            url.pathExtension.lowercased() == "gguf"
                && !url.lastPathComponent.lowercased().hasPrefix("mmproj")
        }
        return ggufs.contains(where: isValidGGUFFile)
    }

    static func isValidGGUFFile(_ url: URL) -> Bool {
        guard isNonEmptyFile(url),
              let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: 24), data.count >= 16 else { return false }
        guard data.prefix(4) == Data([0x47, 0x47, 0x55, 0x46]) else { return false } // "GGUF"
        let version = data.withUnsafeBytes { raw -> UInt32 in
            var v: UInt32 = 0
            _ = withUnsafeMutableBytes(of: &v) { dst in
                raw[4..<8].copyBytes(to: dst)
            }
            return UInt32(littleEndian: v)
        }
        guard version == 2 || version == 3 else { return false }
        let tensors = data.withUnsafeBytes { raw -> UInt64 in
            var t: UInt64 = 0
            _ = withUnsafeMutableBytes(of: &t) { dst in
                raw[8..<16].copyBytes(to: dst)
            }
            return UInt64(littleEndian: t)
        }
        return tensors > 0
    }

    /// Inspects a GGUF v2/v3 file header and validates that it represents a
    /// supported, shard-complete causal language model before native loading.
    static func validateGGUFForAssistant(_ url: URL) throws -> GGUFHeaderMetadata {
        let filename = url.lastPathComponent
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GGUFValidationError.fileNotFound(filename)
        }
        let ext = url.pathExtension.lowercased()
        guard ext == "gguf" else {
            throw GGUFValidationError.unsupportedExtension(ext.isEmpty ? "none" : ext)
        }
        guard isNonEmptyFile(url) else {
            throw GGUFValidationError.emptyFile(filename)
        }
        if filename.lowercased().hasPrefix("mmproj") {
            throw GGUFValidationError.standaloneProjector(filename)
        }

        let metadata = try inspectGGUFHeader(at: url)
        if let arch = metadata.architecture?.lowercased() {
            if arch == "clip" || arch == "mtmd" {
                throw GGUFValidationError.standaloneProjector(filename)
            }
            if unsupportedGGUFArchitectures.contains(arch) {
                throw GGUFValidationError.unsupportedArchitecture(arch)
            }
        }

        let missingShards = missingSplitGGUFShards(for: url, metadata: metadata)
        if !missingShards.isEmpty {
            throw GGUFValidationError.incompleteSplitShards(missing: missingShards)
        }
        return metadata
    }

    /// Parses the GGUF v2/v3 binary header and extracts lightweight general /
    /// architecture / context / split metadata without loading model tensors.
    static func inspectGGUFHeader(at url: URL, maxKVScanEntries: Int = 96) throws -> GGUFHeaderMetadata {
        let filename = url.lastPathComponent
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GGUFValidationError.fileNotFound(filename)
        }
        guard isNonEmptyFile(url) else {
            throw GGUFValidationError.emptyFile(filename)
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var reader = GGUFStreamReader(handle: handle)
        guard let magic = reader.readBytes(count: 4), magic.count == 4 else {
            throw GGUFValidationError.truncatedHeader(filename)
        }
        guard magic == Data([0x47, 0x47, 0x55, 0x46]) else {
            throw GGUFValidationError.invalidMagic(filename)
        }
        guard let version = reader.readUInt32() else {
            throw GGUFValidationError.truncatedHeader(filename)
        }
        guard version == 2 || version == 3 else {
            throw GGUFValidationError.unsupportedVersion(version)
        }
        guard let tensorCount = reader.readUInt64(),
              let kvCount = reader.readUInt64() else {
            throw GGUFValidationError.truncatedHeader(filename)
        }
        guard tensorCount > 0 else {
            throw GGUFValidationError.emptyTensorTable
        }

        var architecture: String?
        var modelName: String?
        var contextLength: Int?
        var chatTemplate: String?
        var splitIndex: Int?
        var splitCount: Int?
        var pendingContextByKey: [String: Int] = [:]

        let limit = min(Int(clamping: kvCount), maxKVScanEntries)
        for _ in 0..<limit {
            guard let key = reader.readString(maxBytes: 1_024),
                  let valueType = reader.readUInt32() else {
                break
            }
            switch key {
            case "general.architecture":
                architecture = reader.readStringValue(type: valueType, maxBytes: 256)
                if let arch = architecture,
                   let found = pendingContextByKey["\(arch).context_length"] {
                    contextLength = found
                }
            case "general.name":
                modelName = reader.readStringValue(type: valueType, maxBytes: 512)
            case "tokenizer.chat_template":
                chatTemplate = reader.readStringValue(type: valueType, maxBytes: 65_536)
            case "split.no":
                splitIndex = reader.readIntValue(type: valueType)
            case "split.count":
                splitCount = reader.readIntValue(type: valueType)
            default:
                if key.hasSuffix(".context_length") {
                    if let parsed = reader.readIntValue(type: valueType), parsed > 0 {
                        pendingContextByKey[key] = parsed
                        if let arch = architecture, key == "\(arch).context_length" {
                            contextLength = parsed
                        } else if contextLength == nil {
                            contextLength = parsed
                        }
                    }
                } else if !reader.skipValue(type: valueType) {
                    return GGUFHeaderMetadata(
                        version: version,
                        tensorCount: tensorCount,
                        metadataKVCount: kvCount,
                        architecture: architecture,
                        modelName: modelName,
                        contextLength: contextLength,
                        chatTemplate: chatTemplate,
                        splitIndex: splitIndex,
                        splitCount: splitCount
                    )
                }
            }
        }

        return GGUFHeaderMetadata(
            version: version,
            tensorCount: tensorCount,
            metadataKVCount: kvCount,
            architecture: architecture,
            modelName: modelName,
            contextLength: contextLength,
            chatTemplate: chatTemplate,
            splitIndex: splitIndex,
            splitCount: splitCount
        )
    }

    private static func missingSplitGGUFShards(
        for url: URL,
        metadata: GGUFHeaderMetadata
    ) -> [String] {
        let filename = url.lastPathComponent
        let dir = url.deletingLastPathComponent()
        // Standard llama.cpp split naming: <prefix>-00001-of-00003.gguf
        let pattern = #"^(.*)-(\d{5})-of-(\d{5})\.gguf$"#
        if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
           let match = regex.firstMatch(
               in: filename,
               options: [],
               range: NSRange(filename.startIndex..<filename.endIndex, in: filename)
           ),
           match.numberOfRanges == 4,
           let prefixRange = Range(match.range(at: 1), in: filename),
           let totalRange = Range(match.range(at: 3), in: filename),
           let total = Int(filename[totalRange]),
           total > 1 {
            let prefix = String(filename[prefixRange])
            let ext = url.pathExtension
            var missing: [String] = []
            for index in 1...total {
                let expected = String(format: "%@-%05d-of-%05d.%@", prefix, index, total, ext)
                let shardURL = dir.appendingPathComponent(expected)
                if !isNonEmptyFile(shardURL) {
                    missing.append(expected)
                }
            }
            return missing
        }

        if let splitCount = metadata.splitCount, splitCount > 1 {
            let siblingGGUFs = (try? FileManager.default.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: nil
            ))?.filter {
                $0.pathExtension.lowercased() == "gguf"
                    && !$0.lastPathComponent.lowercased().hasPrefix("mmproj")
                    && isNonEmptyFile($0)
            } ?? []
            if siblingGGUFs.count < splitCount {
                return ["expected \(splitCount) split GGUF shards, found \(siblingGGUFs.count)"]
            }
        }
        return []
    }

    private struct GGUFStreamReader {
        let handle: FileHandle
        private var offset: UInt64 = 0

        init(handle: FileHandle) {
            self.handle = handle
        }

        mutating func readBytes(count: Int) -> Data? {
            guard count >= 0 else { return nil }
            if count == 0 { return Data() }
            guard let data = try? handle.read(upToCount: count), data.count == count else {
                return nil
            }
            offset += UInt64(count)
            return data
        }

        mutating func skip(bytes: UInt64) -> Bool {
            if bytes == 0 { return true }
            let target = offset + bytes
            do {
                try handle.seek(toOffset: target)
                offset = target
                return true
            } catch {
                return false
            }
        }

        mutating func readUInt32() -> UInt32? {
            guard let data = readBytes(count: 4) else { return nil }
            var value: UInt32 = 0
            _ = withUnsafeMutableBytes(of: &value) { data.copyBytes(to: $0) }
            return UInt32(littleEndian: value)
        }

        mutating func readInt32() -> Int32? {
            guard let raw = readUInt32() else { return nil }
            return Int32(bitPattern: raw)
        }

        mutating func readUInt64() -> UInt64? {
            guard let data = readBytes(count: 8) else { return nil }
            var value: UInt64 = 0
            _ = withUnsafeMutableBytes(of: &value) { data.copyBytes(to: $0) }
            return UInt64(littleEndian: value)
        }

        mutating func readInt64() -> Int64? {
            guard let raw = readUInt64() else { return nil }
            return Int64(bitPattern: raw)
        }

        mutating func readString(maxBytes: Int) -> String? {
            guard let length = readUInt64() else { return nil }
            if length > UInt64(maxBytes) {
                guard skip(bytes: length) else { return nil }
                return nil
            }
            guard let data = readBytes(count: Int(length)) else { return nil }
            return String(decoding: data, as: UTF8.self)
        }

        mutating func readStringValue(type: UInt32, maxBytes: Int) -> String? {
            guard type == 8 else {
                _ = skipValue(type: type)
                return nil
            }
            return readString(maxBytes: maxBytes)
        }

        mutating func readIntValue(type: UInt32) -> Int? {
            switch type {
            case 0: // UINT8
                return readBytes(count: 1).map { Int($0[0]) }
            case 1: // INT8
                return readBytes(count: 1).map { Int(Int8(bitPattern: $0[0])) }
            case 2: // UINT16
                guard let d = readBytes(count: 2) else { return nil }
                var v: UInt16 = 0
                _ = withUnsafeMutableBytes(of: &v) { d.copyBytes(to: $0) }
                return Int(UInt16(littleEndian: v))
            case 3: // INT16
                guard let d = readBytes(count: 2) else { return nil }
                var v: Int16 = 0
                _ = withUnsafeMutableBytes(of: &v) { d.copyBytes(to: $0) }
                return Int(Int16(littleEndian: v))
            case 4: // UINT32
                return readUInt32().map { Int($0) }
            case 5: // INT32
                return readInt32().map { Int($0) }
            case 10: // UINT64
                return readUInt64().flatMap { Int(exactly: $0) }
            case 11: // INT64
                return readInt64().flatMap { Int(exactly: $0) }
            default:
                _ = skipValue(type: type)
                return nil
            }
        }

        private static func fixedElementSize(for type: UInt32) -> UInt64? {
            switch type {
            case 0, 1, 7: return 1
            case 2, 3: return 2
            case 4, 5, 6: return 4
            case 10, 11, 12: return 8
            default: return nil
            }
        }

        mutating func skipValue(type: UInt32, depth: Int = 0) -> Bool {
            guard depth < 4 else { return false }
            if let fixed = Self.fixedElementSize(for: type) {
                return skip(bytes: fixed)
            }
            switch type {
            case 8: // STRING
                guard let length = readUInt64() else { return false }
                return skip(bytes: length)
            case 9: // ARRAY
                guard let elemType = readUInt32(),
                      let count = readUInt64() else { return false }
                if let elemFixed = Self.fixedElementSize(for: elemType) {
                    let (total, overflow) = count.multipliedReportingOverflow(by: elemFixed)
                    guard !overflow else { return false }
                    return skip(bytes: total)
                }
                // Cap variable-width array skipping (e.g. 256k vocab strings)
                guard count <= 300_000 else { return false }
                for _ in 0..<count {
                    if !skipValue(type: elemType, depth: depth + 1) { return false }
                }
                return true
            default:
                return false
            }
        }
    }

    static func isNonEmptyFile(_ url: URL) -> Bool {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? NSNumber else { return false }
        return size.int64Value > 0
    }

    private static func recursiveFiles(in dir: URL) -> [URL]? {
        guard let enumerator = FileManager.default.enumerator(
            at: dir,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }
        var files: [URL] = []
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
            if values?.isRegularFile == true {
                files.append(url)
            }
        }
        return files
    }

    private static func isRecognizedWeightFile(_ name: String) -> Bool {
        let lower = name.lowercased()
        return lower.hasSuffix(".safetensors")
            || lower.hasSuffix(".npz")
            || lower.hasSuffix(".bin")
            || lower.hasSuffix(".gguf")
    }

    private static func validateSafetensorShards(in dir: URL, files: [URL]) -> String? {
        guard let indexURL = files.first(where: { $0.lastPathComponent == "model.safetensors.index.json" }) else {
            return nil
        }
        guard let required = requiredSafetensorShards(indexURL: indexURL), !required.isEmpty else {
            return nil
        }
        let present = Set(files.map(\.lastPathComponent))
        let missing = required.subtracting(present)
        if missing.isEmpty { return nil }
        return "Missing safetensor shard(s): \(missing.sorted().joined(separator: ", "))."
    }

    private static func requiredSafetensorShards(indexURL: URL) -> Set<String>? {
        guard
            let data = try? Data(contentsOf: indexURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = json["weight_map"] as? [String: String]
        else { return nil }
        return Set(weightMap.values)
    }
}
