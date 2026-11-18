import Foundation

enum ModelToolParser: String, Codable, CaseIterable, Sendable {
    case none
    case hermes
    case qwen3Coder = "qwen3_coder"
    case qwen3XML = "qwen3_xml"
    case gemma4 = "gemma4"
    case foundationModels = "foundation_models"

    var displayName: String {
        switch self {
        case .none: return "None"
        case .hermes: return "Hermes"
        case .qwen3Coder: return "Qwen3 Coder"
        case .qwen3XML: return "Qwen3 XML"
        case .gemma4: return "Gemma 4"
        case .foundationModels: return "Foundation Models"
        }
    }
}

enum ModelReasoningParser: String, Codable, CaseIterable, Sendable {
    case none
    case deepSeekR1 = "deepseek_r1"
    case qwen3
    case gemma4 = "gemma4"
    case hermes

    var displayName: String {
        switch self {
        case .none: return "None"
        case .deepSeekR1: return "DeepSeek R1"
        case .qwen3: return "Qwen3"
        case .gemma4: return "Gemma 4"
        case .hermes: return "Hermes"
        }
    }

    /// Chat templates for these parsers typically pre-fill the opening
    /// reasoning marker, so the model only emits a closing tag.
    var assumesPrefilledOpening: Bool {
        switch self {
        case .none, .hermes: return false
        case .deepSeekR1, .qwen3, .gemma4: return true
        }
    }
}

enum ModelParallelToolPolicy: Equatable, Sendable {
    case sequential
    case automatic(maximumCalls: Int)

    var maximumCalls: Int {
        switch self {
        case .sequential: return 1
        case .automatic(let maximumCalls): return maximumCalls
        }
    }

    var displayName: String {
        switch self {
        case .sequential: return "Sequential"
        case .automatic(let maximumCalls): return "Auto · up to \(maximumCalls)"
        }
    }
}

enum LocalAPIParallelToolCallsSetting: Int, CaseIterable, Identifiable, Equatable, Sendable {
    case sequential = 1
    case automaticTwo = 2
    case automaticThree = 3
    case automaticFour = 4

    var id: Int { rawValue }

    var maximumCalls: Int { rawValue }

    var displayName: String {
        switch self {
        case .sequential: return "Sequential"
        case .automaticTwo: return "Auto · up to 2"
        case .automaticThree: return "Auto · up to 3"
        case .automaticFour: return "Auto · up to 4"
        }
    }

    var allowsParallelCalls: Bool {
        self != .sequential
    }
}

/// One source of truth for the model-facing context, parser, and tool policy.
/// The same profile drives MLX generation, `/v1/models`, request validation,
/// and the Home compatibility card so the server never advertises a context
/// window that its active KV cache cannot retain.
struct ModelCapabilityProfile: Equatable, Sendable {
    static let hermesMinimumContextTokens = 65_536

    enum Family: String, Sendable {
        case qwen3
        case qwen35 = "qwen3.5"
        case gemma4 = "gemma-4"
        case ornith
        case bonsai
        case hermes
        case generic

        var displayName: String {
            switch self {
            case .qwen3: return "Qwen3"
            case .qwen35: return "Qwen3.5"
            case .gemma4: return "Gemma 4"
            case .ornith: return "Ornith"
            case .bonsai: return "Bonsai"
            case .hermes: return "Hermes"
            case .generic: return "Generic"
            }
        }
    }

    let family: Family
    /// Published architecture limit. This can be larger than the practical
    /// on-device allocation.
    let modelContextLength: Int
    /// Context the app really retains for this runtime.
    let configuredContextLength: Int
    let maximumKVCacheTokens: Int
    let maximumOutputTokens: Int
    let kvBits: Int
    let prefillStepSize: Int
    let toolParser: ModelToolParser
    let reasoningParser: ModelReasoningParser
    let parallelTools: ModelParallelToolPolicy
    let supportsThinkingToggle: Bool
    let usesChatTemplateOverride: Bool
    let requiresYaRNContextExtension: Bool
    /// Conservative estimate used for an understandable UI warning. MLX
    /// allocates lazily, so this is the fully-populated cache estimate rather
    /// than memory charged immediately when the model loads.
    let estimatedKVCacheBytes: Int64

    var effectiveContextWindow: Int {
        min(configuredContextLength, maximumKVCacheTokens)
    }

    var defaultOutputTokens: Int {
        min(maximumOutputTokens, max(1, min(2_048, effectiveContextWindow / 4)))
    }

    var runtimeLimits: LocalAPIRuntimeLimits {
        LocalAPIRuntimeLimits(
            contextWindow: effectiveContextWindow,
            maximumOutputTokens: maximumOutputTokens,
            defaultOutputTokens: defaultOutputTokens,
            limitSource: .runtime,
            tokenizerIdentifier: nil,
            tokenCountMode: .conservativeApproximation
        )
    }

    /// Whether the OpenAI streaming path should start inside a reasoning
    /// block (template-prefilled open). Ornith uses an explicit
    /// `Thinking Process:` marker instead, so it stays on the implicit path.
    var streamAssumesPrefilledReasoningOpen: Bool {
        guard reasoningParser.assumesPrefilledOpening else { return false }
        return family != .ornith
    }

    var isHermesContextCompatible: Bool {
        configuredContextLength >= Self.hermesMinimumContextTokens
            && maximumKVCacheTokens >= Self.hermesMinimumContextTokens
            && toolParser != .none
    }

    var hermesWarning: String? {
        guard !isHermesContextCompatible else { return nil }
        return "Hermes agents require at least 65,536 real context tokens. This profile is configured for \(configuredContextLength.formatted())."
    }

    func effectiveParallelLimit(
        requestAllowsParallel: Bool,
        globalEnabled: Bool,
        configuredMaximumCalls: Int? = nil
    ) -> Int {
        guard requestAllowsParallel, globalEnabled else { return 1 }
        let configuredLimit = max(1, configuredMaximumCalls ?? parallelTools.maximumCalls)
        return min(parallelTools.maximumCalls, configuredLimit)
    }

    static func resolve(for model: AssistantModel) -> Self {
        if model.runtime == .llamaCpp {
            return resolveGGUF(
                repoID: model.repoID,
                catalogContextLength: model.contextWindowTokens,
                supportsThinking: model.supportsThinking,
                availableBytes: MemoryAdvisor.availableMemoryForModel,
                hasIncreasedMemoryEntitlement: MemoryAdvisor.hasIncreasedMemoryLimitEntitlement
            )
        }
        return resolve(
            repoID: model.repoID,
            catalogContextLength: model.contextWindowTokens,
            supportsThinking: model.supportsThinking
        )
    }

    /// Resolves a conservative 8 GB iPad-safe runtime profile for imported /
    /// downloaded GGUF models so `/v1/models`, API token budgeting, and
    /// `CodingAssistantService` agree on the real `llama.cpp` KV window.
    static func resolveGGUF(
        repoID: String,
        ggufArchitecture: String? = nil,
        metadataContextLength: Int? = nil,
        catalogContextLength: Int = 4_096,
        supportsThinking: Bool = false,
        availableBytes: Int64 = 0,
        hasIncreasedMemoryEntitlement: Bool = true
    ) -> Self {
        let gguf = GGUFExecutionProfile.resolve(
            repoID: repoID,
            ggufArchitecture: ggufArchitecture,
            metadataContextLength: metadataContextLength,
            availableBytes: availableBytes,
            hasIncreasedMemoryEntitlement: hasIncreasedMemoryEntitlement
        )
        let modelMax = max(gguf.contextTokens, metadataContextLength ?? catalogContextLength)
        return .init(
            family: gguf.family,
            modelContextLength: modelMax,
            configuredContextLength: gguf.contextTokens,
            maximumKVCacheTokens: gguf.contextTokens,
            maximumOutputTokens: gguf.maxOutputTokens,
            kvBits: 16,
            prefillStepSize: 64,
            toolParser: gguf.toolParser,
            reasoningParser: gguf.reasoningParser,
            parallelTools: gguf.parallelTools,
            supportsThinkingToggle: gguf.supportsThinking || supportsThinking,
            usesChatTemplateOverride: true,
            requiresYaRNContextExtension: false,
            estimatedKVCacheBytes: gguf.estimatedKVCacheBytes
        )
    }

    static func resolve(
        repoID: String,
        catalogContextLength: Int = 32_768,
        supportsThinking: Bool = false
    ) -> Self {
        let identity = repoID.lowercased()

        // Bonsai must be checked before Qwen3.5 because some imported folders
        // expose the base architecture in their local path. Its smaller and
        // ternary variants have not been reliable with parallel agent calls.
        if identity.contains("bonsai") {
            let context = min(max(2_048, catalogContextLength), 32_768)
            return .init(
                family: .bonsai,
                modelContextLength: max(context, catalogContextLength),
                configuredContextLength: context,
                maximumKVCacheTokens: context,
                maximumOutputTokens: 2_048,
                kvBits: 4,
                prefillStepSize: 128,
                toolParser: .qwen3Coder,
                reasoningParser: .qwen3,
                parallelTools: .sequential,
                supportsThinkingToggle: supportsThinking,
                usesChatTemplateOverride: true,
                requiresYaRNContextExtension: false,
                estimatedKVCacheBytes: Int64(context) * 16_384
            )
        }

        if identity.contains("ornith") {
            return .init(
                family: .ornith,
                modelContextLength: 262_144,
                configuredContextLength: 65_536,
                maximumKVCacheTokens: 65_536,
                maximumOutputTokens: 4_096,
                kvBits: 4,
                prefillStepSize: 128,
                toolParser: .qwen3XML,
                reasoningParser: .qwen3,
                parallelTools: .automatic(maximumCalls: 2),
                supportsThinkingToggle: true,
                usesChatTemplateOverride: true,
                requiresYaRNContextExtension: false,
                estimatedKVCacheBytes: 1_100_000_000
            )
        }

        if identity.contains("gemma-4")
            || identity.contains("gemma4")
            || identity.contains("gemma_4") {
            let context = min(max(2_048, catalogContextLength), 32_768)
            return .init(
                family: .gemma4,
                modelContextLength: max(131_072, context),
                configuredContextLength: context,
                maximumKVCacheTokens: context,
                maximumOutputTokens: min(2_048, max(512, context / 4)),
                kvBits: 8,
                prefillStepSize: 64,
                toolParser: .gemma4,
                reasoningParser: .gemma4,
                parallelTools: .automatic(maximumCalls: 2),
                supportsThinkingToggle: true,
                usesChatTemplateOverride: true,
                requiresYaRNContextExtension: false,
                estimatedKVCacheBytes: Int64(context) * 49_152
            )
        }

        if identity.contains("qwen3.5")
            || identity.contains("qwen3_5")
            || identity.contains("qwen-3.5")
            || identity.contains("qwen35") {
            return .init(
                family: .qwen35,
                modelContextLength: 262_144,
                configuredContextLength: 65_536,
                maximumKVCacheTokens: 65_536,
                maximumOutputTokens: 4_096,
                kvBits: 4,
                prefillStepSize: 128,
                toolParser: .qwen3Coder,
                reasoningParser: .qwen3,
                parallelTools: .automatic(maximumCalls: 2),
                supportsThinkingToggle: true,
                usesChatTemplateOverride: true,
                requiresYaRNContextExtension: false,
                estimatedKVCacheBytes: 1_100_000_000
            )
        }

        if identity.contains("qwen3") {
            let bytesPerToken: Int64
            if identity.contains("8b") {
                bytesPerToken = 65_536
            } else if identity.contains("4b") {
                bytesPerToken = 32_768
            } else {
                bytesPerToken = 16_384
            }
            return .init(
                family: .qwen3,
                modelContextLength: 131_072,
                configuredContextLength: 65_536,
                maximumKVCacheTokens: 65_536,
                maximumOutputTokens: 4_096,
                kvBits: 4,
                prefillStepSize: 128,
                toolParser: .hermes,
                reasoningParser: .qwen3,
                parallelTools: .automatic(maximumCalls: 2),
                supportsThinkingToggle: true,
                usesChatTemplateOverride: true,
                requiresYaRNContextExtension: true,
                estimatedKVCacheBytes: Int64(65_536) * bytesPerToken
            )
        }

        // Hermes models (NousResearch Hermes 2 Pro through Hermes 4) natively
        // emit <tool_call>{json}</tool_call> and <think> reasoning, so they
        // get the Hermes parser and no template override — their own chat
        // templates already carry tool support. Without this branch they fell
        // to the generic profile with no tool parser, disabling native tool
        // calling and failing the Hermes-agent context check.
        if identity.contains("hermes") {
            let context = min(max(2_048, catalogContextLength), 65_536)
            return .init(
                family: .hermes,
                modelContextLength: max(context, catalogContextLength),
                configuredContextLength: context,
                maximumKVCacheTokens: context,
                maximumOutputTokens: 4_096,
                kvBits: 4,
                prefillStepSize: 128,
                toolParser: .hermes,
                reasoningParser: .hermes,
                parallelTools: .automatic(maximumCalls: 2),
                supportsThinkingToggle: supportsThinking,
                usesChatTemplateOverride: false,
                requiresYaRNContextExtension: false,
                estimatedKVCacheBytes: Int64(context) * 16_384
            )
        }

        let context = max(2_048, catalogContextLength)
        return .init(
            family: .generic,
            modelContextLength: context,
            configuredContextLength: context,
            maximumKVCacheTokens: context,
            maximumOutputTokens: min(4_096, max(512, context / 4)),
            kvBits: 8,
            prefillStepSize: 256,
            toolParser: .none,
            reasoningParser: supportsThinking ? .qwen3 : .none,
            parallelTools: .sequential,
            supportsThinkingToggle: supportsThinking,
            usesChatTemplateOverride: false,
            requiresYaRNContextExtension: false,
            estimatedKVCacheBytes: Int64(context) * 32_768
        )
    }
}

enum ModelRuntimeContextConfigurator {
    private static let backupName = ".on-device-las-config-original.json"

    /// Qwen3 is native 32K. The official Qwen guidance requires a YaRN factor
    /// of 2 for a real 65,536-token runtime. Apply that override to the local
    /// model configuration before MLX decodes it, while keeping a private
    /// original beside the model for recovery/export transparency.
    static func applyIfNeeded(directory: URL, profile: ModelCapabilityProfile) throws {
        guard profile.requiresYaRNContextExtension else { return }
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else { return }
        let data = try Data(contentsOf: configURL)
        guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        let existingLength = (object["max_position_embeddings"] as? NSNumber)?.intValue ?? 0
        let existingScaling = object["rope_scaling"] as? [String: Any]
        let existingType = (existingScaling?["rope_type"] ?? existingScaling?["type"]) as? String
        let existingFactor = (existingScaling?["factor"] as? NSNumber)?.doubleValue ?? 0
        if existingLength >= profile.configuredContextLength,
           existingType?.lowercased() == "yarn",
           existingFactor >= 2 {
            return
        }

        let backupURL = directory.appendingPathComponent(backupName)
        if !FileManager.default.fileExists(atPath: backupURL.path) {
            try FileManager.default.copyItem(at: configURL, to: backupURL)
        }
        object["max_position_embeddings"] = profile.configuredContextLength
        object["rope_scaling"] = [
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 32_768
        ]
        let updated = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try updated.write(to: configURL, options: .atomic)
    }
}

/// Conservative 8 GB iPad-safe execution envelope for `llama.cpp` GGUF models.
///
/// Until `com.apple.developer.kernel.increased-memory-limit` is confirmed on a
/// sideloaded real device, GGUF defaults must stay comfortably below iOS's
/// unentitled ~3.2–3.4 GB Jetsam limit while still providing enough room for
/// OpenAI tool declarations, web search passages, and cited responses.
struct GGUFExecutionProfile: Equatable, Sendable {
    let family: ModelCapabilityProfile.Family
    let contextTokens: Int
    let maxOutputTokens: Int
    let inputBudgetTokens: Int
    let toolParser: ModelToolParser
    let reasoningParser: ModelReasoningParser
    let parallelTools: ModelParallelToolPolicy
    let supportsThinking: Bool
    let estimatedKVCacheBytes: Int64

    static func resolve(
        repoID: String,
        ggufArchitecture: String? = nil,
        metadataContextLength: Int? = nil,
        availableBytes: Int64 = 0,
        hasIncreasedMemoryEntitlement: Bool = true
    ) -> Self {
        let lower = repoID.lowercased()
        let arch = (ggufArchitecture ?? "").lowercased()
        let tightHeadroom = availableBytes > 0 && availableBytes < 2_800_000_000

        let isGemma4 = arch == "gemma4"
            || lower.contains("gemma-4")
            || lower.contains("gemma4")
            || lower.contains("gemma_4")
        let isQwen35 = arch == "qwen35"
            || lower.contains("qwen3.5")
            || lower.contains("qwen3_5")
            || lower.contains("qwen-3.5")
            || lower.contains("qwen35")

        let baseContext: Int
        let baseOutput: Int
        let family: ModelCapabilityProfile.Family
        let toolParser: ModelToolParser
        let reasoningParser: ModelReasoningParser
        let parallelTools: ModelParallelToolPolicy
        let supportsThinking: Bool
        let bytesPerToken: Int64

        if isGemma4 {
            // Gemma 4 E4B IT Q4_K_XL combines attention KV + recurrent state +
            // per-layer embeddings. Use 2,048 ctx / 768 output conservatively
            // when unentitled or below 3.6 GB live headroom; step up to 4,096 /
            // 1,024 only when entitled with >= 3.6 GB free (or unknown/0).
            let canUse4K = hasIncreasedMemoryEntitlement
                && (availableBytes == 0 || availableBytes >= 3_600_000_000)
            baseContext = canUse4K ? 4_096 : 2_048
            baseOutput = canUse4K ? 1_024 : 768
            family = .gemma4
            toolParser = .gemma4
            reasoningParser = .gemma4
            parallelTools = .automatic(maximumCalls: 2)
            supportsThinking = true
            bytesPerToken = 98_304
        } else if isQwen35 {
            baseContext = tightHeadroom ? 2_048 : 4_096
            baseOutput = tightHeadroom ? 512 : 1_024
            family = .qwen35
            toolParser = .qwen3Coder
            reasoningParser = .qwen3
            parallelTools = .automatic(maximumCalls: 2)
            supportsThinking = true
            bytesPerToken = 73_728
        } else if lower.contains("qwen3") {
            baseContext = tightHeadroom ? 2_048 : 4_096
            baseOutput = tightHeadroom ? 512 : 1_024
            family = .qwen3
            toolParser = .hermes
            reasoningParser = .qwen3
            parallelTools = .automatic(maximumCalls: 2)
            supportsThinking = true
            bytesPerToken = 81_920
        } else if lower.contains("hermes") {
            baseContext = tightHeadroom ? 2_048 : 4_096
            baseOutput = tightHeadroom ? 512 : 1_024
            family = .hermes
            toolParser = .hermes
            reasoningParser = .hermes
            parallelTools = .automatic(maximumCalls: 2)
            supportsThinking = false
            bytesPerToken = 81_920
        } else {
            baseContext = tightHeadroom ? 2_048 : 4_096
            baseOutput = tightHeadroom ? 512 : 1_024
            family = .generic
            toolParser = (lower.contains("qwen") || lower.contains("llama") || lower.contains("mistral"))
                ? .hermes
                : .none
            reasoningParser = .none
            parallelTools = .sequential
            supportsThinking = false
            bytesPerToken = 81_920
        }

        let clampedContext: Int
        if let metaCtx = metadataContextLength, metaCtx >= 512 {
            clampedContext = min(baseContext, metaCtx)
        } else {
            clampedContext = baseContext
        }
        let clampedOutput = min(baseOutput, max(128, clampedContext / 2))
        let inputBudget = max(384, clampedContext - clampedOutput - 192)
        return .init(
            family: family,
            contextTokens: clampedContext,
            maxOutputTokens: clampedOutput,
            inputBudgetTokens: inputBudget,
            toolParser: toolParser,
            reasoningParser: reasoningParser,
            parallelTools: parallelTools,
            supportsThinking: supportsThinking,
            estimatedKVCacheBytes: Int64(clampedContext) * bytesPerToken
        )
    }
}
