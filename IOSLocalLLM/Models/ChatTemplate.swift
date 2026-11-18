import Foundation

// MARK: - Assistant output cleanup

/// Removes small prompt-boundary artifacts emitted by some converted local
/// models. Keep this deliberately narrow: normal answer text must pass through
/// unchanged.
enum AssistantOutputSanitizer {
    static func clean(_ output: String) -> String {
        var lines = output
            .replacingOccurrences(of: "\r\n", with: "\n")
            .components(separatedBy: "\n")

        lines.removeAll { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let lowered = trimmed.lowercased()
            return lowered == "assistant: off."
                || lowered == "assistant: off"
        }

        while lines.first?.trimmingCharacters(in: .whitespaces).isEmpty == true {
            lines.removeFirst()
        }
        if lines.first?.trimmingCharacters(in: .whitespaces) == "." {
            lines.removeFirst()
            while lines.first?.trimmingCharacters(in: .whitespaces).isEmpty == true {
                lines.removeFirst()
            }
        }
        return lines.joined(separator: "\n")
    }

    /// Foundation's full Markdown parser treats a single newline as a soft
    /// break and may collapse it without leaving visible separation. Local
    /// models frequently use single newlines for compact lists, so promote
    /// those boundaries to Markdown hard breaks while leaving blank-line
    /// paragraph boundaries unchanged.
    static func preservingLineBreaksForMarkdown(_ output: String) -> String {
        let repaired = repairingRunTogetherBoundaries(in: output)
        let lines = repaired
            .replacingOccurrences(of: "\r\n", with: "\n")
            .components(separatedBy: "\n")
        guard lines.count > 1 else { return repaired }

        var result = ""
        for index in lines.indices {
            let line = lines[index]
            result += line
            guard index < lines.index(before: lines.endIndex) else { continue }

            let next = lines[lines.index(after: index)]
            if !line.isEmpty, !next.isEmpty, !line.hasSuffix("  ") {
                result += "  "
            }
            result += "\n"
        }
        return result
    }

    /// Some small converted models omit whitespace at semantic boundaries
    /// even though their token stream looked acceptable while arriving.
    /// Repair only strong prose signals here; fenced code is separated before
    /// this formatter is called, so identifiers and source code stay intact.
    private static func repairingRunTogetherBoundaries(in output: String) -> String {
        var repaired = output
        let replacements: [(String, String)] = [
            (#"([.!?])([A-Z])"#, "$1\n\n$2"),
            (#"([.!?])([—–]{1,3})([A-Z])"#, "$1\n\n—$3"),
            (#":([A-Z])"#, ":\n$1"),
            (#"\)([A-Z])"#, ")\n$1"),
            (
                #"([a-z0-9])((?:Any|Approximate|Based|Calculations|Chip|City|Current|Description|Document|Draft|Elements|Email|Find|Help|Important|Instagram|It|Just|Keep|Key|Let|Memory|Model|Navigation|North|Note|Once|Or|Original|Please|Pricing|Product|Prompt|Provide|Purpose|Removals|Searching|Set|Something|Step|Stock|Storage|Submit|Temperature|The|This|Uses|Visible|Website|West|What|Working|Works|Would|You|Your)\b)"#,
                "$1\n$2"
            ),
            (
                #"([A-Z]{2,})(?=(?:Chip|Description|Elements|Memory|Model|Navigation|Note|Pricing|Product|Stock|Storage|Visible|Website)\b)"#,
                "$1\n"
            ),
        ]
        for (pattern, template) in replacements {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let range = NSRange(repaired.startIndex..., in: repaired)
            repaired = regex.stringByReplacingMatches(
                in: repaired,
                range: range,
                withTemplate: template
            )
        }
        return repaired
    }
}

// MARK: - Image grounding handoff cleanup

/// Cleans text produced by one local model before it is embedded inside
/// another model's chat template. Vision runtimes can occasionally surface
/// their own role/end tokens as ordinary text; passing those through verbatim
/// can close the receiving model's user turn and make it emit EOS immediately.
enum ImageGroundingSanitizer {
    static func clean(_ output: String) -> String {
        var cleaned = output
            .replacingOccurrences(of: "\r\n", with: "\n")

        let controlPatterns = [
            #"<\|[^>\n]{1,80}\|>"#,
            #"</?s>"#,
            #"<end_of_(?:utterance|turn)>"#,
            #"\[/?INST\]"#,
            #"<<\/?SYS>>"#,
        ]
        for pattern in controlPatterns {
            cleaned = cleaned.replacingOccurrences(
                of: pattern,
                with: " ",
                options: .regularExpression
            )
        }

        cleaned = cleaned.replacingOccurrences(
            of: #"(?im)^\s*(?:assistant|system|user)\s*:?\s*$\n?"#,
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: #"(?im)^\s*(?:assistant|system|user)\s*:\s*"#,
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: #"[ \t]+\n"#,
            with: "\n",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: #"\n{3,}"#,
            with: "\n\n",
            options: .regularExpression
        )
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - ChatTemplate
// Describes the prompt formatting rules for a specific model family.
// Used to properly format conversation history before tokenization.
// Different model families (Qwen, Llama, Gemma, Phi) use different
// control tokens and formatting conventions.

struct ChatTemplate: Codable, Hashable {
    /// Template format identifier: "chatml", "qwen35", "llama3", "gemma", "phi", "custom"
    let format: String
    /// Opening tag for a system message turn (e.g. "<|im_start|>system\n")
    let systemPrefix: String
    /// Closing tag for a system message (e.g. "<|im_end|>")
    let systemSuffix: String
    /// Opening tag for a user message turn
    let userPrefix: String
    let userSuffix: String
    /// Opening tag for the assistant message turn
    let assistantPrefix: String
    let assistantSuffix: String
    /// The prompt suffix that tells the model to start generating
    /// (e.g. "<|im_start|>assistant\n" for ChatML)
    let generationPrompt: String
    /// Whether the model supports a "thinking" mode toggle
    let supportsThinking: Bool
    /// Optional tag to suppress thinking (e.g. "/no_think" for Qwen3)
    let noThinkTag: String?
    /// Optional tag to enable thinking (e.g. "/think" for Qwen3)
    let thinkTag: String?
    /// Extra end-of-sequence tokens beyond the standard ones
    let extraEOSTokens: [String]

    // MARK: - Built-in templates

    /// ChatML — used by Qwen2.5, Qwen3, and many Chinese models.
    static let chatML = ChatTemplate(
        format: "chatml",
        systemPrefix: "<|im_start|>system\n",
        systemSuffix: "<|im_end|>\n",
        userPrefix: "<|im_start|>user\n",
        userSuffix: "<|im_end|>\n",
        assistantPrefix: "<|im_start|>assistant\n",
        assistantSuffix: "<|im_end|>\n",
        generationPrompt: "<|im_start|>assistant\n",
        supportsThinking: true,
        noThinkTag: " /no_think",
        thinkTag: " /think",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Qwen 3.5 hybrid-attention models use ChatML turns but start every
    /// assistant response with an explicit thinking block. Bonsai 27B keeps
    /// this tokenizer contract even though its repository name omits Qwen.
    static let qwen35 = ChatTemplate(
        format: "qwen35",
        systemPrefix: "<|im_start|>system\n",
        systemSuffix: "<|im_end|>\n",
        userPrefix: "<|im_start|>user\n",
        userSuffix: "<|im_end|>\n",
        assistantPrefix: "<|im_start|>assistant\n",
        assistantSuffix: "<|im_end|>\n",
        generationPrompt: "<|im_start|>assistant\n",
        supportsThinking: true,
        noThinkTag: "<think>\n\n</think>\n\n",
        thinkTag: "<think>\n",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Llama 3 — uses <|begin_of_text|> and <|eot_id|>
    static let llama3 = ChatTemplate(
        format: "llama3",
        systemPrefix: "<|start_header_id|>system<|end_header_id|>\n\n",
        systemSuffix: "<|eot_id|>",
        userPrefix: "<|start_header_id|>user<|end_header_id|>\n\n",
        userSuffix: "<|eot_id|>",
        assistantPrefix: "<|start_header_id|>assistant<|end_header_id|>\n\n",
        assistantSuffix: "<|eot_id|>",
        generationPrompt: "<|start_header_id|>assistant<|end_header_id|>\n\n",
        supportsThinking: false,
        noThinkTag: nil,
        thinkTag: nil,
        extraEOSTokens: ["<|eot_id|>"]
    )

    /// Gemma — uses <bos>, user/nmodel turns
    static let gemma = ChatTemplate(
        format: "gemma",
        systemPrefix: "",
        systemSuffix: "",
        userPrefix: "<start_of_turn>user\n",
        userSuffix: "<end_of_turn>\n",
        assistantPrefix: "<start_of_turn>model\n",
        assistantSuffix: "<end_of_turn>\n",
        generationPrompt: "<start_of_turn>model\n",
        supportsThinking: false,
        noThinkTag: nil,
        thinkTag: nil,
        extraEOSTokens: ["<end_of_turn>"]
    )

    /// Gemma 4 IT — uses <|turn>role\n...<turn|>\n, <|think|> in the system
    /// turn when thinking is enabled, <|channel>thought\n...<channel|>, and
    /// native <|tool> / <|tool_call> / <|tool_response> blocks.
    static let gemma4 = ChatTemplate(
        format: "gemma4",
        systemPrefix: "<|turn>system\n",
        systemSuffix: "<turn|>\n",
        userPrefix: "<|turn>user\n",
        userSuffix: "<turn|>\n",
        assistantPrefix: "<|turn>model\n",
        assistantSuffix: "<turn|>\n",
        generationPrompt: "<|turn>model\n",
        supportsThinking: true,
        noThinkTag: "<|channel>thought\n<channel|>",
        thinkTag: "<|channel>thought\n",
        extraEOSTokens: ["<turn|>", "<end_of_turn>"]
    )

    /// Phi — uses <|user|>, <|assistant|>, <|system|>
    static let phi = ChatTemplate(
        format: "phi",
        systemPrefix: "<|system|>\n",
        systemSuffix: "<|end|>\n",
        userPrefix: "<|user|>\n",
        userSuffix: "<|end|>\n",
        assistantPrefix: "<|assistant|>\n",
        assistantSuffix: "<|end|>\n",
        generationPrompt: "<|assistant|>\n",
        supportsThinking: false,
        noThinkTag: nil,
        thinkTag: nil,
        extraEOSTokens: ["<|end|>", "<|endoftext|>"]
    )

    /// Generic / unknown — fallback to ChatML which is the most widely used.
    static let generic = ChatTemplate(
        format: "generic",
        systemPrefix: "<|im_start|>system\n",
        systemSuffix: "<|im_end|>\n",
        userPrefix: "<|im_start|>user\n",
        userSuffix: "<|im_end|>\n",
        assistantPrefix: "<|im_start|>assistant\n",
        assistantSuffix: "<|im_end|>\n",
        generationPrompt: "<|im_start|>assistant\n",
        supportsThinking: false,
        noThinkTag: nil,
        thinkTag: nil,
        extraEOSTokens: ["<|im_end|>"]
    )

    // MARK: - Resolution

    /// Detect the best template for a model based on its repo ID, display name,
    /// or inspected GGUF `general.architecture` metadata.
    static func detect(for repoID: String, ggufArchitecture: String? = nil) -> ChatTemplate {
        let lower = repoID.lowercased()
        let arch = (ggufArchitecture ?? "").lowercased()
        if arch == "gemma4"
            || lower.contains("gemma-4")
            || lower.contains("gemma4")
            || lower.contains("gemma_4") {
            return .gemma4
        }
        if arch == "qwen35"
            || lower.contains("qwen3.5")
            || lower.contains("qwen3_5")
            || lower.contains("qwen-3.5")
            || lower.contains("qwen35")
            || (lower.contains("bonsai") && lower.contains("27b")) {
            return .qwen35
        }
        if lower.contains("bonsai") { return .chatML }
        if lower.contains("qwen") || arch.hasPrefix("qwen") { return .chatML }
        if lower.contains("hermes") { return .chatML }
        if lower.contains("llama") || arch == "llama" { return .llama3 }
        if lower.contains("gemma") || arch.hasPrefix("gemma") { return .gemma }
        if lower.contains("phi") || arch.hasPrefix("phi") { return .phi }
        if lower.contains("smol")   { return .chatML }
        return .generic
    }

    // MARK: - Formatting

    /// Format an array of ChatMessage into the model's prompt format.
    func format(messages: [ChatMessage], enableThinking: Bool = false) -> String {
        if format == "gemma4" {
            return formatGemma4(messages: messages, enableThinking: enableThinking)
        }

        var parts: [String] = []
        let usesChatMLTools = (format == "qwen35" || format == "chatml")
        var index = 0

        while index < messages.count {
            let msg = messages[index]
            switch msg.role {
            case .system:
                if !systemPrefix.isEmpty {
                    parts.append("\(systemPrefix)\(msg.contentForModel)\(systemSuffix)")
                } else {
                    // Legacy Gemma 2/3 has no system turn — merge into the next
                    // user message if adjacent, otherwise wrap once as user.
                    if index + 1 < messages.count, messages[index + 1].role == .user {
                        let nextUser = messages[index + 1]
                        parts.append("\(userPrefix)\(msg.contentForModel)\n\n\(nextUser.contentForModel)\(userSuffix)")
                        index += 2
                        continue
                    } else {
                        parts.append("\(userPrefix)System: \(msg.contentForModel)\(userSuffix)")
                    }
                }
            case .user:
                parts.append("\(userPrefix)\(msg.contentForModel)\(userSuffix)")
            case .assistant:
                if usesChatMLTools, let calls = msg.toolCalls, !calls.isEmpty {
                    let prose = AssistantOutputSanitizer.clean(msg.content)
                    let renderedCalls = calls.map { call in
                        format == "qwen35"
                            ? Self.formatQwen35ToolCall(name: call.name, argumentsJSON: call.argumentsJSON)
                            : Self.formatHermesToolCall(name: call.name, argumentsJSON: call.argumentsJSON)
                    }.joined(separator: "\n")
                    let combined = prose.isEmpty ? renderedCalls : "\(prose)\n\(renderedCalls)"
                    parts.append("\(assistantPrefix)\(combined)\(assistantSuffix)")
                } else {
                    // Do not feed a converted model's prompt-boundary leak back
                    // into the next turn, where it tends to be imitated.
                    let content = AssistantOutputSanitizer.clean(msg.contentForModel)
                    parts.append("\(assistantPrefix)\(content)\(assistantSuffix)")
                }
            case .tool:
                if usesChatMLTools {
                    // Official Qwen3 / Qwen3.5 Jinja wraps consecutive tool
                    // responses inside one <|im_start|>user turn using
                    // <tool_response>...</tool_response>.
                    var responses: [String] = []
                    while index < messages.count, messages[index].role == .tool {
                        let raw = messages[index].content.isEmpty
                            ? messages[index].contentForModel
                            : messages[index].content
                        responses.append("<tool_response>\n\(raw)\n</tool_response>")
                        index += 1
                    }
                    parts.append("\(userPrefix)\(responses.joined(separator: "\n"))\(userSuffix)")
                    continue
                } else {
                    parts.append("\(userPrefix)Tool result:\n\(msg.contentForModel)\(userSuffix)")
                }
            }
            index += 1
        }

        // Build the generation prompt — the model responds after this.
        var prompt = generationPrompt
        if supportsThinking && enableThinking, let tag = thinkTag {
            prompt += tag
        } else if supportsThinking && !enableThinking, let tag = noThinkTag {
            prompt += tag
        }
        parts.append(prompt)

        return parts.joined()
    }

    // MARK: - Gemma 4 & Qwen3.5 Native Tool Formatting

    private func formatGemma4(messages: [ChatMessage], enableThinking: Bool) -> String {
        var parts: [String] = []

        // 1. System turn: Gemma 4 places `<|think|>` at the start of the system
        // turn when thinking is enabled, even if no explicit system text exists.
        let systemTexts = messages
            .filter { $0.role == .system }
            .map(\.contentForModel)
            .filter { !$0.isEmpty }
        let combinedSystem = systemTexts.joined(separator: "\n\n")
        if enableThinking {
            if combinedSystem.isEmpty {
                parts.append("\(systemPrefix)<|think|>\(systemSuffix)")
            } else {
                parts.append("\(systemPrefix)<|think|>\n\(combinedSystem)\(systemSuffix)")
            }
        } else if !combinedSystem.isEmpty {
            parts.append("\(systemPrefix)\(combinedSystem)\(systemSuffix)")
        }

        // Map tool_call_id -> function name across the conversation for Gemma 4
        // `<|tool_response>response:NAME{...}<tool_response|>` correlation.
        var callNamesByID: [String: String] = [:]
        for msg in messages {
            for call in msg.toolCalls ?? [] {
                callNamesByID[call.id] = call.name
            }
        }

        let nonSystem = messages.filter { $0.role != .system }
        var inOpenModelTurn = false

        for idx in nonSystem.indices {
            let msg = nonSystem[idx]
            let nextRole: ChatMessage.Role? = (idx + 1 < nonSystem.count) ? nonSystem[idx + 1].role : nil

            switch msg.role {
            case .system:
                break
            case .user:
                if inOpenModelTurn {
                    parts.append(assistantSuffix)
                    inOpenModelTurn = false
                }
                parts.append("\(userPrefix)\(msg.contentForModel)\(userSuffix)")
            case .assistant:
                if !inOpenModelTurn {
                    parts.append(assistantPrefix)
                    inOpenModelTurn = true
                }
                parts.append("<|channel>thought\n<channel|>")
                if let calls = msg.toolCalls, !calls.isEmpty {
                    let prose = AssistantOutputSanitizer.clean(msg.content)
                    if !prose.isEmpty {
                        parts.append(prose)
                    }
                    for call in calls {
                        parts.append(Self.formatGemma4ToolCall(name: call.name, argumentsJSON: call.argumentsJSON))
                    }
                    if nextRole != .tool {
                        parts.append(assistantSuffix)
                        inOpenModelTurn = false
                    }
                } else {
                    let content = AssistantOutputSanitizer.clean(msg.contentForModel)
                    parts.append(content)
                    parts.append(assistantSuffix)
                    inOpenModelTurn = false
                }
            case .tool:
                let toolName = msg.toolCallID.flatMap { callNamesByID[$0] } ?? "tool"
                let rawPayload = msg.content.isEmpty ? msg.contentForModel : msg.content
                if !inOpenModelTurn {
                    parts.append(assistantPrefix)
                    parts.append("<|channel>thought\n<channel|>")
                    inOpenModelTurn = true
                }
                parts.append(Self.formatGemma4ToolResponse(name: toolName, content: rawPayload))
                if nextRole == .user {
                    parts.append(assistantSuffix)
                    inOpenModelTurn = false
                }
            }
        }

        let thoughtOpening = enableThinking
            ? (thinkTag ?? "<|channel>thought\n")
            : (noThinkTag ?? "<|channel>thought\n<channel|>")
        if inOpenModelTurn {
            parts.append(thoughtOpening)
        } else {
            parts.append("\(generationPrompt)\(thoughtOpening)")
        }
        return parts.joined()
    }

    static func formatQwen35ToolCall(name: String, argumentsJSON: String) -> String {
        var lines: [String] = ["<tool_call>", "<function=\(name)>"]
        if let data = argumentsJSON.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            for key in obj.keys.sorted() {
                let val = obj[key]
                let rendered: String
                if let str = val as? String {
                    rendered = str
                } else if let val, JSONSerialization.isValidJSONObject(val),
                          let nested = try? JSONSerialization.data(withJSONObject: val, options: [.sortedKeys]),
                          let nestedStr = String(data: nested, encoding: .utf8) {
                    rendered = nestedStr
                } else if let num = val as? NSNumber {
                    if CFGetTypeID(num) == CFBooleanGetTypeID() {
                        rendered = num.boolValue ? "true" : "false"
                    } else {
                        rendered = num.stringValue
                    }
                } else {
                    rendered = "null"
                }
                lines.append("<parameter=\(key)>")
                lines.append(rendered)
                lines.append("</parameter>")
            }
        }
        lines.append("</function>")
        lines.append("</tool_call>")
        return lines.joined(separator: "\n")
    }

    static func formatHermesToolCall(name: String, argumentsJSON: String) -> String {
        "<tool_call>\n{\"name\": \"\(name)\", \"arguments\": \(argumentsJSON)}\n</tool_call>"
    }

    static func formatGemma4ToolDeclaration(
        name: String,
        description: String?,
        parametersJSON: String
    ) -> String {
        var fields: [String] = []
        if let description, !description.isEmpty {
            fields.append("description:\(formatGemma4Value(description))")
        }
        if let data = parametersJSON.data(using: .utf8),
           let schemaObj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            fields.append("parameters:\(formatGemma4Object(schemaObj, uppercaseSchemaTypes: true))")
        } else {
            fields.append("parameters:{type:<|\"|>OBJECT<|\"|>,properties:{}}")
        }
        return "<|tool>declaration:\(name){\(fields.joined(separator: ","))}<tool|>"
    }

    static func formatGemma4ToolCall(name: String, argumentsJSON: String) -> String {
        if let data = argumentsJSON.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return "<|tool_call>call:\(name)\(formatGemma4Object(obj))<tool_call|>"
        }
        return "<|tool_call>call:\(name){}<tool_call|>"
    }

    static func formatGemma4ToolResponse(name: String, content: String) -> String {
        if let data = content.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return "<|tool_response>response:\(name)\(formatGemma4Object(obj))<tool_response|>"
        }
        return "<|tool_response>response:\(name){value:\(formatGemma4Value(content))}<tool_response|>"
    }

    static func formatGemma4Object(
        _ dict: [String: Any],
        uppercaseSchemaTypes: Bool = false
    ) -> String {
        let entries = dict.keys.sorted().map { key -> String in
            let rawValue = dict[key]
            if uppercaseSchemaTypes, key == "type", let typeStr = rawValue as? String {
                return "\(key):\(formatGemma4Value(typeStr.uppercased()))"
            }
            if uppercaseSchemaTypes, key == "properties", let props = rawValue as? [String: Any] {
                let propEntries = props.keys.sorted().map { propKey -> String in
                    if let sub = props[propKey] as? [String: Any] {
                        return "\(propKey):\(formatGemma4Object(sub, uppercaseSchemaTypes: true))"
                    }
                    return "\(propKey):\(formatGemma4Value(props[propKey], uppercaseSchemaTypes: true))"
                }
                return "properties:{\(propEntries.joined(separator: ","))}"
            }
            return "\(key):\(formatGemma4Value(rawValue, uppercaseSchemaTypes: uppercaseSchemaTypes))"
        }
        return "{\(entries.joined(separator: ","))}"
    }

    static func formatGemma4Value(_ value: Any?, uppercaseSchemaTypes: Bool = false) -> String {
        guard let value, !(value is NSNull) else { return "null" }
        if let str = value as? String {
            return "<|\"|>\(str)<|\"|>"
        }
        if let num = value as? NSNumber {
            if CFGetTypeID(num) == CFBooleanGetTypeID() {
                return num.boolValue ? "true" : "false"
            }
            return num.stringValue
        }
        if let dict = value as? [String: Any] {
            return formatGemma4Object(dict, uppercaseSchemaTypes: uppercaseSchemaTypes)
        }
        if let arr = value as? [Any] {
            let items = arr.map { formatGemma4Value($0, uppercaseSchemaTypes: uppercaseSchemaTypes) }
            return "[\(items.joined(separator: ","))]"
        }
        return "<|\"|>\(String(describing: value))<|\"|>"
    }
}

// MARK: - ChatMessage template extension

extension [ChatMessage] {
    /// Format messages using a specific chat template.
    func formattedWithTemplate(_ template: ChatTemplate, enableThinking: Bool = false) -> String {
        template.format(messages: self, enableThinking: enableThinking)
    }
}
