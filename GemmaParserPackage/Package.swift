// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GemmaParserPackage",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "ToolCallParser", targets: ["ToolCallParser"])
    ],
    targets: [
        .target(
            name: "ToolCallParser",
            path: "Sources/ToolCallParser"
        ),
        .testTarget(
            name: "ToolCallParserTests",
            dependencies: ["ToolCallParser"],
            path: "Tests/ToolCallParserTests"
        )
    ]
)
