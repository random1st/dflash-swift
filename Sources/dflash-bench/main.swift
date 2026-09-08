// Первый настоящий прогон порта: загрузить таргет и драфт, сгенерировать,
// померить принятые токены на раунд и скорость.
//
//   dflash-bench <каталог-модели> <каталог-драфта> [токенов]
import Foundation
import MLX
import MLXLLM
import MLXHuggingFace
import MLXLMCommon
import HuggingFace
import Tokenizers
import DFlashKit

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    print("usage: dflash-bench <model-dir> <drafter-dir> [max-tokens]")
    exit(2)
}
let modelDirectory = URL(fileURLWithPath: arguments[1])
let drafterDirectory = URL(fileURLWithPath: arguments[2])
let maximumTokens = arguments.count > 3 ? Int(arguments[3]) ?? 200 : 200

print("загружаю таргет \(modelDirectory.lastPathComponent)…")
var started = Date()
let tokenizerLoader = #huggingFaceTokenizerLoader()
let context = try await loadModel(from: modelDirectory, using: tokenizerLoader)
print("  за \(Int(Date().timeIntervalSince(started)))s")

guard let target = context.model as? Qwen35TextModel ?? (context.model as? Qwen35Model)?.languageModel
else {
    print("модель не Qwen3.5-совместимая: \(type(of: context.model))")
    exit(1)
}

print("загружаю драфт \(drafterDirectory.lastPathComponent)…")
started = Date()
let drafter = try DFlashDraftModel.load(directory: drafterDirectory)
print("  за \(Int(Date().timeIntervalSince(started)))s, блок \(drafter.configuration.blockSize), слои \(drafter.configuration.targetLayerIds)")

if CommandLine.arguments.contains("--rows") {
    measureRows(target: target, vocabulary: drafter.configuration.vocabularySize)
    exit(0)
}

let prompt = "Write a Python function that parses a semver string into a tuple."
let messages = [Chat.Message.user(prompt)]
let userInput = UserInput(chat: messages, additionalContext: ["enable_thinking": false])
let input = try await context.processor.prepare(input: userInput)
let tokens = input.text.tokens.asArray(Int.self)
print("промпт: \(tokens.count) токенов")

let capArgument = arguments.firstIndex(of: "--cap").flatMap { index -> Int? in
    arguments.count > index + 1 ? Int(arguments[index + 1]) : nil
}
let generator = DFlashSpeculativeGenerator(
    target: target, drafter: drafter, maximumDraftTokens: capArgument)
print("cap \(generator.cap) черновых токенов на раунд, генерирую \(maximumTokens)…")

var produced: [Int] = []
var eos = Set<Int>()
if let id = context.tokenizer.eosTokenId { eos.insert(id) }
let statistics = try generator.generate(
    prompt: tokens, maximumTokens: maximumTokens, stopTokens: eos
) { token in produced.append(token) }

print("")
print("=== результат")
print("  токенов:            \(statistics.tokens)")
print("  раундов:            \(statistics.rounds.count)")
print(String(format: "  принято на раунд:   %.2f", statistics.meanAcceptedPerRound))
print(String(format: "  секунд:             %.2f", statistics.seconds))
print(String(format: "  токенов в секунду:  %.2f", Double(statistics.tokens) / statistics.seconds))
print(String(format: "  префилл:            %.2fs", statistics.prefillSeconds))
print(String(format: "  черновление:        %.2fs", statistics.draftSeconds))
print(String(format: "  проверка:           %.2fs", statistics.verifySeconds))
print(String(format: "  откат:              %.2fs", statistics.rollbackSeconds))
print("")
print(context.tokenizer.decode(tokenIds: produced))
