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

// Ядро small-M читает веса один раз на 6–8 строк; в генераторе оно включено по
// умолчанию, --no-small-m его выключает. Кривую стоимости по ширине меряем на
// голой модели, поэтому там подмена делается руками: --rows --small-m.
let smallM = !arguments.contains("--no-small-m")

if CommandLine.arguments.contains("--rows") {
    if arguments.contains("--small-m") {
        print("ядро small-M: заменено слоёв \(enableSmallMQuantizedMatmul(in: target))")
    }
    measureRows(target: target, vocabulary: drafter.configuration.vocabularySize)
    exit(0)
}

let prompt = "Write a Python function that parses a semver string into a tuple."
// --long повторяет вопрос, пока промпт не станет длинным: на коротком префилл и так
// доли секунды, и prefix-кэшу нечего экономить.
let repeats = arguments.contains("--long") ? 40 : 1
let messages = (0 ..< repeats).map { _ in Chat.Message.user(prompt) }
let userInput = UserInput(chat: messages, additionalContext: ["enable_thinking": false])
let input = try await context.processor.prepare(input: userInput)
let tokens = input.text.tokens.asArray(Int.self)
print("промпт: \(tokens.count) токенов")

let capArgument = arguments.firstIndex(of: "--cap").flatMap { index -> Int? in
    arguments.count > index + 1 ? Int(arguments[index + 1]) : nil
}
let generator = DFlashSpeculativeGenerator(
    target: target, drafter: drafter, maximumDraftTokens: capArgument, useSmallMKernel: smallM,
    prefixCache: arguments.contains("--prefix-cache") ? PrefixCache() : nil)
print("ядро small-M: заменено слоёв \(generator.acceleratedLayers)")
print("cap \(generator.cap) черновых токенов на раунд, генерирую \(maximumTokens)…")

var eos = Set<Int>()
if let id = context.tokenizer.eosTokenId { eos.insert(id) }

func run() throws -> (DFlashGenerationStatistics, [Int]) {
    var produced: [Int] = []
    let statistics = try generator.generate(
        prompt: tokens, maximumTokens: maximumTokens, stopTokens: eos
    ) { token in produced.append(token) }
    return (statistics, produced)
}

func report(_ statistics: DFlashGenerationStatistics, _ label: String) {
    print("")
    print("=== \(label)")
    print("  токенов:            \(statistics.tokens)")
    print("  раундов:            \(statistics.rounds.count)")
    print(String(format: "  принято на раунд:   %.2f", statistics.meanAcceptedPerRound))
    let widths = statistics.rounds.map(\.proposed)
    if let lo = widths.min(), let hi = widths.max() {
        let mean = Double(widths.reduce(0, +)) / Double(max(widths.count, 1))
        print(String(format: "  ширина черновика:   %.1f (от %d до %d)", mean, lo, hi))
    }
    print(String(format: "  секунд:             %.2f", statistics.seconds))
    print(String(format: "  токенов в секунду:  %.2f", Double(statistics.tokens) / statistics.seconds))
    print(String(format: "  префилл:            %.2fs", statistics.prefillSeconds))
    print("  из промпта переиспользовано: \(statistics.reusedPromptTokens) токенов")
    print(String(format: "  черновление:        %.2fs", statistics.draftSeconds))
    print(String(format: "  проверка:           %.2fs", statistics.verifySeconds))
    print(String(format: "  откат:              %.2fs", statistics.rollbackSeconds))
}

let (first, produced) = try run()
report(first, "результат")

// Второй прогон того же промпта: с prefix-кэшем он должен попасть в снапшот и дать
// тот же текст. Отличие в тексте значит, что восстановленное состояние не равно
// холодному префиллу, и никакая экономия этого не оправдывает.
if arguments.contains("--prefix-cache") {
    let (second, again) = try run()
    report(second, "повтор того же промпта")
    print("")
    print("  текст совпал:       \(again == produced ? "да" : "НЕТ")")
    if first.prefillSeconds > 0 {
        print(String(format: "  префилл быстрее в:  %.1fx",
            first.prefillSeconds / max(second.prefillSeconds, 1e-9)))
    }
}

// Второй ход диалога — тот случай, ради которого снапшот снимается ещё и в конце
// генерации. Промпт следующего хода начинается с промпта предыдущего плюс ответ, так
// что он обязан попасть в кэш; и он обязан дать ровно то же, что холодный прогон, иначе
// восстановленное состояние не соответствует токенам, под которыми записано.
if arguments.contains("--multiturn") {
    let reply = context.tokenizer.decode(tokenIds: produced)
    let followUp = messages + [Chat.Message.assistant(reply), Chat.Message.user("Now add tests.")]
    let nextInput = try await context.processor.prepare(
        input: UserInput(chat: followUp, additionalContext: ["enable_thinking": false]))
    let nextTokens = nextInput.text.tokens.asArray(Int.self)
    print("")
    print("=== второй ход: \(nextTokens.count) токенов промпта")

    var warm: [Int] = []
    let warmStatistics = try generator.generate(
        prompt: nextTokens, maximumTokens: maximumTokens, stopTokens: eos
    ) { token in warm.append(token) }

    // Холодный контроль на том же таргете. Ядро уже подменено, поэтому второй генератор
    // насчитает 0 заменённых слоёв — cap задаём явно, иначе он молча возьмёт узкий.
    let cold = DFlashSpeculativeGenerator(
        target: target, drafter: drafter, maximumDraftTokens: generator.cap,
        useSmallMKernel: false, prefixCache: nil)
    var coldTokens: [Int] = []
    let coldStatistics = try cold.generate(
        prompt: nextTokens, maximumTokens: maximumTokens, stopTokens: eos
    ) { token in coldTokens.append(token) }

    // Сколько вообще можно было переиспользовать: где расходятся промпт первого хода
    // вместе с ответом и промпт второго хода.
    let firstTurn = tokens + produced
    let overlap = zip(firstTurn, nextTokens).prefix { $0 == $1 }.count
    print("  общий префикс ходов: \(overlap) при ответе в \(produced.count) токенов")
    print("  переиспользовано:   \(warmStatistics.reusedPromptTokens) из \(nextTokens.count)")
    print(String(format: "  префилл с кэшем:    %.2fs", warmStatistics.prefillSeconds))
    print(String(format: "  префилл холодный:   %.2fs", coldStatistics.prefillSeconds))
    print(String(format: "  быстрее в:          %.1fx",
        coldStatistics.prefillSeconds / max(warmStatistics.prefillSeconds, 1e-9)))
    print("  текст совпал:       \(warm == coldTokens ? "да" : "НЕТ")")
}

print("")
print(context.tokenizer.decode(tokenIds: produced))
