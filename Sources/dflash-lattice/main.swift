// Снимает решётку драфтера по каждой позиции ответа, чтобы посчитать офлайн, что
// дало бы дерево вместо одной жадной цепочки.
//
// Почему это можно мерить офлайн. Все политики принимают только те токены, с
// которыми таргет и так согласен, поэтому зафиксированная последовательность
// одна и та же при любой ширине дерева: жадное продолжение таргета. Значит
// достаточно один раз пройти это продолжение с teacher forcing, на каждой
// позиции снять решётку драфтера - кандидатов по слотам и оценки переходов - и
// дальше любую политику отыграть по записи, включая границы раундов.
//
//   dflash-lattice <каталог-модели> <каталог-драфта> [--tokens N] [--out файл]
//                  [--think] [--prompts файл]
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
    print("usage: dflash-lattice <model-dir> <drafter-dir> [--tokens N] [--out file] [--think] [--prompts file]")
    exit(2)
}

func option(_ name: String) -> String? {
    guard let index = arguments.firstIndex(of: name), arguments.count > index + 1 else { return nil }
    return arguments[index + 1]
}

let modelDirectory = URL(fileURLWithPath: arguments[1])
let drafterDirectory = URL(fileURLWithPath: arguments[2])
let maximumTokens = option("--tokens").flatMap(Int.init) ?? 200
let outputPath = option("--out") ?? "/tmp/dflash-lattice.jsonl"
// Роман работает с включённым рассуждением, и драфтится текст рассуждения иначе,
// чем проза ответа: мерить надо тот режим, в котором модель реально живёт.
let thinking = !arguments.contains("--no-think")

// Восемь промптов, а не один: на одном промпте цифра описывает конкретный ответ,
// а не модель. Код, математика, проза и диалог драфтятся по-разному, и решение
// про дерево должно выдержать все четыре.
let defaultPrompts = [
    "Write a Python function that parses a semver string into a tuple.",
    "Refactor this into idiomatic Swift: for i in 0..<a.count { if a[i] > 0 { b.append(a[i] * 2) } }",
    "A train leaves at 14:20 going 80 km/h, another at 15:05 going 110 km/h on the same track 240 km apart. When do they meet?",
    "Explain why speculative decoding does not change the output of greedy decoding.",
    "Write a SQL query returning the top 3 customers by revenue per region for the last quarter.",
    "Опиши, чем отличается блочная диффузия от авторегрессионного драфтера, простыми словами.",
    "I keep getting 'Address already in use' when restarting my server. Walk me through the causes.",
    "Write a short, warm note to a colleague who just shipped a hard project.",
]
let prompts: [String] = {
    guard let path = option("--prompts"), let text = try? String(contentsOfFile: path, encoding: .utf8)
    else { return defaultPrompts }
    return text.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
}()

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

let drafter = try DFlashDraftModel.load(directory: drafterDirectory, quantizeBits: nil)
let blockSize = drafter.configuration.blockSize
let slots = blockSize - 1
let topK = drafter.configuration.selectorTopK
print("драфтер: блок \(blockSize) (\(slots) слотов), top-K \(topK), слои \(drafter.configuration.targetLayerIds)")
guard drafter.configuration.hasSelector else {
    print("в чекпоинте нет селектора: решётки не существует, мерить нечего")
    exit(1)
}

// Ядро small-M не бит-в-бит совпадает с обычным quantizedMatmul, поэтому жадный
// ответ таргета с ним и без него -- это два разных текста, и принятые токены на них
// разные. Генератор в проде идёт с ядром, значит и решётки снимаем с ядром, иначе
// реплей предсказывает не ту конфигурацию, которая работает у Романа.
if !arguments.contains("--no-small-m") {
    print("ядро small-M: заменено слоёв \(enableSmallMQuantizedMatmul(in: target))")
}

let bridge = Qwen35Bridge(target: target, tapIndices: drafter.configuration.targetLayerIds)
drafter.bind(bridge)

var eos = Set<Int>()
if let id = context.tokenizer.eosTokenId { eos.insert(id) }

FileManager.default.createFile(atPath: outputPath, contents: nil)
guard let output = FileHandle(forWritingAtPath: outputPath) else {
    print("не могу писать в \(outputPath)")
    exit(1)
}
func emit(_ object: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [])
    output.write(data)
    output.write(Data("\n".utf8))
}

let maskToken = Int32(drafter.configuration.maskTokenId)
started = Date()

for (promptIndex, prompt) in prompts.enumerated() {
    let userInput = UserInput(
        chat: [Chat.Message.user(prompt)], additionalContext: ["enable_thinking": thinking])
    let input = try await context.processor.prepare(input: userInput)
    let promptTokens = input.text.tokens.asArray(Int.self)

    let targetCache = try target.newCache(parameters: nil)
    let draftCache = drafter.makeCache()

    // Префилл. Строка позиции i - это контекст драфтера для токена i+1, поэтому
    // в кэш уходят все строки кроме последней, а она переносится дальше.
    let ids = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, promptTokens.count)
    let prefill = target(LMInput.Text(tokens: ids), cache: targetCache, state: bridge.requestState())
    guard let prefillState = prefill.state, let prefillFused = bridge.fuse(prefillState) else {
        print("таргет не отдал tapped-слои - собран ли патч multilayer-tap?")
        exit(1)
    }
    let head = prefillFused[0..., ..<(promptTokens.count - 1), 0...]
    if head.dim(1) > 0 {
        drafter.appendContext(drafter.projectContext(head), cache: draftCache)
    }
    var carried = prefillFused[0..., (promptTokens.count - 1)..., 0...]
    var pending = argMax(prefill.logits[0, -1], axis: -1).item(Int.self)

    var truth = [pending]
    var position = 0
    var stopped = eos.contains(pending)

    while truth.count < maximumTokens && !stopped {
        let anchor = Int32(pending)
        let blockIds = [anchor] + Array(repeating: maskToken, count: blockSize - 1)
        let block = MLXArray(blockIds).reshaped(1, blockSize)
        guard
            let lattice = drafter.draftLattice(
                block, fusedTargetHidden: carried, cache: draftCache, cap: slots, anchorId: anchor)
        else {
            print("решётка не построена")
            exit(1)
        }
        eval(lattice.candidateIds, lattice.scores, lattice.unaryLogits)
        let candidates = lattice.candidateIds.asArray(Int32.self).map(Int.init)
        let scores = lattice.scores.asArray(Float.self)
        let unary = lattice.unaryLogits.asArray(Float.self)

        // Один токен вперёд по истинному продолжению: именно его драфтер и должен
        // был угадать, и именно он определяет, где раунд оборвался бы.
        let step = target(
            LMInput.Text(tokens: MLXArray([anchor]).reshaped(1, 1)), cache: targetCache,
            state: bridge.requestState())
        guard let stepState = step.state, let stepFused = bridge.fuse(stepState) else {
            print("таргет не отдал tapped-слои на шаге")
            exit(1)
        }
        carried = stepFused
        let next = argMax(step.logits[0, -1], axis: -1).item(Int.self)

        try emit([
            "type": "lattice", "prompt": promptIndex, "pos": position, "anchor": Int(anchor),
            "cands": candidates, "scores": scores.map { Double(($0 * 1000).rounded() / 1000) },
            "unary": unary.map { Double(($0 * 1000).rounded() / 1000) },
        ])

        position += 1
        pending = next
        truth.append(next)
        stopped = eos.contains(next)
    }

    try emit([
        "type": "truth", "prompt": promptIndex, "tokens": truth, "promptTokens": promptTokens.count,
        "thinking": thinking,
    ])
    print(
        "промпт \(promptIndex + 1)/\(prompts.count): \(truth.count) токенов, \(position) решёток, "
            + "\(Int(Date().timeIntervalSince(started)))s всего")
}

try output.close()
print("записано в \(outputPath)")
