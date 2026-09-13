// Подгонка драфтера DFlash под конкретные веса таргета.
//
// Готовый чекпоинт `z-lab/Qwen3.6-35B-A3B-DFlash` обучен на оригинальных весах. На
// abliterated-версии тех же весов приёмка падает с 6.96 до 3.07 токенов за раунд — ровно
// на «пол» из статьи, строку «без признаков таргета». Abliteration поворачивает
// направления в весах, снятые скрытые состояния уезжают из пространства, под которое
// обучена проекция `fc`, и блочная диффузия вырождается в почти безусловную.
//
// Три режима:
//
//   dflash-train <таргет> <драфтер>                       — тест на градиенты, минуты
//   dflash-train <таргет> <драфтер> --collect <каталог>    — сбор данных таргетом
//   dflash-train <таргет> <драфтер> --train <каталог>      — обучение по собранному
//
// Сбор и обучение разделены намеренно: таргет нужен живым только на сборе, а перебирать
// гиперпараметры потом хочется по одному драфтеру в 0.4B, без 35B в цикле.
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers
import DFlashKit

let args = CommandLine.arguments
guard args.count >= 3 else {
    print("""
        usage: dflash-train <target-dir> <drafter-dir> [режим]
          --collect <dir>   собрать данные: ответы таргета и снятые под ними состояния
            --samples N     сколько примеров собрать (по умолчанию 2000)
            --prompts FILE  jsonl, по строке-строке на промпт
            --offset N      с какого промпта начать — чтобы дособрать, не повторяясь
          --train <dir>     обучить по собранному
            --steps N       шагов (по умолчанию 2000)
            --batch N       примеров в шаге (по умолчанию 32)
            --lr X          скорость обучения (по умолчанию 1e-4)
            --scope fc|all  что размораживать (по умолчанию fc)
            --out <dir>     куда положить обученный драфтер
        """)
    exit(2)
}

func flag(_ name: String) -> String? {
    args.firstIndex(of: name).flatMap { args.count > $0 + 1 ? args[$0 + 1] : nil }
}
func intFlag(_ name: String, _ fallback: Int) -> Int { flag(name).flatMap { Int($0) } ?? fallback }

let collectPath = flag("--collect")
let trainPath = flag("--train")
let sampleBudget = intFlag("--samples", 2000)
let promptOffset = intFlag("--offset", 0)
let trainSteps = intFlag("--steps", 2000)
let batchSize = intFlag("--batch", 32)
let learningRate = flag("--lr").flatMap { Float($0) } ?? 1e-4
let scope = flag("--scope") ?? "fc"
let outPath = flag("--out")

print("загружаю таргет…")
let context = try await loadModel(from: URL(fileURLWithPath: args[1]),
                                  using: #huggingFaceTokenizerLoader())
guard let target = context.model as? Qwen35TextModel
    ?? (context.model as? Qwen35Model)?.languageModel
else { print("не Qwen3.5-совместимая модель"); exit(1) }

let drafter = try DFlashDraftModel.load(directory: URL(fileURLWithPath: args[2]))
let bridge = Qwen35Bridge(target: target, tapIndices: drafter.configuration.targetLayerIds)
drafter.bind(bridge)
let blockSize = drafter.configuration.blockSize
let labelWidth = blockSize - 1
let maskToken = Int32(drafter.configuration.maskTokenId)
print("блок \(blockSize), слои \(drafter.configuration.targetLayerIds)")

var stopTokens = Set<Int>()
if let id = context.tokenizer.eosTokenId { stopTokens.insert(id) }

/// Промпты по умолчанию — для теста на градиентах. Настоящий сбор идёт по `--prompts`:
/// жадная генерация на один и тот же промпт даёт один и тот же ответ, так что без корпуса
/// данные вырождаются в десяток повторяющихся последовательностей.
let builtinPrompts = [
    "Write a Python function that parses a semver string into a tuple.",
    "Explain what a hash map is in two sentences.",
    "List three differences between TCP and UDP.",
]

func loadPrompts() -> [String] {
    guard let path = flag("--prompts"),
          let text = try? String(contentsOfFile: path, encoding: .utf8)
    else { return builtinPrompts }
    let decoder = JSONDecoder()
    return text.split(separator: "\n").compactMap {
        try? decoder.decode(String.self, from: Data($0.utf8))
    }
}

struct Sample {
    let anchor: Int32
    let targets: [Int32]        // что таргет выдал на позициях блока
    let fused: MLXArray         // строка скрытых состояний под якорь
}

/// Разворачивает ответ таргета и режет его на обучающие примеры.
///
/// Пример — это якорь, следующие за ним токены блока и строка скрытых состояний, снятая
/// под якорем. Ровно то, что драфтер видит в бою: состояния одной позиции на входе, блок
/// продолжения на выходе.
func collect(_ prompt: String, maxTokens: Int) async throws -> [Sample] {
    let input = try await context.processor.prepare(
        input: UserInput(chat: [Chat.Message.user(prompt)],
                         additionalContext: ["enable_thinking": false]))
    let ids = input.text.tokens.asArray(Int32.self)
    let cache = try target.newCache(parameters: nil)

    let prompted = MLXArray(ids).reshaped(1, ids.count)
    let out = target(LMInput.Text(tokens: prompted), cache: cache, state: bridge.requestState())
    guard let state = out.state, let fused = bridge.fuse(state) else {
        throw NSError(domain: "dflash-train", code: 1)
    }
    var next = argMax(out.logits[0, -1], axis: -1).item(Int32.self)
    var carried = fused[0..., (ids.count - 1)..., 0...]

    var samples: [Sample] = []
    var stop = false
    while samples.count * blockSize < maxTokens && !stop {
        let anchor = next
        let anchorRow = carried
        var produced: [Int32] = []
        for _ in 0 ..< labelWidth {
            let step = target(LMInput.Text(tokens: MLXArray([next]).reshaped(1, 1)),
                              cache: cache, state: bridge.requestState())
            guard let st = step.state, let f = bridge.fuse(st) else { stop = true; break }
            next = argMax(step.logits[0, -1], axis: -1).item(Int32.self)
            produced.append(next)
            carried = f[0..., 0..., 0...]
            if stopTokens.contains(Int(next)) { stop = true; break }
        }
        guard produced.count == labelWidth else { break }
        samples.append(Sample(anchor: anchor, targets: produced, fused: anchorRow))
    }
    return samples
}

// --- снятые состояния на диск ---------------------------------------------------------
// Проверка самой гипотезы «abliteration уводит скрытые состояния из пространства fc»:
// один и тот же текст через стоковую и abliterated, потом косинус по слоям и позициям.
if let dumpPath = flag("--dump-hidden") {
    var arrays: [String: MLXArray] = [:]
    for (i, prompt) in loadPrompts().prefix(8).enumerated() {
        let input = try await context.processor.prepare(
            input: UserInput(chat: [Chat.Message.user(prompt)],
                             additionalContext: ["enable_thinking": false]))
        let ids = input.text.tokens.asArray(Int32.self)
        let cache = try target.newCache(parameters: nil)
        let out = target(LMInput.Text(tokens: MLXArray(ids).reshaped(1, ids.count)),
                         cache: cache, state: bridge.requestState())
        guard let state = out.state, let fused = bridge.fuse(state) else { continue }
        arrays["tokens_\(i)"] = MLXArray(ids)
        arrays["fused_\(i)"] = fused[0].asType(.float32)
        arrays["logits_\(i)"] = out.logits[0].asType(.float32)
    }
    try MLX.save(arrays: arrays, url: URL(fileURLWithPath: dumpPath))
    print("записано: \(dumpPath)")
    exit(0)
}

// --- сбор ---------------------------------------------------------------------------
if let collectPath {
    let prompts = loadPrompts()
    let directory = URL(fileURLWithPath: collectPath)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    print("промптов в корпусе: \(prompts.count), цель \(sampleBudget) примеров")

    // Шардами, а не одним файлом: сбор идёт часами, и обрыв на середине не должен
    // стоить всего, что уже сгенерировано.
    let shardSize = 512
    let stamp = Int(Date().timeIntervalSince1970)
    var anchors: [Int32] = []
    var labels: [Int32] = []
    var rows: [MLXArray] = []
    var written = 0
    var shard = 0
    let started = Date()

    func flush() throws {
        guard !anchors.isEmpty else { return }
        let url = directory.appending(
            component: String(format: "shard-%d-%04d.safetensors", stamp, shard))
        try MLX.save(arrays: [
            "anchors": MLXArray(anchors),
            "labels": MLXArray(labels).reshaped(anchors.count, labelWidth),
            "fused": concatenated(rows, axis: 0),
        ], url: url)
        written += anchors.count
        shard += 1
        anchors.removeAll(); labels.removeAll(); rows.removeAll()
        print("  шард \(url.lastPathComponent), всего \(written)")
    }

    var index = promptOffset
    while written + anchors.count < sampleBudget {
        let prompt = prompts[index % prompts.count]
        index += 1
        let batch = try await collect(prompt, maxTokens: 384)
        for sample in batch where written + anchors.count < sampleBudget {
            anchors.append(sample.anchor)
            labels += sample.targets
            rows.append(sample.fused.reshaped(1, -1).asType(.float16))
        }
        if anchors.count >= shardSize { try flush() }
        let done = written + anchors.count
        let rate = Double(done) / max(Date().timeIntervalSince(started), 1)
        print(String(format: "  %d/%d  %.1f примеров/с  осталось ~%.0f мин",
                     done, sampleBudget, rate, Double(sampleBudget - done) / rate / 60))
    }
    try flush()
    print("собрано \(written), промптов израсходовано \(index - promptOffset)")
    exit(0)
}

// --- потери -------------------------------------------------------------------------
// Кросс-энтропия на позициях маски против того, что выдал таргет. Вход батчем: драфтер
// 0.4B, и гонять его по одному примеру — значит платить за запуск ядер больше, чем за
// саму арифметику.
func batchLogits(_ model: DFlashDraftModel, anchors: MLXArray, fused: MLXArray) -> MLXArray {
    let batch = anchors.dim(0)
    let masks = MLXArray.full([batch, labelWidth], values: MLXArray(maskToken))
    let block = concatenated([anchors.reshaped(batch, 1), masks], axis: 1)
    let hidden = model.forwardHidden(block, fusedTargetHidden: fused.reshaped(batch, 1, -1),
                                     cache: model.makeCache(), logitsStart: 1)
    let flat = hidden.reshaped(batch * labelWidth, -1)
    return bridge.logits(flat)[0..., ..<model.configuration.vocabularySize]
}

func batchLoss(_ model: DFlashDraftModel, anchors: MLXArray, fused: MLXArray,
               labels: MLXArray) -> MLXArray {
    crossEntropy(logits: batchLogits(model, anchors: anchors, fused: fused),
                 targets: labels.reshaped(anchors.dim(0) * labelWidth), reduction: .mean)
}

/// Доля позиций, где драфтер угадал токен таргета. Прокси приёмки: настоящая метрика —
/// принято за раунд, она меряется бенчем, но между шагами обучения нужна дешёвая.
func batchAccuracy(_ model: DFlashDraftModel, anchors: MLXArray, fused: MLXArray,
                   labels: MLXArray) -> Float {
    let logits = batchLogits(model, anchors: anchors, fused: fused)
    let hit = argMax(logits, axis: -1).asType(.int32)
        .== labels.reshaped(anchors.dim(0) * labelWidth)
    return hit.asType(.float32).mean().item(Float.self)
}

// --- обучение по собранному ----------------------------------------------------------
if let trainPath {
    let directory = URL(fileURLWithPath: trainPath)
    let shards = try FileManager.default
        .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !shards.isEmpty else { print("в \(trainPath) нет шардов"); exit(1) }

    var anchorParts: [MLXArray] = []
    var labelParts: [MLXArray] = []
    var fusedParts: [MLXArray] = []
    for shard in shards {
        let arrays = try loadArrays(url: shard)
        guard let a = arrays["anchors"], let l = arrays["labels"], let f = arrays["fused"]
        else { continue }
        anchorParts.append(a); labelParts.append(l); fusedParts.append(f)
    }
    let allAnchors = concatenated(anchorParts, axis: 0)
    let allLabels = concatenated(labelParts, axis: 0)
    let allFused = concatenated(fusedParts, axis: 0)
    eval(allAnchors, allLabels, allFused)
    let total = allAnchors.dim(0)

    // Держим отложенный кусок: без него «потери падают» ничего не говорит о том, выучил
    // драфтер отображение или запомнил корпус.
    let holdout = min(512, total / 10)
    let trainCount = total - holdout
    print("примеров \(total): обучение \(trainCount), отложено \(holdout)")

    // Обучение в fp32: AdamW с lr 1e-4 в bf16 теряет обновление в округлении.
    drafter.update(parameters: drafter.parameters().mapValues { $0.asType(.float32) })
    drafter.freeze()
    switch scope {
    case "all":
        drafter.unfreeze()
    default:
        drafter.visit(modules: { key, module in
            if key == "fc" || key == "hidden_norm" { module.unfreeze() }
        })
    }
    eval(drafter)
    let trainable = drafter.trainableParameters().flattened()
    print("размораживаю \(scope): \(trainable.count) тензоров, "
          + "\(trainable.reduce(0) { $0 + $1.1.size } / 1_000_000)M параметров")

    func slice(_ indices: [Int32]) -> (MLXArray, MLXArray, MLXArray) {
        let idx = MLXArray(indices)
        return (allAnchors[idx], allFused[idx].asType(.float32), allLabels[idx].asType(.int32))
    }
    let holdoutIndices = Array(Int32(trainCount) ..< Int32(total))

    let optimizer = AdamW(learningRate: learningRate)
    let lossAndGrad = valueAndGrad(model: drafter) {
        (model: DFlashDraftModel, arrays: [MLXArray]) in
        [batchLoss(model, anchors: arrays[0], fused: arrays[1], labels: arrays[2])]
    }

    func holdoutAccuracy() -> Float {
        guard holdout > 0 else { return .nan }
        var hits: Float = 0
        var seen = 0
        for start in stride(from: 0, to: holdout, by: batchSize) {
            let chunk = Array(holdoutIndices[start ..< min(start + batchSize, holdout)])
            let (ha, hf, hl) = slice(chunk)
            hits += batchAccuracy(drafter, anchors: ha, fused: hf, labels: hl) * Float(chunk.count)
            seen += chunk.count
        }
        return hits / Float(seen)
    }

    var generator = SystemRandomNumberGenerator()
    print("шаг    потери   попадание(отложенное)")
    print(String(format: "%5d  %7s  %6.1f%%", 0, "—", holdoutAccuracy() * 100))
    for step in stride(from: 1, through: trainSteps, by: 1) {
        let picks = (0 ..< batchSize).map { _ in
            Int32.random(in: 0 ..< Int32(trainCount), using: &generator)
        }
        let (a, f, l) = slice(picks)
        let (value, grads) = lossAndGrad(drafter, [a, f, l])
        optimizer.update(model: drafter, gradients: grads)
        eval(drafter, optimizer)

        if step == 1 || step % 50 == 0 {
            print(String(format: "%5d  %7.4f  %6.1f%%", step, value[0].item(Float.self),
                         holdoutAccuracy() * 100))
        }
    }

    if let outPath {
        let out = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        // Рядом с весами нужен config.json — загрузчик читает его первым.
        let source = URL(fileURLWithPath: args[2])
        for name in try FileManager.default.contentsOfDirectory(atPath: source.path)
        where !name.hasSuffix(".safetensors") {
            let destination = out.appending(component: name)
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.copyItem(at: source.appending(component: name),
                                             to: destination)
        }
        let weights = Dictionary(
            uniqueKeysWithValues: drafter.parameters().flattened()
                .map { ($0.0, $0.1.asType(.bfloat16)) })
        try MLX.save(arrays: weights, url: out.appending(component: "model.safetensors"))
        print("записан драфтер: \(out.path)")
    }
    exit(0)
}

// --- тест на градиенты ---------------------------------------------------------------
// Если потери на горстке примеров не падают к нулю, дело не в данных, а в связке
// прямого прохода, потерь и обратного распространения. Это выясняется за минуты.
var dataset: [Sample] = []
for prompt in loadPrompts().prefix(3) { dataset += try await collect(prompt, maxTokens: 64) }
print("собрано примеров: \(dataset.count), длина блока разметки: \(labelWidth)")

let anchors = MLXArray(dataset.map(\.anchor))
let fused = concatenated(dataset.map { $0.fused.reshaped(1, -1) }, axis: 0)
let labels = MLXArray(dataset.flatMap(\.targets)).reshaped(dataset.count, labelWidth)

let optimizer = AdamW(learningRate: learningRate)
let lossAndGrad = valueAndGrad(model: drafter) { (model: DFlashDraftModel, _: [MLXArray]) in
    [batchLoss(model, anchors: anchors, fused: fused, labels: labels)]
}

print("шаг  потери")
for step in 1 ... intFlag("--steps", 60) {
    let (value, grads) = lossAndGrad(drafter, [MLXArray(0)])
    optimizer.update(model: drafter, gradients: grads)
    eval(drafter, optimizer)
    if step == 1 || step % 10 == 0 {
        print(String(format: "%4d  %.4f", step, value[0].item(Float.self)))
    }
}
