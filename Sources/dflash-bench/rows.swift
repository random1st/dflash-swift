// Сколько стоит проход таргета на 1 строке против 8: если веса читаются один раз,
// разница должна быть небольшой, и тогда спекуляция окупается. Если восемь строк
// стоят вчетверо — окупаться нечему, и это ответ на вопрос «почему нет выигрыша».
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

func measureRows(target: Qwen35TextModel, vocabulary: Int) {
    // 5 и 6 обрамляют порог, с которого ядро small-M включается, 8 и 9 — границу
    // между одной плиткой MMA и двумя: если кривая ломается не там, виновата граница,
    // а не ядро.
    let widths = [1, 2, 4, 5, 6, 8, 9, 12, 16]
    print("ширина  секунд/проход  относительно одной строки")
    var single = 0.0
    for width in widths {
        let cache = try! target.newCache(parameters: nil)
        // прогрев
        for _ in 0 ..< 2 {
            let ids = MLXArray((0 ..< width).map { Int32(($0 * 977) % vocabulary) }).reshaped(1, width)
            let out = target(LMInput.Text(tokens: ids), cache: cache, state: nil)
            eval(out.logits)
        }
        let rounds = 12
        let started = Date()
        for _ in 0 ..< rounds {
            let ids = MLXArray((0 ..< width).map { Int32(($0 * 977) % vocabulary) }).reshaped(1, width)
            let out = target(LMInput.Text(tokens: ids), cache: cache, state: nil)
            eval(out.logits)
        }
        let each = Date().timeIntervalSince(started) / Double(rounds)
        if width == 1 { single = each }
        print(String(format: "%5d   %10.4f   %.2fx", width, each, each / single))
    }
}
