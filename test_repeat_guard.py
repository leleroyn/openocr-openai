"""重复循环防护的单元测试（纯 Python，不需要模型，秒级跑完）。

重点覆盖表格场景：表格输出天然高度周期化，防护必须能区分「表格骨架的正常重复」
（<td></td>、|---|、每行结构一样）和「模型真的陷入循环」。

用法: python test_repeat_guard.py
"""
import numpy as np

import server

EOS = 2
VOCAB = 40
CONTENT_NEED = server.REPEAT_MIN_TOKENS
STRUCT_NEED = server.REPEAT_STRUCT_MIN_TOKENS

# 假词表：结构 token 与内容 token 分开
TOK = {1: "<bos>", EOS: "<eos>", 5: "<td>", 6: "</td>", 7: "金额", 8: "合计",
       9: "|", 10: "---", 11: "1", 12: "2", 13: "3", 14: "4"}
INV = {v: k for k, v in TOK.items()}


def emit(script, n):
    """按 script 循环生成 n 个 token。"""
    return [script[i % len(script)] for i in range(n)]


def test_is_structural():
    for t in ["<td></td>", "<table><tbody><tr>", "|---|---|", "|  |  |", "\n", "||", "<br>"]:
        assert server._is_structural(t), f"{t!r} 应判为纯表格骨架"
    for t in ["<td>金额</td>", "|合计|100|", "绿化工程", "<td>1</td>", "|-a-|"]:
        assert not server._is_structural(t), f"{t!r} 含实际内容，不应判为骨架"
    print("  _is_structural: 表格骨架/正文判定用例通过")


def test_content_loop_detected():
    # <td>金额</td> 周期 3，含内容 → 按正文阈值（判定单位是完整重复的 token 总数，用例长度按周期取整）
    seq = emit([5, 7, 6], CONTENT_NEED - 3)
    texts = [TOK[t] for t in seq]
    assert not server._is_repeating(seq, texts), f"重复 {CONTENT_NEED - 3} token 未达正文阈值，不应触发"
    seq = emit([5, 7, 6], CONTENT_NEED + 1)
    texts = [TOK[t] for t in seq]
    assert server._is_repeating(seq, texts), f"含内容重复超过 {CONTENT_NEED} token 应触发"
    # 单 token 刷屏
    seq = emit([7], CONTENT_NEED)
    assert server._is_repeating(seq, [TOK[7]] * len(seq)), "单 token 刷屏应触发"
    print(f"  正文循环: {CONTENT_NEED} token 触发，{CONTENT_NEED - 1} 不触发")


def test_table_skeleton_not_truncated():
    # 空单元格连续重复：纯骨架，阈值放宽到 STRUCT_NEED
    for n in (40, 60, 90, STRUCT_NEED - 1):
        seq = emit([5, 6], n)
        texts = [TOK[t] for t in seq]
        assert not server._is_repeating(seq, texts), f"{n} 个连续空单元格被误判为循环"
    seq = emit([5, 6], STRUCT_NEED)
    texts = [TOK[t] for t in seq]
    assert server._is_repeating(seq, texts), f"骨架重复达 {STRUCT_NEED} token 应判定空转"
    # 每行结构相同、内容不同的正常大表：不能触发
    rows = [[5, 11 + (r % 3), 6, 5, 13, 6] for r in range(60)]
    seq = [t for row in rows for t in row]
    texts = [TOK[t] for t in seq]
    assert not server._is_repeating(seq, texts), "内容变化的正常大表被误判为循环"
    # Markdown 管道表：表头分隔行 + 空行
    md = [9, 10, 10, 9, 9, 10, 10, 9]
    seq = emit(md, 40)
    texts = [TOK[t] for t in seq]
    assert not server._is_repeating(seq, texts), "纯分隔行表格骨架不应触发"
    print(f"  表格场景: 空单元格 {STRUCT_NEED - 1} token 内不误伤；正常大表不误伤；骨架空转 {STRUCT_NEED} 触发")


def test_trim_repeats():
    assert server._trim_repeats("正常一句话。").endswith("。")
    out = server._trim_repeats("表格" + "<td>金额</td>" * 40)
    assert len(out) < 2 + 11 * 40 and out.endswith("…"), f"正文刷屏应截断，实际 {len(out)} 字"
    legit = "表头" + "<td></td>" * 12
    assert server._trim_repeats(legit) == legit, "12 个连续空单元格不应被截断"
    long_empty = "表头" + "<td></td>" * 60
    out2 = server._trim_repeats(long_empty)
    assert out2.endswith("…") and len(out2) < len(long_empty), "60 个连续空单元格应截断"
    print("  _trim_repeats: 正文 8 次截断 / 表格骨架 24 次才截")


def _fake_rec(script):
    class FakeTokenizer:
        eos_token_id = EOS
        id_to_token = dict(TOK)

    class FakeUniRec:
        tokenizer = FakeTokenizer()

        def decode_step(self, input_id, past_length, cross_k, cross_v,
                        past_key_values, padding_idx=1):
            logits = np.zeros((1, 1, VOCAB), dtype=np.float32)
            logits[0, -1, script[past_length % len(script)]] = 1.0
            return logits, past_key_values

    return FakeUniRec


def _run_loop(rec, budget=2048):
    """复刻库的贪心解码循环，返回实际步数。"""
    server._patch_unirec(type(rec))
    ck = np.zeros((1, 1, 4), dtype=np.float32)
    cv = np.zeros((1, 1, 4), dtype=np.float32)
    generated = [1]
    past = []
    steps = 0
    for step in range(budget - 1):
        logits, past = rec.decode_step(generated[-1], step, ck, cv, past)
        nxt = int(np.argmax(logits[0, -1, :]))
        generated.append(nxt)
        steps += 1
        if nxt == EOS:
            break
    return steps, generated[-1] == EOS


def test_decode_patch_stops_content_loop():
    steps, eos = _run_loop(_fake_rec([5, 7, 6])())
    assert eos and steps <= CONTENT_NEED + 2, f"正文循环应在 {CONTENT_NEED + 2} 步内停止，实际 {steps}"
    print(f"  解码补丁: 正文死循环第 {steps} 步切断（上限 2047）")


def test_decode_patch_allows_big_table():
    # 阈值预算内：空表骨架必须一路放行，不能被当成循环
    budget = STRUCT_NEED - 6
    steps, eos = _run_loop(_fake_rec([5, 6])(), budget=budget)
    assert not eos and steps == budget - 1, f"空表骨架在 {budget} token 内应放行，实际 {steps} 步 eos={eos}"
    # 超过阈值：判定为在空转，切断
    steps2, eos2 = _run_loop(_fake_rec([5, 6])(), budget=STRUCT_NEED + 20)
    assert eos2 and steps2 <= STRUCT_NEED + 4, f"骨架空转应在 {STRUCT_NEED + 4} 步内停止，实际 {steps2}"
    print(f"  解码补丁: 空表骨架在阈值内放行 {steps} 步；骨架空转第 {steps2} 步切断（正文阈值 {CONTENT_NEED}，骨架阈值 {STRUCT_NEED}）")


def test_decode_patch_normal_output():
    """token 不断变化（模拟正常识别）不能被误伤。"""
    script = [5, 11, 6, 5, 12, 6, 5, 13, 6, 5, 14, 6, 7, 8, 9, 10]
    steps, eos = _run_loop(_fake_rec(script)(), budget=200)
    assert not eos and steps == 199, f"正常输出被误伤，只跑了 {steps} 步"
    print(f"  解码补丁: 正常输出跑满 {steps} 步未被误伤")


def test_truncation_marker():
    """跑满 token 预算时必须显式标记截断（半张表比空结果更坑调用方）。"""
    def engine(**kw):
        server._repeat_state.steps = kw["max_length"]
        return "<table><tr><td>1</td>", None

    out = server._run_unirec(engine, "x.png", 2048)
    assert "max_tokens=2048" in out and "截断" in out, f"应带截断标记，实际: {out[:120]}"

    def engine2(**kw):
        server._repeat_state.steps = 30
        return "<table><tr><td>1</td></tr></table>", None

    out2 = server._run_unirec(engine2, "x.png", 2048)
    assert "截断" not in out2, "自然收尾不应加截断标记"
    print("  截断标记: 跑满预算加标记，自然收尾不加")


if __name__ == "__main__":
    test_is_structural()
    test_content_loop_detected()
    test_table_skeleton_not_truncated()
    test_trim_repeats()
    test_decode_patch_stops_content_loop()
    test_decode_patch_allows_big_table()
    test_decode_patch_normal_output()
    test_truncation_marker()
    print("\n全部通过")
