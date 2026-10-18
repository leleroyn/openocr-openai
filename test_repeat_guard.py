"""重复循环防护的单元测试（纯 Python，不需要模型，秒级跑完）。

用法: python test_repeat_guard.py
"""
import numpy as np

import server

EOS = 2
VOCAB = 20
N = server.REPEAT_NGRAM
K = server.REPEAT_MIN_TIMES


def toks(seq):
    return list(seq)


def test_repeating_cases():
    # 触发：n-gram 连续重复 K 次
    assert server._is_repeating(toks(list(range(5)) + [7] * 0 + [3] * (N * K))), "单 token 刷屏应触发"
    pat = [1, 2, 3, 4, 5, 6, 7, 8]
    assert server._is_repeating(toks(pat * K)), f"{N}-gram 连续 {K} 次应触发"
    assert server._is_repeating(toks([9] * server.REPEAT_SINGLE)), "单 token 连续刷屏应触发"
    # 不触发：正常文本、只重复 2 次、重复不连续
    assert not server._is_repeating(toks(list(range(60)))), "正常变化不应触发"
    assert not server._is_repeating(toks(pat * (K - 1))), f"只重复 {K - 1} 次不应触发"
    assert not server._is_repeating(toks(pat * 2 + [0] + pat * 2)), "重复被隔开不应触发"
    assert not server._is_repeating(toks([4] * (server.REPEAT_SINGLE - 1))), "单 token 未达阈值不应触发"
    print(f"  _is_repeating: n-gram={N}x{K}, 单 token={server.REPEAT_SINGLE} 用例通过")


def test_trim_repeats():
    assert server._trim_repeats("正常一句话。").endswith("。")
    out = server._trim_repeats("表格" + "<td></td>" * 40)
    assert len(out) < 2 + 9 * 40, f"结尾刷屏应被截断，实际 {len(out)} 字"
    assert out.endswith("…")
    legit = "合计" + "|---" * 4 + "|金额|100"
    assert server._trim_repeats(legit) == legit, "正常短重复不应被截断"
    print("  _trim_repeats: 截断/放行用例通过")


def test_decode_patch_forces_eos():
    """模拟库的贪心解码循环：模型永远吐同一个 token 时，补丁必须让循环提前 break。"""
    class FakeTokenizer:
        eos_token_id = EOS

    class FakeUniRec:
        tokenizer = FakeTokenizer()

        def decode_step(self, input_id, past_length, cross_k, cross_v,
                        past_key_values, padding_idx=1):
            logits = np.zeros((1, 1, VOCAB), dtype=np.float32)
            logits[0, -1, 5] = 1.0          # 死循环：永远输出 token 5
            return logits, past_key_values

    server._patch_unirec(FakeUniRec)
    rec = FakeUniRec()
    ck = np.zeros((1, 1, 4), dtype=np.float32)
    cv = np.zeros((1, 1, 4), dtype=np.float32)

    generated = [0]
    past = []
    steps = 0
    for step in range(2048 - 1):            # 库原版循环
        logits, past = rec.decode_step(generated[-1], step, ck, cv, past)
        nxt = int(np.argmax(logits[0, -1, :]))
        generated.append(nxt)
        steps += 1
        if nxt == EOS:
            break

    assert steps < N * K + 2, f"应在 {N * K + 2} 步内停止，实际跑了 {steps} 步"
    assert generated[-1] == EOS, "最后一步应被强制成 EOS"
    print(f"  解码补丁: 死循环在第 {steps} 步被切断（上限 2047 步）")


def test_decode_patch_passes_normal():
    """正常输出（token 不断变化）不能被补丁打断。"""
    class FakeTokenizer:
        eos_token_id = EOS

    class FakeUniRec:
        tokenizer = FakeTokenizer()

        def decode_step(self, input_id, past_length, cross_k, cross_v,
                        past_key_values, padding_idx=1):
            logits = np.zeros((1, 1, VOCAB), dtype=np.float32)
            # 循环遍历 6..12（周期 7），不构成 8-gram 连续重复，也不构成单 token 刷屏
            logits[0, -1, 6 + (past_length % 7)] = 1.0
            return logits, past_key_values

    server._patch_unirec(FakeUniRec)
    rec = FakeUniRec()
    ck = np.zeros((1, 1, 4), dtype=np.float32)
    cv = np.zeros((1, 1, 4), dtype=np.float32)
    generated = [0]
    past = []
    steps = 0
    for step in range(200):
        logits, past = rec.decode_step(generated[-1], step, ck, cv, past)
        nxt = int(np.argmax(logits[0, -1, :]))
        if nxt == EOS:
            break
        generated.append(nxt)
        steps += 1
    assert steps == 200, f"正常输出被误伤，只跑了 {steps} 步"
    print(f"  解码补丁: 正常输出跑满 {steps} 步未被误伤")


if __name__ == "__main__":
    test_repeating_cases()
    test_trim_repeats()
    test_decode_patch_forces_eos()
    test_decode_patch_passes_normal()
    print("\n全部通过")
