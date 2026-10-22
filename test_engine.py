"""引擎与输出层单测（纯 Python，不需要 GPU；权重缺失时只跳过加载用例）。

覆盖 ScriptMoE 管线里最容易静默出错的地方：
  - 阅读顺序组装（行聚组 / 行内排序 / 中英文分隔符）
  - 逐行重复截断
  - 预处理与 torchvision 的逐像素一致性（本项目刻意不装 torchvision）
  - 模型 id 别名、旧渠道不破
  - 前置缩放与 EXIF 转正

跑法: python test_engine.py
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server  # noqa: E402
from ocr_engine import assemble_text, line_geometry, sort_boxes_scores  # noqa: E402


def L(text, x0, yc, h=20.0):
    """造一条行记录：box 的左上角由 (x0, yc-h/2) 推出，行内排序只看 x0/_yc。"""
    line = {"text": text, "points": [[x0, yc - h / 2], [x0 + 100, yc - h / 2],
                                     [x0 + 100, yc + h / 2], [x0, yc + h / 2]]}
    line.update(line_geometry(line["points"]))
    return line


def test_row_grouping_label_and_value():
    """同一行上的「标签 + 值」两个框必须拼成一行，中间补空格。"""
    out = assemble_text([L("91420600343516164T", 300, 100), L("统一社会信用代码", 20, 100)])
    assert out == "统一社会信用代码 91420600343516164T", repr(out)
    print("  同行标签+值: 拼成一行且补空格")


def test_cjk_adjacent_no_space():
    """两侧都是中文的相邻块不插空格（中文排版无词间空格）。"""
    out = assemble_text([L("有限公司", 200, 100), L("武汉某某科技", 20, 100)])
    assert out == "武汉某某科技有限公司", repr(out)
    print("  中文相邻: 不插空格")


def test_row_order_and_intra_row_sort():
    """行按 y 从上到下，行内按 x 从左到右，输入乱序也要排好。"""
    out = assemble_text([L("第三行", 10, 300), L("第二行B", 200, 200),
                         L("第二行A", 20, 200), L("第一行", 10, 100)])
    assert out.split("\n") == ["第一行", "第二行A 第二行B", "第三行"], repr(out)
    print("  阅读顺序: 行 y 上到下，行内 x 左到右")


def test_far_apart_rows_split():
    """y 中心差超过 0.6 x 行高就是两行：不能把紧凑的两行并成一行。"""
    out = assemble_text([L("上行", 10, 100, h=20), L("下行", 10, 140, h=20)])
    assert out == "上行\n下行", repr(out)
    out2 = assemble_text([L("上半", 10, 100, h=20), L("下半", 10, 110, h=20)])
    assert out2 == "上半下半", repr(out2)   # 两侧都是中文 -> 不插空格
    print("  行分割阈值: y 中心差 40px 分成两行，10px 合成一行")


def test_empty_and_whitespace_lines_dropped():
    out = assemble_text([L("   ", 10, 100), L("有字", 10, 300)])
    assert out == "有字", repr(out)
    assert assemble_text([]) == ""
    print("  空行: 丢弃，空输入返回空串")


def test_legit_repeated_rows_survive():
    """重复截断是逐行做的，整表 20 行内容完全相同也不能被截掉。"""
    lines = [L("1 合格 通过", 10, 100 + 30 * i) for i in range(20)]
    cleaned = [dict(l, text=server._trim_repeats(l["text"])) for l in lines]
    out = assemble_text(cleaned)
    assert out.count("1 合格 通过") == 20, repr(out[:200])
    print("  正常重复行: 20 行同内容表完整保留")


def test_trim_line_loop():
    """单行被同一子串刷屏（'/' 连打 40 次）必须截断。"""
    junk = "1595959595959595959595959595959595959595"      # 周期 3，远超阈值
    out = server._trim_repeats(junk)
    assert out.endswith("…") and len(out) < len(junk), f"应截断，实际 {out!r}"
    normal = "营业执照 统一社会信用代码 91420600343516164T"
    assert server._trim_repeats(normal) == normal, "正常行被误伤"
    seven = server._trim_repeats("表头" + "abcd" * (server.TRIM_MIN_TIMES - 1))
    assert not seven.endswith("…"), f"重复 {server.TRIM_MIN_TIMES - 1} 次未到阈值不该截"
    ten = server._trim_repeats("表头" + "abcd" * (server.TRIM_MIN_TIMES + 2))
    assert ten.endswith("…"), "超过阈值应截断"
    print(f"  逐行截断: 阈值 {server.TRIM_MIN_TIMES} 次（{server.TRIM_MIN_TIMES - 1} 放行 / {server.TRIM_MIN_TIMES + 2} 截断）")


def test_sort_boxes_reading_order():
    """检测框排序：先上到下，同一行（顶边 y 差 <10px）内左到右。"""
    boxes = np.array([[[300, 100], [400, 100], [400, 120], [300, 120]],
                      [[10, 300], [100, 300], [100, 320], [10, 320]],
                      [[10, 101], [100, 101], [100, 121], [10, 121]]], dtype=np.float32)
    scores = np.array([0.3, 0.9, 0.6], dtype=np.float32)
    s_boxes, s_scores = sort_boxes_scores(boxes, scores)
    xs = [float(b[0][0]) for b in s_boxes]
    assert xs == [10.0, 300.0, 10.0], xs
    assert [round(x, 2) for x in s_scores] == [0.6, 0.3, 0.9], s_scores   # 分数跟着框走
    print("  检测框排序: 行序 + 行内左到右，分数同步跟随")


def test_model_id_aliases():
    """旧渠道的模型名必须还能用，但 /v1/models 只列规范 id。"""
    assert server.MODELS == [server.MODEL]
    for legacy in ("opendoc-0.1b", "unirec-0.1b", "scriptmoe-0.1b"):
        assert server.MODEL_ALIASES.get(legacy) == server.MODEL, legacy
    ids = [m["id"] for m in server.list_models()["data"]]
    assert ids == [server.MODEL], ids
    print(f"  模型 id: 规范 {server.MODEL}，旧 id 走别名映射")


def test_preprocess_matches_torchvision():
    """预处理必须与 torchvision 的 BICUBIC + ToTensor + Normalize 逐像素一致。

    本项目刻意不装 torchvision（识别侧只要 torch），这条用例是那个换法的验收线。
    没装 torchvision 的环境（生产容器）自动跳过。
    """
    try:
        from torchvision import transforms as T
        from torchvision.transforms import functional as TF
    except ImportError:
        print("  预处理一致性: 本机没有 torchvision，跳过（生产容器不装）")
        return
    rec = server.get_engine().recognizer
    rng = np.random.default_rng(7)
    worst = 0.0
    for (w, h) in [(400, 24), (60, 120), (120, 120), (900, 30), (40, 40), (500, 45)]:
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        pil = Image.fromarray(arr)
        mine, ratio = rec._preprocess(pil)
        if h > w * 1.5:
            pil2 = pil.transpose(Image.ROTATE_90)
        else:
            pil2 = pil
        w2, h2 = pil2.size
        gr = max(1, round(float(w2) / float(h2)))
        gr = int(np.clip(gr, 1, rec.max_ratio))
        tw, th = server_shape(gr)
        ref = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])(
            TF.resize(pil2, (th, tw), interpolation=T.InterpolationMode.BICUBIC))
        assert mine.shape == ref.shape, (mine.shape, ref.shape, (w, h))
        assert ratio == gr, (ratio, gr, (w, h))
        worst = max(worst, float((mine - ref).abs().max()))
    assert worst < 1e-3, f"预处理与 torchvision 不一致，最大偏差 {worst}"
    print(f"  预处理一致性: 6 种宽高比逐像素一致（最大偏差 {worst:.2e}）")


def server_shape(gen_ratio):
    from ocr_engine.pipeline import BASE_H, BASE_SHAPE
    if gen_ratio <= 4:
        return BASE_SHAPE[gen_ratio - 1]
    return BASE_H * gen_ratio, BASE_H


def test_shrink_and_exif():
    """超限图必须重编码并删除原文件；带 EXIF Orientation=6 的图必须转正。"""
    d = tempfile.mkdtemp(prefix="oocr_test_")
    limit = server.MAX_IMAGE_BYTES
    try:
        server.MAX_IMAGE_BYTES = 20 * 1024
        big = os.path.join(d, "big.jpg")
        # 平滑渐变：JPEG 压得动，才能验证「体积上限」这条路径真的把文件压下去了
        grad = np.tile(np.linspace(0, 255, 1200, dtype=np.uint8), (400, 1))
        Image.fromarray(np.stack([grad, grad, grad], axis=2)).save(big, "JPEG", quality=95)
        out = server._shrink(big)
        assert out.endswith(".c.jpg"), out
        assert not os.path.exists(big), "原文件应删除"
        assert os.path.getsize(out) <= server.MAX_IMAGE_BYTES, "未压到体积上限内"

        exif = Image.Exif()
        exif[274] = 6                                   # Orientation: 顺时针 90°
        rot = os.path.join(d, "rot.jpg")
        Image.new("RGB", (200, 100), (255, 0, 0)).save(rot, "JPEG", exif=exif.tobytes())
        out2 = server._shrink(rot)
        with Image.open(out2) as im:
            assert im.size == (100, 200), f"EXIF 未转正: {im.size}"
        print("  前置缩放: 超限重编码 + 原文件删除；EXIF Orientation=6 已转正")
    finally:
        server.MAX_IMAGE_BYTES = limit
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_weights_present():
    """权重/字典缺失要报人话，不能启动后输出乱码。"""
    for path in (server.SCRIPTMOE_WEIGHTS, server.SCRIPTMOE_DICT):
        if not os.path.exists(path):
            print(f"  权重检查: 跳过（{path} 不存在，权重不进 git）")
            return
        assert os.path.getsize(path) > 1000, path
    print(f"  权重检查: {server.SCRIPTMOE_WEIGHTS} 就位")


if __name__ == "__main__":
    test_row_grouping_label_and_value()
    test_cjk_adjacent_no_space()
    test_row_order_and_intra_row_sort()
    test_far_apart_rows_split()
    test_empty_and_whitespace_lines_dropped()
    test_legit_repeated_rows_survive()
    test_trim_line_loop()
    test_sort_boxes_reading_order()
    test_model_id_aliases()
    test_weights_present()
    test_shrink_and_exif()
    test_preprocess_matches_torchvision()
    print("\n全部通过")
