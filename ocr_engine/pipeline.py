"""ScriptMoE 端到端 OCR：文本检测（OpenOCR repvit_db ONNX）+ ScriptMoE 识别 + 阅读顺序组装。

从上游 E2EOCR/infer.py 移植，为服务改了三处：
  1) 去掉 torchvision 依赖：预处理用 PIL + numpy 复刻 RatioDataSetTVResize
     （BICUBIC → /255 → (x-0.5)/0.5，与 torchvision 路径逐像素一致，见 test_engine.py）。
  2) 检测直接吃落盘路径：服务端图片本来就在磁盘上，省掉上游那趟「数组 → PNG → 解码」。
  3) 新增阅读顺序组装：把带框行拼成纯文本，这是本服务对调用方的输出格式。
"""
from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .crops import get_minarea_rect_crop, get_rotate_crop_image
from .modeling import ScriptMoERecModel
from .postprocess import ARLabelDecoder

# 识别输入分辨率：复刻训练侧 RatioDataSetTVResize，gen_ratio -> (w, h)
BASE_SHAPE = [(64, 64), (96, 48), (112, 40), (128, 32)]
BASE_H = 32
MIN_RATIO = 1
DEFAULT_MAX_TEXT_LENGTH = 100    # 单行最多几个字符（decoder 自回归步数上限）
DEFAULT_MAX_RATIO = 20           # 行宽高比上限：单图显存/耗时随它平方增长

# 中日韩与全角标点：行内相邻两块之间不补空格（中文排版没有词间空格）
_WIDE_RE = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff"
                      r"\uac00-\ud7af\uff00-\uffef]")


def _sep(left: str, right: str) -> str:
    """行内相邻两个文字块之间补什么：两侧都是宽字符就不补，否则补一个空格。"""
    if left and right and _WIDE_RE.match(left[-1]) and _WIDE_RE.match(right[0]):
        return ""
    return " "


def sort_boxes_scores(boxes: np.ndarray, scores: np.ndarray) -> Tuple[List, List[float]]:
    """按阅读顺序排文字框：先上到下，同一行内左到右，分数跟着框走。

    「同一行」判据是两框顶边 y 差 10 像素以内（上游同款阈值），相邻框再冒泡一次理顺行内顺序。
    """
    n = len(boxes)
    if n == 0:
        return [], []
    idxs = sorted(range(n), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
    for i in range(n - 1):
        for j in range(i, -1, -1):
            bj, bj1 = boxes[idxs[j]], boxes[idxs[j + 1]]
            if abs(bj1[0][1] - bj[0][1]) < 10 and bj1[0][0] < bj[0][0]:
                idxs[j], idxs[j + 1] = idxs[j + 1], idxs[j]
            else:
                break
    return [boxes[i] for i in idxs], [float(scores[i]) for i in idxs]


def line_geometry(points) -> Dict[str, float]:
    """多边形框的 x 左缘 / y 中心 / 行高：行聚组与行内排序都靠这三个量。"""
    pts = np.array(points, dtype=np.float32)
    ys, xs = pts[:, 1], pts[:, 0]
    return {"_x0": float(xs.min()), "_yc": float(ys.mean()),
            "_h": max(float(ys.max() - ys.min()), 1.0)}


def assemble_text(lines: Sequence[Dict]) -> str:
    """把带框的识别行按阅读顺序拼成纯文本。

    行框按 y 中心聚成「文本行」（阈值 0.6 x 较高框的行高），行内按 x 左到右拼接，行间换行。
    证照/表单上「标签 + 值」通常是同一行的两个框，聚成一行后调用方按行抽字段最省事；
    纯按框一行一句输出会把「统一社会信用代码」和它的号码拆成两行。
    """
    if not lines:
        return ""
    rows: List[List[Dict]] = []
    for ln in sorted(lines, key=lambda d: d["_yc"]):
        if rows:
            prev = rows[-1]
            tol = 0.6 * max(ln["_h"], max(p["_h"] for p in prev))
            if abs(ln["_yc"] - prev[0]["_yc"]) <= tol:
                prev.append(ln)
                continue
        rows.append([ln])
    out = []
    for row in rows:
        parts = ""
        for ln in sorted(row, key=lambda d: d["_x0"]):
            t = (ln["text"] or "").strip()
            if not t:
                continue
            parts = t if not parts else parts + _sep(parts, t) + t
        if parts:
            out.append(parts)
    return "\n".join(out)


def _empty_det() -> Tuple[np.ndarray, np.ndarray]:
    return (np.zeros((0, 4, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32))


def _to_quads_scores(polys, scores) -> Tuple[np.ndarray, np.ndarray]:
    """检测框统一成 (N, 4, 2) 四边形 + (N,) 分数；缺分数按 1.0 补中性置信度。"""
    if polys is None or len(polys) == 0:
        return _empty_det()
    polys = np.array(polys, dtype=np.float32)
    if polys.ndim == 3 and polys.shape[1] != 4:      # 多边形框 → 最小外接矩形
        quads = [cv2.boxPoints(cv2.minAreaRect(p.astype(np.int32))) for p in polys]
        polys = np.array(quads, dtype=np.float32)
    n = len(polys)
    if scores is None:
        scores = np.ones((n,), dtype=np.float32)
    else:
        scores = np.array(scores, dtype=np.float32).reshape(-1)
        if len(scores) != n:
            scores = np.ones((n,), dtype=np.float32)
    return polys, scores


class TextDetector:
    """OpenOCR 文本检测（repvit_db，ONNX Runtime 会话）：输入落盘图片路径，输出四边形框 + 分数。

    权重首次使用时由 openocr 走 modelscope 下载并缓存；生产上建议提前把
    openocr_det_model.onnx 放进缓存目录，或用 det_model_path 直接指过去。
    """

    def __init__(self, use_gpu: str = "auto", mode: str = "mobile",
                 det_model_path: Optional[str] = None, auto_download: bool = True):
        from openocr import OpenOCR
        kwargs = dict(task="det", mode=mode, use_gpu=use_gpu,
                      auto_download=auto_download)
        if det_model_path:
            if not os.path.exists(det_model_path):
                raise FileNotFoundError(f"检测模型不存在: {det_model_path}")
            kwargs["onnx_det_model_path"] = det_model_path
        self._det = OpenOCR(**kwargs)

    def __call__(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        results = self._det(image_path=image_path)
        if not results or not isinstance(results[0], dict):
            return _empty_det()
        res = results[0]
        scores = None
        for key in ("scores", "det_scores", "box_scores"):
            if res.get(key) is not None:
                scores = res[key]
                break
        return _to_quads_scores(res.get("boxes"), scores)


class ScriptMoERecognizer:
    """ScriptMoE 识别器：输入文字条裁图，输出（文本，置信度）。预处理见模块 docstring。"""

    def __init__(self, weights_path: str, char_dict_path: str, use_gpu: str = "auto",
                 max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
                 max_ratio: int = DEFAULT_MAX_RATIO, threads: int = 0):
        if threads > 0:
            torch.set_num_threads(int(threads))
        if use_gpu == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if use_gpu in ("true", "1", "yes") else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("要求 GPU 推理，但这个 torch 没有可用的 CUDA 设备"
                               "（GTX 10 系等 Pascal 卡需要 cu126 版 torch）")
        self.device = torch.device(device)

        for what, path in (("ScriptMoE 权重", weights_path), ("字符字典", char_dict_path)):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{what} 不存在: {path}")

        self.max_ratio = int(max_ratio)
        self.decoder = ARLabelDecoder(char_dict_path, use_space_char=True)
        self.model = ScriptMoERecModel(num_classes=self.decoder.num_classes,
                                       max_len=int(max_text_length))
        missing, unexpected = self.model.load_state_dict(
            self._load_state_dict(weights_path), strict=False)
        # strict=False 只用来容忍 key 前缀差异，真对不上必须当场报错：随机初始化的权重
        # 照样能跑完一整遍，只是输出全是乱码 —— 最难查的那种故障。
        if missing or unexpected:
            raise RuntimeError(
                f"ScriptMoE 权重与模型结构不匹配：缺 {len(missing)} 个"
                f"（示例 {missing[:2]}），多 {len(unexpected)} 个（示例 {unexpected[:2]}）")
        self.model.to(self.device).eval()

    @staticmethod
    def _load_state_dict(path: str):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        obj = torch.load(path, map_location="cpu")
        return obj.get("state_dict", obj) if isinstance(obj, dict) else obj

    def _preprocess(self, pil_img: Image.Image) -> Tuple[torch.Tensor, int]:
        """单条文字条 → 归一化张量 + 宽高比分桶号。"""
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        if h > w * 1.5:                       # 竖排 → 转 90° 变横排
            pil_img = pil_img.transpose(Image.ROTATE_90)
            w, h = pil_img.size
        gen_ratio = max(1, round(float(w) / float(h)))
        gen_ratio = int(np.clip(gen_ratio, MIN_RATIO, self.max_ratio))
        if gen_ratio <= 4:
            img_w, img_h = BASE_SHAPE[gen_ratio - 1]
        else:
            img_w, img_h = BASE_H * gen_ratio, BASE_H
        arr = np.asarray(pil_img.resize((img_w, img_h), Image.BICUBIC),
                         dtype=np.float32)
        # ToTensor: HWC uint8 → CHW /255；Normalize(0.5, 0.5): (x - 0.5) / 0.5，即 2x - 1
        t = torch.from_numpy(arr.transpose(2, 0, 1) / 255.0)
        return t.sub_(0.5).div_(0.5), gen_ratio    # Normalize(mean=0.5, std=0.5) = 2x - 1

    @torch.no_grad()
    def __call__(self, pil_imgs: Sequence[Image.Image],
                 batch_num: int = 8) -> List[Dict]:
        """按宽高比分桶批量识别（同桶才能拼 batch），返回与输入同序的结果列表。"""
        n = len(pil_imgs)
        if n == 0:
            return []
        tensors: List[torch.Tensor] = []
        buckets: Dict[int, List[int]] = {}
        for i, img in enumerate(pil_imgs):
            t, ratio = self._preprocess(img)
            tensors.append(t)
            buckets.setdefault(ratio, []).append(i)

        results: List[Optional[Dict]] = [None] * n
        for _ratio, idx_list in buckets.items():
            for s in range(0, len(idx_list), max(1, batch_num)):
                sub = idx_list[s:s + batch_num]
                batch = torch.stack([tensors[i] for i in sub]).to(self.device)
                decoded = self.decoder(self.model(batch))
                for k, (text, score) in zip(sub, decoded):
                    results[k] = {"text": text, "score": float(score)}
        return results


class ScriptMoEOcr:
    """检测 + 识别的一体化入口。

    infer_file(path) -> (纯文本, 行级结果, 耗时明细)
      行级结果每项：text / points(四边形) / score(识别置信度) / det_score(检测分数)
                    + _x0/_yc/_h（组装用的框几何，调用方可忽略）
      耗时明细：det_ms / rec_ms / total_ms / num_boxes / num_kept
    """

    def __init__(self, weights_path: str, char_dict_path: str, use_gpu: str = "auto",
                 det_mode: str = "mobile", det_model_path: Optional[str] = None,
                 drop_score: float = 0.5, rec_batch_num: int = 8,
                 max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
                 max_ratio: int = DEFAULT_MAX_RATIO, box_type: str = "quad",
                 threads: int = 0, auto_download: bool = True):
        t0 = time.time()
        self.recognizer = ScriptMoERecognizer(
            weights_path=weights_path, char_dict_path=char_dict_path,
            use_gpu=use_gpu, max_text_length=max_text_length,
            max_ratio=max_ratio, threads=threads)
        self.detector = TextDetector(use_gpu=use_gpu, mode=det_mode,
                                     det_model_path=det_model_path,
                                     auto_download=auto_download)
        self.drop_score = float(drop_score)
        self.rec_batch_num = int(rec_batch_num)
        self.box_type = box_type
        self.device = str(self.recognizer.device)
        self.load_seconds = time.time() - t0

    def infer_file(self, image_path: str) -> Tuple[str, List[Dict], Dict]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"图片无法解码: {image_path}")

        t0 = time.time()
        boxes, det_scores = self.detector(image_path)
        det_ms = (time.time() - t0) * 1000.0
        if len(boxes) == 0:
            return "", [], {"det_ms": det_ms, "rec_ms": 0.0, "total_ms": det_ms,
                            "num_boxes": 0, "num_kept": 0}

        boxes, det_scores = sort_boxes_scores(np.array(boxes), np.array(det_scores))
        crop_fn = get_rotate_crop_image if self.box_type == "quad" else get_minarea_rect_crop
        crops = [crop_fn(img, np.array(b, dtype=np.float32)) for b in boxes]

        t1 = time.time()
        rec = self.recognizer([Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                               for c in crops], batch_num=self.rec_batch_num)
        rec_ms = (time.time() - t1) * 1000.0

        lines: List[Dict] = []
        for box, det_score, r in zip(boxes, det_scores, rec):
            if r is None or r["score"] < self.drop_score:
                continue
            line = {"text": r["text"], "points": np.array(box, dtype=np.float32).tolist(),
                    "score": r["score"], "det_score": float(det_score)}
            line.update(line_geometry(box))
            lines.append(line)
        text = assemble_text(lines)
        return text, lines, {"det_ms": det_ms, "rec_ms": rec_ms,
                             "total_ms": det_ms + rec_ms,
                             "num_boxes": int(len(boxes)), "num_kept": len(lines)}
