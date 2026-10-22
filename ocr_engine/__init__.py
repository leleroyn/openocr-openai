"""ScriptMoE 推理引擎包（检测 + 识别 + 文本组装），server.py 唯一依赖的推理入口。

modeling.py / postprocess.py 是上游 ScriptMoE 的原样 vendor，不要改；
pipeline.py 是服务端适配层（预处理去 torchvision、检测吃落盘路径、阅读顺序组装）。
"""
from .pipeline import (DEFAULT_MAX_RATIO, DEFAULT_MAX_TEXT_LENGTH, ScriptMoEOcr,
                       ScriptMoERecognizer, TextDetector, assemble_text,
                       line_geometry, sort_boxes_scores)

__all__ = ["ScriptMoEOcr", "ScriptMoERecognizer", "TextDetector", "assemble_text",
           "line_geometry", "sort_boxes_scores", "DEFAULT_MAX_RATIO",
           "DEFAULT_MAX_TEXT_LENGTH"]
