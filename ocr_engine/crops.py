"""文本框裁剪：把检测到的四边形/多边形框仿射拉正成水平文字条。

只取上游 E2EOCR/visualize.py 里这两个纯 cv2 函数（画图部分依赖 23MB 字库，服务端用不到）。
"""
from __future__ import annotations

import cv2
import numpy as np


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """把四边形文字框透视变换成水平文字条（4 点顺序：左上、右上、右下、左下）。"""
    points = np.array(points, dtype=np.float32)
    crop_w = int(max(np.linalg.norm(points[0] - points[1]),
                     np.linalg.norm(points[2] - points[3])))
    crop_h = int(max(np.linalg.norm(points[0] - points[3]),
                     np.linalg.norm(points[1] - points[2])))
    crop_w = max(crop_w, 1)
    crop_h = max(crop_h, 1)
    pts_std = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst = cv2.warpPerspective(img, M, (crop_w, crop_h),
                              borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_CUBIC)
    h, w = dst.shape[:2]
    if h * 1.5 > w:      # 竖排裁条转成横排：识别器只见过横排文字（预处理同款规则）
        dst = np.rot90(dst)
    return dst


def get_minarea_rect_crop(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """多边形框 → 最小外接矩形 → 再按四边形裁条（det_box_type=poly 时用）。"""
    bbox = cv2.minAreaRect(np.array(points).astype(np.int32))
    pts = sorted(list(cv2.boxPoints(bbox)), key=lambda x: x[0])
    idx = sorted(range(4), key=lambda i: bbox[1][i % 2])
    box = np.array([pts[idx[0]], pts[idx[1]], pts[idx[2]], pts[idx[3]]],
                   dtype=np.float32)
    return get_rotate_crop_image(img, box)
