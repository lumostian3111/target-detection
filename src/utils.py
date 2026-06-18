"""
工具函数：可视化、文件处理、图片编解码
"""

import base64
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# COCO 数据集 80 类别调色板 (BGR 格式)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
    (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128), (128, 64, 128), (0, 192, 128), (128, 192, 128),
    (64, 64, 0), (192, 64, 0), (64, 192, 0), (192, 192, 0),
    (64, 64, 128), (192, 64, 128), (64, 192, 128), (192, 192, 128),
    (0, 0, 64), (128, 0, 64), (0, 128, 64), (128, 128, 64),
    (0, 0, 192), (128, 0, 192), (0, 128, 192), (128, 128, 192),
    (64, 0, 64), (192, 0, 64), (64, 128, 64), (192, 128, 64),
    (64, 0, 192), (192, 0, 192), (64, 128, 192), (192, 128, 192),
    (0, 64, 64), (128, 64, 64), (0, 192, 64), (128, 192, 64),
    (0, 64, 192), (128, 64, 192), (0, 192, 192), (128, 192, 192),
    (64, 64, 64), (192, 64, 64), (64, 192, 64), (192, 192, 64),
    (64, 64, 192), (192, 64, 192), (64, 192, 192), (192, 192, 192),
    (32, 0, 0), (160, 0, 0), (32, 128, 0), (160, 128, 0),
    (32, 0, 128), (160, 0, 128), (32, 128, 128), (160, 128, 128),
    (96, 0, 0), (224, 0, 0), (96, 128, 0), (224, 128, 0),
    (96, 0, 128), (224, 0, 128), (96, 128, 128), (224, 128, 128),
]


def get_color(class_id: int) -> Tuple[int, int, int]:
    """为每个类别返回一个固定颜色"""
    return COLORS[class_id % len(COLORS)]


def draw_boxes(
    image: np.ndarray,
    detections: List[dict],
    draw_labels: bool = True,
) -> np.ndarray:
    """
    在图片上绘制检测框和标签

    Args:
        image: BGR 格式图片
        detections: 检测结果列表
        draw_labels: 是否绘制类别标签

    Returns:
        标注后的 BGR 图片
    """
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class_name"]
        conf = det["confidence"]
        color = get_color(det["class_id"])

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if draw_labels:
            text = f"{label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def image_to_base64(image: np.ndarray) -> str:
    """将 numpy 图片转为 base64 JPEG 字符串"""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_image(b64_str: str) -> np.ndarray:
    """将 base64 字符串解码为 numpy 图片"""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "webm", "mkv"}


def allowed_file(filename: str, file_type: str = "image") -> bool:
    """检查文件扩展名是否允许"""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    if file_type == "image":
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == "video":
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return ext in ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
