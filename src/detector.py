"""
YOLO 目标检测核心模块
支持图片、视频帧的检测推理
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """YOLOv8 目标检测器，封装模型加载与推理"""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        """
        Args:
            model_name: 模型文件名 (yolov8n.pt ~ yolov8x.pt) 或本地路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU 阈值
            device: 推理设备 cpu / cuda:0
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model: Optional[YOLO] = None

    def load_model(self) -> 'YOLODetector':
        """加载模型（首次调用时自动下载）"""
        self.model = YOLO(self.model_name)
        return self

    def predict(self, image: np.ndarray) -> List[dict]:
        """
        对单张图片执行检测

        Args:
            image: BGR 格式 numpy 数组

        Returns:
            [{"bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int, "class_name": str}, ...]
        """
        if self.model is None:
            self.load_model()

        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
            clss = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

            for box, conf, cls_id in zip(boxes, confs, clss):
                detections.append({
                    "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    "confidence": round(float(conf), 3),
                    "class_id": int(cls_id),
                    "class_name": self.model.names.get(int(cls_id), f"class_{int(cls_id)}"),
                })

        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[dict]]:
        """批量检测多张图片"""
        return [self.predict(img) for img in images]

    def get_available_models(self) -> List[str]:
        """返回可用的 YOLOv8 预训练模型列表"""
        return ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
