"""
YOLOv8 目标检测系统 — Flask Web 后端
"""

import json
import os
import time
import uuid
from pathlib import Path

import cv2
import yaml
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from src.detector import YOLODetector
from src.utils import allowed_file, base64_to_image, draw_boxes, image_to_base64

# —————————————————————————— 初始化 App ——————————————————————————

app = Flask(__name__)

# 加载配置
CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

UPLOAD_DIR = Path(__file__).parent / "uploads"
OUTPUT_DIR = Path(__file__).parent / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB 上限

# —————————————————————————— 全局检测器 ——————————————————————————

detector = YOLODetector(
    model_name=config["model"]["name"],
    conf_threshold=config["detection"]["conf_threshold"],
    iou_threshold=config["detection"]["iou_threshold"],
    device=config["model"]["device"],
)

# 应用启动时预加载模型
print(f"[INFO] 加载模型: {detector.model_name} ...")
detector.load_model()
print("[INFO] 模型加载完成 ✓")


# —————————————————————————— 页面路由 ——————————————————————————

@app.route("/")
def index():
    """主页面"""
    return render_template("index.html")


# —————————————————————————— API 路由 ——————————————————————————

@app.route("/api/models")
def get_models():
    """返回可用的模型列表"""
    return jsonify({
        "models": detector.get_available_models(),
        "current": detector.model_name,
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """动态更新检测参数"""
    data = request.get_json()
    if "model_name" in data:
        detector.model_name = data["model_name"]
        print(f"[INFO] 切换模型: {detector.model_name}")
        detector.load_model()
    if "conf_threshold" in data:
        detector.conf_threshold = float(data["conf_threshold"])
    if "iou_threshold" in data:
        detector.iou_threshold = float(data["iou_threshold"])
    return jsonify({
        "model_name": detector.model_name,
        "conf_threshold": detector.conf_threshold,
        "iou_threshold": detector.iou_threshold,
    })


@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    """上传图片 → 返回检测结果 (base64)"""
    # —— 文件上传模式 ——
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename, "image"):
            return jsonify({"error": "不支持的文件格式"}), 400

        filename = secure_filename(file.filename or "upload.jpg")
        ext = filename.rsplit(".", 1)[1].lower()
        save_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = UPLOAD_DIR / save_name
        file.save(str(save_path))
        image = cv2.imread(str(save_path))

    # —— Base64 模式 (网页端 webcam 截图等) ——
    elif request.is_json:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "缺少 image 字段"}), 400
        image = base64_to_image(data["image"])
        save_name = None
    else:
        return jsonify({"error": "请上传文件或发送 base64 图片"}), 400

    if image is None:
        return jsonify({"error": "无法读取图片"}), 400

    # 检测
    detections = detector.predict(image)

    # 画框
    result_image = draw_boxes(image, detections)

    # 转 base64 返回
    result_b64 = image_to_base64(result_image)
    original_b64 = image_to_base64(image)

    return jsonify({
        "success": True,
        "detections": detections,
        "count": len(detections),
        "original": original_b64,
        "result": result_b64,
    })


@app.route("/api/detect/video", methods=["POST"])
def detect_video():
    """上传视频 → 逐帧检测 → 返回处理后的视频"""
    if "file" not in request.files:
        return jsonify({"error": "请上传视频文件"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename, "video"):
        return jsonify({"error": "不支持的视频格式，支持: mp4/avi/mov/webm/mkv"}), 400

    filename = secure_filename(file.filename or "video.mp4")
    ext = filename.rsplit(".", 1)[1].lower()
    save_name = f"{uuid.uuid4().hex}.{ext}"
    input_path = UPLOAD_DIR / save_name
    file.save(str(input_path))

    # 读取视频
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return jsonify({"error": "无法打开视频文件"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_name = f"{uuid.uuid4().hex}.mp4"
    output_path = OUTPUT_DIR / output_name

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.predict(frame)
        total_detections += len(detections)
        annotated = draw_boxes(frame, detections)
        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()

    return jsonify({
        "success": True,
        "video_url": f"/outputs/{output_name}",
        "filename": filename,
        "frames": frame_count,
        "total_detections": total_detections,
        "fps": round(fps, 1),
    })


@app.route("/api/detect/webcam", methods=["POST"])
def detect_webcam_frame():
    """接收前端 webcam 帧 (base64) → 返回带检测框的帧 (base64)"""
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "缺少 image 字段"}), 400

    image = base64_to_image(data["image"])
    if image is None:
        return jsonify({"error": "图片解码失败"}), 400

    detections = detector.predict(image)
    result = draw_boxes(image, detections)
    result_b64 = image_to_base64(result)

    return jsonify({
        "detections": detections,
        "count": len(detections),
        "result": result_b64,
    })


@app.route("/outputs/<filename>")
def serve_output(filename):
    """提供输出文件下载"""
    return send_from_directory(str(OUTPUT_DIR), filename)


# —————————————————————————— 启动入口 ——————————————————————————

if __name__ == "__main__":
    host = config["server"]["host"]
    port = config["server"]["port"]
    debug = config["server"]["debug"]
    print(f"\n{'='*60}")
    print(f"  🎯 YOLO 目标检测系统")
    print(f"  📍 http://{host}:{port}")
    print(f"  🧠 模型: {detector.model_name}")
    print(f"  📐 设备: {detector.device}")
    print(f"{'='*60}\n")
    app.run(host=host, port=port, debug=debug)
