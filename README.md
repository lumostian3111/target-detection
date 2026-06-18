# 🎯 YOLOv8 目标检测系统

基于 **YOLOv8** 的 Web 端目标检测系统，支持图片、视频和实时摄像头检测。

## ✨ 功能

- 🖼️ **图片检测** — 上传图片，实时返回带标注框的结果
- 🎬 **视频检测** — 上传视频，逐帧检测并输出标注后的视频
- 📷 **实时摄像头** — 调用浏览器摄像头，实时进行目标检测
- ⚙️ **模型切换** — 支持 yolov8n/s/m/l/x 五种模型在线切换
- 🎚️ **参数调节** — 置信度 / IOU 阈值实时可调
- 📊 **结果统计** — 按类别统计检测到的目标数量

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch + ultralytics

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动系统
```bash
python run.py
```

浏览器打开 **http://localhost:5000** 即可使用。

## 📁 项目结构

```
yolo/
├── app.py              # Flask Web 后端
├── run.py              # 一键启动脚本
├── config/
│   └── default.yaml    # 默认配置
├── src/
│   ├── detector.py     # YOLO 检测核心
│   └── utils.py        # 工具函数
├── templates/
│   └── index.html      # Web 前端页面
├── static/
│   ├── css/style.css   # 样式
│   └── js/app.js       # 前端逻辑
├── uploads/            # 上传文件暂存
├── outputs/            # 检测结果输出
├── models/             # 下载的模型权重
└── requirements.txt    # 依赖清单
```

## 🔧 配置说明

修改 `config/default.yaml` 可调整默认参数：

```yaml
model:
  name: "yolov8n.pt"      # 默认模型
  device: "cpu"            # cpu / cuda:0

detection:
  conf_threshold: 0.5      # 置信度阈值
  iou_threshold: 0.45      # IOU 阈值

server:
  host: "0.0.0.0"
  port: 5000
```

## 📝 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页面 |
| POST | `/api/detect/image` | 图片检测（文件上传或 base64）|
| POST | `/api/detect/video` | 视频检测 |
| POST | `/api/detect/webcam` | 摄像头帧检测 |
| POST | `/api/settings` | 更新检测参数 |
| GET | `/api/models` | 获取可用模型列表 |
| GET | `/outputs/<filename>` | 下载检测结果 |

## 🧠 模型说明

| 模型 | 大小 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| yolov8n | ~6MB | 最快 | 最低 | 实时检测 / 边缘设备 |
| yolov8s | ~22MB | 快 | 中 | 一般场景 |
| yolov8m | ~52MB | 中等 | 较高 | 精度要求中等 |
| yolov8l | ~87MB | 较慢 | 高 | 精度要求高 |
| yolov8x | ~136MB | 最慢 | 最高 | 离线高精度分析 |
