#!/usr/bin/env python
"""
启动脚本 — 一键启动 YOLO 目标检测 Web 系统
用法: python run.py
"""

from app import app, config

if __name__ == "__main__":
    host = config["server"]["host"]
    port = config["server"]["port"]
    debug = config["server"]["debug"]

    print(f"""
╔══════════════════════════════════════════════════════╗
║         🎯 YOLOv8 目标检测系统                       ║
║                                                      ║
║  📍 地址:  http://{host}:{port}                      ║
║  🧠 模型:  {config["model"]["name"]}               ║
║  📐 设备:  {config["model"]["device"]}              ║
║                                                      ║
║  在浏览器中打开上方地址即可使用                       ║
╚══════════════════════════════════════════════════════╝
""")

    app.run(host=host, port=port, debug=debug)
