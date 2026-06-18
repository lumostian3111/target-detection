/**
 * YOLOv8 目标检测系统 — 前端逻辑
 */

// ======================== 全局状态 ========================

let currentTab = "image";
let webcamStream = null;
let webcamInterval = null;
let webcamFpsCounter = 0;
let webcamFpsTimer = 0;

// ======================== 初始化 ========================

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initImageUpload();
    initVideoUpload();
    initWebcam();
    initSettings();
    loadModels();
});

// ======================== Tab 切换 ========================

function initTabs() {
    document.querySelectorAll(".tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            currentTab = tab.dataset.tab;

            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            document.getElementById(`tab-${currentTab}`).classList.add("active");

            // 切换 tab 时停止摄像头
            if (currentTab !== "webcam" && webcamStream) {
                stopWebcam();
            }
        });
    });
}

// ======================== Toast 提示 ========================

function showToast(msg, type = "") {
    const toast = document.getElementById("toast");
    toast.textContent = msg;
    toast.className = "toast " + type + " show";
    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => toast.classList.remove("show"), 2800);
}

// ======================== 设置面板 ========================

function initSettings() {
    const confSlider = document.getElementById("conf-slider");
    const iouSlider = document.getElementById("iou-slider");
    document.getElementById("conf-value").textContent = parseFloat(confSlider.value).toFixed(2);
    document.getElementById("iou-value").textContent = parseFloat(iouSlider.value).toFixed(2);

    confSlider.addEventListener("input", () => {
        document.getElementById("conf-value").textContent = parseFloat(confSlider.value).toFixed(2);
    });
    iouSlider.addEventListener("input", () => {
        document.getElementById("iou-value").textContent = parseFloat(iouSlider.value).toFixed(2);
    });

    document.getElementById("btn-apply").addEventListener("click", async () => {
        const modelName = document.getElementById("model-select").value;
        const conf = parseFloat(confSlider.value);
        const iou = parseFloat(iouSlider.value);

        try {
            const res = await fetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model_name: modelName,
                    conf_threshold: conf,
                    iou_threshold: iou,
                }),
            });
            const data = await res.json();
            document.getElementById("model-badge").textContent = data.model_name;
            showToast("✅ 设置已更新");
        } catch (e) {
            showToast("❌ 设置更新失败", "error");
        }
    });
}

async function loadModels() {
    try {
        const res = await fetch("/api/models");
        const data = await res.json();
        document.getElementById("model-badge").textContent = data.current;
        document.getElementById("model-select").value = data.current;
    } catch (e) {
        // ignore
    }
}

// ======================== 拖拽上传辅助 ========================

function setupDragDrop(area, fileInput, callback) {
    area.addEventListener("click", () => fileInput.click());

    area.addEventListener("dragover", e => {
        e.preventDefault();
        area.classList.add("drag-over");
    });
    area.addEventListener("dragleave", () => area.classList.remove("drag-over"));
    area.addEventListener("drop", e => {
        e.preventDefault();
        area.classList.remove("drag-over");
        const files = e.dataTransfer.files;
        if (files.length > 0) callback(files[0]);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) callback(fileInput.files[0]);
    });
}

// ======================== 显示统计 ========================

function showStats(detections) {
    const panel = document.getElementById("stats-panel");
    panel.style.display = "block";
    const content = document.getElementById("stats-content");

    const countByClass = {};
    detections.forEach(d => {
        countByClass[d.class_name] = (countByClass[d.class_name] || 0) + 1;
    });

    let html = `<p>📦 共检测到 <strong>${detections.length}</strong> 个目标</p>`;
    html += '<ul style="margin-top:8px;list-style:none;font-size:0.82rem;">';
    for (const [name, count] of Object.entries(countByClass)) {
        html += `<li style="padding:2px 0;">• ${name}: <strong>${count}</strong></li>`;
    }
    html += '</ul>';
    content.innerHTML = html;
}

// ======================== 图片检测 ========================

function initImageUpload() {
    const area = document.getElementById("image-upload-area");
    const fileInput = document.getElementById("image-file-input");
    const loading = document.getElementById("image-loading");
    const results = document.getElementById("image-results");

    setupDragDrop(area, fileInput, async (file) => {
        // 隐藏上传区，显示加载
        area.style.display = "none";
        results.style.display = "none";
        loading.style.display = "block";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/detect/image", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();

            if (data.error) {
                showToast("❌ " + data.error, "error");
                area.style.display = "block";
                loading.style.display = "none";
                return;
            }

            document.getElementById("original-img").src = "data:image/jpeg;base64," + data.original;
            document.getElementById("result-img").src = "data:image/jpeg;base64," + data.result;

            results.style.display = "flex";
            loading.style.display = "none";
            area.style.display = "block";

            showStats(data.detections);
            showToast(`✅ 检测完成，发现 ${data.count} 个目标`);
        } catch (e) {
            showToast("❌ 检测失败: " + e.message, "error");
            area.style.display = "block";
            loading.style.display = "none";
        }
    });
}

// ======================== 视频检测 ========================

function initVideoUpload() {
    const area = document.getElementById("video-upload-area");
    const fileInput = document.getElementById("video-file-input");
    const loading = document.getElementById("video-loading");
    const results = document.getElementById("video-results");
    const progress = document.getElementById("video-progress");

    setupDragDrop(area, fileInput, async (file) => {
        area.style.display = "none";
        results.style.display = "none";
        loading.style.display = "block";
        progress.textContent = "正在上传并逐帧检测，大视频可能需要几分钟...";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/detect/video", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();

            if (data.error) {
                showToast("❌ " + data.error, "error");
                area.style.display = "block";
                loading.style.display = "none";
                return;
            }

            const video = document.getElementById("result-video");
            video.src = data.video_url;
            video.load();

            const download = document.getElementById("video-download");
            download.href = data.video_url;
            download.download = "detected_" + data.filename;

            results.style.display = "flex";
            loading.style.display = "none";
            area.style.display = "block";

            document.getElementById("stats-content").innerHTML =
                `<p>📹 ${data.frames} 帧</p>
                 <p>🎯 ${data.total_detections} 次检测</p>
                 <p>⚡ ${data.fps} FPS (原始)</p>`;
            document.getElementById("stats-panel").style.display = "block";

            showToast(`✅ 视频处理完成！共 ${data.frames} 帧，${data.total_detections} 次检测`);
        } catch (e) {
            showToast("❌ 视频处理失败: " + e.message, "error");
            area.style.display = "block";
            loading.style.display = "none";
        }
    });
}

// ======================== 实时摄像头 ========================

function initWebcam() {
    const btnStart = document.getElementById("btn-webcam-start");
    const btnStop = document.getElementById("btn-webcam-stop");
    const status = document.getElementById("webcam-status");

    btnStart.addEventListener("click", startWebcam);
    btnStop.addEventListener("click", stopWebcam);
}

async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "environment" },
        });

        const video = document.getElementById("webcam-video");
        video.srcObject = webcamStream;

        document.getElementById("btn-webcam-start").disabled = true;
        document.getElementById("btn-webcam-stop").disabled = false;
        document.getElementById("webcam-status").textContent = "▶ 运行中";
        document.getElementById("webcam-info").style.display = "block";

        // 等待视频元数据加载
        await new Promise(resolve => video.addEventListener("loadedmetadata", resolve, { once: true }));
        video.play();

        const canvas = document.getElementById("webcam-canvas");
        const resultCanvas = document.getElementById("webcam-result-canvas");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resultCanvas.width = video.videoWidth;
        resultCanvas.height = video.videoHeight;

        webcamFpsCounter = 0;
        webcamFpsTimer = performance.now();

        // 每 200ms 发送一帧检测
        webcamInterval = setInterval(() => detectWebcamFrame(video, canvas, resultCanvas), 200);

        showToast("📷 摄像头已开启");
    } catch (e) {
        showToast("❌ 无法访问摄像头: " + e.message, "error");
        console.error(e);
    }
}

function stopWebcam() {
    if (webcamInterval) {
        clearInterval(webcamInterval);
        webcamInterval = null;
    }
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
    document.getElementById("webcam-video").srcObject = null;
    document.getElementById("btn-webcam-start").disabled = false;
    document.getElementById("btn-webcam-stop").disabled = true;
    document.getElementById("webcam-status").textContent = "已停止";
    document.getElementById("webcam-info").style.display = "none";
}

async function detectWebcamFrame(video, canvas, resultCanvas) {
    if (video.readyState < 2) return;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const frameData = canvas.toDataURL("image/jpeg", 0.8);

    try {
        const res = await fetch("/api/detect/webcam", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: frameData }),
        });
        const data = await res.json();

        // 显示检测画面
        if (data.result) {
            const img = new Image();
            img.onload = () => {
                resultCanvas.width = img.width;
                resultCanvas.height = img.height;
                resultCanvas.getContext("2d").drawImage(img, 0, 0);
            };
            img.src = "data:image/jpeg;base64," + data.result;
        }

        // 更新检测列表
        if (data.detections && data.detections.length > 0) {
            const list = document.getElementById("webcam-detections-list");
            list.innerHTML = data.detections.map(d =>
                `<li>${d.class_name} — <strong>${(d.confidence * 100).toFixed(0)}%</strong></li>`
            ).join("");
        } else {
            document.getElementById("webcam-detections-list").innerHTML = "<li>未检测到目标</li>";
        }

        // FPS 计算
        webcamFpsCounter++;
        const now = performance.now();
        const elapsed = now - webcamFpsTimer;
        if (elapsed >= 1000) {
            const fps = Math.round(webcamFpsCounter / (elapsed / 1000));
            document.getElementById("webcam-fps").textContent = fps;
            webcamFpsCounter = 0;
            webcamFpsTimer = now;
        }
    } catch (e) {
        // 静默处理帧检测失败
    }
}
