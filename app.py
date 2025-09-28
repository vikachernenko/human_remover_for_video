import os
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from threading import Thread
from io import BytesIO

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

current_frame = None
progress = {"analyze": 0, "render": 0, "total": 0, "done": False}


class HumanRemover:
    def __init__(self, weights="yolov8s.pt", imgsz=640, conf=0.25,
                 min_area=1000, max_area=1e7, skip=1, pad=0.5):
        self.model = YOLO(weights)
        self.p = dict(imgsz=imgsz, conf=conf, min_area=min_area,
                      max_area=max_area, skip=skip, pad=pad)

    def detect(self, frame):
        res = self.model(frame, imgsz=self.p["imgsz"],
                         conf=self.p["conf"], classes=[0])
        boxes = []
        for b in res[0].boxes:
            try:
                x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy().flatten())
                area = (x2 - x1) * (y2 - y1)
                if self.p["min_area"] <= area <= self.p["max_area"]:
                    boxes.append((x1, y1, x2, y2))
            except Exception:
                continue
        return boxes

    def process(self, video_path, output_path):
        global current_frame, progress
        progress["done"] = False

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress["total"] = total
        mask = np.ones(total, bool)

        # ---- Анализ ----
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % self.p["skip"] == 0 and self.detect(frame):
                mask[idx] = False
            color = (0, 255, 0) if mask[idx] else (0, 0, 255)
            for box in self.detect(frame):
                cv2.rectangle(frame, (box[0], box[1]),
                              (box[2], box[3]), color, 2)
            current_frame = frame.copy()
            progress["analyze"] = idx
            idx += 1
        cap.release()

        # ---- Рендер ----
        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or idx >= total:
                break
            if mask[idx]:
                out.write(frame)
            color = (0, 255, 0) if mask[idx] else (0, 0, 255)
            for box in self.detect(frame):
                cv2.rectangle(frame, (box[0], box[1]),
                              (box[2], box[3]), color, 2)
            current_frame = frame.copy()
            progress["render"] = idx
            idx += 1
        cap.release()
        out.release()

        # ---- Файл готов ----
        progress["done"] = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        output_name = f"result_{filename}"
        output_path = os.path.join(RESULT_FOLDER, output_name)

        # Получаем настройки с формы
        conf = float(request.form.get("conf", 0.25))
        imgsz = int(request.form.get("imgsz", 640))
        skip = int(request.form.get("skip", 1))
        min_area = int(request.form.get("min_area", 1000))
        max_area = int(request.form.get("max_area", int(1e7)))

        Thread(target=lambda: HumanRemover(
            imgsz=imgsz, conf=conf, skip=skip,
            min_area=min_area, max_area=max_area
        ).process(path, output_path), daemon=True).start()
        return jsonify({"filename": filename, "output": output_name})
    return "No file", 400


@app.route("/download/<filename>")
def download(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    return "File not found", 404


@app.route("/preview_frame")
def preview_frame():
    global current_frame
    if current_frame is not None:
        _, buffer = cv2.imencode(".jpg", current_frame)
        return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")
    return "No frame", 404


@app.route("/progress")
def get_progress():
    return jsonify(progress)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
