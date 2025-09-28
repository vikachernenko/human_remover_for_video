FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY yolov8s.pt .
COPY templates/ ./templates/
COPY static/ ./static/
COPY README.md .

EXPOSE 80

CMD ["python", "app.py"]
