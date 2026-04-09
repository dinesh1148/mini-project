# Object Detection Web App

A Flask-based web interface for the Object Detection with Voice Assistance project.

## Features

- Upload images or videos for object detection
- Real-time detection using YOLOv8
- Display bounding boxes and confidence scores
- Web-based interface accessible from any browser

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements_web.txt
   ```

2. **Ensure model is trained:**
   - Run `python train_voc2012.py` to train on VOC2012 dataset
   - Or use the existing `yolov8n.pt` for COCO classes

3. **Run the web app:**
   ```bash
   python run_web.py
   ```
   Or directly:
   ```bash
   python app.py
   ```

4. **Open browser:**
   - Go to `http://localhost:5000`
   - Upload an image or video file
   - Click "Detect Objects"

## Model

- **Trained model:** `runs/voc2012/train/weights/best.pt` (VOC2012 classes)
- **Fallback model:** `yolov8n.pt` (COCO classes)

The app automatically detects which model to use.

## Supported Formats

- **Images:** PNG, JPG, JPEG, GIF
- **Videos:** MP4, AVI, MOV (processes first frame)

## API Endpoints

- `GET /` - Main page
- `POST /detect` - Upload and detect objects in file

## Notes

- Training is still running in the background
- The web app will work with the current best model
- For best results, wait for training to complete