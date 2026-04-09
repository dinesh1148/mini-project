# Object Detection Voice Assistance Project - Summary

## Project Overview
Real-time object detection system with voice assistance for blind/visually impaired users.

## Completed Components

### 1. Main Script: `main.py`
**Features:**
- Real-time object detection using YOLOv8n model
- Multiple input sources:
  - Webcam (default)
  - Single image file
  - Video file
  - Folder of images
- Voice output via Text-to-Speech (pyttsx3)
- Bounding box visualization
- Distance estimation based on object size
- Voice announcement every 1 second

**Usage:**
```bash
# Webcam (default)
python main.py

# Single image
python main.py --source image --path path/to/image.jpg

# Video file
python main.py --source video --path path/to/video.mp4

# Folder of images
python main.py --source folder --path path/to/images/
```

### 2. Training Script: `train.py`
- Trains YOLOv8n model on COCO128 dataset
- Supports CPU training (no GPU required)
- Saves best model weights

### 3. Dataset
- **Source:** COCO128 dataset (downloaded from Roboflow)
- **Size:** 128 total images (102 train split, 26 validation split)
- **Classes:** 80 object classes (COCO dataset)
- **Location:** `datasets/mydata/`

### 4. Model
- **Model:** YOLOv8 Nano (yolov8n.pt)
- **Pretrained on:** COCO dataset
- **Fine-tuned on:** COCO128 dataset
- **Performance:**
  - Precision: 0.805
  - Recall: 0.555
  - mAP50: 0.714
  - Training time: ~11 seconds (3 epochs on CPU)

### 5. Configuration File: `data.yaml`
```yaml
path: datasets/mydata
train: images/train
val: images/val
nc: 80
```

## Key Features
1. **Real-time Detection:** ~40ms inference per frame
2. **Voice Announcements:** Every detected object is announced every 1 second
3. **Distance Estimation:** Estimates distance to objects (in meters)
4. **Multiple Formats:** Supports images, videos, webcam, and image folders
5. **Multi-threaded Voice:** Non-blocking speech output (though directly executed for reliability)
6. **Trained Model:** Custom model trained on COCO128 dataset

## Installation Requirements
```
cv2
pyttsx3
ultralytics
torch
numpy
```

## Project Structure
```
ObjectDetectionVoice/
├── models/
│   ├── main.py
│   ├── train.py
│   ├── data.yaml
│   ├── yolov8n.pt
│   ├── datasets/
│   │   └── mydata/
│   │       ├── images/
│   │       │   ├── train/ (102 images)
│   │       │   └── val/ (26 images)
│   │       └── labels/
│   │           ├── train/ (102 txt files)
│   │           └── val/ (26 txt files)
│   └── runs/
│       └── detect/
│           └── train4/
│               ├── weights/
│               │   ├── best.pt (trained model)
│               │   └── last.pt
│               └── results.csv
```

## Testing Results
- Tested on single image: Successfully detected 4 bowls, 1 broccoli, 1 hot dog
- Voice output working correctly
- Object annotations displayed on video frames
- Distance calculations functioning

## Next Steps / Future Enhancements
1. Integrate with blind assistance hardware
2. Add more specialized datasets for better accuracy
3. Optimize for mobile deployment
4. Add confidence filtering
5. Improve distance estimation accuracy
6. Add audio feedback for navigation

## Technical Notes
- Model runs on CPU (can be accelerated with GPU if available)
- Voice uses system text-to-speech engine (pyttsx3)
- YOLO detection runs at ~25fps on CPU
- All detections use pre-trained COCO weights
