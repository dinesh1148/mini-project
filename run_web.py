#!/usr/bin/env python3
"""
Run the Object Detection Web App

This script starts a Flask web server for object detection.
Make sure training is complete and the model weights are available.
"""

import os
import sys

def main():
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = os.environ.get("FLASK_PORT", "5000")

    print("Starting Object Detection Web App...")
    print("Make sure the trained model is available at: runs/voc2012/train/weights/best.pt")
    print("If not, the app will fall back to yolov8n.pt")
    print()

    # Check if model exists
    model_path = 'runs/voc2012/train/weights/best.pt'
    if os.path.exists(model_path):
        print(f"✓ Found trained model: {model_path}")
    else:
        print(f"⚠ Trained model not found: {model_path}")
        print("  The app will use the default yolov8n.pt model")

    print()
    print(f"Starting Flask server on http://{host}:{port}")
    print("Press Ctrl+C to stop")
    print()

    # Run the Flask app
    os.system("python app.py")

if __name__ == '__main__':
    main()
