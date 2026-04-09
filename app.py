from flask import Flask, render_template, jsonify, Response
import os
import time
import cv2
from main import VisionBackground

app = Flask(__name__)

vision = None
vision_init_error = None


def get_vision():
    global vision, vision_init_error
    if vision is not None:
        return vision
    if vision_init_error is not None:
        raise RuntimeError(vision_init_error)

    try:
        vision = VisionBackground()
        return vision
    except Exception as exc:
        vision_init_error = f"Backend initialization failed: {exc}"
        raise RuntimeError(vision_init_error) from exc

def speak(text):
    try:
        local_vision = get_vision()
    except RuntimeError:
        return

    if hasattr(local_vision, 'speech_queue'):
        local_vision.speech_queue.put(text)
        local_vision.latest_speech_text = text

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        try:
            local_vision = get_vision()
        except RuntimeError:
            time.sleep(0.2)
            continue

        if local_vision.running and getattr(local_vision, 'current_frame', None) is not None:
            ret, buffer = cv2.imencode('.jpg', local_vision.current_frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)
        else:
            time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    try:
        local_vision = get_vision()
    except RuntimeError as exc:
        return jsonify({
            "status": "Failed to start",
            "running": False,
            "latest_speech": "Waiting for detection...",
            "error": str(exc),
        }), 500

    local_vision.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if local_vision.camera_ready:
            speak("Vision system starting")
            return jsonify({
                "status": "Started",
                "running": True,
                "latest_speech": local_vision.latest_speech_text,
                "error": None,
            })
        if local_vision.camera_error:
            return jsonify({
                "status": "Failed to start",
                "running": False,
                "latest_speech": local_vision.latest_speech_text,
                "error": local_vision.camera_error,
            }), 500
        time.sleep(0.1)

    return jsonify({
        "status": "Starting",
        "running": local_vision.running,
        "latest_speech": local_vision.latest_speech_text,
        "error": local_vision.camera_error,
    })

@app.route('/stop', methods=['POST'])
def stop():
    try:
        local_vision = get_vision()
    except RuntimeError:
        return jsonify({
            "status": "Stopped",
            "running": False,
            "latest_speech": "Waiting for detection...",
            "error": None,
        })

    local_vision.stop()
    speak("Vision system stopped")
    return jsonify({
        "status": "Stopped",
        "running": False,
        "latest_speech": "Waiting for detection...",
        "error": None,
    })

@app.route('/status', methods=['GET'])
def status():
    if vision_init_error:
        return jsonify({
            "running": False,
            "camera_ready": False,
            "latest_speech": "Waiting for detection...",
            "error": vision_init_error,
        })

    if vision is None:
        return jsonify({
            "running": False,
            "camera_ready": False,
            "latest_speech": "Waiting for detection...",
            "error": None,
        })

    return jsonify({
        "running": vision.running,
        "camera_ready": getattr(vision, 'camera_ready', False),
        "latest_speech": getattr(vision, 'latest_speech_text', 'Waiting for detection...'),
        "error": getattr(vision, 'camera_error', None),
    })

if __name__ == '__main__':
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    print(f"Combined Web Backend running on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
