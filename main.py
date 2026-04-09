import argparse
from pathlib import Path
import cv2
import pyttsx3
from ultralytics import YOLO
import time
import threading
import queue

# -------------------------------
# CONSTANTS
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "yolov8n.pt"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

FOCAL_LENGTH = 800          # camera focal length
REAL_OBJECT_WIDTH = 0.5     # meters (average assumption)
SPEECH_INTERVAL = 1.0       # seconds

engine = None
VOICE_ENABLED = False
_global_engine_initialized = False

def init_global_engine():
    global engine, VOICE_ENABLED, _global_engine_initialized
    if _global_engine_initialized:
        return
    _global_engine_initialized = True
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        VOICE_ENABLED = True
    except Exception as exc:
        engine = None
        VOICE_ENABLED = False
        print(f"Warning: voice engine unavailable, continuing without speech. {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Object Detection with dataset / source support"
    )
    parser.add_argument(
        '--source',
        choices=['camera', 'image', 'video', 'folder'],
        default='camera',
        help='Input source type: camera, image, video, or folder'
    )
    parser.add_argument(
        '--path',
        type=str,
        default=None,
        help='Path to the input image, video file, or folder of images'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help='Path to the YOLO model file'
    )
    parser.add_argument(
        '--focal-length',
        type=float,
        default=FOCAL_LENGTH,
        help='Camera focal length for distance estimation'
    )
    parser.add_argument(
        '--real-width',
        type=float,
        default=REAL_OBJECT_WIDTH,
        help='Real-world width of the detected object in meters'
    )
    return parser.parse_args()


def get_image_paths(folder_path):
    folder = Path(folder_path)
    image_paths = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(sorted(folder.glob(pattern)))
    return image_paths


def open_camera():
    """Try several camera indexes/backends and allow time for webcam warm-up."""
    backend_options = [("default", None)]
    for backend_name, backend_value in (
        ("directshow", getattr(cv2, "CAP_DSHOW", None)),
        ("msmf", getattr(cv2, "CAP_MSMF", None)),
        ("vfw", getattr(cv2, "CAP_VFW", None)),
    ):
        if backend_value is not None:
            backend_options.append((backend_name, backend_value))

    camera_attempts = []
    for camera_index in range(3):
        for backend_name, backend in backend_options:
            camera_attempts.append((camera_index, backend_name, backend))

    failures = []

    for camera_index, backend_name, backend in camera_attempts:
        try:
            if backend is None:
                cap = cv2.VideoCapture(camera_index)
            else:
                cap = cv2.VideoCapture(camera_index, backend)
        except Exception as exc:
            failures.append(f"index {camera_index} ({backend_name}): {exc}")
            continue

        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            failures.append(f"index {camera_index} ({backend_name}): open failed")
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        frame_received = False
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size:
                frame_received = True
                break
            time.sleep(0.1)

        if frame_received:
            print(f"Using camera index {camera_index} with backend: {backend_name}")
            return cap

        cap.release()
        failures.append(f"index {camera_index} ({backend_name}): no frames received")

    failure_summary = "; ".join(failures[:8])
    raise RuntimeError(
        "Unable to open webcam. Close other camera apps, allow camera access in Windows settings, "
        f"and try again. Attempts: {failure_summary}"
    )


def draw_boxes(frame, results, model, focal_length, real_width):
    current_objects = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_width = max(1, x2 - x1)
            distance = (real_width * focal_length) / box_width
            distance = round(distance, 2)

            current_objects.append({
                "label": label,
                "distance": distance,
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} - {distance}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return current_objects


def speak_all_objects(current_objects):
    """Speak about all detected objects"""
    init_global_engine()
    if not VOICE_ENABLED:
        return

    if not current_objects:
        return

    unique_objects = {}
    for obj in current_objects:
        label = obj['label']
        # Keep only the closest distance for each object label
        if label not in unique_objects or obj['distance'] < unique_objects[label]:
            unique_objects[label] = obj['distance']

    speech_text = ". ".join(
        f"Detected {label} at {distance} meters"
        for label, distance in unique_objects.items()
    ) + "."

    engine.say(speech_text)
    engine.runAndWait()


def speak_once_per_interval(current_objects, last_speech_time, interval=SPEECH_INTERVAL):
    if not current_objects:
        return last_speech_time
    current_time = time.monotonic()
    if (current_time - last_speech_time) >= interval:
        speak_all_objects(current_objects)
        return current_time
    return last_speech_time


def process_frame(frame, model, previous_objects, focal_length, real_width):
    results = model(frame, verbose=False)
    current_objects = draw_boxes(frame, results, model, focal_length, real_width)
    return frame, current_objects


def main():
    args = parse_args()
    model = YOLO(args.model)

    previous_objects = []

    print("Smart Object Detection Started")
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    if args.path:
        print(f"Path: {args.path}")

    if args.source == 'camera':
        cap = open_camera()

        last_speech_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, current_objects = process_frame(
                frame,
                model,
                previous_objects,
                args.focal_length,
                args.real_width,
            )
            previous_objects = current_objects
            
            cv2.imshow("Smart Assistive Vision System", frame)
            last_speech_time = speak_once_per_interval(current_objects, last_speech_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    elif args.source == 'video':
        if not args.path:
            raise ValueError('Video path is required when source is video.')

        cap = cv2.VideoCapture(args.path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {args.path}")

        last_speech_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, current_objects = process_frame(
                frame,
                model,
                previous_objects,
                args.focal_length,
                args.real_width,
            )
            previous_objects = current_objects

            cv2.imshow("Smart Assistive Vision System", frame)
            last_speech_time = speak_once_per_interval(current_objects, last_speech_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    elif args.source == 'image':
        if not args.path:
            raise ValueError('Image path is required when source is image.')

        image_path = Path(args.path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {args.path}")

        frame = cv2.imread(str(image_path))
        frame, current_objects = process_frame(
            frame,
            model,
            previous_objects,
            args.focal_length,
            args.real_width,
        )
        
        # Speak about detected objects
        speak_all_objects(current_objects)
        
        cv2.imshow("Smart Assistive Vision System", frame)
        cv2.waitKey(0)

    elif args.source == 'folder':
        if not args.path:
            raise ValueError('Folder path is required when source is folder.')

        image_paths = get_image_paths(args.path)
        if not image_paths:
            raise FileNotFoundError(f"No images found in folder: {args.path}")

        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            frame, current_objects = process_frame(
                frame,
                model,
                previous_objects,
                args.focal_length,
                args.real_width,
            )
            
            last_speech_time = 0.0
            previous_objects = current_objects

            display_start = time.monotonic()
            while (time.monotonic() - display_start) < 2.0:
                cv2.imshow("Smart Assistive Vision System", frame)
                last_speech_time = speak_once_per_interval(current_objects, last_speech_time)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            if key & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

class VisionBackground:
    def __init__(self, model_path=str(DEFAULT_MODEL_PATH)):
        self.model = YOLO(model_path)
        self.cap = None
        self.running = False
        self.camera_ready = False
        self.camera_error = None
        self.last_speech_time = 0.0
        self.current_frame = None
        self.speech_queue = queue.Queue()
        self.latest_speech_text = "Waiting for detection..."
        self._thread = None
        # Start speech thread once reliably
        threading.Thread(target=self._speech_worker, daemon=True).start()

    def start(self):
        if self.running:
            return

        self.running = True
        self.camera_ready = False
        self.camera_error = None
        self.last_speech_time = 0.0
        self.latest_speech_text = "Starting camera..."

        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self.camera_ready = False
        self.current_frame = None

    def _speech_worker(self):
        # Initialize pyttsx3 in the thread to avoid COM crashes on Windows
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        try:
            local_engine = pyttsx3.init()
            local_engine.setProperty('rate', 150)
            voice_enabled = True
        except Exception:
            local_engine = None
            voice_enabled = False

        while True:
            try:
                item = self.speech_queue.get(timeout=0.1)
                
                # Drain the queue to ensure we only speak the freshest detection frame
                if not isinstance(item, str):
                    while not self.speech_queue.empty():
                        try:
                            item = self.speech_queue.get_nowait()
                            if isinstance(item, str):
                                break
                        except queue.Empty:
                            break

                if not voice_enabled or not item:
                    continue
                    
                if isinstance(item, str):
                    text = item
                else:
                    objects = item
                    unique_objects = {}
                    for obj in objects:
                        lbl = obj['label']
                        if lbl not in unique_objects or obj['distance'] < unique_objects[lbl]:
                            unique_objects[lbl] = obj['distance']
                    text = ". ".join(f"Detected {l} at {d} meters" for l, d in unique_objects.items()) + "."
                    
                local_engine.say(text)
                local_engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                print("Speech worker error:", e)

    def _loop(self):
        previous_objects = []

        def local_speak_interval(objects, last_time, interval=SPEECH_INTERVAL):
            if not objects:
                return last_time
            curr_time = time.monotonic()
            if (curr_time - last_time) >= interval:
                self.speech_queue.put(objects)

                unique_objects = {}
                for obj in objects:
                    lbl = obj['label']
                    if lbl not in unique_objects or obj['distance'] < unique_objects[lbl]:
                        unique_objects[lbl] = obj['distance']
                self.latest_speech_text = ". ".join(
                    f"Detected {l} at {d} meters" for l, d in unique_objects.items()
                ) + "."

                return curr_time
            return last_time

        try:
            self.cap = open_camera()
            self.camera_ready = True
            self.latest_speech_text = "Camera opened. Waiting for objects..."

            while self.running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.camera_error = "Camera opened but no frames were received."
                    self.latest_speech_text = self.camera_error
                    break

                frame, current_objects = process_frame(
                    frame, self.model, previous_objects, FOCAL_LENGTH, REAL_OBJECT_WIDTH
                )

                self.current_frame = frame.copy()
                previous_objects = current_objects

                if current_objects:
                    self.last_speech_time = local_speak_interval(
                        current_objects, self.last_speech_time
                    )
                elif self.camera_ready:
                    self.latest_speech_text = "Camera opened. Waiting for objects..."

                time.sleep(0.03)
        except Exception as exc:
            self.camera_error = str(exc)
            self.latest_speech_text = f"Camera error: {exc}"
            print(f"Vision background error: {exc}")
        finally:
            self.running = False
            self.camera_ready = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.current_frame = None

if __name__ == '__main__':
    main()
