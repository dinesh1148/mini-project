import cv2
import pyttsx3
import time
from ultralytics import YOLO

# Initialize Text-to-Speech Engine
try:
    import pythoncom
    pythoncom.CoInitialize() # Required for some Windows environments
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voice_enabled = True
except Exception as e:
    print(f"Warning: Voice engine could not be initialized. {e}")
    voice_enabled = False

# Load YOLOv8 Model (will auto-download yolov8n.pt if not present)
model = YOLO('yolov8n.pt') 

# Minimum delay between speech outputs (seconds)
SPEECH_INTERVAL = 1.0

def main():
    # Open the default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_speech_time = 0.0

    print("\n" + "="*50)
    print("Starting Object Detection System")
    print("Press 'q' in the video window to quit.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Run YOLO inference on the current frame
        # verbose=False keeps the terminal output clean for our own printed labels
        results = model(frame, verbose=False)
        
        detected_labels = set() # Use a set to avoid duplicating labels if there are multiple of the same object
        
        # Parse results, extract labels, and draw boxes
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get the class name for the object
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                detected_labels.add(label)
                
                # Draw green bounding box and text on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
        # If any objects were detected, handle terminal printing and voice speech
        if detected_labels:
            # 1. Print to terminal
            labels_text = ", ".join(detected_labels)
            print(f"Detected: {labels_text}")
            
            # 2. Convert to voice if interval has passed
            current_time = time.time()
            if voice_enabled and (current_time - last_speech_time >= SPEECH_INTERVAL):
                engine.say(f"I see {labels_text}")
                engine.runAndWait()
                last_speech_time = time.time() # Reset the speech timer

        # Display the resulting frame in OpenCV window
        cv2.imshow("Assistance System for Visually Impaired", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
