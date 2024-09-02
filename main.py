from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import cvzone
import math

app = Flask(__name__)
CORS(app)  # Enable CORS so React can communicate with Flask

# Load YOLO model
model = YOLO('./AntiSpoofingDetector/n_version_1_30.pt')

# Class names
classNames = ["fake", "real"]

# Confidence threshold
confidence = 0.8

@app.route('/')
def home():
    return 'API is running'

@app.route('/run-model', methods=['POST'])
def run_model():
    # Initialize video capture and set resolution
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    results_list = []  # To store results for each frame

    while True:
        success, img = cap.read()
        if not success:
            break

        # Perform object detection using YOLO
        results = model(img, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = box.cls[0]

                name = classNames[int(cls)].upper()

                if conf > confidence:
                    color = (0, 255, 0) if name == "REAL" else (0, 0, 255)
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(img, f'{name} {int(conf*100)}%', (max(0, x1), max(35, y1)), scale=2, thickness=2, colorR=color, colorB=color)

                    # Store results
                    results_list.append({
                        'label': name,
                        'confidence': conf,
                        'box': [x1, y1, x2, y2]
                    })

        # Display the frame (Optional, for debugging or visualization)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return results as JSON
    return jsonify(results_list)

if __name__ == '__main__':
    app.run(debug=True)
