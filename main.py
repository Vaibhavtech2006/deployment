from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

app = Flask(__name__)
CORS(app)  # Enable CORS so React can communicate with Flask

# Load YOLO model
model = YOLO('./AntiSpoofingDetector/n_version_1_30.pt')

# Class names
classNames = ["fake", "real"]

# Confidence threshold
confidence = 0.7

# Global variable to store the result
latest_result = None

def process_model():
    # Initialize video capture and set resolution
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detected_classes = []
    highest_confidences = []

    start_time = time.time()
    run_duration = 4.5 # Duration to run the model in seconds

    while time.time() - start_time < run_duration:
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
                    # Update the detected class if the confidence is higher
                    detected_classes.append(name)
                    highest_confidences.append(conf)

                    color = (0, 255, 0) if name == "REAL" else (0, 0, 255)
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(img, f'{name} ', (max(0, x1), max(35, y1)), scale=2, thickness=2, colorR=color, colorB=color)

        # Display the frame (Optional, for debugging or visualization)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return the result as a list of dictionaries
    result_array = [{'label': detected_classes[i], 'confidence': highest_confidences[i]} for i in range(len(detected_classes))]
    return result_array

@app.route('/')
def home():
    global latest_result
    if latest_result is None:
        return 'No results yet. Use /run-model to start the model.'

    # Render the results in a simple HTML format
    result_html = "<h2>Latest Model Result</h2><ul>"
    for res in latest_result:
        result_html += f"<li>{res['label']}: {res['confidence']}</li>"
    result_html += "</ul>"

    return render_template_string(result_html)

@app.route('/run-model', methods=['POST'])
def run_model():
    global latest_result
    # Call the model processing function and store the result
    latest_result = process_model()
    return jsonify(latest_result)

if __name__ == '__main__':
    app.run(debug=True)
