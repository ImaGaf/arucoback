import cv2
import numpy as np
from flask import Flask, request, jsonify
import io
from PIL import Image
import base64
from flask_cors import CORS
from object_detector import HomogeneousBgDetector

print(cv2.__version__)

app = Flask(__name__)
CORS(app)

parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_5X5_100, 5)

detector = HomogeneousBgDetector()

@app.route('/process_image', methods=['POST'])
def processimage():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    in_memory_file = io.BytesIO(file.read())
    img = Image.open(in_memory_file)
    img = np.array(img)
    
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    if ids is None:
        return jsonify({'error': 'No Aruco marker detected'}), 400

    cv2.aruco.drawDetectedMarkers(img, corners, ids)

    aruco_size_cm = 3.3

    side_length_pixels = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
    pixel_cm_ratio = side_length_pixels / aruco_size_cm

    img_height, img_width = img.shape[:2]
    center_x, center_y = img_width // 2, img_height // 2

    contours = detector.detect_objects(img)
    
    if not contours:
        return jsonify({'error': 'No objects detected'}), 400

    closest_object = None
    min_distance = float('inf')

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect

        object_width = w / pixel_cm_ratio
        object_height = h / pixel_cm_ratio
        
        object_center_x, object_center_y = int(x), int(y)
        distance = np.sqrt((object_center_x - center_x) ** 2 + (object_center_y - center_y) ** 2)

        if distance < min_distance:
            min_distance = distance
            closest_object = {'width': object_width, 'height': object_height}

        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.polylines(img, [box], True, (255, 0, 0), 2)
        
        cv2.putText(img, "Ancho: {} cm".format(round(object_width, 1)), 
                        (int(x), int(y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (150, 0, 255), 7)
        cv2.putText(img, "Alto: {} cm".format(round(object_height, 1)), 
                        (int(x), int(y + 15)), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (200, 0, 200), 7)

    if not closest_object:
        return jsonify({'error': 'No objects close to center detected'}), 400

    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    return jsonify({
        'image': base64.b64encode(img_bytes).decode('utf-8'),  
        'dimensions': closest_object 
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
