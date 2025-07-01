from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import base64
import io
import mediapipe as mp
import os

app = Flask(__name__)

# Pre-load glasses images from static folder
def load_glasses_image(filename):
    img = Image.open(filename).convert("RGBA")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

glasses_images = [
    load_glasses_image("static/glasses/images2.png"),
    load_glasses_image("static/glasses/g_front.png"),
    load_glasses_image("static/glasses/images.png"),
]
current_glasses = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
LEFT_EYE, RIGHT_EYE, NOSE_BRIDGE = 33, 263, 168

def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    overlay_image = img_overlay[y1 - y:y2 - y, x1 - x:x2 - x]
    img_crop = img[y1:y2, x1:x2]
    alpha = overlay_image[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha
    for c in range(0, 3):
        img_crop[:, :, c] = alpha * overlay_image[:, :, c] + alpha_inv * img_crop[:, :, c]
    img[y1:y2, x1:x2] = img_crop
    return img

@app.route("/tryon", methods=["POST"])
def tryon():
    global current_glasses

    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_data = data['image'].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame = np.array(image)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = (int(landmarks[LEFT_EYE].x * w), int(landmarks[LEFT_EYE].y * h))
        right_eye = (int(landmarks[RIGHT_EYE].x * w), int(landmarks[RIGHT_EYE].y * h))
        nose = (int(landmarks[NOSE_BRIDGE].x * w), int(landmarks[NOSE_BRIDGE].y * h))
        eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

        # Resize glasses
        glasses_width = int(eye_dist * 1.8)
        scale = glasses_width / glasses_images[current_glasses].shape[1]
        glasses_height = int(glasses_images[current_glasses].shape[0] * scale)
        resized_glasses = cv2.resize(glasses_images[current_glasses], (glasses_width, glasses_height))

        x = nose[0] - glasses_width // 2
        y = nose[1] - glasses_height // 2 + 15
        frame = overlay_image_alpha(frame, resized_glasses, (x, y))

    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"result": f"data:image/jpeg;base64,{encoded_image}"})


@app.route('/')
def home():
    return "Glasses Try-On API is running"

if __name__ == "__main__":
    app.run(debug=True)
