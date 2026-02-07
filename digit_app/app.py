from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("lenet_mnist.h5")

LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(image_path):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert("L")
    img = np.array(img)

    # ---------- SHADOW REMOVAL ----------
    # Adaptive threshold handles uneven lighting
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # ---------- MORPHOLOGICAL CLEAN ----------
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # ---------- BOUNDING BOX ----------
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    if not rows.any() or not cols.any():
        return np.zeros((1, 28, 28, 1), dtype="float32")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    img = img[rmin:rmax+1, cmin:cmax+1]

    # ---------- RESIZE TO 20x20 ----------
    h, w = img.shape
    if h > w:
        new_h = 20
        new_w = int(20 * w / h)
    else:
        new_w = 20
        new_h = int(20 * h / w)

    img = cv2.resize(img, (new_w, new_h))

    # ---------- CENTER IN 28x28 ----------
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

    # ---------- NORMALIZE ----------
    canvas = canvas.astype("float32") / 255.0
    canvas = canvas.reshape(1, 28, 28, 1)

    return canvas

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img = preprocess_image(image_path)
            preds = model.predict(img, verbose=0)

            digit = np.argmax(preds)
            confidence = float(np.max(preds)) * 100
            prediction = LABELS[digit]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
