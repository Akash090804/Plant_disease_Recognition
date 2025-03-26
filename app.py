import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import requests
import tensorflow as tf

app = Flask(__name__)

MODEL_URL = "https://github.com/Akash090804/Plant_disease_Recognition/releases/download/v1.0/plant_disease_recog_model_pwp.keras"
MODEL_PATH = "models/plant_disease_recog_model_pwp.keras"

os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

model = tf.keras.models.load_model(MODEL_PATH)

with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route("/upload/", methods=["POST"])
def uploadimage():
    image = request.files["img"]
    temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
    image_path = f"{temp_name}_{image.filename}"
    image.save(image_path)
    prediction = model_predict(image_path)
    return render_template("home.html", result=True, imagepath=f"/{image_path}", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
