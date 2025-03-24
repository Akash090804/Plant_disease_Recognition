from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import os
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

label = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
 'Background_without_leaves', 'Blueberry__healthy', 'Cherry_Powdery_mildew', 'Cherry__healthy',
 'Corn__Cercospora_leaf_spot Gray_leaf_spot', 'Corn_Common_rust', 'Corn__Northern_Leaf_Blight',
 'Corn__healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
 'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato__Late_blight',
 'Potato__healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
 'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato__Early_blight',
 'Tomato__Late_blight', 'Tomato_Leaf_Mold', 'Tomato__Septoria_leaf_spot',
 'Tomato__Spider_mites Two-spotted_spider_mite', 'Tomato__Target_Spot',
 'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy']

with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)




@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)



@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')




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



@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image_path = f"{temp_name}_{image.filename}"
        image.save(image_path)
        prediction = model_predict(image_path)
        return render_template('home.html', result=True, imagepath=f'/{image_path}', prediction=prediction)
    else:
        return redirect('/')






if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=10000,debug=True)