import os
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

MODEL_PATH  = "artifacts/training/model.keras"
IMAGE_PATH  = "inputImage.jpg"
IMAGE_SIZE  = (224, 224)
CLASSES     = ["adenocarcinoma", "normal"]   

model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    # Step 1: Decode base64 image and save
    image_data = request.json['image']
    image_bytes = base64.b64decode(image_data)
    with open(IMAGE_PATH, "wb") as f:
        f.write(image_bytes)

    # Step 2: Preprocess
    image = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Step 3: Predict
    preds  = model.predict(image)
    index  = int(np.argmax(preds, axis=1)[0])
    label  = CLASSES[index]

    result = [{"image": label}]
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001)