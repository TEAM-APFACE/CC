from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('model.h5')
model_skin = load_model('skin_type_model.h5')

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_path = request.json.get('img_path')
        if not img_path:
            return jsonify({'error': 'img_path is required'}), 400
        img_array = load_and_preprocess_image(img_path)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])

        class_labels =  ['Acne', 'Comedo', 'Freckles', 'Redness'];

        if predicted_class_index < len(class_labels):
            predicted_class = class_labels[predicted_class_index]
        else:
            predicted_class = "Unknown Class"

        return jsonify({'predicted_condition': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_skin', methods=['POST'])
def predict_skin():
    try:
        img_path = request.json.get('img_path')
        if not img_path:
            return jsonify({'error': 'img_path is required'}), 400
        img_array = load_and_preprocess_image(img_path)

        predictions = model_skin.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])

        class_labels = ['Dry', 'Normal', 'Oily']

        if predicted_class_index < len(class_labels):
            predicted_class = class_labels[predicted_class_index]
        else:
            predicted_class = "Unknown Class"

        return jsonify({'predicted_condition': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return 'App Opened'

if __name__ == '__main__':
    app.run(debug=True, port=5001)
