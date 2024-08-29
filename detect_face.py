from flask import Flask, request, jsonify
import face_recognition
import joblib
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained KNN model
knn_clf = joblib.load('knn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = face_recognition.load_image_file(file)
    img_encoding = face_recognition.face_encodings(img)

    if len(img_encoding) == 0:
        return jsonify({'error': 'No face detected'}), 400

    face_encoding = img_encoding[0]
    matches = knn_clf.predict([face_encoding])
    return jsonify({'name': matches[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
