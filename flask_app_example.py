
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model_bundle = joblib.load('C:\\Users\\hkaluuma\\Desktop\\Machine_learning\\best_iris_model.joblib')
model = model_bundle['model']
le = model_bundle['label_encoder']

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.json
    # Expect JSON with either 'features' (list) or individual keys
    if 'features' in payload:
        arr = np.array(payload['features']).reshape(1, -1)
    else:
        arr = np.array([payload[k] for k in ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]).reshape(1, -1)
    pred = model.predict(arr)[0]
    label = le.inverse_transform([pred])[0]
    return jsonify({'species': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
