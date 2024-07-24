from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Carregar o modelo treinado
model = load_model('modelo_ia.h5')

# Função para preprocessar os dados
def preprocess(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    processed_data = preprocess(df)
    predictions = model.predict(processed_data)
    result = {'predictions': predictions.tolist()}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
