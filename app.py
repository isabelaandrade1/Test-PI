from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Carregue o seu modelo treinado
model = tf.keras.models.load_model('modelo_ia.keras')

# Rota principal para verificar se o servidor está funcionando
@app.route('/')
def home():
    return "Servidor Flask está rodando!"

# Rota para renderizar uma página HTML
@app.route('/pagina')
def pagina():
    return render_template('pagina.html')

# Rota para previsão usando o modelo treinado
@app.route('/prever', methods=['POST'])
def prever():
    dados = request.get_json()
    previsao = model.predict([dados['input']])
    return jsonify({'previsao': previsao.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
