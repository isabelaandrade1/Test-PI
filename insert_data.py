import tensorflow as tf
from tensorflow.keras.models import Sequential
import numpy as np

# Exemplo de modelo simples
model = Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Salvar modelo para garantir que temos um modelo salvo
model.save('modelo_ia.h5')

# Carregar o modelo salvo
loaded_model = tf.keras.models.load_model('modelo_ia.h5')

# Exemplo de dados de entrada
data = np.array([[0.1, 0.2, 0.3, 0.4]])

# Fazer previsões
predictions = loaded_model.predict(data)
print("Previsões:", predictions)
