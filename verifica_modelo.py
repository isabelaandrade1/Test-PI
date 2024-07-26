from tensorflow.keras.models import load_model
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Carregar e pré-processar dados (usando dataset de exemplo Iris)
data = load_iris()
X = data.data
y = tf.keras.utils.to_categorical(data.target, num_classes=3)

# Dividir os dados em conjuntos de treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Carregar o modelo salvo
try:
    model = load_model('modelo_ia.keras')
    print("Modelo carregado com sucesso.")
except OSError as e:
    print(f"Erro ao carregar o arquivo: {e}")

# Avaliar o modelo
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Fazer previsões usando o modelo carregado
predicoes = model.predict(X_val)
predicoes_classes = tf.argmax(predicoes, axis=1)

print("Previsões para o conjunto de validação:")
print(predicoes_classes)
