from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
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

# Definir o modelo
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(4,)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Salvar o modelo no formato Keras
model.save('modelo_ia.keras')
print("Modelo salvo como 'modelo_ia.keras'")
