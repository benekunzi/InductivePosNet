import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('dataprocess.csv', delimiter=',')

scaler = MinMaxScaler()
data[1:,[4,5,6,7,8,9]] = scaler.fit_transform(data[1:,[4,5,6,7,8,9]])

trainset, testset = train_test_split(data[1:, [4,5,6,7,8,9]], test_size=0.2, random_state=0)

X_train = trainset[:,[3, 4, 5]]
X_test = testset[:,[3, 4, 5]]

Y_train = trainset[:,[0, 1, 2]]
Y_test = testset[:,[0, 1, 2]]
print("Y train min", Y_train.min())
print("Y train max", Y_train.max())

input_layer = tf.keras.layers.Input(shape=(3,))  # Entrée : x, y, z
hidden_layer1 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l1_l2')(input_layer)  # Couche intermédiaire
hidden_layer2 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l1_l2')(hidden_layer1)  # Couche intermédiaire
hidden_layer3 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l1_l2')(hidden_layer2)  # Couche intermédiaire
hidden_layer4 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l1_l2')(hidden_layer3)  # Couche intermédiaire
output_layer = tf.keras.layers.Dense(3)(hidden_layer4)  # Sortie : variables d'actionnement

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)


model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


History = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=150, batch_size=512, verbose=2, shuffle = True)
print("history loss ", History.history['loss'])


loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Test accuracy:", accuracy)
print("Test loss:", loss)

model.save_weights('modele_poids.h5')


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(History.history['loss'], linestyle='-', label='train loss')
plt.plot(History.history['val_loss'], linestyle='-', label='test loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(History.history['accuracy'], label='train score')
plt.plot(History.history['val_accuracy'], label='test score')
plt.legend()
plt.show()