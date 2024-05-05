import tensorflow as tf
import numpy as np

np.random.seed(0)
x = np.random.rand(1000, 2)
y = np.random.randint(2, size=1000)

x_train, x_test = x[:800], x[800:]
 y_train, y_test = y[:800], y[800:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_dim=2)
])

model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)