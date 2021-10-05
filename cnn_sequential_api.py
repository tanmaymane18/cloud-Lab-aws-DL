import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.reshape(len(x_train), 28,28, 1)/255.
x_test = x_test.reshape(len(x_test), 28,28, 1)/255.

y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

model = keras.Sequential([
                          keras.layers.Conv2D(input_shape=(28,28,1),filters=32,kernel_size=3, strides=1, padding='same'),
                          keras.layers.MaxPool2D(pool_size=(2,2)),
                          keras.layers.BatchNormalization(),
                          keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
                          keras.layers.MaxPool2D(pool_size=(2,2)),
                          keras.layers.BatchNormalization(),
                          keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
                          keras.layers.MaxPool2D(pool_size=(2,2)),
                          keras.layers.BatchNormalization(),
                          keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
                          keras.layers.MaxPool2D(pool_size=(2,2)),
                          keras.layers.BatchNormalization(),
                          keras.layers.Flatten(),
                          keras.layers.Dropout(0.3),
                          keras.layers.Dense(10, activation='softmax'),

])


#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#loss = tf.keras.losses.CategoricalCrossentropy()
#auc = tf.keras.metrics.AUC()
model.compile(optimizer='adam', loss='categorical_cross_entropy', metrics=['acc'])

model.summary()

with tf.device('/device:GPU:0'):
  model.fit(x=x_train, y=y_train, epochs=20, validation_data=(x_test, y_test), batch_size=64)

with tf.device('/device:GPU:0'):
  model.evaluate(x=x_test, y=y_test, batch_size=128)
