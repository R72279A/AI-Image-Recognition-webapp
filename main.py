import numpy as np
import tensorflow as tf

# Cleaned up imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import layers


# --- 1. Data Loading and Preparation ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train , x_test = x_train / 255.0 , x_test / 255.0


# --- 2. Build the Model Architecture ---
model = Sequential([
    layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(units=128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(units=10, activation='softmax')
])


model.summary()


# --- 3. Compile the Model ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# --- 4. Train the Model ---
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))


# --- 5. Save the Final Model ---
model.save('my_image_classifier.h5')
print("Bravo! Model saved to disk.")