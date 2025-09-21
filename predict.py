import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# --- 1. Define constants ---
MODEL_PATH = 'my_image_classifier.h5'
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- 2. Load the trained model ---
# Check if the model file exists before trying to load it
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    print("Please make sure you have run your training script to save the model.")
else:
    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")

    # --- 3. Get the image path from the user ---
    img_path = input("\nPlease enter the full path to your image and press Enter: ")

    # --- 4. Load and preprocess the image ---
    if not os.path.exists(img_path):
        print(f"Error: The image file at '{img_path}' was not found.")
    else:
        try:
            # Load the image and resize it to the model's expected 32x32 size
            img = image.load_img(img_path, target_size=(32, 32))

            # Convert the image to a NumPy array
            img_array = image.img_to_array(img)

            # Add a "batch" dimension to the array
            img_array = np.expand_dims(img_array, axis=0)

            # Normalize the pixel values (scale them between 0 and 1)
            img_array /= 255.0

            # --- 5. Make a prediction ---
            predictions = model.predict(img_array)

            # --- 6. Decode the prediction and display the result ---
            predicted_index = np.argmax(predictions[0])
            predicted_class_name = CLASS_NAMES[predicted_index]
            confidence = np.max(predictions[0]) * 100

            print("\n--- Prediction Result ---")
            print(f"This image is most likely a: {predicted_class_name}")
            print(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            print(f"An error occurred while processing the image: {e}")