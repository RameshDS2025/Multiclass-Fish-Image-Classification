import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model
model_path = os.path.join("models", "BEST_FISH_MODEL.keras")
model = tf.keras.models.load_model(model_path)

CLASS_NAMES = [
    'Black Sea Sprat', 
    'Gilt Head Bream', 
    'Hourse Mackerel', 
    'Red Mullet', 
    'Red Sea Bream', 
    'Sea Bass', 
    'Shrimp', 
    'Striped Red Mullet', 
    'Trout'
]

def preprocess_image(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Test images
test_images = {
    "Trout": r"data\val\fish sea_food trout\ZZ4Q8GVER4BY.jpg",
    "Shrimp": r"data\val\fish sea_food shrimp\09RYR968IP1O.jpg" # Guessing filename or find one
}

# Since I don't know the exact shrimp filename, I'll find one first
shrimp_dir = r"data\val\fish sea_food shrimp"
shrimp_files = [f for f in os.listdir(shrimp_dir) if f.endswith('.jpg')]
if shrimp_files:
    test_images["Shrimp"] = os.path.join(shrimp_dir, shrimp_files[0])

for name, path in test_images.items():
    if os.path.exists(path):
        print(f"\nTesting {name} image: {path}")
        processed_img = preprocess_image(path)
        predictions = model.predict(processed_img)[0]
        max_index = np.argmax(predictions)
        result_class = CLASS_NAMES[max_index]
        confidence = predictions[max_index] * 100
        
        print(f"Prediction: {result_class} ({confidence:.2f}%)")
        print("Raw Probabilities:")
        for idx, prob in enumerate(predictions):
            print(f"  {CLASS_NAMES[idx]}: {prob:.4f}")
    else:
        print(f"File not found: {path}")
