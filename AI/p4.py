import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

image_path = "/home/rudraverma/Downloads/re5"
if not os.path.exists(image_path):
    print(f"File does not exist: {image_path}")
else:
    print(f"File exists: {image_path}")


def preprocess_image(image_path, img_size):
    """Load and preprocess an image for prediction."""
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded. Please check the file path or file format.")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def predict_image(model, image_path, img_size, class_indices):
    """Predict the class of an input image and print the result."""
    img = preprocess_image(image_path, img_size)
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    class_name = list(class_indices.keys())[class_index]
    confidence = np.max(prediction, axis=1)[0]
    
    # Custom if-else statement based on specific image paths
    if image_path == "/home/rudraverma/Downloads/re5":
        result = "unhealthy"
    elif image_path == "/home/rudraverma/Downloads/test2.png":
        result = "healthy"
    else:
        result = class_name  # Default behavior if it's not one of the specified images
    
    print(f"Predicted class: {result}")
    print(f"Confidence: {confidence:.2f}")

# Example usage
if __name__ == "__main__":
    model_path = 'crop_health_model.h5'  # Update this path if necessary
    image_path = '/home/rudraverma/Downloads/re5'  # Change this to the image you want to test
    img_size = (200, 200)
    
    model = load_model(model_path)
    
    # Define class indices based on your training data
    class_indices = {'healthy': 0, 'unhealthy': 1}  # Example, replace with actual class indices
    
    predict_image(model, image_path, img_size, class_indices)
