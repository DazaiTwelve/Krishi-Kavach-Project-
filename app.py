import os
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = None

# Load your trained model
def load_model():
    global model
    model = tf.keras.models.load_model('EfficientNetB3.h5')  # Assuming you saved your model as 'EfficientNetB3.h5'

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Preprocess the image as per your TensorFlow model
            img_size = (200, 200)  # Your model's input size
            img = load_img(image_path, target_size=img_size)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0  # Normalize if your model requires it

            # Make predictions
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)

            # Assuming you have a mapping of classes
            classes = ['healthy', 'unhealthy']  # Replace with your actual class labels
            predicted_label = classes[predicted_class]

            # Calculate accuracy and loss (mock values for demonstration)
            accuracy = np.random.uniform(0.7, 1.0)  # Mock accuracy
            loss = np.random.uniform(0.1, 0.5)      # Mock loss

            return render_template(
                'result.html', 
                label=predicted_label, 
                confidence=confidence, 
                accuracy=accuracy, 
                loss=loss, 
                image_path=image_file.filename
            )
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
