import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
IMAGE_PATH = '/home/rudraverma/Downloads/Image Data base/'
ENV_DATA_PATH = '/home/rudraverma/Downloads/enviromental_data.csv'

# Load and preprocess environmental data
env_data = pd.read_csv(ENV_DATA_PATH)
numerical_features = ['temperature', 'humidity', 'rainfall', 'ph', 'N', 'P', 'K']

# Ensure all features are numeric
for feature in numerical_features:
    env_data[feature] = pd.to_numeric(env_data[feature], errors='coerce')

# Remove any rows with NaN values
env_data = env_data.dropna()

scaler = StandardScaler()
env_data[numerical_features] = scaler.fit_transform(env_data[numerical_features])

# Split environmental data
train_env, test_env = train_test_split(env_data, test_size=0.2, random_state=42)

# Image data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load image data
train_generator = datagen.flow_from_directory(
    IMAGE_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    IMAGE_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

def create_improved_model(num_env_features):
    img_input = layers.Input(shape=(224, 224, 3))
    env_input = layers.Input(shape=(num_env_features,))

    # Image processing branch
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Environmental data branch
    y = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(env_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)

    # Combine branches
    combined = layers.concatenate([x, y])

    z = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)

    z = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)

    output = layers.Dense(1, activation='sigmoid')(z)

    model = models.Model(inputs=[img_input, env_input], outputs=output)
    return model
def combined_generator(img_gen, env_data):
    while True:
        img_batch, label_batch = next(img_gen)
        env_batch = env_data.sample(n=img_batch.shape[0])[numerical_features].values
        yield (img_batch, env_batch), label_batch

# Create and compile the improved model
model = create_improved_model(len(numerical_features))

# Compiling the model with a lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Learning rate scheduler
lr_reducer = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1)

history = model.fit(
    combined_generator(train_generator, train_env),
    steps_per_epoch=len(train_generator),
    epochs=10,  # Number of epochs as requested
    validation_data=combined_generator(validation_generator, test_env),
    validation_steps=len(validation_generator),
    callbacks=[lr_reducer]
)

# Save the model
model.save('crop_health_model1.h5')

# Function to make predictions
def predict_crop_health(model, image_path, env_data):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, 0)  # Create batch axis

    env_array = env_data[numerical_features].values.reshape(1, -1)

    prediction = model.predict([img_array, env_array])
    return "Unhealthy" if prediction[0][0] > 0.5 else "Healthy", prediction[0][0]

# Example usage
test_image_path = '/home/rudraverma/Downloads/test2.png'
test_env_data = pd.DataFrame({
    'temperature': [25],
    'humidity': [60],
    'rainfall': [100],
    'ph': [6.5],
    'N': [50],
    'P': [30],
    'K': [20]
})
test_env_data[numerical_features] = scaler.transform(test_env_data[numerical_features])

model = tf.keras.models.load_model('crop_health_model.h5')
prediction, confidence = predict_crop_health(model, test_image_path, test_env_data)
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}")