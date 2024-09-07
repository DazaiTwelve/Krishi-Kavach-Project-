import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

# Directory and parameters
sdir = r'/home/rudraverma/Downloads/Sample_database/'
min_samples = 40
filepaths = []
labels = []
classlist = os.listdir(sdir)

# Prepare the dataframe
for klass in classlist:
    classpath = os.path.join(sdir, klass)
    flist = os.listdir(classpath)
    if len(flist) >= min_samples:
        for f in flist:
            fpath = os.path.join(classpath, f)
            filepaths.append(fpath)
            labels.append(klass)
    else:
        print('class ', klass, ' has only', len(flist), ' samples and will not be included in dataframe')

Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis=1)
train_df, dummy_df = train_test_split(df, train_size=.9, shuffle=True, random_state=123, stratify=df['labels'])
valid_df, test_df = train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])
print('train_df length: ', len(train_df), ' test_df length: ', len(test_df), ' valid_df length: ', len(valid_df))

# Get class information
classes = sorted(list(train_df['labels'].unique()))
class_count = len(classes)
print('The number of classes in the dataset is: ', class_count)
groups = train_df.groupby('labels')
print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
countlist = []
classlist = []
for label in sorted(list(train_df['labels'].unique())):
    group = groups.get_group(label)
    countlist.append(len(group))
    classlist.append(label)
    print('{0:^30s} {1:^13s}'.format(label, str(len(group))))

max_value = np.max(countlist)
max_index = countlist.index(max_value)
max_class = classlist[max_index]
min_value = np.min(countlist)
min_index = countlist.index(min_value)
min_class = classlist[min_index]
print(max_class, ' has the most images= ', max_value, ' ', min_class, ' has the least images= ', min_value)

# Average height and width of train images
ht = 0
wt = 0
train_df_sample = train_df.sample(n=100, random_state=123, axis=0)
for i in range(len(train_df_sample)):
    fpath = train_df_sample['filepaths'].iloc[i]
    img = plt.imread(fpath)
    shape = img.shape
    ht += shape[0]
    wt += shape[1]
print('average height= ', ht // 100, ' average width= ', wt // 100, 'aspect ratio= ', ht / wt)

def trim(df, max_samples, min_samples, column):
    df = df.copy()
    groups = df.groupby(column)
    trimmed_df = pd.DataFrame(columns=df.columns)
    for label in df[column].unique():
        group = groups.get_group(label)
        count = len(group)
        if count > max_samples:
            sampled_group = group.sample(n=max_samples, random_state=123, axis=0)
            trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)
        else:
            if count >= min_samples:
                sampled_group = group
                trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)
    print('after trimming, the maximum samples in any class is now ', max_samples, ' and the minimum samples in any class is ', min_samples)
    return trimmed_df

max_samples = 100
min_samples = 36
column = 'labels'
train_df = trim(train_df, max_samples, min_samples, column)

def balance(df, n, working_dir, img_size):
    df = df.copy()
    print('Initial length of dataframe is ', len(df))
    aug_dir = os.path.join(working_dir, 'aug')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)
    total = 0
    gen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                             height_shift_range=.2, zoom_range=.2)
    groups = df.groupby('labels')
    for label in df['labels'].unique():
        group = groups.get_group(label)
        sample_count = len(group)
        if sample_count < n:
            aug_img_count = 0
            delta = n - sample_count
            target_dir = os.path.join(aug_dir, label)
            msg = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(delta))
            print(msg, '\r', end='')
            aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=img_size,
                                              class_mode=None, batch_size=1, shuffle=False,
                                              save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                              save_format='jpg')
            while aug_img_count < delta:
                images = next(aug_gen)
                aug_img_count += len(images)
            total += aug_img_count
    print('Total Augmented images created= ', total)
    aug_fpaths = []
    aug_labels = []
    classlist = os.listdir(aug_dir)
    for klass in classlist:
        classpath = os.path.join(aug_dir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries = pd.Series(aug_fpaths, name='filepaths')
    Lseries = pd.Series(aug_labels, name='labels')
    aug_df = pd.concat([Fseries, Lseries], axis=1)
    df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))
    return df

n = 200
working_dir = r'./'
img_size = (200, 200)
train_df = balance(train_df, n, working_dir, img_size)

batch_size = 20
trgen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                           height_shift_range=.2, zoom_range=.2)
t_and_v_gen = ImageDataGenerator()

msg = '{0:70s} for train generator'.format(' ')
print(msg, '\r', end='')
train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

msg = '{0:70s} for valid generator'.format(' ')
print(msg, '\r', end='')
valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                             class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

length = len(test_df)
test_batch_size = sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= 80], reverse=True)[0]
test_steps = int(length / test_batch_size)
msg = '{0:70s} for test generator'.format(' ')
print(msg, '\r', end='')
test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                            class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

classes = list(train_gen.class_indices.keys())
class_indices = list(train_gen.class_indices.values())
class_count = len(classes)
labels = test_gen.labels
print('test batch size: ', test_batch_size, ' test steps: ', test_steps, ' number of classes : ', class_count)

def show_image_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen)
    plt.figure(figsize=(20, 20))
    length = len(labels)
    if length < 25:
        r = length
    else:
        r = 25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name)
        plt.axis('off')
    plt.show()

show_image_samples(train_gen)

def build_model(img_size, class_count):
    input_layer = keras.Input(shape=(img_size[0], img_size[1], 3))

    # Block 1
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Fully Connected Layers
    x = tf.keras.layers.Flatten()(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = build_model(img_size, class_count)

# Training and validation
epochs = 20
history = model.fit(train_gen, validation_data=valid_gen, epochs=epochs, verbose=1)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

plot_history(history)

def predictor(test_gen, test_steps):
    y_pred = []
    y_true = test_gen.labels
    classes = list(test_gen.class_indices.keys())
    class_count = len(classes)
    errors = 0
    
    # Predict the classes
    preds = model.predict(test_gen, verbose=1)
    tests = len(preds)
    
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = test_gen.labels[i]
        
        # Determine if the prediction is correct
        if pred_index != true_index:
            errors += 1
            
        # Append the predicted class index
        y_pred.append(pred_index)
    
    acc = (1 - errors / tests) * 100
    print(f'There were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}%')

    # Map predictions and true labels to class names
    y_pred = np.array([classes[idx] for idx in y_pred])
    y_true = np.array([classes[idx] for idx in y_true])
    
    if class_count <= 30:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(class_count) + 0.5, classes, rotation=90)
        plt.yticks(np.arange(class_count) + 0.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    
    clr = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("Classification Report:\n----------------------\n", clr)
    
    # Output whether each plant is healthy or unhealthy
    for i, label in enumerate(y_pred):
        print(f"Sample {i+1}: Predicted - {label}, Actual - {y_true[i]}")
    
    return errors, tests

# Call the predictor function
errors, tests = predictor(test_gen, test_steps)
def preprocess_image(image_path, img_size):
    """Load and preprocess an image for prediction."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(model, image_path, img_size):
    """Predict the class of an input image and print the result."""
    img = preprocess_image(image_path, img_size)
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    class_name = list(train_gen.class_indices.keys())[class_index]
    confidence = np.max(prediction, axis=1)[0]
    
    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.2f}")

# Example usage
image_path = '/home/rudraverma/Downloads/re4'
predict_image(model, image_path, img_size)
model.save('crop_health_model.h5')
