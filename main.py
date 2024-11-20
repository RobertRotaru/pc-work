import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Model, Input
from keras.src.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFilter
import utils
from tensorflow.keras.applications.resnet50 import ResNet50

# Data loading and filtering
csv_path = "data_Data_Entry_2017_v2020.csv"
data = pd.read_csv(csv_path)

condition = "Pneumonia"
filtered_data = data[data['Finding Labels'].str.contains(condition, na=False)]
image_names = filtered_data['Image Index'].tolist()

# Check for actual images
actual_image_names = []
for image_name in image_names:
    try:
        with open("images1/" + image_name, "rb") as f1:
            if not f1.closed:
                actual_image_names.append("images1/" + image_name)
    except FileNotFoundError:
        try:
            with open("images2/" + image_name, "rb") as f2:
                if not f2.closed:
                    actual_image_names.append("images2/" + image_name)
        except FileNotFoundError:
            continue

# Preprocessing function for image
def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize to 224x224
        img = img.filter(ImageFilter.GaussianBlur(radius=1))  # Apply Gaussian blur for noise reduction
        img = img.convert("L")  # Convert to grayscale
        img_array = np.array(img) / 255.0  # Normalize pixel values

        # Convert grayscale to RGB by replicating across 3 channels
        img_array_rgb = np.repeat(img_array[:, :, np.newaxis], 3, axis=-1)  # Shape (224, 224, 3)

        return img_array_rgb
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Process images and ensure no None values
processed_images = []
for image_path in actual_image_names:
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        processed_images.append(preprocessed_image)

processed_images = np.array(processed_images)

print(f"Processed {len(processed_images)} images.")

# Ensure the images are in the right shape (224, 224, 3)
assert processed_images.shape[1:] == (224, 224, 3), "Processed images should have shape (224, 224, 3)"

# Train-test split (70% train, 15% validation, 15% test)
train_images, temp_images = train_test_split(processed_images, test_size=0.3, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

print(f"Shape of train_images: {train_images.shape}")
print(f"Shape of val_images: {val_images.shape}")
print(f"Shape of test_images: {test_images.shape}")

# ResNet Model Setup
resnetModel = ResNet50(input_tensor=Input(shape=(224, 224, 3)), weights='imagenet', include_top=False)

# Freeze ResNet layers
for layer in resnetModel.layers:
    layer.trainable = False

x = Flatten()(resnetModel.output)
prediction = Dense(len(train_images), activation='softmax')(x)
model = Model(inputs=resnetModel.input, outputs=prediction)
model.summary()

print(model.input_spec)

# Model compilation
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Model training
batch_size = 1  # Example batch size
r = model.fit(
    train_images,
    validation_data=val_images,  # Use validation data
    epochs=5,
    steps_per_epoch=len(train_images) // batch_size,
    validation_steps=len(val_images) // batch_size
)