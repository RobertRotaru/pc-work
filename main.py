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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Data loading and filtering
csv_path = "data_Data_Entry_2017_v2020.csv"
data = pd.read_csv(csv_path)

condition = "Pneumonia"
image_names = data['Image Index'].tolist()
labels = np.array([1 if condition in name else 0 for name in data['Finding Labels'].tolist()])

# Check for actual images
actual_image_names = []
actual_labels = []
for i in range(len(image_names)):
    try:
        with open("images1/" + image_names[i], "rb") as f1:
            if not f1.closed:
                actual_image_names.append("images1/" + image_names[i])
                actual_labels.append(labels[i])
    except FileNotFoundError:
        try:
            with open("images2/" + image_names[i], "rb") as f2:
                if not f2.closed:
                    actual_image_names.append("images2/" + image_names[i])
                    actual_labels.append(labels[i])
        except FileNotFoundError:
            continue

print(f"Image fetch succeded with {len(actual_image_names)} images")

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
        print(f"preprocessed {len(processed_images)} image")

processed_images = np.array(processed_images)

print(f"Processed {len(processed_images)} images.")

# Ensure the images are in the right shape (224, 224, 3)
assert processed_images.shape[1:] == (224, 224, 3), "Processed images should have shape (224, 224, 3)"

subset_size = 2000
processed_images = processed_images[:subset_size]
actual_labels = actual_labels[:subset_size]

print("Got subsets")

# Train-test split with labels
train_images, temp_images, train_labels, temp_labels = train_test_split(processed_images, actual_labels, test_size=0.3, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=42)

# ResNet Model Setup
resnetModel = ResNet50(input_tensor=Input(shape=(224, 224, 3)), weights='imagenet', include_top=False)

# Freeze ResNet layers
for layer in resnetModel.layers:
    layer.trainable = False

x = Flatten()(resnetModel.output)
prediction = Dense(2, activation='softmax')(x)  # Update number of classes here
model = Model(inputs=resnetModel.input, outputs=prediction)

# Model compilation
model.compile(
    loss='sparse_categorical_crossentropy',  # Use sparse if labels are integers
    optimizer='adam',
    metrics=['accuracy']
)

assert train_images.ndim == 4 and train_images.shape[1:] == (224, 224, 3)
assert len(train_labels) == train_images.shape[0]

# Model training
batch_size = 16  # Example batch size
r = model.fit(
    x=np.array(train_images),  # Convert to NumPy array if not already
    y=np.array(train_labels),  # Ensure labels are NumPy arrays
    validation_data=(np.array(val_images), np.array(val_labels)),  # Ensure validation data is correct
    epochs=5,
    batch_size=batch_size
)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Generate predictions
predictions = model.predict(test_images)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Print a sample of actual vs predicted labels
for i in range(5):  # Display the first 5 samples
    print(f"Actual: {test_labels[i]}, Predicted: {predicted_labels[i]}")


cm = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Pneumonia", "Pneumonia"], yticklabels=["No Pneumonia", "Pneumonia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot Training vs Validation Accuracy
plt.plot(r.history['accuracy'], label='Training Accuracy')
plt.plot(r.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Compute ROC curve
fpr, tpr, _ = roc_curve(test_labels, predictions[:, 1])  # Use probabilities for class 1 (Pneumonia)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random chance line
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()