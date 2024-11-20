import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFilter
import utils

csv_path = "data_Data_Entry_2017_v2020.csv"
data = pd.read_csv(csv_path)

condition = "Pneumonia"
filtered_data = data[data['Finding Labels'].str.contains(condition, na=False)]

image_names = filtered_data['Image Index'].tolist()

actual_image_names = []
for image_name in image_names:
    try:
        with open("images1/"+image_name, "rb") as f1:
            if not f1.closed:
                actual_image_names.append("images1/"+image_name)
    except FileNotFoundError:
        try:
            with open("images2/"+image_name, "rb") as f2:
                if not f2.closed:
                    actual_image_names.append("images2/"+image_name)
        except FileNotFoundError:
            continue



def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.convert("L")
        img_array = np.array(img) / 255.0

        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


processed_images = []
for image_path in actual_image_names:
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        processed_images.append(preprocessed_image)

processed_images = np.array(processed_images)

print(f"Processed {len(processed_images)} images.")


plt.imshow(processed_images[0], cmap="gray")
plt.show()