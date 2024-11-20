import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
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
                actual_image_names.append(image_name)
    except FileNotFoundError:
        try:
            with open("images2/"+image_name, "rb") as f2:
                if not f2.closed:
                    actual_image_names.append(image_name)
        except FileNotFoundError:
            continue
