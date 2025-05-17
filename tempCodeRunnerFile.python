import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images = []
    labels = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for emotion_idx, emotion in enumerate(emotions):
        folder = os.path.join(data_dir, emotion)
        if not os.path.exists(folder):
            print(f"Error: Directory {folder} does not exist.")
            continue
        img_count = 0
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error: Could not load image {img_path}")
                continue
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(emotion_idx)
            img_count += 1
        print(f"Loaded {img_count} images from {emotion}")
    if not images:
        raise ValueError("No images loaded. Check dataset path and structure.")
    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels, emotions

# Path to your dataset
data_dir = r'D:\ML tranning\data\fer2013\train'
try:
    images, labels, emotions = load_data(data_dir)
    # Check dataset balance
    unique, counts = np.unique(labels, return_counts=True)
    print("Dataset balance (emotion: count):")
    print(dict(zip([emotions[i] for i in unique], counts)))
    
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_val = X_val.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)
    print("Data loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
except Exception as e:
    print(f"Error: {e}")