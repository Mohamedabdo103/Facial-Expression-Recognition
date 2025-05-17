import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
import time

# Load data
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

# Main execution
data_dir = r'D:\ML tranning\data\fer2013\train'
try:
    # Load data
    images, labels, emotions = load_data(data_dir)
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

    # Load the pre-trained model
    model = load_model('models/best_model.keras')
    print("Pre-trained model loaded successfully.")

    # Evaluate the model
    start_time = time.time()
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    inference_time = time.time() - start_time
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Inference Time for Test Set: {inference_time:.2f} seconds")
    print(f"Average Inference Time per Image: {inference_time / len(X_test):.4f} seconds")

    # Visualize Performance
    # Confusion Matrix
    start_time = time.time()
    y_pred = model.predict(X_test, batch_size=32)
    inference_time = time.time() - start_time
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotions, yticklabels=emotions, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Classification Report (Precision, Recall, F1-Score)
    report = classification_report(y_test, y_pred_classes, target_names=emotions)
    print("Classification Report:\n", report)
    with open('classification_report.txt', 'w') as f:
        f.write(report)

    # Bar Chart for PRF Scores
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average=None, labels=range(len(emotions)))
    plt.figure(figsize=(10, 5))
    x = np.arange(len(emotions))
    width = 0.25
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')
    plt.xlabel('Emotion')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score per Emotion')
    plt.xticks(x, emotions, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('prf_scores.png')
    plt.show()

    # Confidence Score Distribution
    confidences = np.max(y_pred, axis=1)
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.savefig('confidence_distribution.png')
    plt.show()

    # Dataset Class Distribution
    plt.figure(figsize=(8, 5))
    plt.bar(emotions, counts)
    plt.xlabel('Emotion')
    plt.ylabel('Number of Images')
    plt.title('Dataset Class Distribution')
    plt.xticks(rotation=45)
    plt.savefig('class_distribution.png')
    plt.show()

except Exception as e:
    print(f"Error: {e}")