import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model('models/best_model.keras')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load video from MP4 file
video_path = r'D:\ML tranning\examples\0518(2).mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if video loaded successfully
if not cap.isOpened():
    print(f"Error: Could not open video from {video_path}")
    exit()

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer for output
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    # Convert the frame to RGB as MediaPipe expects RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract bounding box from landmarks
            h, w, _ = frame.shape
            x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Ensure the bounding box is square and within frame bounds
            size = max(x_max - x_min, y_max - y_min)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            x_min = max(0, x_center - size // 2)
            x_max = min(w, x_center + size // 2)
            y_min = max(0, y_center - size // 2)
            y_max = min(h, y_center + size // 2)

            # Extract the face region
            face = frame[y_min:y_max, x_min:x_max]
            if face.size == 0:
                continue

            # Preprocess the face for the model
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(gray, (48, 48))
            face_normalized = face_resized / 255.0
            face_input = face_normalized.reshape(1, 48, 48, 1)

            # Predict emotion
            pred = model.predict(face_input)
            emotion_idx = np.argmax(pred)
            confidence = pred[0][emotion_idx] * 100
            emotion = emotions[emotion_idx]

            # Draw bounding box and label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            label = f"{emotion} ({confidence:.1f}%)"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    else:
        # Display message if no face is detected
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the frame to output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Facial Expression Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q' key press

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
face_mesh.close()