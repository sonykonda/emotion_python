import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime

# Load face detection model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load trained emotion model (64x64 input)
classifier = load_model('emotion_model.hdf5')

# Emotion labels (7 classes - FER2013)
emotion_labels = ['Angry', 'Disgust', 'Fear', 
                  'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam
cap = cv2.VideoCapture(0)

# Store emotion history
emotion_history = []

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Extract face
        roi_gray = gray[y:y+h, x:x+w]

        # Resize to 64x64 (IMPORTANT)
        roi_gray = cv2.resize(roi_gray, (64, 64))

        # Normalize
        roi_gray = roi_gray / 255.0

        # Reshape for model
        roi_gray = np.reshape(roi_gray, (1, 64, 64, 1))

        # Predict
        prediction = classifier.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]
        confidence = float(np.max(prediction)) * 100

        # Save emotion with timestamp
        emotion_history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Emotion": emotion,
            "Confidence": round(confidence, 2)
        })

        label = f"{emotion} ({confidence:.2f}%)"

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put text
        cv2.putText(frame, label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2)

    cv2.imshow("Live Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save history to CSV
df = pd.DataFrame(emotion_history)
df.to_csv("emotion_history.csv", index=False)

cap.release()
cv2.destroyAllWindows()

print("Emotion history saved to emotion_history.csv")