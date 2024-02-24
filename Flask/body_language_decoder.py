import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

# Load the machine learning model
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = holistic.process(image)

        # Draw landmarks on the copy of the frame
        image_copy = image.copy()
        mp_drawing.draw_landmarks(image_copy, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image_copy, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image_copy, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image_copy, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Convert frame back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Export coordinates
        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            row = pose_row + face_row
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]

            # Grab ear coordinates
            coords = tuple(np.multiply(np.array((
                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), [640, 480]).astype(int))

            cv2.rectangle(image, (coords[0], coords[1] + 5),
                          (coords[0] + len(body_language_class) * 20, coords[1] - 30), (245, 117, 16), -1)

            cv2.putText(image, body_language_class, coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)

        # Display the frame
        cv2.imshow("Holistic Model Detections", image)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
