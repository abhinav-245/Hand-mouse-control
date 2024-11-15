# run_model.py
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import joblib

# Load trained model
model = joblib.load('gesture_model.joblib')

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

def get_finger_position(landmarks):
    # Adjusted to fix mirror direction issue
    x = (1 - landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)  # Flip horizontal
    y = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    return int(x * screen_width), int(y * screen_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
            landmarks = (landmarks - np.mean(landmarks)) / np.std(landmarks)  # Normalize

            # Predict gesture
            predicted_gesture = model.predict([landmarks])[0]

            # Control mouse based on gesture
            if predicted_gesture == 'move':
                x, y = get_finger_position(hand_landmarks)
                pyautogui.moveTo(x, y, duration=0.1)  # Smooth movement
            elif predicted_gesture == 'click':
                pyautogui.click()

            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
