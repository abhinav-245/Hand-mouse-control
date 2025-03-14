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

# Smooth movement variables
prev_x, prev_y = None, None
smooth_factor = 0.2  # Adjust for smoothness

def get_finger_position(landmarks):
    x = (1 - landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)  # Flip horizontal
    y = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    return int(x * screen_width), int(y * screen_height)

def is_fist(landmarks):
    fingers_open = sum([landmarks.landmark[i].y < landmarks.landmark[i - 2].y for i in range(8, 21, 4)])
    return fingers_open == 0  # All fingers closed

def is_thumb_up(landmarks):
    return landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y

def is_thumb_down(landmarks):
    return landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y > landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y

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

            if predicted_gesture == 'move':
                x, y = get_finger_position(hand_landmarks)
                if prev_x is None or prev_y is None:
                    prev_x, prev_y = x, y
                
                # Smooth cursor movement
                smooth_x = int(prev_x + smooth_factor * (x - prev_x))
                smooth_y = int(prev_y + smooth_factor * (y - prev_y))
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)
                prev_x, prev_y = smooth_x, smooth_y
            
            if predicted_gesture == 'click':
                pyautogui.click()
            
            if predicted_gesture == 'volume_up' and is_thumb_up(hand_landmarks):
                pyautogui.press('volumeup')
            elif predicted_gesture == 'volume_down' and is_thumb_down(hand_landmarks):
                pyautogui.press('volumedown')

            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
