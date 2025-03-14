import cv2
import mediapipe as mp
import pandas as pd

# Initialize Mediapipe and variables
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)
data = []
labels = []

def collect_data(label):
    global data
    global labels
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw landmarks and collect data
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                data.append([coord for lm in landmarks for coord in lm])  # Flatten
                labels.append(label)  # Add label for the gesture

                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Data Capture - Press "q" to exit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Get labels from user
while True:
    label = input("Enter gesture label (or 'exit' to stop): ")
    if label == 'exit':
        break
    print(f"Capturing data for gesture '{label}'. Press 'q' to finish.")
    collect_data(label)

# Save data to CSV
df = pd.DataFrame(data)
df['gesture'] = labels
df.to_csv('gesture_data.csv', index=False)

cap.release()
cv2.destroyAllWindows()
print("Data captured and saved to 'gesture_data.csv'.")
