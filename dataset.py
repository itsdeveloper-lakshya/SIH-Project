import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set static_image_mode to True for processing static images and allow detection of both hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Directory containing the collected hand gesture images
DATA_DIR ='C:\\Users\\gupta\\OneDrive\\Documents\\Desktop\\ekdum'
data = []
labels = []

# Loop through each class directory (gestures)
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Store landmark data for both hands

        # Read and convert image to RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image and detect hands
        results = hands.process(img_rgb)
        
        # Initialize placeholders for left and right hands
        left_hand_data = [0] * 42  # Placeholder if left hand is missing (21 landmarks x 2 coordinates)
        right_hand_data = [0] * 42  # Placeholder if right hand is missing

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_data_aux = []  # Store data for each hand individually

                # Determine if it's the left or right hand
                handedness = hand_handedness.classification[0].label
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                for i in range(len(hand_landmarks.landmark)):
                    hand_data_aux.append(x_coords[i] - min(x_coords))
                    hand_data_aux.append(y_coords[i] - min(y_coords))

                # Check if it's left or right hand and store the corresponding data
                if handedness == "Left":
                    left_hand_data = hand_data_aux
                elif handedness == "Right":
                    right_hand_data = hand_data_aux

        # Combine left and right hand data (or placeholders if a hand is missing)
        data_aux.extend(left_hand_data)
        data_aux.extend(right_hand_data)

        data.append(data_aux)
        labels.append(dir_)

# Save the processed data and labels using pickle
with open('ISLdata.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("File Created Successfully")