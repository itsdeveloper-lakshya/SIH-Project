import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the trained model
model_dict = pickle.load(open('./ISLmodel.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change this to 0 if it doesn't work

# Initialize Mediapipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Allow detection of both hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Define labels
labels_dict = {
    0:'E',1:'H',2:'L',3:'O'
}

# Define the central square ROI with smaller size
def get_square_roi(frame_width, frame_height, square_size):
    """Calculate the coordinates for a central square ROI"""
    start_x = (frame_width - square_size) // 2
    start_y = (frame_height - square_size) // 2
    end_x = start_x + square_size
    end_y = start_y + square_size
    return start_x, start_y, end_x, end_y

# Set a smaller square size
square_size = 350  # Adjusted smaller size

sentence = ""
last_update_time = time.time()
hand_detected = False
no_hand_last_time = time.time()

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    H, W, _ = frame.shape

    # Get the coordinates for the central square ROI
    ROI_START_X, ROI_START_Y, ROI_END_X, ROI_END_Y = get_square_roi(W, H, square_size)

    # Draw ROI rectangle on the frame
    frame_with_roi = frame.copy()  # Create a copy of the frame to draw the ROI
    cv2.rectangle(frame_with_roi, (ROI_START_X, ROI_START_Y), (ROI_END_X, ROI_END_Y), (0, 255, 0), 2)

    # Create a mask for the ROI
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[ROI_START_Y:ROI_END_Y, ROI_START_X:ROI_END_X] = 255

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Crop the masked frame to the ROI
    roi_frame = masked_frame[ROI_START_Y:ROI_END_Y, ROI_START_X:ROI_END_X]

    # Convert the cropped frame to RGB
    frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

    # Process the cropped frame to detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_detected = True
        no_hand_last_time = time.time()

        hand_data_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                hand_data_aux.append(x - min(x_))
                hand_data_aux.append(y - min(y_))
            hand_data_list.append(hand_data_aux)

        # If only one hand is detected, pad the data for the second hand with zeros
        if len(hand_data_list) == 1:
            data_aux.extend(hand_data_list[0])
            data_aux.extend([0] * 42)  # Append 42 zeros for the second hand
        elif len(hand_data_list) == 2:
            data_aux.extend(hand_data_list[0])
            data_aux.extend(hand_data_list[1])

        if len(data_aux) == 84:  # Ensure full data for one or both hands
            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            # Bounding box coordinates for display (relative to the ROI)
            x1 = int(min(x_) * roi_frame.shape[1]) - 10
            y1 = int(min(y_) * roi_frame.shape[0]) - 10
            x2 = int(max(x_) * roi_frame.shape[1]) - 10
            y2 = int(max(y_) * roi_frame.shape[0]) - 10

            # Adjust bounding box coordinates to the original frame
            x1 += ROI_START_X
            y1 += ROI_START_Y
            x2 += ROI_START_X
            y2 += ROI_START_Y

            # Draw rectangle and put the predicted character
            cv2.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame_with_roi, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Update the sentence if more than 3 seconds have passed since the last update
            current_time = time.time()
            if current_time - last_update_time > 3:
                sentence += predicted_character
                last_update_time = current_time
    else:
        current_time = time.time()
        if hand_detected and current_time - no_hand_last_time > 3:
            sentence += ' '
            hand_detected = False
            last_update_time = current_time

    # Resize the frame to make it bigger
    scale_factor = 1.5  # Scale factor for resizing
    frame_resized = cv2.resize(frame_with_roi, (0, 0), fx=scale_factor, fy=scale_factor)

    # Display the sentence with a label
    cv2.putText(frame_resized, 'Sentence: ' + sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

    # Show the resized frame
    cv2.imshow('frame', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the sentence to a file
with open('output_sentence.txt', 'w') as f:
    f.write(sentence)

cap.release()
cv2.destroyAllWindows()