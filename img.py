import os
import cv2

# Path to store collected images
DATA_DIR ='C:\\Users\\gupta\\OneDrive\\Documents\\Desktop\\haukdum'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1 # Number of classes for different gestures
dataset_size = 2000 # Number of images per class

cap = cv2.VideoCapture(0)  # Capture from the default camera
cap.set(3, 1280)  # Set frame width (adjust for better hand visibility)
cap.set(4, 720)   # Set frame height

# Loop through each class for data collection
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for user readiness
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Display instructions on the frame
        cv2.putText(frame, 'Position both hands in frame. Press "Q" when ready.', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Adjusted bounding boxes for both hands (larger area for two hands)
        cv2.rectangle(frame, (250, 150), (1030, 570), (255, 0, 0), 2)  # Larger bounding box

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture dataset_size images for the class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save image to class folder
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()