import cv2
import mediapipe as mp
import subprocess

# Initialize MediaPipe hand detection module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Create a hand detection object
hands = mp_hands.Hands()

# Open the camera (index 2)
cam = cv2.VideoCapture(2)

# Define camera resolution
res_x = 640
res_y = 480

# Set camera width, height, and FPS
cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_x)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_y)
cam.set(cv2.CAP_PROP_FPS, 30)

obsidian = False

def hand_recognition(img, inverted_side=False):
    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the image to detect hands
    results = hands.process(img_rgb)
    all_hands = []

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_side, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
            hand_info = {}
            coord = []
            # Extract the coordinates of the hand landmarks
            for marks in hand_landmarks.landmark:
                coord_x, coord_y, coord_z = int(marks.x * res_x), int(marks.y * res_y), int(marks.z * res_x)
                coord.append((coord_x, coord_y, coord_z))

            hand_info['landmarks'] = coord

            # Check if the hand side should be inverted
            if inverted_side:
                if hand_side.classification[0].label == 'Left':
                    hand_info['side'] = 'Right'
                else:
                    hand_info['side'] = 'Left'
            else:
                hand_info['side'] = hand_side.classification[0].label

            all_hands.append(hand_info)

            # Draw the landmarks and connections on the image
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return img, all_hands

def fingers_raised(hand):
    fingers = []

    if hand['side'] == 'Right':
        if hand['landmarks'][4][0] < hand['landmarks'][3][0]:
            fingers.append(True)
        else:
            fingers.append(False)

    for fingertip in [8, 12, 16, 20]:
        if hand['landmarks'][fingertip][1] < hand['landmarks'][fingertip - 2][1]:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers

def fingers_numbers(fingers):
    return sum(fingers)

while True:
    # Read a frame from the camera
    success, img = cam.read()
    # Flip the image horizontally
    img = cv2.flip(img, 1)

    if not success:
        break

    # Recognize hands in the image
    img, all_hands = hand_recognition(img)

    if len(all_hands) == 1:
        finger_hand_info = fingers_raised(all_hands[0])
        num_fingers = fingers_numbers(finger_hand_info)
        print(num_fingers)
        # Plot the number of fingers on the image
        cv2.putText(img, f'Fingers: {num_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    # Show the image with detected hands
    cv2.imshow("Image", img)

    # Wait for 1 ms for a key press
    key = cv2.waitKey(1)

    # If the 'ESC' key is pressed, exit the loop
    if key == 27:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
