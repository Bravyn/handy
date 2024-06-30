# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2

# STEP 2: Create a HandLandmarker object.
mp_hands = mp.solutions.hands

# STEP 3: Load the input image.
image_path = "image.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# STEP 4: Detect hand landmarks from the input image.
with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
    results = hands.process(image_rgb)

    # STEP 5: Process the detection results and visualize.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# STEP 6: Display the annotated image.
cv2.imshow('Hand Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
