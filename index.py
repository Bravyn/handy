import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
#initialize model path
model_path = 'D:\\projects\\mach\\hand_landmarker.task'

'''#model configuration
BaseOptions = mp.tasks.BaseOptions
HandLandMarker = mp.tasks.vision.HandLandmarker
HandLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#other modes include video mode and live videa stream mode

options = HandLandMarkerOptions(
    base_options = BaseOptions(model_asset_path = model_path, )
)'''

#creating an instance of a hand landmarker with image mode
base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options = base_options,
                                                    num_hands = 2)
detector = vision.HandLandmarker.create_from_options(options)

image = mp.Image.create_from_file("image.jpg")

#detect hand landmarkers from the input image
detection_result = detector.detect(image)

#visualize classification result
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))