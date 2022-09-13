import cv2
import mediapipe as mp
import time


img=cv2.imread("chair2.jpg")
imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

mpObjectron=mp.solutions.objectron
OBJECTRON=mpObjectron.Objectron(static_image_mode=False,max_num_objects=2,
                          min_detection_confidence=0.5,min_tracking_confidence=0.8,model_name="Chair")

mpDraw=mp.solutions.drawing_utils

results=OBJECTRON.process(imgRGB)

if results.detected_objects:
    for detected_Object in  results.detected_objects:
        mpDraw.draw_landmarks(img,detected_Object.landmarks_2d,mpObjectron.BOX_CONNECTIONS)
        mpDraw.draw_axis(img,detected_Object.rotation,detected_Object.translation)

cv2.imwrite("Chair1_Image.jpg",img)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()





