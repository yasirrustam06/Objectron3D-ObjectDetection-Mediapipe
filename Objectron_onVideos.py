import cv2
import mediapipe as mp
import time
mpobj=mp.solutions.objectron
mpDraw=mp.solutions.drawing_utils
pTime=0
cTime=0

cap=cv2.VideoCapture(0)

objectron=mpobj.Objectron(static_image_mode=False,max_num_objects=2,
                          min_detection_confidence=0.5,min_tracking_confidence=0.8,model_name="Cup")


while True:
    ret,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results=objectron.process(imgRGB)
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mpDraw.draw_landmarks(img,detected_object.landmarks_2d,mpobj.BOX_CONNECTIONS)
            mpDraw.draw_axis(img,detected_object.rotation,detected_object.translation)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # cv2.putText(img,str(int(fps)),(70,60),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),-1)
    cv2.imshow("Image",img)

    cv2.waitKey(1)

