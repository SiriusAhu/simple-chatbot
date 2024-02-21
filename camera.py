import cv2
import os
from settings import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--camera", type=int, default=0, help="Camera to use: 0 for the first camera | Index is based on your system's camera index | Try to change this value if you meet errors")

args = parser.parse_args()
camera = args.camera

cap = cv2.VideoCapture(camera)

set_camera_lock()

while cap.isOpened():
    try:
        # Read the frame
        ret, frame = cap.read()

        cv2.imshow('frame', frame)

        if os.path.exists(IMAGE_TRIGGER):
            # save the frame
            cv2.imwrite(PTH_LAST_FRAME, frame)
            os.remove(IMAGE_TRIGGER)

        # Press Q or close the window to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    except KeyboardInterrupt:
        break

# Release the camera and close the window
remove_camera_lock()
cap.release()
cv2.destroyAllWindows()