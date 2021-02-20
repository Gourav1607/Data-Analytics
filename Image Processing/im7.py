import cv2
import sys
import numpy as np

if len(sys.argv) < 2:
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(sys.argv[1])

ret, last_frame = video_capture.read()
ret, current_frame = video_capture.read()
gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

i = 0
while(i < 1000):
    # We want two frames- last and current, so that we can calculate the different between them.
    # Store the current frame as last_frame, and then read a new one
    last_frame = current_frame
    ret, current_frame = video_capture.read()
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(last_frame, current_frame)
    if np.mean(diff) > 10:
        print("Achtung! Motion detected.")
        cv2.imshow('Video', diff)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()
