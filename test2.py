import cv2
import time

cap = cv2.VideoCapture("http://10.5.32.254:4747/video")
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    # Get number of frames in buffer
    frames_in_buffer = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print('frames in buffer:', frames_in_buffer)
    
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()