import cv2 # only for accessing webcam and playing back the feed
import numpy as np
import detection_helpers as dh

THRESHOLD = 50
MIN_AREA = 500

webcam = cv2.VideoCapture(0) # 0 refers to default webcam
ret, frame = webcam.read() # ret is whether the frame was captured successfully
if not ret:
    raise Exception("Could not read from webcam.")

frame = np.asarray(frame, dtype=np.float32)
background = np.mean(frame, axis=2).astype(np.float32)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    _frame = np.asarray(frame, dtype=np.float32)
    gray_frame = np.mean(_frame, axis=2).astype(np.float32)
    # gray_frame = dh.convolve(gray_frame, kernel)
    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 1.0)
    
    diff = np.abs(gray_frame - background)
    # gets the "edges" where motion is detected and turn into black/white, white pixels are motion
    mask = diff > THRESHOLD 
    mask = mask.astype(np.uint8) * 255
    
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # kind of running average of all backgrounds, can slowly adapt to changes
    background = (1 - 0.05) * background + 0.05 * gray_frame
        
    boxes = dh.connected_components(mask)
    boxes = [box for box in boxes if (box[2] - box[0]) * (box[3] - box[1]) > MIN_AREA]
    
    for xmin, ymin, xmax, ymax in boxes:
        # last parameter is thickness
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Motion Mask", mask)
    
    if cv2.waitKey(1) & 0xFF == 27: # press escape to close
        break
    
webcam.release()
cv2.destroyAllWindows()