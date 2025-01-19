from ultralytics import YOLO

import cv2
import math
import numpy as np
import time

model = YOLO('weapon_detection.pt')

cap = cv2.VideoCapture("sample/test-weapon.mp4")
PeopleStayed = 0

while True:
    start_time = time.time()
    s, img = cap.read()
    result = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box[0].conf * 100)) / 100
            class_index = int(box.cls[0])
            # if  class_index == 0:
            #     class_name = coco_classes[class_index]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.putText(img, "Gun Detected: " + str(conf), (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)

    end_time = time.time()
    total_time = end_time - start_time 
    fps = 1/total_time
    # cv2.putText(img, f"{fps}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
