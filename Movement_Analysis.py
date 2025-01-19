from ultralytics import YOLO
import cv2
import queue
import threading
import time
import math
import numpy as np
from sort import Sort

class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # Read frames as soon as they are available, keeping only the most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

cap = cv2.VideoCapture("sample/test-5.mp4")
model = YOLO('yolov8n.pt')

tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.3)
PeopleStayed = 0

# Dictionary to store person IDs and their stay durations
person_stay_duration = {}

dictionary_time = {
    
}
dictionaries_time = []
Time1 = []
results_list = []
# Threshold for considering a person's stay as long
stay_duration_threshold = 10  # In seconds
goneID= []
Id_list = []
while True:
    used_ids = []
    s, img = cap.read()
    overlay = np.zeros_like(img)
    result = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box[0].conf * 100)) / 100
            if box.cls[0] == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                currentarray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentarray))

    resultstrack = tracker.update(detections)

    for results in resultstrack:
        x1, y1, x2, y2, Id = results
        midX = int((x1 + x2)/2)
        midY = int((y1 + y2)/2)
        cv2.circle(img, (midX, midY), 4, (255, 255, 255), -1)
        results_list.append([midX, midY])
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        #newID
        used_ids.append(Id)
        if Id not in Id_list:
            dictionaries_time.append([Id, time.time(), time.time()])
            Id_list.append(Id)
        else:
            for index, sublist in enumerate(dictionaries_time):
                if sublist[0] == Id:
                    time_spend = int(time.time() - dictionaries_time[index][1])
                    dictionaries_time[index][2] = time.time()
                    cv2.putText(img, str(time_spend) + " s", (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    break

    for x in Id_list :
        if x not in used_ids :
            for index, sublist in enumerate(dictionaries_time):
                f = time.time() - dictionaries_time[index][2]
                if sublist[0] == x and f > 5:
                    dictionary_time[x] = dictionaries_time[index][2] - dictionaries_time[index][1]
                    dictionaries_time.pop(index)
                    Id_list.remove(x)
                    break
    
    # print(Id_list)
    # print(used_ids)
    coordinate_count = {}
    radius = 1
    base_color = (0, 255, 0)
    for coordinate in results_list:
        coordinate = tuple(coordinate)
        if coordinate in coordinate_count:
            coordinate_count[coordinate] += 1
        else:
            coordinate_count[coordinate] = 1

    # Draw circles on the overlay
    for coordinate, count in coordinate_count.items():
        cv2.circle(overlay, coordinate, 4, (0, 0, 255), -1)
    # cv2.putText(img, "total number of people :" + str(len(Id_list)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    output = cv2.addWeighted(img, 0.8, overlay, 1, 0)
    print(dictionary_time)
    cv2.imshow("Image", output)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
