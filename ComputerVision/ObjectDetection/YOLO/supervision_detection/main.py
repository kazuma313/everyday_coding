import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
import numpy as np
# import requests

WA_NUMBER = '628xxxxxxxxxx'
WA_MESSAGE = 'Orang yang memakai top biru masuk!'

show_img = True
model = YOLO('best.pt')
tracker = Tracker()
title = 'Demo Deteksi Warna ketika Masuk'
cv2.namedWindow(title)
cap = cv2.VideoCapture('demo.mp4')
my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0
area1 = [(350, 495), (570, 495), (570, 485), (350, 485)]
area2 = [(350, 470), (570, 470), (570, 460), (350, 460)]
going_out = {}
going_in = {}
counter1 = []
counter2 = []
print("Proses deteksi sedang berlangsung...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        color = (255, 255, 255)
        if c == 'htblue':
            color = (247, 253, 62)
        elif c == 'htyellow':
            color = (48, 150, 239)
        if show_img:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (x2, y2), 3, (255, 0, 255), -1)
        if c == 'htblue':
            list.append([x1, y1, x2, y2])
    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        result = cv2.pointPolygonTest(
            np.array(area1, np.int32), (x4, y4), False)
        print(result)
        if result >= 0:
            going_in[id] = (x4, y4)
        if id in going_in:
            result1 = cv2.pointPolygonTest(
                np.array(area2, np.int32), (x4, y4), False)
            # if result1 >= 0:
            #     if counter1.count(id) == 0:
            #         counter1.append(id)
            #         print("Biru Masuk! Kirim ke Whatsapp.")
            #         url = 'http://localhost:9001/instant-messages'
            #         myobj = {'to': WA_NUMBER, 'msg': WA_MESSAGE}
            #         requests.post(url, json = myobj)
    indata = len(counter1)
    if show_img:
        cvzone.putTextRect(frame, f'MASUK (BIRU): {indata}', (0, 20), 1, 1)
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)
        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
