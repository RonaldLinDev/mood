from ultralytics import YOLO
import sys
import os
# load model
model = YOLO(r'models/yolov8n.pt')
# person, chair, dining table, laptop
def predict(file):
    results = model(file, classes = [1, 57, 61, 64])
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    for ele in zip(boxes, classes):
            box = ele[0]
            label = names[ele[1]]
            print(box, label)



for file in os.listdir('/'):
      if not file.endswith('.json'):
            predict(file)