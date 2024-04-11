from ultralytics import YOLO
import json


with open('mood_eval-1/test/_annotations.json') as f:
    data = json.load(f)
    labels = data['categories']
    imgs = data['images']
    imgs['annotations']


model = YOLO('models/yolov8n.pt')


