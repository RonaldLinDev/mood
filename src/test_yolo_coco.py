from ultralytics import YOLO
import sys
import os
import numpy as np
import pandas as pd
# load model
model = YOLO(r'models/yolov8x.pt')
coco_to_mood = {'person' : 'people', 'chair' : 'chair', 'dining table' : 'table', 'laptop' : 'laptop', 'backpack' : 'bag'}
# person, chair, dining table, laptop
def predict(file):
    results = model(source = file, classes = [0, 24, 56, 61, 64], device = 'cuda',)
    # boxes = results[0].boxes.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()
    # results[0].show()
    names = results[0].names
    return [coco_to_mood[names[id]] for id in labels]



os.chdir(r'mood_eval-1/test/')
results = {}
for file in os.listdir():
      if not file.endswith('.json'):
            instances = predict(file)
            counts = {}
            for object in instances:
                counts[object] = counts.get(object, 0) + 1

            results[file] = counts
            
df = pd.DataFrame(results).transpose().replace(np.nan, 0)
df['laptop'] = 0
df['table'] = 0
df = df.sort_index(axis = 1)
df.loc[len(df.index)] = df.sum(axis = 0)
df.rename(index = {9 : "sum"}, inplace = True)

os.chdir(r'..')
os.chdir(r'..')
df.to_csv(r'tables/yolo_coco.csv')