
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import yolov8
import cv2
import supervision as sv
import os
import pandas as pd
import numpy as np



def predict(image_path, base_model = GroundingDINO(CaptionOntology({"person" : "person",
                        "laptop" : "laptop",
                        "chair" : "chair",
                        "table" : "table",
                        "backpack" : "backpack"})

)): 
    image = cv2.imread(image_path)
    classes = ["person", "laptop", "chair", "table", "backpack"]
    detections = base_model.predict(image_path)
    labels = [classes[class_id] for _, _, confidence, class_id, _ in detections]
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    sv.plot_image(annotated_frame)
    return labels

results = {}

os.chdir('mood_eval-1/test')
for file in os.listdir():
      if not file.endswith('.json'):
            instances = predict(file)
            counts = {}
            for object in instances:
                counts[object] = counts.get(object, 0) + 1

            results[file] = counts
            
df = pd.DataFrame(results).transpose().replace(np.nan, 0)

df = df.sort_index(axis = 1)
df.loc[len(df.index)] = df.sum(axis = 0)
df.rename(index = {9 : "sum"}, inplace = True)
os.chdir('..')
os.chdir('..')
df.to_csv('tables/grounding_dino.csv')