import matplotlib as plt
import pandas as pd
import json
import os

with open('mood_eval-1/test/_annotations.json') as f:
    data = json.load(f)
id_to_label = {ele['id'] : ele['name'] for ele in data['categories']}
imgs_to_id =  {ele ['id'] : ele['file_name'] for ele in data['images']}
annotations_by_image = {imgs : {label : 0 for label in id_to_label.values()} for imgs in imgs_to_id.keys()}
for ele in data['annotations']:
    annotations_by_image[ele['image_id']][id_to_label[ele['category_id']]] += 1
df = pd.DataFrame(annotations_by_image).transpose().sort_index(axis = 1)

df.loc[len(df.index)] = df.sum(axis = 0)
df.rename(index = {9 : "sum"}, inplace = True)


print(df)
os.chdir(r'tables/')
df.to_csv(r'eval.csv')

