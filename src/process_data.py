import os
from PIL import Image

labelToNumber = {'person' : 0, 'laptop' : 1, 'chair' : 2, 'table' : 3, 'backpack' : 4}

#assumes yolov8 format
class dataset_sublabel:
    def __init__(self, label, input_dir, flipped, output_folder = "dataset"):
        try:
            self.label_id = labelToNumber[label]
            self.input_dir = input_dir
            self.output_folder = output_folder
            self.flipped = flipped
        except:
            print('label not valid') 
            self.label_id = label

    def convert_to_jpg(self):
        for file in os.listdir(self.input_dir):
            img_png = Image.open(self.input_dir + file)
            img_png = img_png.convert("RGB")
            img_png.save(f'{self.output_folder}/images/{int(sorted(os.listdir(self.output_folder + "/images"), key=lambda x: int(os.path.splitext(x)[0]))[-1][:-4]) + 1}.jpg')
            os.remove(self.input_dir + file)
    def verify(self):
        splits = ['/train', '/valid']
        for split in splits:
            for file in os.listdir(self.output_folder + split + '/labels/'):
                remove = False
                with open(file:= self.output_folder + split + '/labels/' + file) as f:
                    labels = set()
                    for line in f.readlines():
                        if self.flipped and line[0] == self.label_id:
                            remove = not remove
                        else:
                            labels.add(int(line[0]))
                    if self.label_id not in labels:
                        remove = not remove
                if (remove):
                    self.remove_pair(file)

    def remove_pair(self, path_to_annotation):
        print(f'removed {path_to_annotation}')
        os.remove(path_to_annotation)
        os.remove(path_to_annotation.replace('/labels/', '/images/').replace('.txt', '.jpg')) 




