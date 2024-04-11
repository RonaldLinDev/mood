import os

os.chdir('imgs/')

for i, file in enumerate(os.listdir('.')):
    os.rename(file, f"{i}.jpg")