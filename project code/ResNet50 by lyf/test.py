import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from  tensorflow.keras.models import load_model
import time


norm_size=224
imagelist=[]
emotion_labels = {
    0: 'sketchrnn_bear',
    1: 'sketchrnn_camel',
    2: 'sketchrnn_cat',
    3: 'sketchrnn_cow',
    4: 'sketchrnn_crocodile',
    5: 'sketchrnn_dog',
    6: 'sketchrnn_elephant',
    7: 'sketchrnn_flamingo',
    8: 'sketchrnn_giraffe',
    9: 'sketchrnn_hedgehog',
    10: 'sketchrnn_horse',
    11: 'sketchrnn_kangaroo',
    12: 'sketchrnn_lion'
    13: 'sketchrnn_monkey'
    14: 'sketchrnn_owl'
    15: 'sketchrnn_panda'
    16: 'sketchrnn_penguin'
    17: 'sketchrnn_pig'
    18: 'sketchrnn_raccoon'
    19: 'sketchrnn_rhinoceros'
    20: 'sketchrnn_sheep'
    21: 'sketchrnn_squirrel'
    22: 'sketchrnn_tiger'
    23: 'sketchrnn_whale'
    24: 'sketchrnn_zebra'
}

emotion_classifier=load_model("my_model.h5")

image = cv2.imdecode(np.fromfile('dataset/dataset', dtype=np.uint8), -1)
image = val_transform(image=image)['image']
image = cv2.resize(image, (norm_size, norm_size), interpolation=cv2.INTER_LANCZOS4)
image = img_to_array(image)
imagelist.append(image)
imageList = np.array(imagelist, dtype="float")

pre=np.argmax(emotion_classifier.predict(imageList))
emotion = emotion_labels[pre]
t2=time.time()
print(emotion)
t3=t2-t1
print(t3)

