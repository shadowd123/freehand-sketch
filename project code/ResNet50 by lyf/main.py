import numpy as np
from tensorflow.keras.optimizers import Adam
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
import os
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


norm_size = 224
datapath = 'dataset/dataset'
EPOCHS = 20
INIT_LR = 3e-4
labelList = []
dicClass = {'sketchrnn_bear': 0, 'sketchrnn_camel': 1, 'sketchrnn_cat': 2, 'sketchrnn_cow': 3, 'sketchrnn_crocodile': 4, 'sketchrnn_dog': 5, 'sketchrnn_elephant': 6,
            'sketchrnn_flamingo': 7, 'sketchrnn_giraffe': 8, 'sketchrnn_hedgehog': 9, 'sketchrnn_horse': 10, 'sketchrnn_kangaroo': 11, 'sketchrnn_lion': 12,
            'sketchrnn_monkey': 13, 'sketchrnn_owl': 14, 'sketchrnn_panda': 15, 'sketchrnn_penguin': 16, 'sketchrnn_pig': 17, 'sketchrnn_raccoon': 18, 'sketchrnn_rhinoceros': 19,
            'sketchrnn_sheep': 20, 'sketchrnn_squirrel': 21, 'sketchrnn_tiger': 22, 'sketchrnn_whale': 23, 'sketchrnn_zebra': 24}
classnum = 25
batch_size = 4
np.random.seed(42)


#加载图片
def loadImageData():
    imageList = []
    listClasses = os.listdir(datapath)  # 类别文件夹
    print(listClasses)
    for class_name in listClasses:
        label_id = dicClass[class_name]
        class_path = os.path.join(datapath, class_name)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            image_full_path = os.path.join(class_path, image_name)
            labelList.append(label_id)
            imageList.append(image_full_path)
    return imageList


print("开始加载数据")
imageArr = loadImageData()
labelList = np.array(labelList)
print("加载数据完成")

def generator(file_pathList,labels,batch_size,train_action=False):
    L = len(file_pathList)
    while True:
        input_labels = []
        input_samples = []
        for row in range(0, batch_size):
            temp = np.random.randint(0, L)
            X = file_pathList[temp]
            Y = labels[temp]
            image = cv2.imdecode(np.fromfile(X, dtype=np.uint8), -1)
            if image.shape[2] > 3:
                image = image[:, :, :3]
            if train_action:
                image=train_transform(image=image)['image']
            else:
                image = val_transform(image=image)['image']
            image = cv2.resize(image, (norm_size, norm_size), interpolation=cv2.INTER_LANCZOS4)
            image = img_to_array(image)
            input_samples.append(image)
            input_labels.append(Y)
        batch_x = np.asarray(input_samples)
        batch_y = np.asarray(input_labels)
        yield (batch_x, batch_y)

checkpointer = ModelCheckpoint(filepath='best_model.hdf5',
                               monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

reduce = ReduceLROnPlateau(monitor='val_accuracy', patience=10,
                           verbose=1,
                           factor=0.5,
                           min_lr=1e-6)
#建立模型
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Dense(classnum, activation='softmax'))
optimizer = Adam(learning_rate=INIT_LR)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(generator(trainX,trainY,batch_size,train_action=True),
                              steps_per_epoch=len(trainX) / batch_size,
                              validation_data=generator(valX,valY,batch_size,train_action=False),
                              epochs=EPOCHS,
                              validation_steps=len(valX) / batch_size,
                              callbacks=[checkpointer, reduce])
model.save('my_model.h5')
print(history)


#存储结果
loss_trend_graph_path = r"WW_loss.jpg"
acc_trend_graph_path = r"WW_acc.jpg"
import matplotlib.pyplot as plt

print("Now,we start drawing the loss and acc trends graph...")

fig = plt.figure(1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(acc_trend_graph_path)
plt.close(1)

fig = plt.figure(2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(loss_trend_graph_path)
plt.close(2)
print("We are done, everything seems OK...")


