#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#training of model with data augmentation 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from simpledatasetoader import SimpleDatasetLoader
from minivggnet import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# parsing the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-d" , "--dataset" , required =True , help = "path to the cola dataset file")

args = vars(ap.parse_args())

print("Info loading images....")

imagePaths = list(paths.list_images(args["dataset"]))

classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

#initialize the image preprocessors

aap = AspectAwarePreprocessor(64,64)
iap = ImageToArrayPreprocessor()


#load the dataset from the disk and load the raw pixel intensities

sdl =SimpleDatasetLoader(preprocessors= [aap, iap])
(data , labels) =sdl.load(imagePaths , verbose =500)

data = data.astype("float")/255.0


# lets start training the data by splitting into test and train data

(trainX , testX, trainY, testY) = train_test_split(data, labels ,test_size= 0.25 ,random_state =42)

#convert the labels from integers to vectors
lb = LabelBinarizer()
#le.fit(trainY)
trainY = lb.fit_transform(trainY)
testY= lb.transform(testY)

labelNames =["P.diet" , "P.orig", "P.Rsugar", "P.Zero"]
#construct the imagegenerator for data augmentation

aug = ImageDataGenerator(rotation_range =30 ,width_shift_range =0.1 ,height_shift_range =0.1, shear_range =0.2 ,zoom_range =0.2 ,horizontal_flip =True , fill_mode ="nearest")

#initialize the optimization model

print("INFO compiling model...")
opt = SGD(lr= 0.05)

model = MiniVGGNet.build(width =64 ,height= 64 ,depth =3 ,classes = len(classNames))

model.compile(loss = "categorical_crossentropy" , optimizer = opt, metrics = ["accuracy"])


print("info training network....")

#train the network

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
     validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
     epochs=100, verbose=1)


#evaluate the network

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=classNames))
# plot the training loss and accuracy

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history[ "loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label=  "val_acc")
plt.title("training loss and accuracy ")
plt.xlabel("epochs")
plt.ylabel("loss/accuracy")
plt.legend()
plt.savefig(args["output"])
         
         
 

