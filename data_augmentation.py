#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse


#parsing the argument 

ap = argparse.ArgumentParser()
ap.add_argument("-i" , "--image" , required = True , help = "path to the image file")
ap.add_argument("-o" , "--output" , required = True , help = "path to the output file ")
ap.add_argument("-p", "--prefix" , type =str , default ="image " , help = "output prefix name of the file")
args = vars(ap.parse_args())

print ("loading images file .......")
image = load_img(args["images"])
image = img_to_array(image)
image = np.expand_dims(image , axis =0)

#lets initialize our ImageDataGenerator 

aug = ImageDataGenerator(rotation_range= 30, width_shift_range =0.1, height_shift_range  =0.1, shear_range =0.2 ,
                        zoom_range =0.2 ,horizontal_flip= True , fill_mode ="nearest")

total =0

print("INFO generating images....")
imageGen =aug.flow(image, batch_size =1, save_to_dir = args["output"],  save_prefix = args["prefix"], save_format="jpg")

#loop over examples from our image data augmentation generator

for image in imageGen:
    total +=1
    
    if total == 10:
        break



















# In[ ]:


_

