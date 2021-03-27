#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Training a VGG Net

#from keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
#from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import MaxPooling2D
#from keras.layers.core import Activation
#from keras.layers.core import Flatten
#from keras.layers.core import Dropout
#from keras.layers.core import Dense
#from keras import backend as K

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras import activations
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import backend as K



class MiniVGGNet:
    def build(width , height , depth , classes):
        model =Sequential()
        inputShape =(height , width , depth)
        chanDim = -1
        
        
        if K.image_data_format() == "channels_first":
            inputShape =(depth , height , width)
            chanDim =1
        
    #defining the first layer of VGGNet
        model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
        model.add(Dense(10, activation='relu'))
        model.add(BatchNormalization(axis =chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Dense(10, activation='relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size =(2,2)))
        model.add(Dropout(0.25))
        
        
        #adding the second layer to VGGmininet
        
        model.add(Conv2D(64 ,(3,3) , padding="same"))
        model.add(Dense(10, activation='relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(64, (3,3) , padding ="same"))
        model.add(Dense(10, activation='relu'))
        model.add(BatchNormalization(axis =chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        #add Fc layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dense(10, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        #Adding softmax classifier
        model.add(Dense(classes))
        model.add(Dense(10, activation='softmax'))
        
        return model
    




















# In[ ]:




