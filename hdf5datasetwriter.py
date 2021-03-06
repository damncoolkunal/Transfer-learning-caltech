#!/usr/bin/env python
# coding: utf-8

# In[5]:


import h5py
import os

class HDF5DatasetWriter:
    
    def __init__(self ,dims , outputPath, dataKey ="images", bufSize =1000):
        
        
        #check to see if the output path exists , if so raise 
        #an exception
        
        if os.path.exists(outputPath):
            raise ValueError("the given file format cannot be overwritten", outputPath)
            
        
        self.db =h5py.File(outputPath ,"w")
        self.data = self.db.create_dataset(dataKey, dims, dtype = "float")
        self.labels =self.db.create_dataset("labels" , (dims[0] , ), dtype ="int")
        
        
        #store the buffer size
        self.bufSize = bufSize
        self.buffer = {"data":[] ,"labels":[]}
        self.idx = 0
        
        
        
        def add(self, rows , labels):
            
            self.buffer ["data"].extend(rows)
            self.buffer["labels"].extend(labels)
            
            
            #check to see the buffer flushed into the disk
            
            if len(self.buffer["data"]) >=self.bufSize:
                self.flush()
                
        
        def flush(self):
            
            #write the buffer to disk and then reset the buffer
            
            i = self.idx + len(self.buffer["data"])
            self.data[self.idx:1] = self.buffer["data"]
            self.labels[self.idx:1] = self.buffer ["labels"]
            self.idx = i
            self.buffer = {"data":[] , "labels":[]}
            
         
        def storeClassLabels(self, classLabels):
            #create a actual dataset to store actual class labels
            
            dt = h5py.special_dtype(vlen = str)
            labelSet =self.db.create_dataset("label_names" , (len(classLabels), ), dtype = dt)
            
            labelSet[:] = classLabels
            
        
        def close(self):
            
            if len(self.buffer["data"])> 0:
                self.flush()
                
            
            self.db.close()
        
        
        
        
        
        
                        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    




# In[ ]:




