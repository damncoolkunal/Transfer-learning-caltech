#!/usr/bin/env python
# coding: utf-8

# In[1]:


#training a classifier on the extracted features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

#Parsing the argument 

ap = argparse.ArgumentParser()
ap.add_argument("-d" , "--db", required = True , help = "path to the HDF5 dataset")
ap.add_argument("-m" , "--model" , required = True , help  = "path to the output file")
ap.add_argument("-j" ,"--jobs" , type = int , default = -1 , help = "path of jobs when tuning hyperparametres")
args = vars(ap.parse_args())



#open the HDF5 dataset for reading the index

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0]* 0.75)


print("info running into hyperparameters...")

params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}

model = GridSearchCV(LogisticRegression(solver="lbfgs" ,
        multi_class= "auto"), params, cv=3, n_jobs=args["jobs"])

model.fit(db["features"][:i], db[ "labels"][:i])

print("Info best hyperparameters: {}".format(model.best_params_))


#evaluate the model

print("info evaluating....")

preds = model.predict(db["features"][i:] )
print(classification_report(db["labels"][i:], preds, target_names= db ["label_names"]))


print("INFO saving model...")

f= open(args["model"], "wb")

f.write(pickle.dumps(model.best_estimator_))
f.close()


#close the database

db.close()



























# In[ ]:




