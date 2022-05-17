#!/usr/bin/env python3
# encoding: utf-8

#%%
from __future__ import absolute_import, division, print_function, unicode_literals
# Install TensorFlow
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten , Convolution2D, MaxPooling2D , Lambda, Conv2D, Activation,Concatenate
from tensorflow.keras.layers import ActivityRegularization
from tensorflow.keras.optimizers import Adam , SGD , Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import regularizers , initializers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import NumpyArrayIterator



gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
except RuntimeError as e:
# Visible devices must be set before GPUs have been initialized
    print(e)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
# from xgboost import XGBClassifier



#%%
from sklearn import metrics

# !pip3 install keras-tuner --upgrade
# !pip3 install autokeras
# import kerastuner as kt
# import autokeras as ak

#Plot's Making  Packages
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib import font_manager


# Import local libraries
import numpy as np
import h5py
import time
import pandas as pd
import importlib
from scipy import interpolate
import os
from tqdm import tqdm 

import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '64'

print("Tensorflow Version is {}".format(tf.__version__))
print("Keras Version is {}".format(tf.keras.__version__))
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# tf.device('/device:GPU:0')
# !nvidia-smi

time.sleep(5)

# %%
"""
Define Generator
"""

def data_generator(imagepath: str, data_dict: pd.DataFrame, nb_samples: int, batch_size: int):
    while True:
        for start in range(0, nb_samples, batch_size):
            x_leanding_jet_batch = []
            x_subleanding_jet_batch = []
            x_rotated_event_batch = []
            y_batch = []

            end = min(start + batch_size, nb_samples)
            for img_index in range(start, end):
                
                x_train_path = imagepath + data_dict["Image"].iloc[img_index]

                x_train_leanding_jet = np.load(x_train_path)["leading_jet_image"]
                x_train_leanding_jet = np.nan_to_num(x_train_leanding_jet)
                x_leanding_jet_batch.append(x_train_leanding_jet)

                x_train_subleanding_jet = np.load(x_train_path)["subleading_jet_image"]
                x_train_subleanding_jet = np.nan_to_num(x_train_subleanding_jet)
                x_subleanding_jet_batch.append(x_train_subleanding_jet)

                x_train_rotated_event = np.load(x_train_path)["rotated_event_image"]
                x_train_rotated_event = np.nan_to_num(x_train_rotated_event)
                x_rotated_event_batch.append(x_train_rotated_event)
        
                
                if data_dict["Y"].iloc[img_index] == 0:
                    y_batch.append(["0"])
                if data_dict["Y"].iloc[img_index] != 0:
                     y_batch.append(["1"])

            yield ([np.asarray(x_leanding_jet_batch), np.asarray(x_subleanding_jet_batch), np.asarray(x_rotated_event_batch)], to_categorical(np.asarray(y_batch)))


"""
Define Collector
"""

def loading_data(imagepath: str, data_dict: pd.DataFrame, start: int=0, stop: int=20000): 
    x_leanding_jet = []
    x_subleanding_jet = []
    x_rotated_event = []
    y = []

    logging.info("Collect Data from {} to {}.".format(start,stop))
    time.sleep(0.5)
    for img_index in tqdm(range(start,stop)):
        try:
            x_train_path = imagepath + data_dict["Image"].iloc[img_index]

            x_train_leanding_jet = np.load(x_train_path)["leading_jet_image"]
            x_train_leanding_jet = np.nan_to_num(x_train_leanding_jet)
            x_leanding_jet.append(x_train_leanding_jet)

            x_train_subleanding_jet = np.load(x_train_path)["subleading_jet_image"]
            x_train_subleanding_jet = np.nan_to_num(x_train_subleanding_jet)
            x_subleanding_jet.append(x_train_subleanding_jet)

            x_train_rotated_event = np.load(x_train_path)["rotated_event_image"]
            x_train_rotated_event = np.nan_to_num(x_train_rotated_event)
            x_rotated_event.append(x_train_rotated_event)

            # x_jet_tmp = np.divide((x_jet_tmp - norm_dict[0]), (np.sqrt(norm_dict[1])+1e-5))#[0].reshape(1,40,40)
            if data_dict["Y"].iloc[img_index] == 0:
                y.append(["0"])
            if data_dict["Y"].iloc[img_index] != 0:
                y.append(["1"])
        except:
            break

        # if img_index == stop:
        #     break

    return [np.asarray(x_leanding_jet), np.asarray(x_subleanding_jet), np.asarray(x_rotated_event)], to_categorical(np.asarray(y))



#%%
HOMEPATH = "/AICourse2022/alan_THDM/MC_Data/"
ImagePath =  HOMEPATH + "Image_Directory_Test/"
savepath = HOMEPATH + "Image_Directory_Test/"

#%%
process = {
            "ppHhh" : 0,
            "ttbar" : 0,
            # "ppbbbb" : 0,
            # "ppjjjb" : 0,
            # "ppjjjj" : 0,
              }  
    
for i, element in enumerate(process):
    process[element] = pd.read_csv(savepath + str(element) + "_dict.csv")

#%%
process_train_test = {
                    "ppHhh" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                    "ttbar" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                    # "ppbbbb" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                    # "ppjjjb" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                    # "ppjjjj" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                     }  

for i, element in enumerate(process_train_test):

    process_train_test[element]["training"]["X"], \
    process_train_test[element]["test"]["X"], \
    process_train_test[element]["training"]["Y"], \
    process_train_test[element]["test"]["Y"], \
     = train_test_split( process[element]["Image"], process[element]["Y"], test_size=0.10, random_state=42)

#%%
training_pd = shuffle(pd.DataFrame(process_train_test["ppjjjj"]["training"]))[:len(process["ppHhh"])]
for element in process_train_test:
    if element == "ppjjjj":
        continue
    else:
        training_pd = pd.concat([training_pd, pd.DataFrame(process_train_test[element]["training"])], ignore_index=True)

test_pd = shuffle(pd.DataFrame(process_train_test["ppjjjj"]["test"]))[:int(len(process["ppHhh"])/10)]
for element in process_train_test:
    if element == "ppjjjj":
        continue
    else:
        test_pd = pd.concat([test_pd, pd.DataFrame(process_train_test[element]["test"])], ignore_index=True)


training_pd = shuffle(training_pd)
test_pd = shuffle(test_pd)

logging.info("\n")
logging.info("There are {} sig and {} bkg in training dataset.".format(len(training_pd[training_pd["Y"]==0]),len(training_pd[training_pd["Y"]!=0])))
logging.info("There are {} sig and {} bkg in test dataset.".format(len(test_pd[test_pd["Y"]==0]),len(test_pd[test_pd["Y"]!=0])))
logging.info("\n")

x_test, y_test = loading_data(imagepath = ImagePath, data_dict = test_pd, start=0, stop=len(test_pd))


#%%
"""
Learning Curve
"""
learning_curve = pd.read_csv("/AICourse2022/alan_THDM/Model_3CNN/training_log_300.csv")
fig, ax = plt.subplots(1,1, figsize=(5,5))

plt.plot(learning_curve["loss"], label='training data',c='blue',linewidth = 3)
plt.plot(learning_curve["val_loss"], label='validation data',c='red',linewidth = 3)

plt.title("3CNN", fontsize=15)

ax.set_ylabel('loss', fontsize=15,horizontalalignment='right',y=1)
ax.set_xlabel('epoch', fontsize=15,horizontalalignment='right',x=1)
plt.legend(loc='best', prop={'size':15}, edgecolor = "w",fancybox=False, framealpha=0)


plt.show()


#%%

model_3cnn = load_model("/AICourse2022/alan_THDM/Model_3CNN/model_3cnn_300.h5")

for pro in process:
    logging.info("Process: {}".format(pro))

    # if pro != "ppbbbb":
    #     continue

    for i, element in enumerate(range(0,len(process[pro]), 30000)):
        logging.info(i)
        x_test, y_test = loading_data(imagepath = ImagePath, data_dict = process[pro], start= element, stop=int(element+30000))
        prediction = model_3cnn.predict(x_test)
        np.save("./prediction/"+str(pro)+"_3cnn_prediction_"+str(i)+"_30000_test", prediction)


#%%
for pro in process:
    logging.info("Process: {}".format(pro))
    logging.info("Data total length: {}".format(len(process[pro])))

    # if pro != "ppbbbb":
    #     continue

    prediction = np.load("./prediction/"+str(pro)+"_3cnn_prediction_"+str(0)+"_30000_test.npy")
    for i, element in enumerate(range(30000,len(process[pro]), 30000)):
        pre_tmp = np.load("./prediction/"+str(pro)+"_3cnn_prediction_"+str(i+1)+"_30000_test.npy")
        # logging.info("Shape of pre_tmp {}".format(pre_tmp.shape))
        prediction = np.concatenate([prediction,pre_tmp])


    logging.info("Shape of prediction {}".format(prediction.shape))
    logging.info("Prediction total length: {}".format(len(prediction)))

    np.save("./prediction/"+str(pro)+"_3cnn_prediction_test", prediction)
    
    logging.info("{}".format("Done!"))
    logging.info("\n")


# %%
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Please code in this cell

prediction_test =  model_3cnn.predict(x_test)

#%%
confusion_ = confusion_matrix(y_test[:,1], np.argmax(prediction_test,axis=1))


confusion = np.array([[confusion_[0][0]/np.sum(confusion_[0]),confusion_[0][1]/np.sum(confusion_[0])],
                        [confusion_[1][0]/np.sum(confusion_[0]),confusion_[1][1]/np.sum(confusion_[1])]])


truelist = ["sig","bkg"]
likelist = ["sig-like","bkg-like"]

s = len(truelist)
f, ax = plt.subplots(1,1, figsize=(s+4, s+4))

aa = ax.imshow(confusion.T, cmap="Oranges", origin= "upper", vmin=0, vmax=1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad="1%")
cbar = plt.colorbar(aa, cax=cax)
cbar.ax.tick_params(labelsize=15)
# cbar.ax.yaxis.set_major_locator(MaxNLocator(6))
cbar.set_label("Ratio", rotation=270, fontsize=15, labelpad=30, y=0.5)
# cbar.set_ticks([0,500,1000,1500,2000])
# cbar.ax.set_yticklabels(["0","500","1000","1500","2000"])
cbar.set_ticks([0,0.25,0.5,0.75,1])
cbar.ax.set_yticklabels(["0","0.25","0.5","0.75","1"])

ax.set_xticks(range((confusion.T).shape[1]))
ax.set_xticklabels(truelist, fontsize=15, rotation=0)
ax.set_yticks(range((confusion.T).shape[1]))
ax.set_yticklabels(likelist, fontsize=15, rotation=45)

my_colors = ["green","red"]
ax.xaxis.tick_top()
for ticklabel, tickcolor in zip(ax.get_xticklabels(), my_colors):
    ticklabel.set_color(tickcolor)
    
for ticklabel, tickcolor in zip(ax.get_yticklabels(), my_colors):
    ticklabel.set_color(tickcolor)

Terminology = np.array([["TP","FP"],["FN","TN"]])
    
for (i, j), z in np.ndenumerate(confusion.T):
    ax.text(j, i, '{:^3s}: {:0.2f}'.format(Terminology[i,j],z), ha='center', va='center',fontsize=15,color="k")
    
plt.tight_layout()
plt.show()
# %%
