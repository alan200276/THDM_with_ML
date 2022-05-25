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
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24000)])
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

def data_generator(imagepath: str, data_dict: pd.DataFrame, nb_samples: int, batch_size: int, Norm_dict: pd.DataFrame):
    while True:
        for start in range(0, nb_samples, batch_size):
            x_leading_jet_batch = []
            x_subleading_jet_batch = []
            x_rotated_event_batch = []
            y_batch = []

            end = min(start + batch_size, nb_samples)
            for img_index in range(start, end):
                
                x_train_path = imagepath + data_dict["X"].iloc[img_index]

                x_train_leading_jet = np.load(x_train_path)["leading_jet_image"]
                x_train_leading_jet = np.nan_to_num(x_train_leading_jet)

                x_train_leading_jet = np.divide((x_train_leading_jet - Norm_dict["leading_jet"][0]), (np.sqrt(Norm_dict["leading_jet"][1])+1e-5))
                x_train_leading_jet = np.nan_to_num(x_train_leading_jet)

                x_leading_jet_batch.append(x_train_leading_jet)

                x_train_subleading_jet = np.load(x_train_path)["subleading_jet_image"]
                x_train_subleading_jet = np.nan_to_num(x_train_subleading_jet)

                x_train_subleading_jet = np.divide((x_train_subleading_jet - Norm_dict["subleading_jet"][0]), (np.sqrt(Norm_dict["subleading_jet"][1])+1e-5))
                x_train_subleading_jet = np.nan_to_num(x_train_subleading_jet)

                x_subleading_jet_batch.append(x_train_subleading_jet)

                x_train_rotated_event = np.load(x_train_path)["rotated_event_image"]
                x_train_rotated_event = np.nan_to_num(x_train_rotated_event)

                x_train_rotated_event = np.divide((x_train_rotated_event - Norm_dict["full_event"][0]), (np.sqrt(Norm_dict["full_event"][1])+1e-5))
                x_train_rotated_event = np.nan_to_num(x_train_rotated_event)

                x_rotated_event_batch.append(x_train_rotated_event)
        
                
                if data_dict["Y"].iloc[img_index] == 0:
                    y_batch.append(["0"])
                if data_dict["Y"].iloc[img_index] != 0:
                     y_batch.append(["1"])

            yield ([np.asarray(x_leading_jet_batch), np.asarray(x_subleading_jet_batch), np.asarray(x_rotated_event_batch)], to_categorical(np.asarray(y_batch)))


"""
Define Collector
"""

def loading_data(imagepath: str, data_dict: pd.DataFrame , Norm_dict: pd.DataFrame, start: int=0, stop: int=20000): 
    x_leading_jet = []
    x_subleading_jet = []
    x_rotated_event = []
    y = []

    # time.sleep(0.5)
    # for img_index in tqdm(range(start,len(data_dict))):

    logging.info("Collect Data from {} to {}.".format(start,stop))
    time.sleep(0.5)
    for img_index in tqdm(range(start,stop)):
        try:
            x_train_path = imagepath + data_dict["X"].iloc[img_index]

            x_train_leading_jet = np.load(x_train_path)["leading_jet_image"]
            x_train_leading_jet = np.nan_to_num(x_train_leading_jet)

            x_train_leading_jet = np.divide((x_train_leading_jet - Norm_dict["leading_jet"][0]), (np.sqrt(Norm_dict["leading_jet"][1])+1e-5))
            x_train_leading_jet = np.nan_to_num(x_train_leading_jet)

            x_leading_jet.append(x_train_leading_jet)

            x_train_subleading_jet = np.load(x_train_path)["subleading_jet_image"]
            x_train_subleading_jet = np.nan_to_num(x_train_subleading_jet)

            x_train_subleading_jet = np.divide((x_train_subleading_jet - Norm_dict["subleading_jet"][0]), (np.sqrt(Norm_dict["subleading_jet"][1])+1e-5))
            x_train_subleading_jet = np.nan_to_num(x_train_subleading_jet)

            x_subleading_jet.append(x_train_subleading_jet)

            x_train_rotated_event = np.load(x_train_path)["rotated_event_image"]
            x_train_rotated_event = np.nan_to_num(x_train_rotated_event)

            x_train_rotated_event = np.divide((x_train_rotated_event - Norm_dict["full_event"][0]), (np.sqrt(Norm_dict["full_event"][1])+1e-5))
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

    return [np.asarray(x_leading_jet), np.asarray(x_subleading_jet), np.asarray(x_rotated_event)], to_categorical(np.asarray(y))



#%%
HOMEPATH = "/AICourse2022/alan_THDM/"
ImagePath =  HOMEPATH + "Image_Directory/"
savepath = HOMEPATH + "Image_Directory/"

#%%
process = {
            "ppHhh" : 0,
            "ttbar" : 0,
            # "ppbbbb" : 0,
            # "ppjjjb" : 0,
            "ppjjjj" : 0,
              }  
    
for i, element in enumerate(process):
    process[element] = pd.read_csv(savepath + str(element) + "_dict.csv")


#%%
Norm_dict ={
            "leading_jet" : [0,0],
            "subleading_jet" : [0,0],
            "full_event" : [0,0],
            }  

for i, element in enumerate(Norm_dict):

    l = 0

    for j, pro in enumerate(process):
        l += len(process[pro])
        average = np.load(savepath + "average" + "_" + str(element) + "_"+str(pro)+".npy")
        variance = np.load(savepath + "variance" + "_" + str(element) + "_"+str(pro)+".npy")
        
        Norm_dict[element][0] += average #*len(process[pro])
        Norm_dict[element][1] += variance

    Norm_dict[element][0] = Norm_dict[element][0]/(j+1)


#%%
process_train_test = {
                    "ppHhh" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                    "ttbar" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                    # "ppbbbb" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                    # "ppjjjb" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                    "ppjjjj" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                     }  

for i, element in enumerate(process_train_test):

    process_train_test[element]["training"]["X"], \
    process_train_test[element]["test"]["X"], \
    process_train_test[element]["training"]["Y"], \
    process_train_test[element]["test"]["Y"], \
     = train_test_split( process[element]["Image"], process[element]["Y"], test_size=0.10, random_state=42)

#%%
# training_pd = shuffle(pd.DataFrame(process_train_test["ppHhh"]["training"]))[:len(process["ppbbbb"])]
# for element in process_train_test:
#     if element == "ppHhh":
#         continue
#     else:
#         training_pd = pd.concat([training_pd, pd.DataFrame(process_train_test[element]["training"])], ignore_index=True)

# test_pd = shuffle(pd.DataFrame(process_train_test["ppHhh"]["test"]))[:int(len(process["ppbbbb"])/10)]
# for element in process_train_test:
#     if element == "ppHhh":
#         continue
#     else:
#         test_pd = pd.concat([test_pd, pd.DataFrame(process_train_test[element]["test"])], ignore_index=True)


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

#%%
"""
Model
"""
def return_pad_me(padding):
    def pad_me(x):
        #FRANK# x[:,:,:y,:] slice x off from y at the given axis.
        return(tf.concat((x,x[:,:,:padding,:]),2))
#         return(tf.concat((2,x,x[:,:,:padding,:])))
    return(pad_me)



def Model_3CNN():

    input_shape = (3, 40,40)

    model_leadingjet = Sequential(name = 'Leadingjet')
    model_leadingjet.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    data_format='channels_first',input_shape=input_shape, name = 'leadingjet'))
    model_leadingjet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),data_format='channels_first', name = 'leadingjet_MaxPooling_1'))
    model_leadingjet.add(Conv2D(64, (5, 5), activation='relu',data_format='channels_first', name = 'leadingjet_2D_1'))
    model_leadingjet.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first', name = 'leadingjet_MaxPooling_2'))
    model_leadingjet.add(Flatten(name = 'leadingjet_flatten'))
    model_leadingjet.add(Dense(300, activation='relu', name = 'leadingjet_dense_1'))
    model_leadingjet.add(Dropout(0.1))

    model_subleadingjet = Sequential(name = 'SubLeadingjet')
    model_subleadingjet.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    data_format='channels_first',input_shape=input_shape, name = 'subleadingjet'))
    model_subleadingjet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),data_format='channels_first', name = 'subleadingjet_MaxPooling_1'))
    model_subleadingjet.add(Conv2D(64, (5, 5), activation='relu',data_format='channels_first', name = 'subleadingjet_2D_1'))
    model_subleadingjet.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first', name = 'subleadingjet_MaxPooling_2'))
    model_subleadingjet.add(Flatten(name = 'subleadingjet_flatten'))
    model_subleadingjet.add(Dense(300, activation='relu', name = 'subleadingjet_dense_1'))
    model_subleadingjet.add(Dropout(0.1))

    model_event = Sequential(name = 'Event')
    model_event.add(Lambda(return_pad_me(4),
                    input_shape=input_shape, name = 'event'))
    model_event.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    data_format='channels_first', name = 'event_2D_1'))
    model_event.add(Lambda(return_pad_me(1),
                    input_shape=input_shape, name = 'event_padding_1'))
    model_event.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),data_format='channels_first', name = 'event_MaxPooling_1'))
    model_event.add(Lambda(return_pad_me(4),input_shape=input_shape, name = 'event_padding_2'))
    model_event.add(Conv2D(64, (5, 5), activation='relu',data_format="channels_first", name = 'event_2D_2'))
    model_event.add(Lambda(return_pad_me(1),input_shape=input_shape, name = 'event_padding_3'))
    model_event.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first", name = 'event_MaxPooling_2'))
    model_event.add(Flatten(name = 'event_flatten'))
    model_event.add(Dense(300, activation='relu', name = 'event_dense_1'))
    model_event.add(Dropout(0.1))



    mergedOut = Concatenate()([model_leadingjet.output, model_subleadingjet.output,model_event.output])
    # mergedOut = Dense(1, activation='sigmoid')(mergedOut)
    mergedOut = Dense(2, activation='softmax')(mergedOut)

    newModel = Model([model_leadingjet.input, model_subleadingjet.input,model_event.input], mergedOut,name = '3CNN')


    model_opt = keras.optimizers.Adadelta()
    # model_opt = keras.optimizers.Adam()

    newModel.compile(loss="categorical_crossentropy",#keras.losses.binary_crossentropy
                optimizer=model_opt,
                metrics=['accuracy'])
    

    return newModel
#%%
"""
Call Model
"""
model_3cnn = Model_3CNN()
model_3cnn.summary()



#%%
"""
Collecting Data
"""

# x_train, y_train = loading_data(imagepath = ImagePath, data_dict = training_pd, Norm_dict = Norm_dict, start=0, stop=558519)
# x_test, y_test = loading_data(imagepath = ImagePath, data_dict = test_pd, Norm_dict = Norm_dict, start=0, stop=58977)
x_test, y_test = loading_data(imagepath = ImagePath, data_dict = test_pd, Norm_dict = Norm_dict, start=0, stop=58977)

for i, element in enumerate(range(0,len(training_pd), 100000)):
    logging.info("{}/{}".format(i, len(training_pd)//100000))

    x_train, y_train = loading_data(imagepath = ImagePath, data_dict = training_pd, Norm_dict = Norm_dict, start= element, stop=int(element+100000))


    #%%
    # time counter
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    ticks_1 = time.time()
    ############################################################################################################################################################

    """
    Model Training
    """

    nb_train_samples = 558519 #558519
    nb_val_samples = 58977
    nb_test_samples = 58977 #58977

    batch_size = 512

    # train_generator = data_generator(imagepath = ImagePath,  data_dict = training_pd, nb_samples = nb_train_samples, batch_size = batch_size, Norm_dict = Norm_dict)
    # val_generator = data_generator(imagepath = ImagePath,  data_dict = test_pd, nb_samples = nb_test_samples, batch_size = batch_size, Norm_dict = Norm_dict)


    check_list=[]
    csv_logger = CSVLogger('/AICourse2022/alan_THDM/Model_3CNN/training_log_500_norm_ppjjjj_'+str(i)+'.csv')
    checkpoint = ModelCheckpoint(
                        filepath='/AICourse2022/alan_THDM/Model_3CNN/checkmodel_500_norm_ppjjjj_'+str(i)+'.h5',
                        save_best_only=True,
                        verbose=1)
    check_list.append(checkpoint)
    check_list.append(csv_logger)

    # # generator method
    # history_model_3cnn = model_3cnn.fit(
    #                                     train_generator,
    #                                     epochs= 500,
    #                                     steps_per_epoch= nb_train_samples // batch_size,
    #                                     validation_data = val_generator,
    #                                     validation_steps = nb_val_samples // batch_size,
    #                                     callbacks=check_list,
    #                                     verbose=1
    #                                     )

    # normal method
    history_model_3cnn = model_3cnn.fit(
                                        x = x_train, 
                                        y = y_train ,
                                        validation_data= (x_test, y_test),
                                        batch_size=512,
                                        epochs= 500,
                                        callbacks=check_list,
                                        verbose=1
                                        )

    model_3cnn.save("/AICourse2022/alan_THDM/Model_3CNN/model_3cnn_500_norm_ppjjjj_"+str(i)+".h5")

    # %%
    ############################################################################################################################################################
    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))

# %%
