#/bin/python3

#%%
import numpy as np
import copy
import importlib
import time
import pandas as pd
import os
from tqdm import tqdm
import glob
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

"""
Self-define Function
"""
from function import loading_data


os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '64'
# %%
HOMEPATH = "/AICourse2022/alan_THDM/"
ImagePath =  HOMEPATH + "Image_Directory/"
savepath = HOMEPATH + "Image_Directory/"
# %%
process = {
            "ppHhh" : 0,
            "ttbar" : 0,
            # "ppbbbb" : 0,
            # "ppjjjb" : 0,
            "ppjjjj" : 0,
              }  
    
for i, element in enumerate(process):
    logging.info("Process: {}".format(element))

    process[element] = pd.read_csv(savepath + str(element) + "_dict.csv")

    logging.info("length: {}".format(len(process[element])))
# %%
def Get_average_var(image_lists):
    tmp_av, tmp_var = np.zeros((3,40,40)), np.zeros((3,40,40))
    for i in range(3):
        tmp_av[i] = np.average(image_lists[:,i], axis=0)

        tmp_var[i] = np.var(image_lists[:,i], axis=0)

    return tmp_av, tmp_var

#%%
for pro in process:
    logging.info("Process: {}".format(pro))

    av_leading_jet, var_leading_jet = np.zeros((3,40,40)), np.zeros((3,40,40))
    av_subleading_jet, var_subleading_jet = np.zeros((3,40,40)), np.zeros((3,40,40))
    av_full_event, var_full_event = np.zeros((3,40,40)), np.zeros((3,40,40))
    length = 0

    for i, element in enumerate(range(0,len(process[pro]), 30000)):
        logging.info("{}/{}".format(i, len(process[pro])//30000))

        x_batch, _ = loading_data(imagepath = ImagePath, data_dict = process[pro], start= element, stop=int(element+30000))
        
        tmp_av_leading_jet, tmp_var_leading_jet = Get_average_var(x_batch[0])
        av_leading_jet += tmp_av_leading_jet*len(x_batch)
        var_leading_jet += tmp_var_leading_jet
        
        tmp_av_subleading_jet, tmp_var_subleading_jet = Get_average_var(x_batch[1])
        av_subleading_jet += tmp_av_subleading_jet*len(x_batch)
        var_subleading_jet += tmp_var_subleading_jet

        tmp_av_full_event, tmp_var_full_event = Get_average_var(x_batch[2])
        av_full_event += tmp_av_full_event*len(x_batch)
        var_full_event += tmp_var_full_event

        length += len(x_batch)

    av_leading_jet = av_leading_jet/length
    av_subleading_jet = av_subleading_jet/length
    av_full_event = av_full_event/length

    np.save(savepath + "average_leading_jet" + "_" + str(pro) + ".npy", av_leading_jet)
    np.save(savepath + "variance_leading_jet" + "_" + str(pro) + ".npy", var_leading_jet)

    np.save(savepath + "average_subleading_jet" + "_" + str(pro) + ".npy", av_subleading_jet)
    np.save(savepath + "variance_subleading_jet" + "_" + str(pro) + ".npy", var_subleading_jet)

    np.save(savepath + "average_full_event" + "_" + str(pro) + ".npy", av_full_event)
    np.save(savepath + "variance_full_event" + "_" + str(pro) + ".npy", var_full_event)


# %%
