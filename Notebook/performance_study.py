#%%
import numpy as np
import pandas as pd

#Common packages
import copy
from tqdm import tqdm
 
#Plot's Making  Packages
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

#System Packages
import importlib
import time
import os
from os.path import isdir, isfile, join
import glob
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

# Learning packages
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import dump, load

"""
Self-define Function
"""
from function import Basic_Preselection

# %%
path = "/AICourse2022/alan_THDM/MC_Data/"
# path = "/AICourse2022/alan_THDM/Data_High_Level_Features/"
path_ppjjjj = "/AICourse2022/alan_THDM/Data_High_Level_Features/"
   
"""
Load Original Data
"""
sig_ppHhh = pd.read_csv(path+"ppHhh_test.csv")
bkg_ttbar = pd.read_csv(path+"ttbar_test.csv")
# sig_ppHhh = pd.read_csv(path_ppjjjj+"ppHhh.csv")
# bkg_ttbar = pd.read_csv(path_ppjjjj+"ttbar.csv")
# bkg_ppbbbb = pd.read_csv(path+"ppbbbb.csv")
# bkg_ppjjjb = pd.read_csv(path+"ppjjjb.csv")
bkg_ppjjjj = pd.read_csv(path_ppjjjj+"ppjjjj.csv")

logging.info("Before Preselection")
logging.info("Signal(ppHhh) Length: {}".format(len(sig_ppHhh)))
logging.info("BKG(ttabr) Length: {}".format(len(bkg_ttbar)))
# logging.info("BKG(ppbbbb) Length: {}".format(len(bkg_ppbbbb)))
logging.info("BKG(ppjjjj) Length: {}".format(len(bkg_ppjjjj)))
#%%
"""
Basic Preselection
"""
sig_ppHhh = Basic_Preselection(sig_ppHhh)
bkg_ttbar = Basic_Preselection(bkg_ttbar)
# bkg_ppbbbb = Basic_Preselection(bkg_ppbbbb)
# bkg_ppjjjb = Basic_Preselection(bkg_ppjjjb)
bkg_ppjjjj = Basic_Preselection(bkg_ppjjjj)

logging.info("After Preselection")
logging.info("Signal(ppHhh) Length: {}".format(len(sig_ppHhh)))
logging.info("BKG(ttabr) Length: {}".format(len(bkg_ttbar)))
# logging.info("BKG(ppbbbb) Length: {}".format(len(bkg_ppbbbb)))
logging.info("BKG(ppjjjj) Length: {}".format(len(bkg_ppjjjj)))

#%%
process = {
            "ppHhh" : sig_ppHhh,
            "ttbar" : bkg_ttbar,
            # "ppbbbb" : bkg_ppbbbb,
            # "ppjjjb" : bkg_ppjjjb,
            "ppjjjj" : bkg_ppjjjj,
        }  

#%%
for pro in process:
    if pro == "ppHhh":
        label = 1
    else:
        label =0

    logging.info("Process: {}".format(pro))
    process[pro]["label"] = np.full(len(process[pro]),label)
    process[pro]["BDT_pred"] =  np.load("/AICourse2022/alan_THDM/THDM_with_ML/Model/prediction/"+str(pro)+"_BDT_prediction_trail2000_ppjjjj_test.npy")[:,1]
    process[pro]["3CNN_pred"] =  np.load("/AICourse2022/alan_THDM/prediction/"+str(pro)+"_3cnn_prediction_test_ppjjjjmodel.npy")[:,0]


#%%
process_mclength = {
            "ppHhh" : 2000000,
            "ttbar" : 2000000,
            # "ppbbbb" : 2000000,
            # "ppjjjb" : 2000000,
            "ppjjjj" : 12790000,
        }  

#%%
def calculate_eff(cut_BDT: float=0.919, cut_3CNN: float=0.0, Luminosity: int=3000, b_tag_eff: float=0.77**4) -> pd.DataFrame:

    process_eff = {
                    "preselection" : np.zeros(len(process)),
                    "four_b_tag" : np.zeros(len(process)),
                    "BDT" : np.zeros(len(process)),
                    "3CNN" : np.zeros(len(process)),
                    "MJJ" : np.zeros(len(process)),
                    "survival" : np.zeros(len(process)),
                }

    expected_event = {
                        "ppHhh" : (0.81186/1000)*0.8715*(0.3560**2)*Luminosity*1000*b_tag_eff,
                        # "ppHhh" : (1./1000)*Luminosity*1000*0.5824*0.5824*b_tag_eff,
                        "ttbar" : 260.3554*Luminosity*1000*b_tag_eff*0.192,
                        # "ttbar" : 225.71*139*1000*b_tag_eff,*0.192,
                        "ppbbbb" : 0.4070 *Luminosity*1000*b_tag_eff,
                        # "ppjjjb" : 450.04*Luminosity*1000,
                        "ppjjjj" : 11087.8358304*Luminosity*1000*b_tag_eff*0.015,
                        # "ppjjjj" : 8299*139*1000*b_tag_eff*0.015,
                    }  

    for j , element in enumerate(process):
        tmp = process[element]
        
        """
        Preselection
        """
        process_eff["preselection"][j] = len(tmp)/process_mclength[element]#/len(process[element])


        """
        4b-tag 
        """
        tmp = tmp[(tmp["four_b_tag"] == 1)]
        process_eff["four_b_tag"][j] = len(tmp)/process_mclength[element]


        """
        BDT cut
        """
        tmp = tmp[(tmp["BDT_pred"] > cut_BDT)]
        process_eff["BDT"][j] = len(tmp)/process_mclength[element]


        """
        3CNN cut
        """
        tmp = tmp[(tmp["3CNN_pred"] > cut_3CNN)]
        process_eff["3CNN"][j] = len(tmp)/process_mclength[element]


        """
        1100 GeV > M(J1,J2) > 900 GeV
        """
        tmp = tmp[(tmp["MJJ"] > 900) & (tmp["MJJ"] < 1100)]
        process_eff["MJJ"][j] = len(tmp)/process_mclength[element]


        process_eff["survival"][j] = expected_event[element]*process_eff["MJJ"][j]


    logging.info("\r")       
    logging.info("Preselection Efficiency")
    logging.info("\r")

    preselection_process = {"ppHhh":[],
                            "ttbar":[],
                            # "ppbbbb":[],
                            # "ppjjjb":[],
                            "ppjjjj":[], 
                        }


    for element in process_eff:
        tmp = process_eff[element]
            
        for i, var in enumerate(preselection_process):
            preselection_process[var].append(np.round(tmp[i],10))


    preselection_process = pd.DataFrame(preselection_process,
                index=["preselection","four_b_tag","BDT","3CNN","MJJ","survival"]
                )

    return preselection_process

# %%
selection_1 = calculate_eff(cut_BDT=0.919, cut_3CNN=0.0, Luminosity=3000, b_tag_eff=0.77**4)
selection_1
# %%
selection_2 = calculate_eff(cut_BDT=0.964, cut_3CNN=0.0, Luminosity=3000, b_tag_eff=0.77**4)
selection_2
# %%
selection_3 = calculate_eff(cut_BDT=0.0, cut_3CNN=0.99, Luminosity=3000, b_tag_eff=0.77**4)
selection_3
# %%
