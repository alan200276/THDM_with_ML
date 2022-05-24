#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
import importlib
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

#%%
"""
Load Data Based on Path
"""
def High_Level_Features(csv_file: pd.DataFrame = []) -> pd.DataFrame:
    
    if len(csv_file) < 1:
        raise ValueError("Please check high-level features files!!")
        
    high_level_feature = pd.read_csv(csv_file[0])

    for i, file in enumerate(csv_file):
        if i == 0:
            continue
        else:
            dataframe = pd.read_csv(file)
            
            high_level_feature = pd.concat([high_level_feature, dataframe], ignore_index=True, axis=0,join='inner')
            
    
    logging.info( "\033[3;43m Total File Length: {} \033[0;m".format(len(high_level_feature)))
    logging.info("\r")
    
    return high_level_feature

#%%
"""
Basic Preselection
"""
def Basic_Preselection(data : pd.DataFrame)-> pd.DataFrame:
    def ET(pt, m):
        ET = np.sqrt(m**2 + pt**2)
        return  ET

    def XHH(jet1_mass, jet2_mass):
        m1, m2 = jet1_mass, jet2_mass
        XHH = np.sqrt( ((m1-124)/(0.1*(m1+1e-5)))**2 +  ((m2-115)/(0.1*(m2+1e-5)))**2 )
        return  np.nan_to_num(XHH)

    """
    Mass Cut and PT cut
    """
    data["ET"] = ET(data["PTJ1_0"], data["MJ1_0"])

    data["Xhh_0"] = XHH(data["MJ1_0"], data["MJ2_0"])
    data["Xhh"] = XHH(data["MJ1"], data["MJ2"])

    """
    Trigger
    """
    data = data[(data["ET"] > 420) & (data["MJ1_0"] > 35)]

    """
    PT(J1) > 450 GeV 
    """

    data = data[(data["PTJ1"] > 450)]


    # """
    # PT(J1) > 325 GeV (for M(H)=800)
    # """

    # data = data[(data["PTJ1_0"] > 325)] 

    """
    PT(J2) > 250 GeV 
    """

    data = data[(data["PTJ2"] > 250)]

    """
    |Eta(J1)| < 2 & |Eta(J2)| < 2 
    """

    data = data[(abs(data["eta1"]) < 2) & (abs(data["eta2"]) < 2)]

    """
    M(J1) > 50 GeV &  M(J2) > 50 GeV
    """

    data = data[(data["MJ1"] > 50) & (data["MJ2"] > 50)]


    return data


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
