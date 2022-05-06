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

#%%
start = time.time()

HOMEPATH = "/home/u5/THDM/"
path =  HOMEPATH + "Data_High_Level_Features/"
leadingjet_imagespath =  HOMEPATH + "Leading_Jet_Images_trimmed/"
subleadingjet_imagespath =  HOMEPATH + "SubLeading_Jet_Images_trimmed/"
rotated_event_imagespath =  HOMEPATH + "Rotated_Event_Images/"
nonrotated_event_imagespath =  HOMEPATH + "NonRotated_Event_Images/"
savepath = HOMEPATH

process_path_ppHhh_leadingjet = sorted(glob.glob(leadingjet_imagespath+"ppHhh"+"*.npz"))
process_path_ttbar_leadingjet = sorted(glob.glob(leadingjet_imagespath+"ttbar"+"*.npz"))
process_path_ppbbbb_leadingjet = sorted(glob.glob(leadingjet_imagespath+"ppbbbb"+"*.npz"))
process_path_jjjb_leadingjet = sorted(glob.glob(leadingjet_imagespath+"ppjjjb"+"*.npz"))
process_path_jjjj_leadingjet = sorted(glob.glob(leadingjet_imagespath+"ppjjjj"+"*.npz"))

process_path_ppHhh_subleadingjet = sorted(glob.glob(leadingjet_imagespath+"ppHhh"+"*.npz"))
process_path_ttbar_subleadingjet = sorted(glob.glob(leadingjet_imagespath+"ttbar"+"*.npz"))
process_path_ppbbbb_subleadingjet = sorted(glob.glob(leadingjet_imagespath+"ppbbbb"+"*.npz"))
process_path_jjjb_subleadingjet = sorted(glob.glob(leadingjet_imagespath+"ppjjjb"+"*.npz"))
process_path_jjjj_subleadingjet = sorted(glob.glob(leadingjet_imagespath+"ppjjjj"+"*.npz"))

process_path_ppHhh_rotated_event = sorted(glob.glob(rotated_event_imagespath+"ppHhh"+"*.npz"))
process_path_ttbar_rotated_event = sorted(glob.glob(rotated_event_imagespath+"ttbar"+"*.npz"))
process_path_ppbbbb_rotated_event = sorted(glob.glob(rotated_event_imagespath+"ppbbbb"+"*.npz"))
process_path_jjjb_rotated_event = sorted(glob.glob(rotated_event_imagespath+"ppjjjb"+"*.npz"))
process_path_jjjj_rotated_event = sorted(glob.glob(rotated_event_imagespath+"ppjjjj"+"*.npz"))
# process_path_jjjj_2 = []
# process_path_jjjj_2 = sorted(glob.glob("/home/u5/THDM/sample_flow/Data_High_Level_Features/ppjjjj"+"*.npz"))
# process_path_jjjj.extend(process_path_jjjj_2)


process = {
            "ppHhh" : [process_path_ppHhh_leadingjet, process_path_ppHhh_subleadingjet, process_path_ppHhh_rotated_event],
            "ttbar" : [process_path_ttbar_leadingjet, process_path_ttbar_subleadingjet, process_path_ttbar_rotated_event],
            "ppbbbb" : [process_path_ppbbbb_leadingjet, process_path_ppbbbb_subleadingjet, process_path_ppbbbb_rotated_event],
            "ppjjjb" : [process_path_jjjb_leadingjet, process_path_jjjb_subleadingjet, process_path_jjjb_rotated_event],
            "ppjjjj" : [process_path_jjjj_leadingjet, process_path_jjjj_subleadingjet, process_path_jjjj_rotated_event]
        }  



#%%        
for element in process:
    logging.info("Process is {}".format(element))
    logging.info("\n")
    if element == "ppHhh": #ppHhh event
        label = 0
    elif  element == "ttbar": #ttbar event
        label = 1
    elif  element == "ppbbbb": #ppbbbb event
        label = 2
    # elif  element == "ppjjjb": #ppjjjb event
    #     label = 3
    elif  element == "ppjjjj": #ppjjjj event
        label = 4
    else:
        continue
    
    total_length = 0
    folder_index = 0
    
        
    for i, (leadingjet_path, subleadingjet_path, rotated_eventpath) in enumerate(zip(process[element][0],process[element][1],process[element][2])):  

        leadingjetimages = np.load(leadingjet_path)["jet_images"]
        subleadingjetimages = np.load(subleadingjet_path)["jet_images"]
        rotated_eventimages = np.load(rotated_eventpath)["event_images"]

        logging.info("{} {} {}".format(leadingjet_path, subleadingjet_path, rotated_eventpath))
        logging.info("{}'s leadingjet number: {}".format(element + "_" + str(i),len(leadingjetimages)))
        logging.info("{}'s subleadingjet number: {}".format(element + "_" + str(i),len(subleadingjetimages)))
        logging.info("{}'s rotated_event number: {}".format(element + "_" + str(i),len(rotated_eventimages)))
        time.sleep(1)

        ######################################################################################
        """
        Storing Each Image
        """
        logging.info("Storing Each Image")
        logging.info("\r")

        for j, (leadingjetimage, subleadingjetimage, rotated_eventimage) in enumerate(tqdm(zip(leadingjetimages,subleadingjetimages,rotated_eventimages)): 
            
            if total_length%25000 == 0 :
                folder_index += 1

                if os.path.exists(savepath + str("Image_Directory") + "/" + str(element) + "_" + str(folder_index)) == 0:
                    os.mkdir(savepath + str("Image_Directory") + "/" + str(element) + "_" +str(folder_index))

                jet_filepath = savepath + str("Image_Directory") + "/" + str(element) + "_" + str(folder_index)
                logging.info("\n")
                logging.info("folder index = {}".format( folder_index))
                logging.info("jet_filepath= {}".format(jet_filepath))
                logging.info("\n")
            np.savez_compressed(jet_filepath+"/x_"+str(total_length)+".npz", 
                                leading_jet_image = leadingjetimage, 
                                subleading_jet_image = subleadingjetimage, 
                                rotated_event_image = rotated_eventimage, 
                                label=label,
                                index = total_length
                                )
            total_length += 1
            
                    logging.info("total_length: {}".format(total_length))
            
                    if total_length == 5:
                        break
                    
    break




final = time.time()
logging.info("total time: {:.3f} min".format((final-start)/60))
# %%
