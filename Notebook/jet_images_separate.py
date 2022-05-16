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
def Preselection(data : pd.DataFrame)-> pd.DataFrame:
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


#%%
path = "/home/u5/THDM/"


sig_ppHhh = pd.read_csv(path+"ppHhh.csv")
bkg_ttbar = pd.read_csv(path+"ttbar.csv")
bkg_ppbbbb = pd.read_csv(path+"ppbbbb.csv")
bkg_ppjjjb = pd.read_csv(path+"ppjjjb.csv")
bkg_ppjjjj = pd.read_csv(path+"ppjjjj.csv")

sig_ppHhh_index = Preselection(sig_ppHhh).index
bkg_ttbar_index = Preselection(bkg_ttbar).index
bkg_ppbbbb_index = Preselection(bkg_ppbbbb).index
bkg_ppjjjb_index = Preselection(bkg_ppjjjb).index
bkg_ppjjjj_index = Preselection(bkg_ppjjjj).index


#%%
start = time.time()

HOMEPATH = "/home/u5/THDM/"
path =  HOMEPATH + "Data_High_Level_Features/"
leadingjet_imagespath =  HOMEPATH + "Leading_Jet_Images_trimmed/"
subleadingjet_imagespath =  HOMEPATH + "SubLeading_Jet_Images_trimmed/"
rotated_event_imagespath =  HOMEPATH + "Rotated_Event_Images/"
nonrotated_event_imagespath =  HOMEPATH + "NonRotated_Event_Images/"
savepath = HOMEPATH


process_path_ppHhh_csv = sorted(glob.glob(path+"ppHhh"+"*.csv"))
process_path_ttbar_csv = sorted(glob.glob(path+"ttbar"+"*.csv"))
process_path_ppbbbb_csv = sorted(glob.glob(path+"ppbbbb"+"*.csv"))
process_path_jjjb_csv = sorted(glob.glob(path+"ppjjjb"+"*.csv"))
process_path_jjjj_csv = sorted(glob.glob(path+"ppjjjj"+"*.csv"))
# process_path_jjjj_2 = []
process_path_jjjj_2 = sorted(glob.glob("/home/u5/THDM/sample_flow/Data_High_Level_Features/ppjjjj"+"*.csv"))
process_path_jjjj_csv.extend(process_path_jjjj_2)


process = {
            "ppHhh" : [[], [] , [], process_path_ppHhh_csv],
            "ttbar" : [[], [] , [], process_path_ttbar_csv],
            "ppbbbb" : [[], [] , [], process_path_ppbbbb_csv],
            "ppjjjb" : [[], [] , [], process_path_jjjb_csv],
            "ppjjjj" : [[], [] , [], process_path_jjjj_csv],
        }  
for pro in process:
    if pro == "ppjjjj":
        tmp = []
        for subpath in process[pro][3]:
            tmp.append(subpath.split("_")[-1].split(".")[-2])
        
        for i, file_number in enumerate(tmp):
            if i < 20:
                process[pro][0].append(leadingjet_imagespath+str(pro)+"_"+str(file_number)+"_trimmed.npz")
                process[pro][1].append(subleadingjet_imagespath+str(pro)+"_"+str(file_number)+"_trimmed.npz")
                process[pro][2].append(rotated_event_imagespath+str(pro)+"_"+str(file_number)+".npz")
            else:
                process[pro][0].append("/home/u5/THDM/sample_flow/Leading_Jet_Images_trimmed/ppjjjj_"+str(file_number)+"_trimmed.npz")
                process[pro][1].append("/home/u5/THDM/sample_flow/SubLeading_Jet_Images_trimmed/ppjjjj_"+str(file_number)+"_trimmed.npz")
                process[pro][2].append("/home/u5/THDM/sample_flow/Rotated_Event_Images/ppjjjj_"+str(file_number)+".npz")
    else:
        tmp = []
        for subpath in process[pro][3]:
            tmp.append(subpath.split("_")[-1].split(".")[-2])
        
        for i, file_number in enumerate(tmp):
            process[pro][0].append(leadingjet_imagespath+str(pro)+"_"+str(file_number)+"_trimmed.npz")
            process[pro][1].append(subleadingjet_imagespath+str(pro)+"_"+str(file_number)+"_trimmed.npz")
            process[pro][2].append(rotated_event_imagespath+str(pro)+"_"+str(file_number)+".npz")

# process_path_ppHhh_leadingjet = sorted(glob.glob(leadingjet_imagespath+"ppHhh"+"*.npz"))
# process_path_ttbar_leadingjet = sorted(glob.glob(leadingjet_imagespath+"ttbar"+"*.npz"))
# process_path_ppbbbb_leadingjet = sorted(glob.glob(leadingjet_imagespath+"ppbbbb"+"*.npz"))
# process_path_jjjb_leadingjet = sorted(glob.glob(leadingjet_imagespath+"ppjjjb"+"*.npz"))
# process_path_jjjj_leadingjet = sorted(glob.glob(leadingjet_imagespath+"ppjjjj"+"*.npz"))
# process_path_jjjj_leadingjet_2 = []
# process_path_jjjj_leadingjet_2 = sorted(glob.glob("/home/u5/THDM/sample_flow/Leading_Jet_Images_trimmed/ppjjjj"+"*.npz"))
# process_path_jjjj_leadingjet.extend(process_path_jjjj_leadingjet_2)

# process_path_ppHhh_subleadingjet = sorted(glob.glob(subleadingjet_imagespath+"ppHhh"+"*.npz"))
# process_path_ttbar_subleadingjet = sorted(glob.glob(subleadingjet_imagespath+"ttbar"+"*.npz"))
# process_path_ppbbbb_subleadingjet = sorted(glob.glob(subleadingjet_imagespath+"ppbbbb"+"*.npz"))
# process_path_jjjb_subleadingjet = sorted(glob.glob(subleadingjet_imagespath+"ppjjjb"+"*.npz"))
# process_path_jjjj_subleadingjet = sorted(glob.glob(subleadingjet_imagespath+"ppjjjj"+"*.npz"))
# process_path_jjjj_subleadingjet_2 = []
# process_path_jjjj_subleadingjet_2 = sorted(glob.glob("/home/u5/THDM/sample_flow/SubLeading_Jet_Images_trimmed/ppjjjj"+"*.npz"))
# process_path_jjjj_subleadingjet.extend(process_path_jjjj_subleadingjet_2)


# process_path_ppHhh_rotated_event = sorted(glob.glob(rotated_event_imagespath+"ppHhh"+"*.npz"))
# process_path_ttbar_rotated_event = sorted(glob.glob(rotated_event_imagespath+"ttbar"+"*.npz"))
# process_path_ppbbbb_rotated_event = sorted(glob.glob(rotated_event_imagespath+"ppbbbb"+"*.npz"))
# process_path_jjjb_rotated_event = sorted(glob.glob(rotated_event_imagespath+"ppjjjb"+"*.npz"))
# process_path_jjjj_rotated_event = sorted(glob.glob(rotated_event_imagespath+"ppjjjj"+"*.npz"))
# process_path_jjjj_rotated_event_2 = []
# process_path_jjjj_rotated_event_2 = sorted(glob.glob("/home/u5/THDM/sample_flow/Rotated_Event_Images/ppjjjj"+"*.npz"))
# process_path_jjjj_rotated_event.extend(process_path_jjjj_rotated_event_2)


# process = {
#             "ppHhh" : [process_path_ppHhh_leadingjet, process_path_ppHhh_subleadingjet, process_path_ppHhh_rotated_event],
#             "ttbar" : [process_path_ttbar_leadingjet, process_path_ttbar_subleadingjet, process_path_ttbar_rotated_event],
#             "ppbbbb" : [process_path_ppbbbb_leadingjet, process_path_ppbbbb_subleadingjet, process_path_ppbbbb_rotated_event],
#             "ppjjjb" : [process_path_jjjb_leadingjet, process_path_jjjb_subleadingjet, process_path_jjjb_rotated_event],
#             "ppjjjj" : [process_path_jjjj_leadingjet, process_path_jjjj_subleadingjet, process_path_jjjj_rotated_event]
#         }  

#%%
index_dict = {"ppHhh" : sig_ppHhh_index,
              "ttbar" : bkg_ttbar_index,
              "ppbbbb" : bkg_ppbbbb_index,
              "ppjjjb" : bkg_ppjjjb_index,
              "ppjjjj" : bkg_ppjjjj_index,
            }

length = {"ppHhh" : len(sig_ppHhh),
              "ttbar" : len(bkg_ttbar),
              "ppbbbb" : len(bkg_ppbbbb),
              "ppjjjb" : len(bkg_ppjjjb),
              "ppjjjj" : len(bkg_ppjjjj),
            }


#%%        
for element in process:
    logging.info("Process is {}".format(element))
    logging.info("\n")


    # if element == "ppjjjj":
    #     continue

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
    file_index = 0
    
    for_dict_image = []
    for_dict_label = []
    for_dict_index = []

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
        ############################################################################################################
        for j, (leadingjetimage, subleadingjetimage, rotated_eventimage) in enumerate(tqdm(zip(leadingjetimages,subleadingjetimages,rotated_eventimages))): 
            
            if (total_length not in index_dict[element]):
                total_length += 1
                continue

            if file_index%25000 == 0 :
                folder_index += 1

            if os.path.exists(savepath + str("Image_Directory") + "/" + str(element) + "_" + str(folder_index)) == 0:
                os.mkdir(savepath + str("Image_Directory") + "/" + str(element) + "_" +str(folder_index))

            jet_filepath = savepath + str("Image_Directory") + "/" + str(element) + "_" + str(folder_index)
            logging.info("\n")
            logging.info("folder index = {}".format( folder_index))
            logging.info("jet_filepath= {}".format(jet_filepath))
            logging.info("\n")

            np.savez_compressed(jet_filepath+"/x_"+str(file_index)+".npz", 
                                leading_jet_image = leadingjetimage, 
                                subleading_jet_image = subleadingjetimage, 
                                rotated_event_image = rotated_eventimage, 
                                label=label,
                                index = total_length
                                )

            for_dict_image.append(str(element) + "_" + str(folder_index)+"/x_"+str(file_index)+".npz")
            for_dict_label.append(label)
            for_dict_index.append(total_length)
            
            logging.info("total_length: {}".format(total_length))
            logging.info("file_index: {}".format(file_index))

            total_length += 1
            file_index += 1
        ############################################################################################################
        #     if total_length == 10:
        #         break
        # break

        ############################################################################################################
        # for j in range(length[element]):

        #     if (total_length not in index_dict[element]):
        #         total_length += 1
        #         continue

        #     if file_index%25000 == 0 :
        #         folder_index += 1

        #     for_dict_image.append(str(element) + "_" + str(folder_index)+"/x_"+str(file_index)+".npz")
        #     for_dict_label.append(label)
        #     for_dict_index.append(total_length)
            
        #     logging.info("total_length: {}".format(total_length))
        #     logging.info("file_index: {}".format(file_index))

        #     total_length += 1
        #     file_index += 1
        ############################################################################################################

    dict_pd = pd.DataFrame()
    dict_pd["Image"] = for_dict_image
    dict_pd["Y"] = for_dict_label
    dict_pd["index"] = for_dict_index
    dict_pd.to_csv(savepath + str("Image_Directory") + "/" + str(element) + "_dict.csv", index = 0)

    # break


final = time.time()
logging.info("total time: {:.3f} min".format((final-start)/60))

# %%

# list_leading = []
# list_subleading = []
# list_event = []
# list_csv = []
# for leadingjet_path in process["ppjjjj"][0]:  
#     list_leading.append(leadingjet_path.split("_")[-2])
# for subleadingjet_path in process["ppjjjj"][1]:  
#     list_subleading.append(subleadingjet_path.split("_")[-2])
# for rotated_eventpath in process["ppjjjj"][2]:  
#     list_event.append(rotated_eventpath.split("_")[-1].split(".")[-2])

# for aaaa in process["ppjjjj"][3]:  
#     list_csv.append(aaaa.split("_")[-1].split(".")[-2])
#     # break
# # %%
# # list_leading = np.sort(np.array(list_leading))
# # list_subleading = np.sort(np.array(list_subleading))
# # list_event = np.sort(np.array(list_event))
# list_leading = np.array(list_leading)
# list_subleading = np.array(list_subleading)
# list_event = np.array(list_event)
# list_csv = np.array(list_csv)

# # %%
# for i, element in enumerate(list_leading):
#     if element != list_subleading[i]:
#         print(element)

# # %%
# for i, element in enumerate(list_leading):
#     if element != list_event[i]:
#         print(i, element)
# # %%
# for i, element in enumerate(list_event):
#     if element != list_leading[i]:
#         print(i, element)
# # %%
# for i, element in enumerate(list_csv):
#     if element != list_subleading[i]:
#         print("csv", i, element)
#         print("subleading", i, list_subleading[i])
# # %%
