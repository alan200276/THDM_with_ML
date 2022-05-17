#!/bin/python3
# %%
import uproot
import pyjet
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import importlib
import time
import re
from tqdm import tqdm
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

from BranchClass import *

import jet_trimming 
import JSS 
from make_jet_image import make_jet_image 
from make_event_image import make_event_image,  Rotate_Event_List


import h5py

# Returns the difference in phi between phi, and phi_center
# as a float between (-PI, PI)
def dphi(phi,phi_c):

    dphi_temp = phi - phi_c
    while dphi_temp > np.pi:
        dphi_temp = dphi_temp - 2*np.pi
    while dphi_temp < -np.pi:
        dphi_temp = dphi_temp + 2*np.pi
    return (dphi_temp)

def MJJ(j1,j2):
    pt1, eta1, phi1, m1 = j1.pt,j1.eta,j1.phi,j1.mass
    pt2, eta2, phi2, m2 = j2.pt,j2.eta,j2.phi,j2.mass
    
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    
    return np.sqrt((e1+e2)**2-(px1+px2)**2-(py1+py2)**2-(pz1+pz2)**2)


def ET(jet):
    pt = jet.pt
    m = jet.mass
    ET = np.sqrt(m**2 + pt**2)
    return  ET

def XHH(jet1, jet2):
    m1, m2 = jet1.mass, jet2.mass
    XHH = np.sqrt( (m1-124)**2/(0.1*(m1+1e-5)) + (m2-115)**2/(0.1*(m2+1e-5)))
    return  XHH
# %%
###################################################################################
"""
Input Check and Setting
"""
###################################################################################
logging.info("Input Check and Setting")

if len(sys.argv) < 4:
    raise ValueError("********* Usage: python3 preprocess.py <path-of-file>/XXXX.h5 PRO file_number *********")
    

try:
    data_path = str(sys.argv[1])
    
    PRO = str(sys.argv[2])
    
    file_number = int(sys.argv[3])

except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 preprocess.py <path-of-file>/XXXX.h5 PRO file_number *********")
    sys.exit(1)


# %%
###################################################################################
"""
Read Data and Jet Clustering 
"""
###################################################################################

logging.info("Read Data and Jet Clustering ")
logging.info("\n")


hf_read = h5py.File(data_path, 'r')

process_list_clustered = []
weight_list = []

four_b_tag, Higgs_candidate_4b, four_b_raw_weight = [], [], []
three_b_tag, Higgs_candidate_3b, three_b_raw_weight = [], [], []
two_b_tag, Higgs_candidate_2b, two_b_raw_weight = [], [], []

logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
time.sleep(1)
ticks_1 = time.time()

for i in tqdm(range(len(hf_read["GenParticle"]))):

    """
    Jet clustering 
    Fat jet: R = 1
    Anti-kt
    """
    to_cluster = np.core.records.fromarrays(hf_read.get('GenParticle/dataset_'+ str(i))[:9], 
                                            names="pt, eta, phi, mass, PID, Status, Charge, B_tag, weight",
                                            formats = "f8, f8, f8, f8, f8, f8, f8, f8,  f8"
                                           )
    pt_min = 25
    sequence_cluster = pyjet.cluster(to_cluster, R = 1, p = -1) # p = -1: anti-kt , 0: Cambridge-Aachen(C/A), 1: kt
    jets_cluster = sequence_cluster.inclusive_jets(pt_min)
    process_list_clustered.append(jets_cluster)
    for constituent in jets_cluster[0]:
        weight_list.append(constituent.weight)
        break


    """
    4b category: have two b-tagged jets associated with each $H$ candidate
    """
    Higgs_candidate_tmp = []
    if len(jets_cluster) >=2:
        for jet in jets_cluster[:2]:
            B_tag = 0
            for constituent in jet:
                if constituent.B_tag == 1:
                    B_tag += 1
            if B_tag >= 2:
                Higgs_candidate_tmp.append(jet)

    if len(Higgs_candidate_tmp) >= 2:
        Higgs_candidate_4b.append(Higgs_candidate_tmp)
        four_b_tag.append(1)
        for constituent in Higgs_candidate_tmp[0]:
                four_b_raw_weight.append(constituent.weight)
                break

    else:
        four_b_tag.append(0)


    """
    3b category: have two $b$-tagged jets associated with one $H$ candidate 
    and exactly one $b$-tagged jet associated with the other $H$ candidate.
    """
    Higgs_candidate_tmp_2b, Higgs_candidate_tmp_1b = [], []
    if len(jets_cluster) >=2:
        B_tag = 0
        for constituent in jets_cluster[0]:
            if constituent.B_tag == 1:
                B_tag += 1
        if B_tag >= 2:
            Higgs_candidate_tmp_2b.append(jets_cluster[0])

        B_tag = 0
        for constituent in jets_cluster[1]:
            if constituent.B_tag == 1:
                B_tag += 1
        if B_tag >= 2:
            Higgs_candidate_tmp_2b.append(jets_cluster[1])

        B_tag = 0
        for constituent in jets_cluster[0]:
            if constituent.B_tag == 1:
                B_tag += 1
        if B_tag == 1:
            Higgs_candidate_tmp_1b.append(jets_cluster[0])

        B_tag = 0  
        for constituent in jets_cluster[1]:
            if constituent.B_tag == 1:
                B_tag += 1
        if B_tag == 1:
            Higgs_candidate_tmp_1b.append(jets_cluster[1])


    if len(Higgs_candidate_tmp_2b) == 1 and len(Higgs_candidate_tmp_1b) == 1:
        Higgs_candidate_3b.append([Higgs_candidate_tmp_2b[0],Higgs_candidate_tmp_1b[0]])
        three_b_tag.append(1)
        for constituent in Higgs_candidate_tmp[0]:
            three_b_raw_weight.append(constituent.weight)
            break

    else:
        three_b_tag.append(0)


    """
    2b category: have exactly one $b$-tagged jet associated with each $H$ candidate
    """
    Higgs_candidate_tmp = []
    if len(jets_cluster) >=2:
        for jet in jets_cluster[:2]:
            B_tag = 0
            for constituent in jet:
                if constituent.B_tag == 1:
                    B_tag += 1
            if B_tag == 1:
                Higgs_candidate_tmp.append(jet)

    if len(Higgs_candidate_tmp) >= 2:
        Higgs_candidate_2b.append(Higgs_candidate_tmp)
        two_b_tag.append(1)
        for constituent in Higgs_candidate_tmp[0]:
            two_b_raw_weight.append(constituent.weight)
            break
            
    else:
        two_b_tag.append(0)

        
    # if i == 1000:
    #     break

logging.info("There are {} events (process_list_clustered).".format(len(process_list_clustered)))
logging.info("There are {} events (4b Higgs_candidate).".format(len(Higgs_candidate_4b)))
logging.info("There are {} events (3b Higgs_candidate).".format(len(Higgs_candidate_3b)))
logging.info("There are {} events (2b Higgs_candidate).".format(len(Higgs_candidate_2b)))

logging.info("\n")
ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
logging.info("\n")

logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
ticks_1 = time.time()


# %%
###################################################################################
"""
Create Pandas DataFrame
"""
###################################################################################

logging.info("Create Pandas DataFrame")
logging.info("\n")

HOMEPATH = "/home/u5/THDM/sample_flow/"
path =  HOMEPATH + "Data_High_Level_Features/"
leadingjet_imagespath =  HOMEPATH + "Leading_Jet_Images_trimmed/"
subleadingjet_imagespath =  HOMEPATH + "SubLeading_Jet_Images_trimmed/"
rotated_event_imagespath =  HOMEPATH + "Rotated_Event_Images/"
nonrotated_event_imagespath =  HOMEPATH + "NonRotated_Event_Images/"

dataframe = pd.DataFrame()

###################################################################################
    
logging.info("Selection and Trimming")
logging.info("\n")    
logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
time.sleep(1)
ticks_1 = time.time()
    
features = [
            "PRO",
            "MJJ_0","delta_eta_0","XHH_0",
            "MJ1_0","PTJ1_0","eta1_0","phi1_0",
            "t211_0","D211_0","D221_0","C211_0","C221_0",
            "MJ2_0","PTJ2_0","eta2_0","phi2_0",
            "t212_0","D212_0","D222_0","C212_0","C222_0",
            "MJJ","delta_eta","XHH",
            "MJ1","PTJ1","eta1","phi1",
            "t211","D211","D221","C211","C221",
            "MJ2","PTJ2","eta2","phi2",
            "t212","D212","D222","C212","C222",
            "four_b_tag","three_b_tag","two_b_tag","weight",
            "eventindex"
           ]

event_list = []
leading_trimmed_jet, subleading_trimmed_jet = [], []
leading_trimmed_jet_eta_phi, subleading_trimmed_jet_eta_phi = [], []

k = 0
for N in tqdm(range(len(process_list_clustered))):
    
    """
    Trigger
    """
    if ET(process_list_clustered[N][0]) < 420 or process_list_clustered[N][0].mass < 35:
        continue
        
    """
    >= 2 jets
    """
    if len(process_list_clustered[N]) < 2:
        continue

    jet_1_untrimmed = process_list_clustered[N][0] #leading jet's information
    jet_2_untrimmed = process_list_clustered[N][1] #subleading jet's information
    
#     """
#     Basic Selection
#     """

#     if (jet_1_untrimmed.pt < 450) or (jet_2_untrimmed.pt < 250) :
#         continue

#     if (abs(jet_1_untrimmed.eta) >= 2) or (jet_1_untrimmed.mass <= 50) :
#         continue

#     if (abs(jet_2_untrimmed.eta) >= 2) or (jet_2_untrimmed.mass <= 50) :
#         continue
        
        
#     """
#     |\Delta\eta| < 1.3
#     """        
#     if (abs(jet_1_untrimmed.eta - jet_2_untrimmed.eta) > 1.3) :
#         continue
        
#     """
#     M(jj) > 700 GeV 
#     """    
#     if MJJ(jet_1_untrimmed,jet_2_untrimmed) < 700:
#         continue


    """
    for event images
    """
    # pt, eta, phi, mass, PID, Status, Charge, B_tag, weight
    event = hf_read.get('GenParticle/dataset_'+ str(N))[:9]
    event = event.T
    event_list.append(event)

        
    var = []



    var.append(PRO)

    var.append(MJJ(jet_1_untrimmed,jet_2_untrimmed))
    var.append(abs(jet_1_untrimmed.eta-jet_2_untrimmed.eta))
    var.append(XHH(jet_1_untrimmed,jet_2_untrimmed))
    
   


    t1 = JSS.tn(jet_1_untrimmed, n=1)
    t2 = JSS.tn(jet_1_untrimmed, n=2)
    t21_untrimmed = t2 / t1 if t1 > 0.0 else 0.0

    ee2 = JSS.CalcEECorr(jet_1_untrimmed, n=2, beta=1.0)
    ee3 = JSS.CalcEECorr(jet_1_untrimmed, n=3, beta=1.0)
    d21_untrimmed = ee3/(ee2**3) if ee2>0 else 0
    d22_untrimmed = ee3**2/((ee2**2)**3) if ee2>0 else 0
    c21_untrimmed = ee3/(ee2**2) if ee2>0 else 0
    c22_untrimmed = ee3**2/((ee2**2)**2) if ee2>0 else 0 

    var.append(jet_1_untrimmed.mass)
    var.append(jet_1_untrimmed.pt)
    var.append(jet_1_untrimmed.eta)
    var.append(jet_1_untrimmed.phi)
    var.append(t21_untrimmed)
    var.append(d21_untrimmed)
    var.append(d22_untrimmed)
    var.append(c21_untrimmed)
    var.append(c22_untrimmed)



    t1 = JSS.tn(jet_2_untrimmed, n=1)
    t2 = JSS.tn(jet_2_untrimmed, n=2)
    t21_untrimmed = t2 / t1 if t1 > 0.0 else 0.0

    ee2 = JSS.CalcEECorr(jet_2_untrimmed, n=2, beta=1.0)
    ee3 = JSS.CalcEECorr(jet_2_untrimmed, n=3, beta=1.0)
    d21_untrimmed = ee3/(ee2**3) if ee2>0 else 0
    d22_untrimmed = ee3**2/((ee2**2)**3) if ee2>0 else 0
    c21_untrimmed = ee3/(ee2**2) if ee2>0 else 0
    c22_untrimmed = ee3**2/((ee2**2)**2) if ee2>0 else 0 

    var.append(jet_2_untrimmed.mass)
    var.append(jet_2_untrimmed.pt)
    var.append(jet_2_untrimmed.eta)
    var.append(jet_2_untrimmed.phi)
    var.append(t21_untrimmed)
    var.append(d21_untrimmed)
    var.append(d22_untrimmed)
    var.append(c21_untrimmed)
    var.append(c22_untrimmed)



    """
    Jet Trimming
    """
    jet_1_trimmed = jet_trimming.jet_trim(jet_1_untrimmed)[0]   #trimming jet's information
    jet_2_trimmed = jet_trimming.jet_trim(jet_2_untrimmed)[0]   #trimming jet's information

    """
    for jet images
    """
    leading_trimmed_jet.append(jet_1_trimmed)
    subleading_trimmed_jet.append(jet_2_trimmed)

    """
    for rotated event images
    """
    leading_trimmed_jet_eta_phi.append([jet_1_trimmed.eta,jet_1_trimmed.phi])
    subleading_trimmed_jet_eta_phi.append([jet_2_trimmed.eta,jet_2_trimmed.phi])

    var.append(MJJ(jet_1_trimmed,jet_2_trimmed))
    var.append(abs(jet_1_trimmed.eta-jet_2_trimmed.eta))
    var.append(XHH(jet_1_trimmed,jet_2_trimmed))

    t1 = JSS.tn(jet_1_trimmed, n=1)
    t2 = JSS.tn(jet_1_trimmed, n=2)
    t21_trimmed = t2 / t1 if t1 > 0.0 else 0.0

    ee2 = JSS.CalcEECorr(jet_1_trimmed, n=2, beta=1.0)
    ee3 = JSS.CalcEECorr(jet_1_trimmed, n=3, beta=1.0)
    d21_trimmed = ee3/(ee2**3) if ee2>0 else 0
    d22_trimmed = ee3**2/((ee2**2)**3) if ee2>0 else 0
    c21_trimmed = ee3/(ee2**2) if ee2>0 else 0
    c22_trimmed = ee3**2/((ee2**2)**2) if ee2>0 else 0 

    var.append(jet_1_trimmed.mass)
    var.append(jet_1_trimmed.pt)
    var.append(jet_1_trimmed.eta)
    var.append(jet_1_trimmed.phi)
    var.append(t21_trimmed)
    var.append(d21_trimmed)
    var.append(d22_trimmed)
    var.append(c21_trimmed)
    var.append(c22_trimmed)


    t1 = JSS.tn(jet_2_trimmed, n=1)
    t2 = JSS.tn(jet_2_trimmed, n=2)
    t21_trimmed = t2 / t1 if t1 > 0.0 else 0.0

    ee2 = JSS.CalcEECorr(jet_2_trimmed, n=2, beta=1.0)
    ee3 = JSS.CalcEECorr(jet_2_trimmed, n=3, beta=1.0)
    d21_trimmed = ee3/(ee2**3) if ee2>0 else 0
    d22_trimmed = ee3**2/((ee2**2)**3) if ee2>0 else 0
    c21_trimmed = ee3/(ee2**2) if ee2>0 else 0
    c22_trimmed = ee3**2/((ee2**2)**2) if ee2>0 else 0 

    var.append(jet_2_trimmed.mass)
    var.append(jet_2_trimmed.pt)
    var.append(jet_2_trimmed.eta)
    var.append(jet_2_trimmed.phi)
    var.append(t21_trimmed)
    var.append(d21_trimmed)
    var.append(d22_trimmed)
    var.append(c21_trimmed)
    var.append(c22_trimmed)

    var.append(four_b_tag[N])
    var.append(three_b_tag[N])
    var.append(two_b_tag[N])
    var.append(weight_list[N])

    
    var.append(k)

    dataframe_tmp = pd.DataFrame([var],columns=features)
    dataframe = dataframe.append(dataframe_tmp, ignore_index = True)

    k += 1

    # if k >= 10000:
    #     break
        

logging.info("There are {} jets.".format(len(dataframe)))
    
dataframe.to_csv( path + str(PRO) + "_" + str(file_number) + ".csv", index = 0)

ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
logging.info("\n")
# %%

"""
Making Jet Images
"""

print("\n")
print("Making Jet Images")
print("=====START=====")
t1_time = time.time()
time.sleep(1)

make_jet_image(leading_trimmed_jet,leadingjet_imagespath,PRO,file_number)
make_jet_image(subleading_trimmed_jet,subleadingjet_imagespath,PRO,file_number)

t2_time = time.time()
print("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2_time-t1_time)/60.))
print("=====Finish=====")
print("\n")

# %%
from make_event_image import make_event_image,  Rotate_Event_List

# %%
"""
Making Event Images
"""
print("\n")
print("Making Event Images")
print("=====START=====")
t1_time = time.time()
time.sleep(1)

make_event_image(event_list, 
                imagespath = nonrotated_event_imagespath, 
                PRO = PRO, 
                file_number = file_number)

event_list_rotated = Rotate_Event_List(event_list, leading_trimmed_jet_eta_phi)

make_event_image(event_list_rotated, 
                imagespath = rotated_event_imagespath, 
                PRO = PRO, 
                file_number = file_number)


t2_time = time.time()
print("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2_time-t1_time)/60.))
print("=====Finish=====")
print("\n")
