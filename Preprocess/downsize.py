#!/bin/python3

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

from BranchClass import *

import Event_List 
import jet_trimming 
import JSS 
from tqdm import tqdm
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)





# Returns the difference in phi between phi, and phi_center
# as a float between (-PI, PI)
def dphi(phi,phi_c):

    dphi_temp = phi - phi_c
    while dphi_temp > np.pi:
        dphi_temp = dphi_temp - 2*np.pi
    while dphi_temp < -np.pi:
        dphi_temp = dphi_temp + 2*np.pi
    return (dphi_temp)


logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
ticks_1 = time.time()
###################################################################################
"""
Input Check and Setting
"""
###################################################################################
logging.info("Input Check and Setting")
logging.info("\n")

if len(sys.argv) < 5:
    raise ValueError("********* Usage: python3 downsize.py <path-of-file>/XXXX.root mc_type save_path file_number *********")
    
try:
    data_path = str(sys.argv[1])
    
    mc_type = str(sys.argv[2])
    
    save_path = str(sys.argv[3])

    file_number = int(sys.argv[4])
    
    MCData = uproot.open(data_path)["Delphes;1"]
    
except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 downsize.py <path-of-file>/XXXX.root mc_type save_path file_number *********")
    sys.exit(1)
    

    
###################################################################################
"""
Read Data and Jet Clustering 
"""
###################################################################################

logging.info("Read Data and Downsize into h5 format")
logging.info("\n")

# HOMEPATH = "/home/u5/THDM/"
# path =  HOMEPATH + "DownsizeData/"
path = save_path + "/"

eventpath = path + "EventList_" + str(mc_type)+ "_" + str(file_number) + ".h5"
GenParticle = BranchGenParticles(MCData)
Jet10 = BranchParticleFlowJet10(MCData)
EventWeight = Event_Weight(MCData)
EventList = Event_List.Event_List(GenParticle, Jet10, EventWeight, path=eventpath)



logging.info("\n")
ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))