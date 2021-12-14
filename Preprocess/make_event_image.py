#!/usr/bin/python3
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
import copy
from tqdm import tqdm
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

from BranchClass import *

import jet_trimming 
import JSS 

import h5py
# %%
def ET(jet):
    pt = jet.pt
    m = jet.mass
    ET = np.sqrt(m**2 + pt**2)
    return  ET

def dphi(phi,phi_c):

    dphi_temp = phi - phi_c
    while dphi_temp > np.pi:
        dphi_temp = dphi_temp - 2*np.pi
    while dphi_temp < -np.pi:
        dphi_temp = dphi_temp + 2*np.pi
    return (dphi_temp)


# %%
def make_event_image(event_list, imagespath = "path", PRO = "PRO", file_number = 0):

    logging.info("Make Event Images")
    logging.info("\n")    
    logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    time.sleep(1)
    ticks_1 = time.time()


    # event_list shape - >(N,X,9) 
    # N events, X constituents, 9 physical information
    # 9 physical information: pt, eta, phi, mass, PID, Status, Charge, B_tag, weight"
    
    width, height= 40, 40

    image_list = []
    image_0 = np.zeros((width,height)) #Charged pt 
    image_1 = np.zeros((width,height)) #Neutral pt
    image_2 = np.zeros((width,height)) #Charged multiplicity

    for event in event_list:
        for x in range(len(event)):
            phi_index = math.floor(width*event[x,2]//(2*math.pi)+width//2)
            eta_index = math.floor(height*event[x,1]//10+height/2) 
            eta_index = min(eta_index,height-1)
            eta_index = max(0,eta_index)
            phi_index = int(phi_index);eta_index = int(eta_index)
            if (event[x,6] == 0):  # neutral
                image_0[phi_index,eta_index] = image_0[phi_index,eta_index] + event[x,0]
            elif (event[x,6] == 1):  # charged
                image_1[phi_index,eta_index] = image_1[phi_index,eta_index] + event[x,0]
                image_2[phi_index,eta_index] = image_2[phi_index,eta_index] + 1
                
        image_0 = np.divide(image_0,np.sum(image_0))
        image_1 = np.divide(image_1,np.sum(image_1))
        image_2 = np.divide(image_2,np.sum(image_2))
        image_list.append(np.array([image_0,image_1,image_2]))

    logging.info("There are {} Event images.".format(len(image_list)))
    logging.info("\n")
    np.savez(imagespath + str(PRO)+ "_" + str(file_number)+".npz",  event_images = image_list)


    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
    logging.info("\n")  
    
 

def Rotate_Event_List(event_list, leading_jet_list):
    # event_list shape - >(N,X,9) 
    # N events, X constituents, 9 physical information
    # 9 physical information: pt, eta, phi, mass, PID, Status, Charge, B_tag, weight"

    event_list_rotated = copy.deepcopy(event_list)
    
    # move phi to pi/2
    for i in tqdm(range(len(event_list_rotated))):
        event_list_rotated[i][:,2] = event_list_rotated[i][:,2] - np.array(leading_jet_list)[i,1] + np.pi/2 
        for j in range(len(event_list_rotated[i][:,2])):
            if event_list_rotated[i][j,2] > np.pi:
                event_list_rotated[i][j,2] = event_list_rotated[i][j,2] - 2*np.pi

        # flip Eta(Leading Jet) to positive position 
        if np.array(leading_jet_list)[i,0] < 0:
            event_list_rotated[i][:,1] = -1*event_list_rotated[i][:,1]

    return event_list_rotated
