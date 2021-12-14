#!/bin/python3

import time
import pyjet
import numpy as np

from BranchClass import *

def jet_trim(jet0, pt_cut= 0.05, R1 = 0.2):
    # Define a cut threshold that the subjets have to meet (i.e. 5% of the original jet pT)
    jet0_max = jet0.pt
    jet0_cut = jet0_max*pt_cut

    # Grab the subjets by clustering with R1
    # subjets = pyjet.cluster(jet0.constituents_array(), R=R1, p=1) # p = -1: anti-kt , 0: Cambridge-Aachen(C/A), 1: kt
    
    jet_constituent_list = []
    for constituent in jet0:
        tmp = []
        tmp.append(constituent.pt)
        tmp.append(constituent.eta)
        tmp.append(constituent.phi)
        tmp.append(constituent.mass)
        tmp.append(constituent.PID)
        tmp.append(constituent.Status)
        tmp.append(constituent.Charge)
        tmp.append(constituent.B_tag)
        tmp.append(constituent.weight)
        jet_constituent_list.append(tmp)


    to_cluster = np.core.records.fromarrays(np.array(jet_constituent_list).T, 
                                        names="pt, eta, phi, mass, PID, Status, Charge, B_tag, weight",
                                        formats = "f8, f8, f8, f8, f8, f8, f8, f8,  f8"
                                        )
    subjets = pyjet.cluster(to_cluster, R=R1, p=1) # p = -1: anti-kt , 0: Cambridge-Aachen(C/A), 1: kt
    subjet_array = subjets.inclusive_jets()
    j0 = []
    if (subjet_array[0].pt >= jet0_cut):
        for ij, subjet in enumerate(subjet_array):
            if subjet.pt < jet0_cut:
                # subjet doesn't meet the percentage cut on the original jet pT
                continue
            if subjet.pt >= jet0_cut:
                # Get the subjets pt, eta, phi constituents
                for constituent in subjet:
                    tmp = []
                    tmp.append(constituent.pt)
                    tmp.append(constituent.eta)
                    tmp.append(constituent.phi)
                    tmp.append(constituent.mass)
                    tmp.append(constituent.PID)
                    tmp.append(constituent.Status)
                    tmp.append(constituent.Charge)
                    tmp.append(constituent.B_tag)
                    tmp.append(constituent.weight)
                    j0.append(tmp)
            else:
                for constituent in subjet_array[0]:
                    tmp = []
                    tmp.append(constituent.pt)
                    tmp.append(constituent.eta)
                    tmp.append(constituent.phi)
                    tmp.append(constituent.mass)
                    tmp.append(constituent.PID)
                    tmp.append(constituent.Status)
                    tmp.append(constituent.Charge)
                    tmp.append(constituent.B_tag)
                    tmp.append(constituent.weight)
                j0 = np.array(tmp)*0
    
    to_cluster = np.core.records.fromarrays(np.array(j0).T, 
                                            names="pt, eta, phi, mass, PID, Status, Charge, B_tag, weight",
                                            formats = "f8, f8, f8, f8, f8, f8, f8, f8,  f8"
                                           )
    sequence = pyjet.cluster(to_cluster, R=1.0, p=-1)
    jet = sequence.inclusive_jets()
    return jet

