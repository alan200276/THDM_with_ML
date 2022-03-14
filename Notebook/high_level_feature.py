#%%
# Numerical Packages
import numpy as np
import pandas as pd

#Common packages
import copy
from tqdm import tqdm
import re 
import glob
 
#Plot's Making  Packages
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

#System Packages
import importlib
import time
import os
from os.path import isdir, isfile, join
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)


logging.info("Numpy Version: {}".format(np.__version__))
logging.info("Pnadas Version: {}".format(pd.__version__))
#%%
class Samples:
    def __init__(self,\
                 ppHhh,\
                 ttbar,\
                 ppbbbb,\
                 ppjjjb,\
                 ppjjjj,\
                 four_b_tag=False,\
                 three_b_tag=False,\
                 two_b_tag=False,\
                ):
        
        if four_b_tag == True:
            self.ppHhh = ppHhh[ppHhh["four_b_tag"]==1]
            self.ttbar = ttbar[ttbar["four_b_tag"]==1]
            self.ppbbbb = ppbbbb[ppbbbb["four_b_tag"]==1]
            self.ppjjjb = ppjjjb[ppjjjb["four_b_tag"]==1]
            self.ppjjjj = ppjjjj[ppjjjj["four_b_tag"]==1]
            
        elif three_b_tag == True:
            self.ppHhh = ppHhh[ppHhh["three_b_tag"]==1]
            self.ttbar = ttbar[ttbar["three_b_tag"]==1]
            self.ppbbbb = ppbbbb[ppbbbb["three_b_tag"]==1]
            self.ppjjjb = ppjjjb[ppjjjb["three_b_tag"]==1]
            self.ppjjjj = ppjjjj[ppjjjj["three_b_tag"]==1]
        
        elif two_b_tag == True:
            self.ppHhh = ppHhh[ppHhh["two_b_tag"]==1]
            self.ttbar = ttbar[ttbar["two_b_tag"]==1]
            self.ppbbbb = ppbbbb[ppbbbb["two_b_tag"]==1]
            self.ppjjjb = ppjjjb[ppjjjb["two_b_tag"]==1]
            self.ppjjjj = ppjjjj[ppjjjj["two_b_tag"]==1]
            
        else:
            self.ppHhh = ppHhh
            self.ttbar = ttbar
            self.ppbbbb = ppbbbb
            self.ppjjjb = ppjjjb
            self.ppjjjj = ppjjjj
        
    def Signal_only(self,feature):
        H = [self.ppHhh[feature]]
        return H
    
    def Background_only(self,feature):
        QCD = [self.ttbar[feature],self.ppbbbb[feature],self.ppjjjb[feature] ,self.ppjjjj[feature]]
        return QCD
    
    def Signal_Background(self,feature):
        Both = [self.ppHhh[feature],self.ttbar[feature],self.ppbbbb[feature],self.ppjjjb[feature],self.ppjjjj[feature]]
        return Both
#%%
def High_Level_Features(csv_file=[]):
    
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
path = "/home/u5/THDM/Data_High_Level_Features/"
    
    
"""
Signal
"""
process_path_ppHhh = sorted(glob.glob(path+"ppHhh"+"*.csv"))


"""
Backgound
"""
process_path_ttbar = sorted(glob.glob(path+"ttbar"+"*.csv"))
process_path_ppbbbb = sorted(glob.glob(path+"ppbbbb"+"*.csv"))
process_path_jjjb = sorted(glob.glob(path+"ppjjjb"+"*.csv"))
process_path_jjjj = sorted(glob.glob(path+"ppjjjj"+"*.csv"))


#%%
sig_ppHhh = High_Level_Features(process_path_ppHhh)

bkg_ttbar = High_Level_Features(process_path_ttbar)
bkg_ppbbbb = High_Level_Features(process_path_ppbbbb)
bkg_ppjjjb = High_Level_Features(process_path_jjjb)
bkg_ppjjjj = High_Level_Features(process_path_jjjj)


process = {
            "ppHhh" : sig_ppHhh,
            "ttbar" : bkg_ttbar,
            "ppbbbb" : bkg_ppbbbb,
            "ppjjjb" : bkg_ppjjjb,
            "ppjjjj" : bkg_ppjjjj,
        }  
# %%
def ET(pt, m):
    ET = np.sqrt(m**2 + pt**2)
    return  ET



preselection = {
                "Trigger" : np.zeros(len(process)),
                "PT_J1" : np.zeros(len(process)),
                "PT_J2" : np.zeros(len(process)),
                "Eta" : np.zeros(len(process)),
                "M_J" : np.zeros(len(process)),
                "Delta_Eta" : np.zeros(len(process)),
                "XHH" : np.zeros(len(process)),
                "MJJ" : np.zeros(len(process)),
                "four_b_tag" : np.zeros(len(process)),
                "three_b_tag" : np.zeros(len(process)),
                "two_b_tag" : np.zeros(len(process)),
              }


process_selected = {
                    "ppHhh" : 0,
                    "ttbar" : 0,
                    "ppbbbb" : 0,
                    "ppjjjb" : 0,
                    "ppjjjj" : 0,
                    }  

"""
Mass Cut and PT cut
"""
#######################
for j , element in enumerate(process):
    tmp = process[element]
    tmp["ET_0"] = ET(tmp["PTJ1_0"], tmp["MJ1_0"])
    
    """
    Trigger
    """
    tmp = tmp[(tmp["ET_0"] > 420) & (tmp["MJ1_0"] > 35)]
    preselection["Trigger"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])
    
    # """
    # PT(J1) > 450 GeV 
    # """
    
    # tmp = tmp[(tmp["PTJ1_0"] > 450)]
    # preselection["PT_J1"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])
    
    # """
    # PT(J2) > 250 GeV 
    # """
    
    # tmp = tmp[(tmp["PTJ2_0"] > 250)]
    # preselection["PT_J2"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])
    
    # """
    # |Eta(J1)| < 2 & |Eta(J2)| < 2 
    # """
    
    # tmp = tmp[(abs(tmp["eta1_0"]) < 2) & (abs(tmp["eta2_0"]) < 2)]
    # preselection["Eta"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])
    
    # """
    # M(J1) > 50 GeV &  M(J2) > 50 GeV
    # """
    
    # tmp = tmp[(tmp["MJ1_0"] > 50) & (tmp["MJ2_0"] > 50)]
    # preselection["M_J"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])
    
    # """
    # |Delta[Eta(J1),Eta(J2)]| < 1.3
    # """
    
    # tmp = tmp[(abs(tmp["delta_eta_0"]) < 1.3)]
    # preselection["Delta_Eta"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])
    
    
    # """
    # X(H,H) < 5
    # """
    
    # tmp = tmp[(tmp["XHH_0"] < 5)]
    # preselection["XHH"][j] = len(tmp)/len(process[element])
    
    
    # """
    # X(H,H) < 1.6
    # """
    
    # tmp = tmp[(tmp["XHH_0"] < 1.6)]
    # preselection["XHH"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])



    # """
    # M(J1,J2) > 700 GeV
    # """
    
    # tmp = tmp[(tmp["MJJ_0"] > 700)]
    # preselection["MJJ"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])


    # """
    # 1200 GeV > M(J1,J2) > 900 GeV
    # """
    
    # tmp = tmp[(tmp["MJJ_0"] > 900) & (tmp["MJJ_0"] 2< 1200)]
    # preselection["MJJ"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])

    # # """
    # # 4b-tag 
    # # """
    
    # # tmp = tmp[(tmp["four_b_tag"] == 1)]
    # # preselection["four_b_tag"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])
    
    # """
    # 3b-tag 
    # """
    
    # tmp = tmp[(tmp["three_b_tag"] == 1)]
    # preselection["three_b_tag"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])
    
    # """
    # 2b-tag 
    # """
    
    # tmp = tmp[(tmp["two_b_tag"] == 1)]
    # preselection["two_b_tag"][j] = len(tmp)/(len(process_path_ppHhh)*100000)#/len(process[element])
    
    process_selected[element] = tmp
#######################
#%%



logging.info("\r")       
logging.info("Preselection Efficiency")
logging.info("\r")


preselection_process = {"ppHhh":[],
                        "ttbar":[],
                        "ppbbbb":[],
                        "ppjjjb":[],
                        "ppjjjj":[],
                       }

for element in preselection:
    tmp = preselection[element]
        
    for i, var in enumerate(preselection_process):
        preselection_process[var].append(np.round(tmp[i],10))


preselection_process = pd.DataFrame(preselection_process,
            index=["Trigger","PT_J1","PT_J2","Eta","M_J","Delta_Eta","XHH","MJJ","four_b_tag","three_b_tag","two_b_tag"]
            )


#%%
preselection_process
# %%
TotalSamples = Samples(process_selected["ppHhh"],
                       process_selected["ttbar"],
                       process_selected["ppbbbb"],
                       process_selected["ppjjjb"],
                       process_selected["ppjjjj"],
                       four_b_tag = 0,
                       three_b_tag = 0,
                       two_b_tag = 0,
                      )
# %%
def HIST(process, length, title, colors, linestyle,xpo=1,ypo=1):
    hist, bins = np.histogram(process, bins=length)
    plt.step(bins[:-1], hist.astype(np.float32) / hist.sum(), linestyle ,color= colors ,where='mid',linewidth=5, alpha=0.7, label=title[i])
#     plt.legend(bbox_to_anchor=(xpo, ypo),ncol=1,fontsize=30, edgecolor = "w",fancybox=False, framealpha=0)
    plt.legend(loc="best",ncol=1,fontsize=20, edgecolor = "w",fancybox=False, framealpha=0)


title = ["ppHhh","ttbar","ppbbbb","ppjjjb","ppjjjj",
        ]
colors = ["green","red","blue","purple","Orange"
#           "cyan","black","Orange","lightblue"
         ]


linestyle = ["-","-.",":","--","o",
#              "--","o","v",":"
            ]
# %%
Mjj= TotalSamples.Signal_Background("MJJ_0")

fig, ax = plt.subplots(1,1, figsize=(12,9))
for i, element in enumerate(Mjj):
    length = np.linspace(0,2000,101)
    HIST(element, length, title,colors[i],linestyle[i])
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.xlim((0,2000))
    plt.xlabel("$M_{hh}$ GeV", fontsize=25,horizontalalignment='right',x=1) 

    
# plt.ylim((0.))
# plt.yscale("log")
plt.ylabel("1/N dN/d$M_{hh}$ / 20 GeV", fontsize=25, horizontalalignment='right',y=1)
# plt.savefig("../Figures/m_hh_ihtmin_850.pdf", transparent=True, bbox_inches='tight')  #save figure as png
plt.show()   
# %%
jet_kinematic = [
                  "MJJ","MJ1","PTJ1","MJ2","PTJ2",
                  "delta_eta", "XHH"
                ]
jet_kinematic_name = [
                       "$M_{JJ}$", "$M_{J_1}$", "$p_{T_{J_1}}$", "$M_{J_2}$", "$p_{T_{J_2}}$",
                       "$\Delta\eta$", "$X_{HH}$"
                     ]


for index, kinematic in enumerate(jet_kinematic):
    Kinematic= TotalSamples.Signal_Background(kinematic)


    fig, ax = plt.subplots(1,1, figsize=(12,9))
    for i, element in enumerate(Kinematic):

#         xmin, xmax = 0, np.max(process)
        xmin = np.sort(Kinematic[0])[int(len(Kinematic[0])*1/2000)] 
        xmax = np.sort(Kinematic[0])[int(len(Kinematic[0])*1990/2000)]
        length = np.linspace(xmin,xmax,201)
        HIST(element, length, title,colors[i],linestyle[i])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlim((xmin,xmax*0.9))
        plt.xlabel(jet_kinematic_name[index]+" [GeV]", fontsize=25,horizontalalignment='right',x=1) 


    # plt.ylim((0.))
    unit = np.around((xmax-xmin)/200, decimals=2)
    
    plt.ylabel("1/N dN/d" +jet_kinematic_name[index]+ "/ "+str(unit) , fontsize=25, horizontalalignment='right',y=1)
    # plt.savefig("./Plots/m_ww_parton.png", transparent=True, bbox_inches='tight')  #save figure as png
    plt.show()  
# %%
jet_substructure = [
                't211', 'D211', 'D221', 'C211', 'C221', 
                't212', 'D212', 'D222', 'C212', 'C222'
                ]
jet_substructure_name = [
                       '$\\tau^{\\beta =1}_{21}$($J_1$)', 
                       '$D^{\\beta =1}_{2}$($J_1$)', 
                       '$D^{\\beta =2}_{2}$($J_1$)', 
                       '$C^{\\beta =1}_{2}$($J_1$)', 
                       '$C^{\\beta =2}_{2}$($J_1$)', 
                       '$\\tau^{\\beta =1}_{21}$($J_2$)', 
                       '$D^{\\beta =1}_{2}$($J_2$)', 
                       '$D^{\\beta =2}_{2}$($J_2$)', 
                       '$C^{\\beta =1}_{2}$($J_2$)', 
                       '$C^{\\beta =2}_{2}$($J_2$)', 
                     ]


for index, substructure in enumerate(jet_substructure):
    Substructure= TotalSamples.Signal_Background(substructure)


    fig, ax = plt.subplots(1,1, figsize=(12,9))
    for i, element in enumerate(Substructure):

#         xmin, xmax = 0, np.max(process)
        xmin = np.sort(Substructure[0])[int(len(Substructure[0])*1/2000)] 
        xmax = np.sort(Substructure[0])[int(len(Substructure[0])*1990/2000)]
        length = np.linspace(xmin,xmax,201)
        HIST(element, length, title,colors[i],linestyle[i])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlim((xmin,xmax*0.9))
        plt.xlabel(jet_substructure_name[index] , fontsize=25,horizontalalignment='right',x=1) 


    # plt.ylim((0.))
    unit = np.around((xmax-xmin)/200, decimals=2)
    
    plt.ylabel("1/N dN/d" +jet_substructure_name[index]+ "/ "+str(unit) , fontsize=25, horizontalalignment='right',y=1)
    # plt.savefig("./Plots/m_ww_parton.png", transparent=True, bbox_inches='tight')  #save figure as png
    plt.show()  

# %%
