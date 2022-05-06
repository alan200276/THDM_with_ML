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
# process_path_ppHhh = sorted(glob.glob("/home/u5/THDM/sample_flow/Data_High_Level_Features/ppHhh"+"*.csv"))


"""
Backgound
"""
process_path_ttbar = sorted(glob.glob(path+"ttbar"+"*.csv"))
process_path_ppbbbb = sorted(glob.glob(path+"ppbbbb"+"*.csv"))
process_path_jjjb = sorted(glob.glob(path+"ppjjjb"+"*.csv"))
process_path_jjjj = sorted(glob.glob(path+"ppjjjj"+"*.csv"))
# process_path_jjjj_2 = []
process_path_jjjj_2 = sorted(glob.glob("/home/u5/THDM/sample_flow/Data_High_Level_Features/ppjjjj"+"*.csv"))
process_path_jjjj.extend(process_path_jjjj_2)

#%%
# sig_ppHhh = High_Level_Features(process_path_ppHhh)

# bkg_ttbar = High_Level_Features(process_path_ttbar)
# bkg_ppbbbb = High_Level_Features(process_path_ppbbbb)
# bkg_ppjjjb = High_Level_Features(process_path_jjjb)
# bkg_ppjjjj = High_Level_Features(process_path_jjjj)


sig_ppHhh = pd.read_csv(path+"ppHhh.csv")
bkg_ttbar = pd.read_csv(path+"ttbar.csv")
bkg_ppbbbb = pd.read_csv(path+"ppbbbb.csv")
bkg_ppjjjb = pd.read_csv(path+"ppjjjb.csv")
bkg_ppjjjj = pd.read_csv(path+"ppjjjj.csv")

#%%
process_length = {
            "ppHhh" : len(process_path_ppHhh)*100000,
            # "ppHhh" : len(process_path_ppHhh)*10000,
            "ttbar" : len(process_path_ttbar)*100000,
            "ppbbbb" : len(process_path_ppbbbb)*100000,
            "ppjjjb" : len(process_path_jjjb)*100000,
            "ppjjjj" : (len(process_path_jjjj)-len(process_path_jjjj_2))*100000+len(process_path_jjjj_2)*10000,
        }  

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


def XHH(jet1_mass, jet2_mass):
    m1, m2 = jet1_mass, jet2_mass
    XHH = np.sqrt( ((m1-124)/(0.1*(m1+1e-5)))**2 +  ((m2-115)/(0.1*(m2+1e-5)))**2 )
    return  np.nan_to_num(XHH)


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
                "survival" : np.zeros(len(process)),
              }


process_selected = {
                    "ppHhh" : 0,
                    "ttbar" : 0,
                    "ppbbbb" : 0,
                    "ppjjjb" : 0,
                    "ppjjjj" : 0,
                    }  

# expected_event = {
#                     "ppHhh" : (0.81186/1000)*0.8715*(0.3560**2)*3000*1000,
#                     "ttbar" : 260.3554*3000*1000,
#                     "ppbbbb" : 0.4070 *3000*1000,
#                     "ppjjjb" : 450.04*3000*1000,
#                     "ppjjjj" : 11087.8358304*3000*1000,
#                 }  

expected_event = {
                    "ppHhh" : (0.81186/1000)*0.8715*(0.3560**2)*139*1000*(0.77**4),
                    # "ppHhh" : (1./1000)*139*1000*0.6*0.6*(0.77**4),
                    "ttbar" : 260.3554*139*1000*(0.77**4)*0.27515,
                    "ppbbbb" : 0.4070 *139*1000,
                    "ppjjjb" : 450.04*139*1000,
                    # "ppjjjj" : 11087.8358304*139*1000*(0.77**4)*0.02365529,
                    "ppjjjj" : 8299*139*1000*(0.77**4)*0.02365529,
                }  


"""
Mass Cut and PT cut
"""
#######################
for j , element in enumerate(process):
    tmp = process[element]
    tmp["ET"] = ET(tmp["PTJ1_0"], tmp["MJ1_0"])

    tmp["Xhh_0"] = XHH(tmp["MJ1_0"], tmp["MJ2_0"])
    tmp["Xhh"] = XHH(tmp["MJ1"], tmp["MJ2"])
    
    """
    Trigger
    """
    tmp = tmp[(tmp["ET"] > 420) & (tmp["MJ1_0"] > 35)]
    preselection["Trigger"][j] = len(tmp)/process_length[element]#/len(process[element])
    
    """
    PT(J1) > 450 GeV 
    """
    
    tmp = tmp[(tmp["PTJ1"] > 450)]
    preselection["PT_J1"][j] = len(tmp)/process_length[element]#/len(process[element])

    # """
    # PT(J1) > 325 GeV  (for M(H)=800)
    # """
    
    # tmp = tmp[(tmp["PTJ1_0"] > 325)]
    # preselection["PT_J1"][j] = len(tmp)/process_length[element]#/len(process[element])
    
    
    """
    PT(J2) > 250 GeV 
    """
    
    tmp = tmp[(tmp["PTJ2"] > 250)]
    preselection["PT_J2"][j] = len(tmp)/process_length[element]#/len(process[element])
    
    """
    |Eta(J1)| < 2 & |Eta(J2)| < 2 
    """
    
    tmp = tmp[(abs(tmp["eta1"]) < 2) & (abs(tmp["eta2"]) < 2)]
    preselection["Eta"][j] = len(tmp)/process_length[element]#/len(process[element])
    
    """
    M(J1) > 50 GeV &  M(J2) > 50 GeV
    """
    
    tmp = tmp[(tmp["MJ1"] > 50) & (tmp["MJ2"] > 50)]
    preselection["M_J"][j] = len(tmp)/process_length[element]#/len(process[element])
    
    """
    |Delta[Eta(J1),Eta(J2)]| < 1.3
    """
    
    tmp = tmp[(abs(tmp["delta_eta"]) < 1.3)]
    preselection["Delta_Eta"][j] = len(tmp)/process_length[element]#/len(process[element])
    
    
    # """
    # X(H,H) < 5
    # """
    
    # tmp = tmp[(tmp["XHH"] < 5)]
    # preselection["XHH"][j] = len(tmp)/len(process[element])
    
    
    """
    X(H,H) < 1.6
    """
    
    # tmp = tmp[(tmp["XHH_0"] < 1.6)]
    tmp = tmp[(tmp["Xhh"] < 1.6)]   
    preselection["XHH"][j] = len(tmp)/process_length[element]#/len(process[element])




    # """
    # M(J1,J2) > 700 GeV
    # """
    
    # tmp = tmp[(tmp["MJJ"] > 700)]
    # preselection["MJJ"][j] = len(tmp)/process_length[element]#/len(process[element])


    """
    1200 GeV > M(J1,J2) > 900 GeV
    """
    
    # tmp = tmp[(tmp["MJJ"] > 900) & (tmp["MJJ"] < 1200)]
    tmp = tmp[(tmp["MJJ"] > 900) & (tmp["MJJ"] < 1200)]
    preselection["MJJ"][j] = len(tmp)/process_length[element]#/len(process[element])


    # """
    # 850 GeV > M(J1,J2) > 750 GeV  (for M(H)=800)
    # """
    
    # tmp = tmp[(tmp["MJJ_0"] > 750) & (tmp["MJJ_0"] < 850)]
    # preselection["MJJ"][j] = len(tmp)/process_length[element]#/len(process[element])

    """
    4b-tag 
    """
    
    tmp = tmp[(tmp["four_b_tag"] == 1)]
    preselection["four_b_tag"][j] = len(tmp)/process_length[element]#/len(process[element])
    
    # """
    # 3b-tag 
    # """
    
    # tmp = tmp[(tmp["three_b_tag"] == 1)]
    # preselection["three_b_tag"][j] = len(tmp)/process_length[element]#/len(process[element])
    
    # """
    # 2b-tag 
    # """
    
    # tmp = tmp[(tmp["two_b_tag"] == 1)]
    # preselection["two_b_tag"][j] = len(tmp)/process_length[element]#/len(process[element])
    
    preselection["survival"][j] = expected_event[element]*preselection["four_b_tag"][j]

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
            index=["Trigger","PT_J1","PT_J2","Eta","M_J","Delta_Eta","XHH","MJJ","four_b_tag","three_b_tag","two_b_tag","survival"]
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
Mjj= TotalSamples.Signal_Background("MJJ")

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
                  "delta_eta", "Xhh_0",#"XHH"
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
        if kinematic == "Xhh_0":
            xmin, xmax = 0, 10
        else:
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

# %%

# %%
