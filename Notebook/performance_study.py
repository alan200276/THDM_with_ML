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
path = "/home/u5/THDM/MC_Data/"
# path = "/AICourse2022/alan_THDM/Data_High_Level_Features/"
path_ppjjjj = "/home/u5/THDM/"
   
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
    process[pro]["BDT_pred"] =  np.load("/home/alan/ML_Analysis/THDM/Model/prediction/"+str(pro)+"_BDT_prediction_trail2000_ppjjjj_test.npy")[:,1]
    process[pro]["3CNN_pred"] =  np.load("/home/alan/ML_Analysis/THDM/Model/prediction/"+str(pro)+"_3cnn_prediction_test.npy")[:,0]
    process[pro]["BDT_argmax"] = np.argmax(np.load("/home/alan/ML_Analysis/THDM/Model/prediction/"+str(pro)+"_BDT_prediction_trail2000_ppjjjj_test.npy"),axis=1)
    process[pro]["3CNN_argmax"] =  1 - np.argmax(np.load("/home/alan/ML_Analysis/THDM/Model/prediction/"+str(pro)+"_3cnn_prediction_test.npy"),axis=1)


#%%
process_mclength = {
            "ppHhh" : 2000000,
            "ttbar" : 5000000,
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

def train_test_data(feature: str = "BDT_argmax", process: dict=process):
    process_train_test = {
                        "ppHhh" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                        "ttbar" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                        # "ppbbbb" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                        # "ppjjjb" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                        "ppjjjj" : {"training": {"X": 0, "Y": 0}, "test": {"X": 0, "Y": 0}},
                        }  

    for i, element in enumerate(process_train_test):

        process_train_test[element]["training"]["X"], \
        process_train_test[element]["test"]["X"], \
        process_train_test[element]["training"]["Y"], \
        process_train_test[element]["test"]["Y"], \
        = train_test_split( process[element][feature], process[element]["label"], test_size=0.1, random_state=42)

    #%%
    training_pd = shuffle(pd.DataFrame(process_train_test["ppjjjj"]["training"]))[:len(process["ppHhh"])]
    for element in process_train_test:
        if element == "ppjjjj":
            continue
        else:
            training_pd = pd.concat([training_pd, pd.DataFrame(process_train_test[element]["training"])], ignore_index=True)


    test_pd = shuffle(pd.DataFrame(process_train_test["ppjjjj"]["test"]))[:int(len(process["ppHhh"])/10)]
    for element in process_train_test:
        if element == "ppjjjj":
            continue
        else:
            test_pd = pd.concat([test_pd, pd.DataFrame(process_train_test[element]["test"])], ignore_index=True)

    logging.info("\n")
    logging.info("There are {} sig and {} bkg in training dataset.".format(len(training_pd[training_pd["Y"]==0]),len(training_pd[training_pd["Y"]!=0])))
    logging.info("There are {} sig and {} bkg in test dataset.".format(len(test_pd[test_pd["Y"]==0]),len(test_pd[test_pd["Y"]!=0])))
    logging.info("\n")

    return shuffle(training_pd), shuffle(test_pd)



#%%
# training_pd, test_pd = train_test_data("BDT_argmax", process)
training_pd, test_pd = train_test_data("3CNN_argmax", process)

#%%
confusion_ = confusion_matrix(training_pd["Y"], training_pd["X"])

# confusion = np.array([[confusion_[0][0]/np.sum(confusion_[0]),confusion_[0][1]/np.sum(confusion_[0])],
#                         [confusion_[1][0]/np.sum(confusion_[0]),confusion_[1][1]/np.sum(confusion_[1])]])
confusion = np.array([[confusion_[0][0]/np.sum(confusion_[:,0]),confusion_[0][1]/np.sum(confusion_[:,0])],
                        [confusion_[1][0]/np.sum(confusion_[:,0]),confusion_[1][1]/np.sum(confusion_[:,1])]])

truelist = ["sig","bkg"]
likelist = ["sig-like","bkg-like"]

s = len(truelist)
f, ax = plt.subplots(1,1, figsize=(s+4, s+4))

aa = ax.imshow(confusion.T, cmap="Oranges", origin= "upper", vmin=0, vmax=1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad="1%")
cbar = plt.colorbar(aa, cax=cax)
cbar.ax.tick_params(labelsize=15)
# cbar.ax.yaxis.set_major_locator(MaxNLocator(6))
cbar.set_label("Ratio", rotation=270, fontsize=15, labelpad=30, y=0.5)
# cbar.set_ticks([0,500,1000,1500,2000])
# cbar.ax.set_yticklabels(["0","500","1000","1500","2000"])
cbar.set_ticks([0,0.25,0.5,0.75,1])
cbar.ax.set_yticklabels(["0","0.25","0.5","0.75","1"])

ax.set_xticks(range((confusion.T).shape[1]))
ax.set_xticklabels(truelist, fontsize=15, rotation=0)
ax.set_yticks(range((confusion.T).shape[1]))
ax.set_yticklabels(likelist, fontsize=15, rotation=45)

my_colors = ["green","red"]
ax.xaxis.tick_top()
for ticklabel, tickcolor in zip(ax.get_xticklabels(), my_colors):
    ticklabel.set_color(tickcolor)
    
for ticklabel, tickcolor in zip(ax.get_yticklabels(), my_colors):
    ticklabel.set_color(tickcolor)

Terminology = np.array([["TP","FP"],["FN","TN"]])
    
for (i, j), z in np.ndenumerate(confusion.T):
    ax.text(j, i, '{:^3s}: {:0.2f}'.format(Terminology[i,j],z), ha='center', va='center',fontsize=15,color="k")
    
plt.tight_layout()
plt.show()
# %%
training_pd_3CNN, test_pd_3CNN = train_test_data("3CNN_pred", process)
training_pd_BDT, test_pd_BDT = train_test_data("BDT_pred", process)

auc_3CNN = metrics.roc_auc_score(training_pd_3CNN["Y"], training_pd_3CNN["X"]/(np.max(training_pd_3CNN["X"])))
FalsePositiveFull_3CNN, TruePositiveFull_3CNN, _ = metrics.roc_curve(training_pd_3CNN["Y"], training_pd_3CNN["X"]/(np.max(training_pd_3CNN["X"])))

auc_BDT = metrics.roc_auc_score(training_pd_BDT["Y"],training_pd_BDT["X"]/(np.max(training_pd_BDT["X"])))
FalsePositiveFull_BDT, TruePositiveFull_BDT, _ = metrics.roc_curve(training_pd_BDT["Y"], training_pd_BDT["X"]/(np.max(training_pd_BDT["X"])))

fig, ax = plt.subplots(1,1, figsize=(10,10))

plt.plot(TruePositiveFull_3CNN,1-FalsePositiveFull_3CNN,"-", color='red', linewidth = 3,label='3CNN: AUC={0:.2f}'.format(auc_3CNN))
plt.plot(TruePositiveFull_BDT,1-FalsePositiveFull_BDT,"-", color='blue', linewidth = 3,label='BDT: AUC={0:.2f}'.format(auc_BDT))


# eff_1n1c1c = sigimp_1n1c1c["ggF_eff"][np.where(sigimp_1n1c1c["sig_impro"] == max(sigimp_1n1c1c["sig_impro"]))]
# rejection_rate_1n1c1c = sigimp_1n1c1c["rejection_rate"]

# plt.scatter(eff_0n1c1c,rejection_rate_0n1c1c, marker="D", s=200 , c='green')

plt.xlim((0,1))
# plt.ylim((1,1E+4))
plt.ylim((0,1))
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
# plt.yscale("log")
plt.ylabel('Background Rejection Rate', fontsize=30,horizontalalignment='right',y=1)
plt.xlabel('Signal Efficiency ', fontsize=30,horizontalalignment='right',x=1)
plt.legend(ncol=1,fontsize=20, edgecolor = "w",fancybox=False, framealpha=0) #bbox_to_anchor=(0.7, 0.1),
plt.tight_layout()


# plt.savefig("./Plots/Comparison_ROC_all.pdf", transparent=True, bbox_inches='tight')
plt.show()
# %%
plt.figure(figsize=(10,8))
xbin = np.linspace(0,1,51)
Datatest = pd.DataFrame()
Datatest["target"] = training_pd_3CNN["Y"]
Datatest["pre"] = training_pd_3CNN["X"]/(np.max(training_pd_3CNN["X"]))
inner = Datatest[Datatest["target"]==1]
outter = Datatest[Datatest["target"]==0]

ggh_hist, ggh_bins = np.histogram(inner["pre"], bins=xbin)
plt.step(ggh_bins[:-1], ggh_hist.astype(np.float32)/sum(ggh_hist)/0.02 ,color = "blue", where='mid',linewidth=2, alpha=0.7,label="Sig (3CNN)") 
other_hist, other_bins = np.histogram(outter["pre"], bins=xbin)
plt.step(other_bins[:-1], other_hist.astype(np.float32)/sum(other_hist)/0.02 ,color = "red", where='mid',linewidth=2, alpha=0.7,label="Bkg (3CNN)") 


Datatrain = pd.DataFrame()
Datatrain["target"] =  training_pd_BDT["Y"]
Datatrain["pre_train"] = training_pd_BDT["X"]/(np.max(training_pd_BDT["X"]))
inner = Datatrain[Datatrain["target"]==1]
outter = Datatrain[Datatrain["target"]==0]

ggh_hist, ggh_bins = np.histogram(inner["pre_train"], bins=xbin)  #*ggh_new_weight[1]*0.5824
plt.scatter(ggh_bins[:-1], ggh_hist.astype(np.float32)/sum(ggh_hist)/0.02 ,marker = "+",c ="b",s=120,label="Sig (BDT)")
other_hist, other_bins = np.histogram(outter["pre_train"], bins=xbin)  #*ggh_new_weight[1]*0.5824
plt.scatter(other_bins[:-1], other_hist.astype(np.float32)/sum(other_hist)/0.02 ,marker = "+",c ="r",s=120,label="Bkg (BDT)")


# plt.xlim(0,10)
plt.xlabel("ML score", fontsize=20,horizontalalignment='right',x=1)
plt.ylabel("$1/N\,\,dN/d (ML score)$",fontsize=20,horizontalalignment='right',y=1)
plt.legend(loc="best",ncol=2,fontsize=20)
# plt.savefig("./Higgs_Pt/BDT_score.pdf", transparent=True)
plt.show()
# %%
