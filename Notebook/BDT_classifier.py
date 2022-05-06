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
path = "/home/u5/THDM/Data_High_Level_Features/"
    
    
"""
Signal
"""
process_path_ppHhh = sorted(glob.glob(path+"ppHhh"+"*.csv"))
# process_path_ppHhh = sorted(glob.glob("/home/u5/THDM/sample_flow/Data_High_Level_Features/ppHhh"+"*.csv"))

#%%
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
sig_ppHhh = Preselection(sig_ppHhh)
#%%
bkg_ttbar = Preselection(bkg_ttbar)
bkg_ppbbbb = Preselection(bkg_ppbbbb)
bkg_ppjjjb = Preselection(bkg_ppjjjb)
bkg_ppjjjj = Preselection(bkg_ppjjjj)


#%%
sig_ppHhh["label"] = np.full(len(sig_ppHhh),1)
bkg_ttbar["label"] = np.full(len(bkg_ttbar),0)
bkg_ppbbbb["label"] = np.full(len(bkg_ppbbbb),0)
bkg_ppjjjb["label"] = np.full(len(bkg_ppjjjb),0)
bkg_ppjjjj["label"] = np.full(len(bkg_ppjjjj),0)

#%%
Data = pd.concat([sig_ppHhh, bkg_ttbar], ignore_index=True, axis=0,join='inner')
Data = pd.concat([Data, bkg_ppbbbb], ignore_index=True, axis=0,join='inner')

# %%
Data = Preselection(Data)
#%%
X = Data
Y = Data["label"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)


# %%
def BDT_Model():
    
    rand = np.random.randint(1000000)
    clf_GBDT = GradientBoostingClassifier(
                n_estimators=1000,
                learning_rate=0.005,
                max_depth=5, 
                min_samples_split = 0.25,
                min_samples_leaf = 0.05,
    #             min_impurity_split = 0.00001,
    #             validation_fraction = 0.1,
                random_state= rand,  #np.random,
                verbose = 1
                )

    return clf_GBDT

#%%
features = ['MJJ', 'delta_eta', 'Xhh', #'XHH', 
            'MJ1','t211', 'D211', 'D221', 'C211', 'C221', 
            'MJ2','t212', 'D212', 'D222', 'C212', 'C222'
            ] 
# %%
clf_GBDT = BDT_Model()
clf_GBDT.fit(np.asarray(X_train[features]), np.asarray(Y_train))
# %%
prediction_test =  clf_GBDT.predict_proba(np.asarray(X_test[features]))
discriminator_test = prediction_test[:,1]
# discriminator_test = discriminator_test/(max(discriminator_test))

auc = metrics.roc_auc_score(Y_test, discriminator_test)
FalsePositiveFull, TruePositiveFull, _ = metrics.roc_curve(Y_test, discriminator_test)
last = np.where(TruePositiveFull > 0.095)[0][0]

logging.info("AUC: {}".format(auc))
logging.info("Rejection Rate: {}".format(FalsePositiveFull[last]))

#%%
dump(clf_GBDT, "./clf_GBDT_1000_Xhh.h5")

#%%
clf_GBDT = load("./clf_GBDT_1000_Xhh.h5")
#%%
"""
Prediction 
"""
sig_ppHhh["prediction"] = clf_GBDT.predict_proba(np.nan_to_num(sig_ppHhh[features]))[:,1]
bkg_ttbar["prediction"] = clf_GBDT.predict_proba(np.nan_to_num(bkg_ttbar[features]))[:,1]
bkg_ppbbbb["prediction"] = clf_GBDT.predict_proba(np.nan_to_num(bkg_ppbbbb[features]))[:,1]
bkg_ppjjjb["prediction"] = clf_GBDT.predict_proba(np.nan_to_num(bkg_ppjjjb[features]))[:,1]
bkg_ppjjjj["prediction"] = clf_GBDT.predict_proba(np.nan_to_num(bkg_ppjjjj[features]))[:,1]

# %%
sig_ppHhh_4_catal = sig_ppHhh[sig_ppHhh["four_b_tag"] == 1 ]
bkg_ttbar_4_catal = bkg_ttbar[bkg_ttbar["four_b_tag"] == 1 ]
bkg_ppbbbb_4_catal = bkg_ppbbbb[bkg_ppbbbb["four_b_tag"] == 1 ]
bkg_ppjjjb_4_catal = bkg_ppjjjb[bkg_ppjjjb["four_b_tag"] == 1 ]
bkg_ppjjjj_4_catal = bkg_ppjjjj[bkg_ppjjjj["four_b_tag"] == 1 ]

#%%
Luminosity = 3000 * (0.77**4)
Xection_ppHhh = 0.81186/1000 #pb
# Xection_ppHhh = 3.070/1000 #pb
Xection_ttbar = 260.3554 *0.27515 #pb
Xection_ppbbbb = 0.4070 #pb
Xection_ppjjjb = 450.04 #pb
Xection_ppjjjj = 11087.8358304 *0.02365529 #pb 
BRHhh = 0.8715
BRhbb = 0.3560
# BRHhh = 0.9413
# BRhbb = 0.3558



#%%
nset_simulation = 20
# cut_threshold = 0.9925
cut_threshold = 0.98#52972814739865

sig_eff = len(sig_ppHhh_4_catal)/(nset_simulation*100000) 
bkg_ttbar_eff = len(bkg_ttbar_4_catal)/(nset_simulation*100000)
bkg_ppbbbb_eff = len(bkg_ppbbbb_4_catal)/(nset_simulation*100000)
bkg_ppjjjb_eff = len(bkg_ppjjjb_4_catal)/(nset_simulation*100000)
bkg_ppjjjj_eff = len(bkg_ppjjjj_4_catal)/(nset_simulation*100000+len(process_path_jjjj_2)*10000)


logging.info("Efficiency Before BDT")
logging.info("Efficiency in 4b Category")
logging.info("sig_4_catal_eff : {} # of events: {:.2f}".format(sig_eff, sig_eff*Xection_ppHhh*BRHhh*BRhbb*BRhbb*Luminosity*1000))
logging.info("bkg_ttbar_4_catal_eff : {} # of events: {:.2f}".format(bkg_ttbar_eff, bkg_ttbar_eff*Xection_ttbar*Luminosity*1000))
logging.info("bkg_ppbbbb_4_catal_eff : {} # of events: {:.2f}".format(bkg_ppbbbb_eff, bkg_ppbbbb_eff*Xection_ppbbbb*Luminosity*1000))
logging.info("bkg_ppjjjb_4_catal_eff : {} # of events: {:.2f}".format(bkg_ppjjjb_eff, bkg_ppjjjb_eff*Xection_ppjjjb*Luminosity*1000))
logging.info("bkg_ppjjjj_4_catal_eff : {} # of events: {:.2f}".format(bkg_ppjjjj_eff, bkg_ppjjjj_eff*Xection_ppjjjj*Luminosity*1000))
logging.info("\n")

sig_eff = len(sig_ppHhh_4_catal[sig_ppHhh_4_catal["prediction"] > cut_threshold])/(nset_simulation*100000) 
bkg_ttbar_eff = len(bkg_ttbar_4_catal[bkg_ttbar_4_catal["prediction"] > cut_threshold])/(nset_simulation*100000)
bkg_ppbbbb_eff = len(bkg_ppbbbb_4_catal[bkg_ppbbbb_4_catal["prediction"] > cut_threshold])/(nset_simulation*100000)
bkg_ppjjjb_eff = len(bkg_ppjjjb_4_catal[bkg_ppjjjb_4_catal["prediction"] > cut_threshold])/(nset_simulation*100000)
bkg_ppjjjj_eff = len(bkg_ppjjjj_4_catal[bkg_ppjjjj_4_catal["prediction"] > cut_threshold])/(nset_simulation*100000+len(process_path_jjjj_2)*10000)

logging.info("Efficiency After BDT")
logging.info("Efficiency in 4b Category")
logging.info("sig_4_catal_eff : {} # of events: {:.2f}".format(sig_eff, sig_eff*Xection_ppHhh*BRHhh*BRhbb*BRhbb*Luminosity*1000))
logging.info("bkg_ttbar_4_catal_eff : {} # of events: {:.2f}".format(bkg_ttbar_eff, bkg_ttbar_eff*Xection_ttbar*Luminosity*1000))
logging.info("bkg_ppbbbb_4_catal_eff : {} # of events: {:.2f}".format(bkg_ppbbbb_eff, bkg_ppbbbb_eff*Xection_ppbbbb*Luminosity*1000))
logging.info("bkg_ppjjjb_4_catal_eff : {} # of events: {:.2f}".format(bkg_ppjjjb_eff, bkg_ppjjjb_eff*Xection_ppjjjb*Luminosity*1000))
logging.info("bkg_ppjjjj_4_catal_eff : {} # of events: {:.2f}".format(bkg_ppjjjj_eff, bkg_ppjjjj_eff*Xection_ppjjjj*Luminosity*1000))
logging.info("\n")
# %%
nset_simulation = 20
# cut_threshold = 0.9948#70815049923983
# cut_threshold = 0.9905#6164489563258
cut_threshold = 0.98#52972814739865

sig_eff = len(sig_ppHhh)/(nset_simulation*100000)
bkg_ttbar_eff = len(bkg_ttbar)/(nset_simulation*100000)
bkg_ppbbbb_eff = len(bkg_ppbbbb)/(nset_simulation*100000)
bkg_ppjjjb_eff = len(bkg_ppjjjb)/(nset_simulation*100000)
bkg_ppjjjj_eff = len(bkg_ppjjjj)/(nset_simulation*100000+len(process_path_jjjj_2)*10000)

logging.info("Efficiency Before BDT")
logging.info("sig_eff : {} # of events: {:.2f}".format(sig_eff, sig_eff*Xection_ppHhh*BRHhh*BRhbb*BRhbb*Luminosity*1000))
logging.info("bkg_ttbar_eff : {} # of events: {:.2f}".format(bkg_ttbar_eff, bkg_ttbar_eff*Xection_ttbar*Luminosity*1000))
logging.info("bkg_ppbbbb_eff : {} # of events: {:.2f}".format(bkg_ppbbbb_eff, bkg_ppbbbb_eff*Xection_ppbbbb*Luminosity*1000))
logging.info("bkg_ppjjjb_eff : {} # of events: {:.2f}".format(bkg_ppjjjb_eff, bkg_ppjjjb_eff*Xection_ppjjjb*Luminosity*1000))
logging.info("bkg_ppjjjj_eff : {} # of events: {:.2f}".format(bkg_ppjjjj_eff, bkg_ppjjjj_eff*Xection_ppjjjj*Luminosity*1000))
logging.info("\n")


sig_eff = len(sig_ppHhh[sig_ppHhh["prediction"] > cut_threshold])/(nset_simulation*100000) 
bkg_ttbar_eff = len(bkg_ttbar[bkg_ttbar["prediction"] > cut_threshold])/(nset_simulation*100000)
bkg_ppbbbb_eff = len(bkg_ppbbbb[bkg_ppbbbb["prediction"] > cut_threshold])/(nset_simulation*100000)
bkg_ppjjjb_eff = len(bkg_ppjjjb[bkg_ppjjjb["prediction"] > cut_threshold])/(nset_simulation*100000)
bkg_ppjjjj_eff = len(bkg_ppjjjj[bkg_ppjjjj["prediction"] > cut_threshold])/(nset_simulation*100000+len(process_path_jjjj_2)*10000)

logging.info("Efficiency After BDT")
logging.info("sig_eff : {} # of events: {:.2f}".format(sig_eff, sig_eff*Xection_ppHhh*BRHhh*BRhbb*BRhbb*Luminosity*1000))
logging.info("bkg_ttbar_eff : {} # of events: {:.2f}".format(bkg_ttbar_eff, bkg_ttbar_eff*Xection_ttbar*Luminosity*1000))
logging.info("bkg_ppbbbb_eff : {} # of events: {:.2f}".format(bkg_ppbbbb_eff, bkg_ppbbbb_eff*Xection_ppbbbb*Luminosity*1000))
logging.info("bkg_ppjjjb_eff : {} # of events: {:.2f}".format(bkg_ppjjjb_eff, bkg_ppjjjb_eff*Xection_ppjjjb*Luminosity*1000))
logging.info("bkg_ppjjjj_eff : {} # of events: {:.2f}".format(bkg_ppjjjj_eff, bkg_ppjjjj_eff*Xection_ppjjjj*Luminosity*1000))
logging.info("\n")
# %%
