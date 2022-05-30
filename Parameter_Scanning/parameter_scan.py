#/bin/python3

#%%
import numpy as np
from tqdm import tqdm
import time
import os
import importlib
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

#%%
"""
ghU(cb_a, tb, type)
"""
def ghU(cb_a, tb, type):
    b_a = np.arccos(cb_a)
    b = np.arctan(tb)
    a = np.arctan(tb) - np.arccos(cb_a)
    
    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    else:
        return np.cos(a)/np.sin(b)

    
"""
ghD(cb_a, tb, type)
"""
def ghD(cb_a, tb, type):
    b_a = np.arccos(cb_a)
    b = np.arctan(tb)
    a = np.arctan(tb) - np.arccos(cb_a)

    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    elif type == 1:
        return np.cos(a)/np.sin(b)  
    elif type == 2:
        return -1*np.sin(a)/np.cos(b)  
    elif type == 3:
        return np.cos(a)/np.sin(b) 
    elif type == 4:
        return -1*np.sin(a)/np.cos(b)   


"""
ghL(cb_a, tb, type)
"""
def ghL(cb_a, tb, type):
    b_a = np.arccos(cb_a)
    b = np.arctan(tb)
    a = np.arctan(tb) - np.arccos(cb_a)
    
    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    elif type == 1:
        return np.cos(a)/np.sin(b)  
    elif type == 2:
        return -1*np.sin(a)/np.cos(b)  
    elif type == 3:
        return -1*np.sin(a)/np.cos(b)
    elif type == 4:
        return np.cos(a)/np.sin(b)  



"""
gHU(cb_a, tb, type)
"""
def gHU(cb_a, tb, type):
    b_a = np.arccos(cb_a)
    b = np.arctan(tb)
    a = np.arctan(tb) - np.arccos(cb_a)

    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    else:
        return np.sin(a)/np.sin(b)

    
"""
gHD(cb_a, tb, type)
"""
def gHD(cb_a, tb, type):
    b_a = np.arccos(cb_a)
    b = np.arctan(tb)
    a = np.arctan(tb) - np.arccos(cb_a)
    
    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    elif type == 1:
        return np.sin(a)/np.sin(b)  
    elif type == 2:
        return np.cos(a)/np.cos(b)  
    elif type == 3:
        return np.sin(a)/np.sin(b)
    elif type == 4:
        return np.cos(a)/np.cos(b) 
    
"""
gHL(cb_a, tb, type)
"""
def gHL(cb_a, tb, type):
    b_a = np.arccos(cb_a)
    b = np.arctan(tb)
    a = np.arctan(tb) - np.arccos(cb_a)
    
    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    elif type == 1:
        return np.sin(a)/np.sin(b)  
    elif type == 2:
        return np.cos(a)/np.cos(b)  
    elif type == 3:
        return np.cos(a)/np.cos(b)
    elif type == 4:
        return np.sin(a)/np.sin(b) 

"""
gAU(cb_a, tb, type)
"""
def gAU(cb_a, tb, type):
    b_a = np.arccos(cb_a)
    b = np.arctan(tb)
    a = np.arctan(tb) - np.arccos(cb_a)

    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    else:
        return np.cos(b)/np.sin(b)

    
"""
gAD(cb_a, tb, type)
"""
def gAD(cb_a, tb, type):
    b_a = np.arccos(cb_a)
    b = np.arctan(tb)
    a = np.arctan(tb) - np.arccos(cb_a)
    
    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    elif type == 1:
        return -np.cos(b)/np.sin(b)
    elif type == 2:
        return tb
    elif type == 3:
        return -np.cos(b)/np.sin(b)
    elif type == 4:
        return tb
    
"""
gAL(cb_a, tb, type)
"""
def gAL(cb_a, tb, type):
    b_a = np.arccos(cb_a)
    b = np.arctan(tb)
    a = np.arctan(tb) - np.arccos(cb_a)
    
    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    elif type == 1:
        return -np.cos(b)/np.sin(b)
    elif type == 2:
        return tb
    elif type == 3:
        return tb
    elif type == 4:
        return -np.cos(b)/np.sin(b)



"""
gU(tb, type)
"""
def gU(tb, type):

    Mu, Mc, Mt = 0.0, 0.0, 173.07
    Md, Ms, Mb = 0.0, 0.0, 4.78
    Me, Mmu, mta = 5.10998918e-04, 1.05658367e-01, 1.77684000e+00
    vev = 246

    b = np.arctan(tb)
    
    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    else:
        return np.array([Mu, Mc, Mt])*np.sqrt(2)/vev*(np.cos(b)/np.sin(b))*(-1)

"""
gD(tb, type)
"""
def gD(tb, type):

    Mu, Mc, Mt = 0.0, 0.0, 173.07
    Md, Ms, Mb = 0.0, 0.0, 4.78
    Me, Mmu, mta = 5.10998918e-04, 1.05658367e-01, 1.77684000e+00
    vev = 246

    b = np.arctan(tb)
    
    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    elif type == 1:
        return np.array([Md, Ms, Mb])*np.sqrt(2)/vev*(np.cos(b)/np.sin(b))*(-1)
    elif type == 2:
        return -np.array([Md, Ms, Mb])*np.sqrt(2)/vev*tb*(-1)
    elif type == 3:
        return -np.array([Md, Ms, Mb])*np.sqrt(2)/vev*tb*(-1)
    elif type == 4:
        return np.array([Md, Ms, Mb])*np.sqrt(2)/vev*(np.cos(b)/np.sin(b))*(-1)

"""
gL(tb, type)
"""
def gL(tb, type):

    Mu, Mc, Mt = 0.0, 0.0, 173.07
    Md, Ms, Mb = 0.0, 0.0, 4.78
    Me, Mmu, mta = 5.10998918e-04, 1.05658367e-01, 1.77684000e+00
    vev = 246

    b = np.arctan(tb)
    
    if type not in [1,2,3,4]:
        raise ValueError("Please let 'type' in [1,2,3,4]")
    elif type == 1:
        return np.array([Me, Mmu, mta])*np.sqrt(2)/vev*(np.cos(b)/np.sin(b))*(-1)
    elif type == 2:
        return -np.array([Me, Mmu, mta ])*np.sqrt(2)/vev*tb*(-1)
    elif type == 3:
        return np.array([Me, Mmu, mta ])*np.sqrt(2)/vev*(np.cos(b)/np.sin(b))*(-1)
    elif type == 4:
        return -np.array([Me, Mmu, mta ])*np.sqrt(2)/vev*tb*(-1)



#%%
rand = str(int(np.random.rand()*100000))+"1"
cb_a, tb, type = 8.80000000e-02 , 5, 2 #Our Benchmark
sba = np.sqrt(1-cb_a**2)
mh, mH, mA, mHp, lambda_6, lambda_7, m_12s = 125, 1000, 1001, 1001, 0, 0, 400000


#%%
def Calculate_Xection_BranhingRatio(rand, cb_a, tb, type, sba, mh, mH, mA, mHp, lambda_6, lambda_7, m_12s):
    #%%
    logging.info("Random: {}".format(rand))
    logging.info("cb_a: {}".format(cb_a))
    logging.info("tb: {}".format(tb))
    logging.info("Yukawas_type: {}".format(type))
    logging.info("sba: {}".format(sba))
    logging.info("mh: {}".format(mh))
    logging.info("mH: {}".format(mH))
    logging.info("mA: {}".format(mA))
    logging.info("mHp: {}".format(mHp))
    logging.info("lambda_6: {}".format(lambda_6))
    logging.info("lambda_7: {}".format(lambda_7))
    logging.info("m_12s: {}".format(m_12s))


    #%%
    print("ghU: {}".format(ghU(cb_a, tb, type )))
    print("gHU: {}".format(gHU(cb_a, tb, type )))
    print("gAU: {}".format(gAU(cb_a, tb, type )))
    print("ghD: {}".format(ghD(cb_a, tb, type )))
    print("gHD: {}".format(gHD(cb_a, tb, type )))
    print("gAD: {}".format(gAD(cb_a, tb, type )))
    print("ghL: {}".format(ghL(cb_a, tb, type )))
    print("gHL: {}".format(gHL(cb_a, tb, type )))
    print("gAL: {}".format(gAL(cb_a, tb, type )))
    print("\n")

    print("{:^9s}{:^10s} {:^10s} {:^10s}".format("Coupling", "1x1","2x2","3x3"))
    print("{:^9s}{:^.9f} {:^.9f} {:^.9f}".format("gU:",gU(tb, type)[0],gU(tb, type)[1],gU(tb, type)[2]))
    print("{:^9s}{:^.9f} {:^.9f} {:^.9f}".format("gD:",gD(tb, type)[0],gD(tb, type)[1],gD(tb, type)[2]))
    print("{:^9s}{:^.9f} {:^.9f} {:^.9f}".format("gL:",gL(tb, type)[0],gL(tb, type)[1],gL(tb, type)[2]))



    #%%
    """
    Execute 2HDMC
    """
    THDMC_parameter_output_path = "/home/alan/ML_Analysis/THDM/Parameter_Scanning/THDMC_output/"

    cmd = "/root/THDM_Tools/2HDMC-1.8.0/CalcPhys "+str(mh)+" "+str(mH)+" "+str(mA)+" "+str(mHp)+" "+str(sba)+" "+str(lambda_6)+" "+str(lambda_7)+" "+str(m_12s)+" "+str(tb)+" "+str(type)+" "+str(THDMC_parameter_output_path)+"parameters_"+str(rand)+".out > "+str(THDMC_parameter_output_path)+"THDM_"+str(rand)+".log"
    os.system(cmd)

    #%%
    """
    Read Parameter From 2HDMC Outputs
    """
    THDMC_output_path = THDMC_parameter_output_path + "parameters_"+str(rand)+".out"

    with open(THDMC_output_path,'r') as f:
        for i, line in enumerate(f):
                # print(line.strip())
            if "lambda_2" in line.strip():
                l2 = line.strip().split()[1]
            if "lambda_3" in line.strip():
                l3 = line.strip().split()[1]
            if "sin(beta-alpha)" in line.strip():
                sba = line.strip().split()[1]
                # ref: http://feynrules.irmp.ucl.ac.be/attachment/wiki/2HDM/typeIItbeta.rst
                # mixh = Pi/2-ArcSin[sinbma]
                mixh = np.pi/2 - np.arcsin(float(sba))

            if "DECAY  25" in line.strip():
                # print(line.strip())
                branchin_ratio_start = i 
                branchin_ratio_25 = i 

            if "DECAY  35" in line.strip():
                # print(line.strip())
                branchin_ratio_35 = i 
            
            if "DECAY  36" in line.strip():
                # print(line.strip())
                branchin_ratio_36 = i 

            if "BLOCK MGUSER" in line.strip():
                # print(line.strip())
                branchin_ratio_end = i 

    with open(THDMC_output_path,'r') as f:
        lines = f.readlines()


    #%%
    """
    Modify Parameter Card for MG5
    """
    parameter_card_home_path = "/home/alan/ML_Analysis/THDM/Parameter_Scanning/Parameter_card/"
    parameter_card_origin_path = "/home/alan/ML_Analysis/THDM/Parameter_Scanning/"
    cmd = "cp "+str(parameter_card_origin_path)+"param_card_origin.dat "+str(parameter_card_home_path)+"param_card_"+str(rand)+".dat"
    os.system(cmd)

    parameter_path = str(parameter_card_home_path)+"param_card_"+str(rand)+".dat"

    """
    BLOCK HIGGS
    """
    cmd = "sed -i -e s/param_l2/"+str(l2)+"/g " + parameter_path
    os.system(cmd)

    cmd = "sed -i -e s/param_l3/"+str(l3)+"/g " + parameter_path
    os.system(cmd)

    cmd = "sed -i -e s/param_mixh/"+str(mixh)+"/g " + parameter_path
    os.system(cmd)


    """
    BLOCK LOOP
    """
    param_ytrs1 = ghU(cb_a, tb, type )
    cmd = "sed -i -e s/param_ytrs1/"+str(param_ytrs1)+"/g " + parameter_path
    os.system(cmd)

    param_ytrs2 = gHU(cb_a, tb, type )
    cmd = "sed -i -e s/param_ytrs2/"+str(param_ytrs2)+"/g " + parameter_path
    os.system(cmd)

    param_ytrs3 = gAU(cb_a, tb, type )
    cmd = "sed -i -e s/param_ytrs3/"+str(param_ytrs3)+"/g " + parameter_path
    os.system(cmd)

    param_ytrhp = gAU(cb_a, tb, type )
    cmd = "sed -i -e s/param_ytrhp/"+str(param_ytrhp)+"/g " + parameter_path
    os.system(cmd)

    param_ybrs1 = ghD(cb_a, tb, type )
    cmd = "sed -i -e s/param_ybrs1/"+str(param_ybrs1)+"/g " + parameter_path
    os.system(cmd)

    param_ybrs2 = gHD(cb_a, tb, type )
    cmd = "sed -i -e s/param_ybrs2/"+str(param_ybrs2)+"/g " + parameter_path
    os.system(cmd)

    param_ybrs3 = gAD(cb_a, tb, type )
    cmd = "sed -i -e s/param_ybrs3/"+str(param_ybrs3)+"/g " + parameter_path
    os.system(cmd)

    param_ybrhp = gAD(cb_a, tb, type )
    cmd = "sed -i -e s/param_ybrhp/"+str(param_ybrhp)+"/g " + parameter_path
    os.system(cmd)


    """
    INFORMATION FOR MASS
    """
    cmd = "sed -i -e s/param_mh1/"+str(mh)+"/g " + parameter_path
    os.system(cmd)

    cmd = "sed -i -e s/param_mh2/"+str(mH)+"/g " + parameter_path
    os.system(cmd)

    cmd = "sed -i -e s/param_mh3/"+str(mA)+"/g " + parameter_path
    os.system(cmd)

    cmd = "sed -i -e s/param_mhc/"+str(mHp)+"/g " + parameter_path
    os.system(cmd)

    """
    INFORMATION FOR YUKAWAGDI
    """
    param_gdr33 = gD(tb, type)[2]
    cmd = "sed -i -e s/param_gdr33/"+str(param_gdr33)+"/g " + parameter_path
    os.system(cmd)

    param_glr11 = gL(tb, type)[0]
    cmd = "sed -i -e s/param_glr11/"+str(param_glr11)+"/g " + parameter_path
    os.system(cmd)

    param_glr22 = gL(tb, type)[1]
    cmd = "sed -i -e s/param_glr22/"+str(param_glr22)+"/g " + parameter_path
    os.system(cmd)

    param_glr33 = gL(tb, type)[2]
    cmd = "sed -i -e s/param_glr33/"+str(param_glr33)+"/g " + parameter_path
    os.system(cmd)

    param_gur33 = gU(tb, type)[2]
    cmd = "sed -i -e s/param_gur33/"+str(param_gur33)+"/g " + parameter_path
    os.system(cmd)

    with open(parameter_path,'a') as f:
        f.writelines("\n")
        f.writelines(lines[branchin_ratio_start:branchin_ratio_end])
    #%%
    """
    Create MG5 Process Card
    """
    mg5_card_home_path = "/home/alan/ML_Analysis/THDM/Parameter_Scanning/MG5/"
    mg5_card_origin_path = "/home/alan/ML_Analysis/THDM/Parameter_Scanning/"
    cmd = "cp "+str(mg5_card_origin_path)+"proc_ppHhh.txt "+str(mg5_card_home_path)+"proc_ppHhh_"+str(rand)+".txt"
    os.system(cmd)

    mg5_card_path = str(mg5_card_home_path)+"proc_ppHhh_"+str(rand)+".txt"


    cmd = "sed -i -e s/randomseed/"+str(rand)+"/g " + mg5_card_path
    os.system(cmd)
    #%%
    """
    Execute MG5 
    """
    cmd = "python /root/MG5_aMC_v2_7_3/bin/mg5_aMC "+mg5_card_home_path+"proc_ppHhh_"+str(rand)+".txt"
    os.system(cmd)


    #%%
    """
    Read run_01_tag_1_banner.txt 
    """
    run_banner_path = mg5_card_home_path+"proc_ppHhh_"+str(rand)+"/Events/run_01/run_01_tag_1_banner.txt"
    with open(run_banner_path,'r') as f:
        for i, line in enumerate(f):
                # print(line.strip())
            if "#  Integrated weight (pb)  :" in line.strip():
                xection = line.strip().split()[-1]
    #%%
    """
    Read Branching Ratio from 2HDMC Output
    """
    for line in lines[branchin_ratio_25:branchin_ratio_35]:
        if "5    -5" in line:
            # print(line.strip().split()[0])
            BRhbb = line.strip().split()[0]

    for line in lines[branchin_ratio_35:branchin_ratio_36]:
        if "25    25" in line:
            # print(line.strip().split()[0])
            BRHhh = line.strip().split()[0]
    #%%
    logging.info("Xection: {} (fb)".format(float(xection)*1000))
    logging.info("BRhbb: {}".format(BRhbb))
    logging.info("BRHhh: {}".format(BRHhh))
    #%%
    """
    Remove Files
    """
    cmd = "rm -rf " + mg5_card_path
    os.system(cmd)
    cmd = "rm -rf " + mg5_card_home_path+"proc_ppHhh_"+str(rand)
    os.system(cmd)

    cmd = "rm -rf " + THDMC_parameter_output_path+"parameters_"+str(rand)+".out"
    os.system(cmd)
    cmd = "rm -rf " + THDMC_parameter_output_path+"THDM_"+str(rand)+".log"
    os.system(cmd)

    cmd = "rm -rf " + parameter_path
    os.system(cmd)


    return float(xection), float(BRhbb), float(BRHhh)


#%%
rand = str(int(np.random.rand()*100000))+"1"
cb_a, tb, type = 8.80000000e-02 , 5, 2 #Our Benchmark
sba = np.sqrt(1-cb_a**2)
mh, mH, mA, mHp, lambda_6, lambda_7, m_12s = 125, 1000, 1001, 1001, 0, 0, 400000

#%%
Calculate_Xection_BranhingRatio(rand, cb_a, tb, type, sba, mh, mH, mA, mHp, lambda_6, lambda_7, m_12s)
    

#%%
from multiprocessing import Process, Pool

nb_threads = 1

tmp_para = [[rand, cb_a, tb, type, sba, mh, mH, mA, mHp, lambda_6, lambda_7, m_12s]]

if __name__ == '__main__':

    pool = Pool(nb_threads)

    pool_outputs = pool.starmap_async(Calculate_Xection_BranhingRatio, tmp_para)
    # print('將不會阻塞並和 pool.map_async 並行觸發')

    # close 和 join 是確保主程序結束後，子程序仍然繼續進行
    pool.close()
    pool.join()

#%%
# constraint_n = np.array(pool_outputs.get()).reshape(n_slice, n_slice)