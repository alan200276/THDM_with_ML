Ref: [HiggsBounds](https://gitlab.com/higgsbounds/higgsbounds)
Ref: [HiggsSignals](https://gitlab.com/higgsbounds/higgssignals)
Ref: [Additional_Tools](https://gitlab.com/higgsbounds/additional_tools)
Ref: [2HDMC - Two-Higgs-Doublet Model Calculator](https://arxiv.org/abs/0902.0851)


In order to get branching ratio and corresponding paramters in $\lambda$ basis from mass basis, we can use [2HDMC](https://arxiv.org/abs/0902.0851) for calculation.   
The following is the procedure for installation and setting.

* Environment: [alan200276/ubuntu:HEPtools](https://hub.docker.com/layers/126824214/alan200276/ubuntu/HEPtools/images/sha256-4493b662288826ca93545ffb66572e796701a634ef1871da900e86177ea489c9?context=explore)

---

* Installation:
    * install relevent package  
        `
        apt-get install libgsl-dev  
        `
    * download HiggsBounds, unpack and compile  
        `
        wget https://gitlab.com/higgsbounds/higgsbounds/-/archive/master/higgsbounds-master.tar.gz  
        `    
        `  
        tar -xvf higgsbounds-master.tar.gz
        `     
        `  
        cd higgsbounds-master
        `     
        `  
        mkdir build && cd build
        `    
        `
        cmake ..        
        `   
        `  
        make
        `  


    * download HiggsSignals, unpack and compile  
        `
        wget https://gitlab.com/higgsbounds/higgssignals/-/archive/master/higgssignals-master.tar.gz
        `   
        `
        tar -xvf higgssignals-master.tar.gz
        `   
        `
        cd higgssignals-master
        `  
        `  
        mkdir build && cd build
        `    
        `
        cmake ..        
        `   
        `
        make
        `
    
    * download 2HDMC, unpack and compile  
        `
        wget https://2hdmc.hepforge.org/downloads/2HDMC-1.8.0.tar.gz    
        `   
        `
        tar -xvf 2HDMC-1.8.0.tar.gz 
        `   
        `
        cd 2HDMC-1.8.0   
        `   
        `  
        cp [path-to-higgsbounds-lib]/libHB.a ./lib
        `  
        `  
        cp [path-to-higgssignals-lib]/libHS.a ./lib
        `  
          
        Uncomment following two lines in the  *Makefile* and modify path
        
        `
            # HiggsBounds_DIR=path-to-higgsbounds/build   
        `  
        `  
            # HiggsSignals_DIR=path-to-higgssignals/build  
        `
    
        `
        make  CFLAGS=-fpermissive
        `
        
    * use 2HDMC to calculate
        We can use `CalcPhys` to get parameters in $\lambda$ basis from mass basis  
        `
        Usage: ./CalcPhys mh mH mA mHp sin(beta-alpha) lambda_6 lambda_7 m_12^2 tan_beta yukawas_type output_filename  
        `
        
        Our benchmark point is   
        mh = 125  
        mH = 1000  
        mA = 1001  
        mHp = 1001 
        sin(beta-alpha)= 0.99612  
        lambda_6 = 0  
        lambda_7 = 0  
        m_12^2 = 400000  
        tan_beta = 5 
        yukawas_type = 2 (Type II)  
        `
        ./CalcPhys 125 1000 1001 1001 0.99612  0 0 400000  5 2 parameters.out > THDM.log  
        `
        
        Then what we need are parameters.out and THDM.log in the 2HDMC-1.8.0 folder