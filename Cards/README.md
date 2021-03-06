


* Signal (heavy resonant pair production of Higgs bosons in $b\bar{b}b\bar{b}$ final state)  
    * UFO [2HDM: 2HDMtII_NLO](http://feynrules.irmp.ucl.ac.be/attachment/wiki/2HDM/2HDMtII_NLO.tar.gz)  
    * process:   
    `generate p p > h2 [QCD]`     
    * decay in Madspin: 
    ```
    set spinmode none   
    decay h2 > h1 h1, h1 > b b~
    ```
    
    * run_card setting:  
    ```
    set run_card nevents 100000
    set run_card ebeam1 7000.0
    set run_card ebeam2 7000.0

    set run_card pdlabel lhapdf 
    set run_card lhaid 260000  #NNPDF30_nlo_as_0118
    set run_card fixed_ren_scale True
    set run_card fixed_fac_scale True
    set run_card scale 500
    set run_card dsqrt_q2fact1 500
    set run_card dsqrt_q2fact2 500
    ```
    
    * param_card setting:  
    ```
    set param_card tanbeta 5
    set param_card sinbma 0.99612
    set param_card mh1 125
    set param_card mh2 1000
    set param_card mh3 1001
    set param_card mhc 1001

    set param_card l2 -0.90347
    set param_card l3 -10.40733
    set param_card lr7 0
    ```
    * process card:
    ```
    proc_ppHhh.txt
    ```




* Background (QCD multijet (only b) )   
    * process:   
    ```
    define p = p b b~    
    generate p p > b b~ b b~  
    ```  
    
    * run_card setting:
    ```
    set run_card nevents 100000
    set run_card ebeam1 7000.0
    set run_card ebeam2 7000.0
    set run_card pdlabel lhapdf 
    set run_card lhaid 260000  #NNPDF30_nlo_as_0118

    set run_card ihtmin 850
    ```
    * process card:
    ```
    proc_ppbbbb.txt
    ```

* Background (QCD multijet (inclusive) )   

    * process:   
    ```
    define p = p b b~
    define j = j b b~

    generate p p > j j j j 
    ```  


    * run_card setting:
    ```
    set run_card nevents 100000
    set run_card ebeam1 7000.0
    set run_card ebeam2 7000.0
    set run_card pdlabel lhapdf 
    set run_card lhaid 260000  #NNPDF30_nlo_as_0118

    set run_card ihtmin 850
    ```
    * process card:
    ```
    proc_jjjj.txt
    ```

* Background (QCD multijet (one b) )   

    * process:   
    ```
    define p = p b b~
    define j = j b b~
    define bb = b b~

    generate p p > j j j bb
    ```  
    
    * run_card setting:
    ```
    set run_card nevents 100000
    set run_card ebeam1 7000.0
    set run_card ebeam2 7000.0
    set run_card pdlabel lhapdf 
    set run_card lhaid 260000  #NNPDF30_nlo_as_0118

    set run_card ihtmin 850
    ```
    * process card:
    ```
    proc_jjjb.txt
    ```



* Background ($t\bar{t}$ )   
    * process:   
    ```
    define p = p b b~
    define j = j b b~

    generate p p > t t~, (t > w+ b, w+ > j j), (t~ > w- b~, w- > j j) @0
    add process p p > t t~ j, (t > w+ b, w+ > j j), (t~ > w- b~, w- > j j)  @1
    add process p p > t t~ j j, (t > w+ b, w+ > j j), (t~ > w- b~, w- > j j)  @2
    ```  
    
    * run_card setting:
    ```
    set run_card nevents 100000
    set run_card ebeam1 7000.0
    set run_card ebeam2 7000.0
    set run_card pdlabel lhapdf 
    set run_card lhaid 247000  #NNPDF23_lo_as_0130_qed
    set run_card ickkw 1 #MLM
    set run_card xqcut 20.0
    ```
    * process card:
    ```
    proc_ttbar.txt
    ```