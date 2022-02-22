#!/bin/bash

cardpath="/home/alan/ML_Analysis/THDM/Cards"

outpath="/home/alan/ML_Analysis/THDM/Log"

mcdatapath="/home/u5/THDM"


# UFO resource: https://feynrules.irmp.ucl.ac.be/wiki/2HDM
#cd /root/MG5_aMC_v2_7_3/models
#wget https://feynrules.irmp.ucl.ac.be/raw-attachment/wiki/2HDM/2HDMtII_NLO.tar.gz
#tar -xvf 2HDMtII_NLO.tar.gz


echo "Start Running"

i=6
while [ $i != 11 ]
do
   echo i=$i

   date +"%Y %b %m"
   date +"%r"
   
   echo "PP H hh"
   python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ppHhh.txt > $outpath/proc_ppHhh_"$i".log 

   echo "ttbar"
   python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ttbar.txt > $outpath/proc_ttbar_"$i".log
   
   echo "ppbbbb"
   python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ppbbbb.txt > $outpath/proc_ppbbbb_"$i".log   
   
   echo "ppjjjj"
   python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ppjjjj.txt > $outpath/proc_ppjjjj_"$i".log
   
   echo "PP jjjb"
   python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_jjjb.txt > $outpath/proc_ppjjjb_"$i".log

   
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))

done

# gzip -d $mcdatapath/proc_*/Events/run_*/unweighted_events.lhe.gz
rm -rf $mcdatapath/proc_*/Events/run_*/*.hepmc.gz

echo "Finish"

date
