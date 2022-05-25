#!/bin/bash

mg5cardpath="/home/alan/THDM_with_ML/Cards/sample_flow"
outpath="/home/alan/MC_Data/Log"
mcdatapath="/home/alan/MC_Data"

preprocesspath="/home/alan/THDM_with_ML/Preprocess"
downsizesavepath="/home/alan/MC_Data/Downsized"


echo "Start Running"
date


# Madgraph Part

rand=$RANDOM
rand="$rand"1

echo "Random Seed =  $rand "

cp $mg5cardpath/proc_ppHhh.txt  $mg5cardpath/proc_ppHhh_"$rand".txt 

sed -i -e "s/randomseed/"$rand"/g" $mg5cardpath/proc_ppHhh_"$rand".txt 

python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $mg5cardpath/proc_ppHhh_"$rand".txt  > $outpath/proc_ppHhh_"$rand".log

sed -i -e "s/"$rand"/randomseed/g" $mg5cardpath/proc_ppHhh_"$rand".txt 


# Downsize Part

python3 $preprocesspath/downsize.py $mcdatapath/proc_ppHhh_"$rand"/Events/run_01_decayed_1/tag_1_delphes_events.root ppHhh $downsizesavepath  $rand >  $outpath/ppHhh_downsize_"$rand".log



# Preprocess Part

python3 $preprocesspath/preprocess_sample_flow.py $downsizesavepath/EventList_ppHhh_"$rand".h5 ppHhh $rand > $outpath/ppHhh_preprocess_"$rand".log


rm -rf $mg5cardpath/proc_ppHhh_"$rand".txt 
rm -rf $mcdatapath/proc_ppHhh_"$rand"

echo "Finish"

date