#!/bin/bash

mg5cardpath="/home/alan/ML_Analysis/THDM/Cards/sample_flow"
outpath="/home/alan/ML_Analysis/THDM/Cards/sample_flow/Log"
mcdatapath="/home/u5/THDM/sample_flow"

preprocesspath="/home/alan/ML_Analysis/THDM/Preprocess"
downsizesavepath="/home/u5/THDM/sample_flow/Downsized"


echo "Start Running"
date


# Madgraph Part

rand=$RANDOM
rand="$rand"1

echo "Random Seed =  $rand "

cp $mg5cardpath/proc_ppjjjj.txt  $mg5cardpath/proc_ppjjjj_"$rand".txt 

sed -i -e "s/randomseed/"$rand"/g" $mg5cardpath/proc_ppjjjj_"$rand".txt 

python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $mg5cardpath/proc_ppjjjj_"$rand".txt  > $outpath/proc_ppjjjj_"$rand".log

sed -i -e "s/"$rand"/randomseed/g" $mg5cardpath/proc_ppjjjj_"$rand".txt 


# Downsize Part

python3 $preprocesspath/downsize.py $mcdatapath/proc_ppjjjj_"$rand"/Events/run_01/tag_1_delphes_events.root ppjjjj $downsizesavepath  $rand >  $outpath/ppjjjj_downsize_"$rand".log



# Preprocess Part

python3 $preprocesspath/preprocess_sample_flow.py $downsizesavepath/EventList_ppjjjj_"$rand".h5 ppjjjj $rand > $outpath/ppjjjj_preprocess_"$rand".log


rm -rf $mg5cardpath/proc_ppjjjj_"$rand".txt 
rm -rf $mcdatapath/proc_ppjjjj_"$rand"

echo "Finish"

date