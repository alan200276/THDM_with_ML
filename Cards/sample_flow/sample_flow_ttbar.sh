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

cp $mg5cardpath/proc_ttbar.txt  $mg5cardpath/proc_ttbar_"$rand".txt 

sed -i -e "s/randomseed/"$rand"/g" $mg5cardpath/proc_ttbar_"$rand".txt 

python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $mg5cardpath/proc_ttbar_"$rand".txt  > $outpath/proc_ttbar_"$rand".log

sed -i -e "s/"$rand"/randomseed/g" $mg5cardpath/proc_ttbar_"$rand".txt 


# Downsize Part

python3 $preprocesspath/downsize.py $mcdatapath/proc_ttbar_"$rand"/Events/run_01/tag_1_delphes_events.root ttbar $downsizesavepath  $rand >  $outpath/ttbar_downsize_"$rand".log



# Preprocess Part

python3 $preprocesspath/preprocess_sample_flow.py $downsizesavepath/EventList_ttbar_"$rand".h5 ttbar $rand > $outpath/ttbar_preprocess_"$rand".log


rm -rf $mg5cardpath/proc_ttbar_"$rand".txt 
rm -rf $mcdatapath/proc_ttbar_"$rand"

echo "Finish"

date