#/bin/bash

HOMEPATH="/home/alan/ML_Analysis/THDM"
datapath="/home/u5/THDM/Downsized"
outpath="/home/alan/ML_Analysis/THDM/Log"

date

echo "Start Running"

date +"%Y %b %m"


i=1
while [ $i != 2 ] 
do
#========================================================================================
    
    nohup python3 $HOMEPATH/Preprocess/preprocess.py $datapath/EventList_ppHhh_"$i".h5 ppHhh $i > $outpath/ppHhh_preprocess_"$i".log &
    nohup python3 $HOMEPATH/Preprocess/preprocess.py $datapath/EventList_ttbar_"$i".h5 ttbar $i > $outpath/ttbar_preprocess_"$i".log &
    nohup python3 $HOMEPATH/Preprocess/preprocess.py $datapath/EventList_ppbbbb_"$i".h5 ppbbbb $i > $outpath/ppbbbb_preprocess_"$i".log &
    nohup python3 $HOMEPATH/Preprocess/preprocess.py $datapath/EventList_ppjjjj_"$i".h5 ppjjjj $i > $outpath/ppjjjj_preprocess_"$i".log &

    date +"%Y %b %m"


    
    i=$(($i+1))

done
#========================================================================================



echo "Finish"

date
