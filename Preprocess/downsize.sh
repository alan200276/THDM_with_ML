#/bin/bash

HOMEPATH="/home/alan/ML_Analysis/THDM"
datapath="/home/u5/THDM"
savepath="/home/u5/THDM/Downsized"
outpath="/home/alan/ML_Analysis/THDM/Log"

date

echo "Start Running"


date +"%Y %b %m"

i=1
while [ $i != 2 ] 
do
#========================================================================================
    if [ "$i" -lt "10" ];then

    nohup python3 ./downsize.py $datapath/proc_ppHhh/Events/run_0"$i"_decayed_1/tag_1_delphes_events.root ppHhh $savepath $i  >  $outpath/ppHhh_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_ttbar/Events/run_0"$i"/tag_1_delphes_events.root ttbar $savepath  $i >  $outpath/ttbar_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_ppbbbb/Events/run_0"$i"/tag_1_delphes_events.root ppbbbb $savepath  $i  >  $outpath/ppbbbb_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_jjjj/Events/run_0"$i"/tag_1_delphes_events.root ppjjjj $savepath  $i >  $outpath/ppjjjj_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_jjjb/Events/run_0"$i"/tag_1_delphes_events.root ppjjjb $savepath  $i >  $outpath/ppjjjb_downsize_"$i".log &


    elif [ "$i" -eq "10" ];then

    nohup python3 ./downsize.py $datapath/proc_ppHhh/Events/run_10_decayed_1/tag_1_delphes_events.root ppHhh $savepath $i  >  $outpath/ppHhh_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_ttbar/Events/run_10/tag_1_delphes_events.root ttbar $savepath  $i >  $outpath/ttbar_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_ppbbbb/Events/run_10/tag_1_delphes_events.root ppbbbb $savepath  $i  >  $outpath/ppbbbb_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_jjjj/Events/run_10/tag_1_delphes_events.root ppjjjj $savepath  $i >  $outpath/ppjjjj_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_jjjb/Events/run_10/tag_1_delphes_events.root ppjjjb $savepath  $i >  $outpath/ppjjjb_downsize_"$i".log &


    elif [ "$i" -gt "10" ];then

    nohup python3 ./downsize.py $datapath/proc_ppHhh/Events/run_"$i"_decayed_1/tag_1_delphes_events.root ppHhh $savepath $i  >  $outpath/ppHhh_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_ttbar/Events/run_"$i"/tag_1_delphes_events.root ttbar $savepath  $i >  $outpath/ttbar_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_ppbbbb/Events/run_"$i"/tag_1_delphes_events.root ppbbbb $savepath  $i  >  $outpath/ppbbbb_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_jjjj/Events/run_"$i"/tag_1_delphes_events.root ppjjjj $savepath  $i >  $outpath/ppjjjj_downsize_"$i".log &

    nohup python3 ./downsize.py $datapath/THDM/proc_jjjb/Events/run_"$i"/tag_1_delphes_events.root ppjjjb $savepath  $i >  $outpath/ppjjjb_downsize_"$i".log &

    fi



    date +"%Y %b %m"


    
    i=$(($i+1))

done
#========================================================================================



echo "Finish"

date
