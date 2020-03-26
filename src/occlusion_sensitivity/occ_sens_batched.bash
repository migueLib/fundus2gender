#!/bin/bash

declare -i COUNTER=0
BATCH='./occlusion_sensitivity/ukbb_test_set/female/batch'
OUT='./occlusion_sensitivity/Out/Full/female/batch'
while [ $COUNTER -lt 10 ]; do
        CBATCH="$BATCH${COUNTER}/"
        COUT="$OUT${COUNTER}/"
        echo $CBATCH
        python occ_sens.py -m ./inceptionv3_2out_normalizedCropped_ukbb_b80_e40_4dataAug_210319.pth -i $CBATCH -o $COUT
        COUNTER=$COUNTER+1   
        
      
        
done
        
