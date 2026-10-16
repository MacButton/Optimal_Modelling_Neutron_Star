#!/bin/bash

EVENT_list=("170817" "190425") # for the typical injection
#EVENT_list=("rho") #for no signal injected 
#EVENT_list=("rand") #for random variables

SCRIPT="./shell.sh"
NOISE_list=("GWINC" "GWOSC")
WF_list=("Easter" "NRPMw") 

# The parameters to be looped over
for SNR in 2 3 4 5 6 7 8 9 10 15 20; do
for SEED in {1..30}; do
for EVENT in "${EVENT_list[@]}"; do
for NOISE in "${NOISE_list[@]}"; do
for WF in "${WF_list[@]}"; do
for DISTANCEPERCENT in 100; do # For uniform distance priors, ignore for delta priors

echo "Running: EVENT=$EVENT SNR=$SNR SEED=$SEED NOISE=$NOISE WF=$WF"

# Pass as arguments
$SCRIPT "$SNR" "$SEED" "$NOISE" "$WF" "$DISTANCEPERCENT" "$EVENT"

done
done
done
done
done
done
