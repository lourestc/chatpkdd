#!/bin/sh

INFILE=$1 #"timestamps/train (2).csv"
OUTPATH=$2 #"predictions/"

python prepare_data.py $INFILE "manipulated_data.csv"
python run_LSTM.py "manipulated_data.csv" $OUTPATH+'preds.csv'
