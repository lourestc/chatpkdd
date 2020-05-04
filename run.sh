#!/bin/sh

INFILE=$1 #"timestamps/train (2).csv"
OUTPATH=$2 #"predictions/"

mkdir "prepared"

python prepare_data.py first $INFILE "prepared"
python run_LSTM.py hsearch "prepared" $OUTPATH
