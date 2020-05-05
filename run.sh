#!/bin/sh

INFILE=$1 #"timestamps/train (2).csv"
OUTPATH=$2 #"predictions/"

mkdir "prepared"

INNLINES=$(wc -l < "$INFILE")

python prepare_data.py first $INFILE "prepared" $INNLINES
#python run_LSTM.py hsearch "prepared" $OUTPATH $INNLINES
python batched_model.py train "prepared/train0000.csv" 100000 "prepared/train0001.csv" 100000 $OUTPATH
