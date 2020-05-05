#!/bin/sh

INFILE=$1 #"timestamps/train (2).csv"
OUTPATH=$2 #"predictions/"

mkdir "prepared"

INNLINES=$(wc -l < "$INFILE")

python prepare_data.py file $INFILE "prepared" False
#python run_LSTM.py hsearch "prepared" $OUTPATH $INNLINES
#python batched_model.py train "prepared/train0001.csv" 100000 "prepared/train0000.csv" 100000 $OUTPATH
python batched_model.py test "prepared" $OUTPATH

#tail -n +2 train0000.csv | split -l 50 - --filter='sh -c "{ head -n1 train0000.csv; cat; } > $FILE"' x