#!/bin/sh

INFILE=$1 #"timestamps/train (2).csv"
OUTPATH=$2 #"predictions/"

mkdir "splitted"

tail -n +2 $INFILE | split -l 10000 - --filter='sh -c "{ head -n1 $INFILE; cat; } > $FILE"' "splitted/x"

mkdir "prepared"

python prepare_data.py all "splitted" "prepared" False

#INNLINES=$(wc -l < "$INFILE")

#python run_LSTM.py hsearch "prepared" $OUTPATH $INNLINES
#python batched_model.py train "prepared/train0001.csv" 100000 "prepared/train0000.csv" 100000 $OUTPATH
python batched_model.py test "prepared" $OUTPATH