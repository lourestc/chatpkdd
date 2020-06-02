#!/bin/bash

INFILE=$1 #"timestamps/train (2).csv"
OUTPATH=$2 #"predictions/"

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda init
conda activate pkddchat

mkdir "splitted"

FLINES=100000
tail -n +2 $INFILE | split -l $FLINES - --filter='sh -c "{ head -n1 $INFILE; cat; } > $FILE"' "splitted/x"

mkdir "test_prepared"

python prepare_data.py all "splitted" "test_prepared" False

#INNLINES=$(wc -l < "$INFILE")

#python run_LSTM.py hsearch "prepared" $OUTPATH $INNLINES
#python batched_model.py train "prepared/train_split.csv" 20000000 "prepared/val_split.csv" 9539420

mkdir $OUTPATH

python batched_model.py test "test_prepared" $OUTPATH