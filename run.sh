#!/bin/bash

INFILE=$1 #"timestamps/train (2).csv"
OUTPATH=$2 #"predictions/"

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda init
#conda activate pkddchat

rm -r splitted
mkdir splitted

rm -r test_prepared
mkdir test_prepared

mkdir $OUTPATH

INNLINES=$(wc -l < "$INFILE")
FLINES=3
NSPLITS=$(( ($INNLINES + ($FLINES - 1)) / $FLINES ))

echo "Innlines:" $INNLINES
echo "Flines:" $FLINES
echo "Nsplits:" $NSPLITS

for ((i=0;i<NSPLITS;i++));
do
	FIRSTLINE=$(( 1+($i*$FLINES) ))
	LASTLINE=$(( (1+$i)*$FLINES ))
	STOPLINE=$(( $LASTLINE + 1 ))
	sed -n '${FIRSTLINE},${LASTLINE}p;${STOPLINE}q' $INFILE #> splitted/xaa
	echo '...'
	
	#python prepare_data.py all "splitted" "test_prepared" False
	
	#python batched_model.py test "test_prepared" $OUTPATH
done


#split -l $FLINES $INFILE splitted/
# tail -n +2 $INFILE | split -l $FLINES - --filter='sh -c "{ head -n1 $INFILE; cat; } > $FILE"' "splitted/x"
# head -$FLINES $INFILE > "splitted/xaa"

#python run_LSTM.py hsearch "prepared" $OUTPATH $INNLINES
#python batched_model.py train "prepared/train_split.csv" 20000000 "prepared/val_split.csv" 9539420
