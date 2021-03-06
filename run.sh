#!/bin/bash

INFILE=$1"/test.json"
OUTPATH=$2

#CONDA_BASE=$(conda info --base)
#source ${CONDA_BASE}/etc/profile.d/conda.sh
#conda init

rm -rf splitted
mkdir splitted

rm -rf test_prepared
mkdir test_prepared

mkdir -p $OUTPATH

#echo 'user,channel,subscribed' > $OUTPATH'/predictions.csv'

INNLINES=$(wc -l < "$INFILE")
FLINES=10000
NSPLITS=$(( ($INNLINES + ($FLINES - 1)) / $FLINES ))

echo "Innlines:" $INNLINES
echo "Flines:" $FLINES
echo "Nsplits:" $NSPLITS

for ((i=0;i<NSPLITS;i++));
do
	FIRSTLINE=$(( 1+($i*$FLINES) ))
	LASTLINE=$(( (1+$i)*$FLINES ))
	STOPLINE=$(( $LASTLINE + 1 ))
	sed -n "${FIRSTLINE},${LASTLINE}p;${STOPLINE}q" $INFILE > splitted/xaa
	
	python3 prepare_data.py all "splitted" "test_prepared" False
	
	python3 batched_model.py test "test_prepared" $OUTPATH
done


#split -l $FLINES $INFILE splitted/
# tail -n +2 $INFILE | split -l $FLINES - --filter='sh -c "{ head -n1 $INFILE; cat; } > $FILE"' "splitted/x"
# head -$FLINES $INFILE > "splitted/xaa"

#python run_LSTM.py hsearch "prepared" $OUTPATH $INNLINES
#python batched_model.py train "prepared/train_split.csv" 20000000 "prepared/val_split.csv" 9539420
