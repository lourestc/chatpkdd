from pathlib import Path
import pandas as pd
import sklearn

from run_LSTM import data_fnames

if __name__ == '__main__':

	analisys = {}
	
	fnames = data_fnames('predictions')
	merged_f = Path('merged_results.csv')
	
	if merged_f.is_file():
		df_gt = pd.read_csv(merged_f)
	else:
		df_gt = pd.read_csv('../ChAT/train_truth.csv')
		for fname in fnames:
			df_preds = pd.read_csv(fname, index_col=0)
			df_gt = pd.merge( df_gt, df_preds, on=['channel','user'], suffixes=['','::'+fname.stem] )
		df_gt.to_csv( merged_f, index=False )
	
	scores = {}
	for col in df_gt.columns[3:]:	
			sc = sklearn.metrics.precision_recall_fscore_support( y_true=df_gt['subscribed'], y_pred=df_gt[col] )
			scores[col] = [ item for sublist in sc for item in sublist ]
	df_scores = pd.DataFrame.from_dict( scores, columns=['0precision','1precision','0recall','1recall','0fscore','1fscore','0support','1support'], orient='index' )
	df_scores.to_csv( 'result_scores.csv', index_label='run' )