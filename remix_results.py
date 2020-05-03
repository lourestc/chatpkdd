import pandas as pd

from run_LSTM import data_fnames

if __name__ == '__main__':

	analisys = {}
	
	fnames = data_fnames('predictions')
	
	df_gt = pd.read_csv('../ChAT/train_truth.csv')
	
	for fname in fnames:
		df_preds = pd.read_csv(fname)
		pd.merge( df_gt, df_preds, on=['channel','user'], suffixes=['',fname.split('/')[-1].split('.')[0]] )
		
	scores = {}
	for col in df_gt.columns[3:]:	
			scores[col] = [] + sklearn.metrics.precision_recall_fscore_support( y_true=df_gt['subscribed'], y_pred=df_gt[col] )
	df_scores = pd.DataFrame.from_dict( scores, columns=['precision','recall','fscore','support'], orient='index' )
	df_scores.to_csv( 'result_scores.csv', index_label='run' )