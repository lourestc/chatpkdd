from batched_model import *

#python3 find_unknown_users.py train_prepared/train0000.csv test_prepared/test.csv test_prepared/test_kn.csv test_prepared/test_unkn.csv
if __name__ == '__main__':

	trainfile = sys.argv[1]
	intestfile = sys.argv[2]
	outtestfile1 = sys.argv[3]
	outtestfile2 = sys.argv[4]
	
	train_df = read_data(trainfile)
	test_df = read_data(intestfile)
	
	test_df[test_df.user.isin(train_df.user)].to_csv(outtestfile1, index=False)
	test_df[~test_df.user.isin(train_df.user)].to_csv(outtestfile2, index=False)