from batched_model import *

#python3 find_unknown_users.py user train_prepared/train0000.csv test_prepared/test.csv test_prepared/test_u_kn.csv test_prepared/test_u_unkn.csv
#python3 find_unknown_users.py channel train_prepared/train0000.csv test_prepared/test.csv test_prepared/test_c_kn.csv test_prepared/test_c_unkn.csv
if __name__ == '__main__':

	field = sys.argv[1]
	trainfile = sys.argv[2]
	intestfile = sys.argv[3]
	outtestfile1 = sys.argv[4]
	outtestfile2 = sys.argv[5]
	
	train_df = read_data(trainfile)
	test_df = read_data(intestfile)
	
	if field=='user':
		test_df[test_df.user.isin(train_df.user)].to_csv(outtestfile1, index=False)
		test_df[~test_df.user.isin(train_df.user)].to_csv(outtestfile2, index=False)
	elif
		field=='channel'
		test_df[test_df.channel.isin(train_df.channel)].to_csv(outtestfile1, index=False)
		test_df[~test_df.channel.isin(train_df.channel)].to_csv(outtestfile2, index=False)