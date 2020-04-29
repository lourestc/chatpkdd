import pandas as pd
import json
import csv
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

import sys

def read_ground_truth(csv_filename='train_truth.csv'):

	subscribed = {}
	with open(csv_filename) as f:
		csv_reader = csv.DictReader(f)
		for line in csv_reader:
			subscribed.setdefault( line['channel'], {} )[line['user']] = line['subscribed']
	
	return subscribed

def convert_json_DataFrame(json_filename, ground_truth):
	UserChannels = []
	with open(json_filename, 'r') as f:
		for line in f:
			UserChannels.append(json.loads(line))
			
	final_results = dict()
	cont = 0
	
	for item in UserChannels:
		final_results[cont] = list()
		
		channel = item['c']
		user = item['u']
		
		messages = sorted( item['ms'], key=lambda m:m['t'] )
		concatenatedm = ' '.join([ m['m'] for m in messages ])
		
		ts_list = [ m['t'] for m in messages ]
		delta_ts = [0]+[ ts2-ts1 for ts1,ts2 in zip(ts_list[:-1],ts_list[1:]) ]
		
		subscribed = ground_truth[channel][user]
	
		final_results[cont].extend([ channel, user, concatenatedm, json.dumps(delta_ts), subscribed ])		
		cont += 1
		
	return pd.DataFrame.from_dict(final_results, columns=['channel', 'user', 'concatenated_m', 'delta_ts', 'subscribed'], orient='index')

def remove_stopwords(dataframe):
	stoplist =  stopwords.words('english')
	punc = set(punctuation)
	for id, msg in enumerate(dataframe.concatenated_m):
		if str(msg) != "nan":
			text_tokens = word_tokenize(msg)
			tokens_without_sw = [word for word in text_tokens if not word in stoplist]
			s = ''.join(w if set(w) <= punc else ' '+w for w in tokens_without_sw).lstrip()
			dataframe.loc[id, 'concatenated_m'] = s
	return dataframe

def extracting_features(data):

	dts = pd.DataFrame(columns=['delta_ts'])
	dts.loc[:,'delta_ts'] = data['delta_ts'].apply(eval)
	dts.loc[:,'delta_ts'] = dts['delta_ts'].apply(np.asarray)
	
	data.loc[:,'length(delta_ts)'] = dts.apply(lambda row: row['delta_ts'].size, axis=1)
	data.loc[:,'sum(delta_ts)'] = dts.apply(lambda row: np.sum(row['delta_ts']) if row['delta_ts'].size<=1 else np.sum(np.delete(row['delta_ts'],0)), axis=1)
	data.loc[:,'average(delta_ts)'] = dts.apply(lambda row: np.mean(row['delta_ts']) if row['delta_ts'].size<=1 else np.mean(np.delete(row['delta_ts'],0)), axis=1)
	data.loc[:,'std(delta_ts)'] = dts.apply(lambda row: np.std(row['delta_ts']) if row['delta_ts'].size<=1 else np.std(np.delete(row['delta_ts'],0)), axis=1)
	data.loc[:,'min(delta_ts)'] = dts.apply(lambda row: np.amin(row['delta_ts']) if row['delta_ts'].size<=1 else np.amin(np.delete(row['delta_ts'],0)), axis=1)
	data.loc[:,'max(delta_ts)'] = dts.apply(lambda row: np.amax(row['delta_ts']) if row['delta_ts'].size<=1 else np.amax(np.delete(row['delta_ts'],0)), axis=1)	  
	data.loc[:,'length(concatenated_m)'] = data.apply(lambda row: len(row['concatenated_m'].split()), axis=1)

	grouped_by_user = data.groupby('user')
	avg_msgsize_per_user = grouped_by_user['length(concatenated_m)'].mean()
	channels_per_user = grouped_by_user['length(delta_ts)'].count()
	data.loc[:,'average(SizeMsgChannels_User)'] = data.apply(lambda row: avg_msgsize_per_user.loc[row['user']], axis=1)
	data.loc[:,'count(Channels_User)'] = data.apply(lambda row: channels_per_user.loc[row['user']], axis=1)

	grouped_by_channel = data.groupby('channel')
	avg_msgsize_per_channel = grouped_by_channel['length(concatenated_m)'].mean()
	users_per_channel = grouped_by_channel['length(delta_ts)'].count()
	data.loc[:,'average(SizeMsgUsers_Channel)'] = data.apply(lambda row: avg_msgsize_per_channel.loc[row['channel']], axis=1)
	data.loc[:,'count(Users_Channel)'] = data.apply(lambda row: users_per_channel.loc[row['channel']], axis=1)

def prepare_data_from_json(json_filename):

	gt = read_ground_truth('../ChAT/train_truth.csv')
	dataframe = convert_json_DataFrame(json_filename, gt)
	remove_stopwords(dataframe)
	extracting_features(dataframe)
	return dataframe
	
if __name__ == '__main__':

	json_filename = sys.argv[1]
	csv_filename = sys.argv[2]
	
	df = prepare_data_from_json(json_filename)
	df.to_csv(csv_filename)