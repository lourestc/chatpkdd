from pathlib import Path
import pandas as pd
import json
import csv
import numpy as np
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

import sys

class UCFeatures:
	self.user = None
	self.channel = None
	self.subscribed = None
	self.length_delta_ts = None
	self.sum_delta_ts = None
	self.average_delta_ts = None
	self.std_delta_ts = None
	self.min_delta_ts = None
	self.max_delta_ts = None
	self.length_concatenated_m = None
	self.average_SizeMsgChannels_User = None
	self.count_Channels_User = None
	self.average_SizeMsgUsers_Channel = None
	self.count_Users_Channel = None

def data_fnames(folderpath='timestamps'):
	train_path = Path(folderpath)
	return [ fname for fname in train_path.glob('*') ]

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
		
		if ground_truth:
			subscribed = ground_truth[channel][user]
			final_results[cont].extend([ channel, user, concatenatedm, json.dumps(delta_ts), subscribed ])		
			cont += 1
		else:
			final_results[cont].extend([ channel, user, concatenatedm, json.dumps(delta_ts) ])		
			cont += 1
			
	if ground_truth:
		return pd.DataFrame.from_dict(final_results, columns=['channel', 'user', 'concatenated_m', 'delta_ts', 'subscribed'], orient='index')
	else:
		return pd.DataFrame.from_dict(final_results, columns=['channel', 'user', 'concatenated_m', 'delta_ts'], orient='index')

def remove_stopwords(dataframe):
	stoplist =  stopwords.words('english')
	punc = set(punctuation)
	for id, msg in enumerate(dataframe.concatenated_m):
		if str(msg) != "nan":
			text_tokens = word_tokenize(msg)
			tokens_without_sw = [word for word in text_tokens if not word in stoplist]
			s = ''.join(w if set(w) <= punc else ' '+w+' ' for w in tokens_without_sw)
			s = ' '.join(s.split())
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

def prepare_data_from_json(json_filename,gt):

	print("Converting JSON to DataFrame... file:", json_filename)
	dataframe = convert_json_DataFrame(json_filename, gt)
	
	print("Removing stopwords...")
	remove_stopwords(dataframe)
	
	print("Extracting features...")
	extracting_features(dataframe)
	
	return dataframe
	
def base_process_json_line(uc_json, uc_features):
	
	uc_features.user = uc_json['user']
	uc_features.channel = uc_json['channel']
	
	messages = sorted( uc_json['ms'], key=lambda m:m['t'] )
	concatenatedm = ' '.join([ m['m'] for m in messages ])
	
	ts_list = [ m['t'] for m in messages ]
	delta_ts = [0]+[ ts2-ts1 for ts1,ts2 in zip(ts_list[:-1],ts_list[1:]) ]
	
	return concatenatedm, delta_ts

def remove_stopwords_line(concatenatedm):
	stoplist =  stopwords.words('english')
	punc = set(punctuation)
	s = concatenatedm
	if str(s) != "nan":
		text_tokens = word_tokenize(s)
		tokens_without_sw = [word for word in text_tokens if not word in stoplist]
		s = ''.join(w if set(w) <= punc else ' '+w+' ' for w in tokens_without_sw)
		s = ' '.join(s.split())
	return s

def extract_features_line(uc_features, concatenatedm, delta_ts, UStats, CStats):

	dts = np.asarray(eval(delta_ts))
	
	uc_features.length_delta_ts = dts.size
	uc_features.sum_delta_ts = np.sum(dts) if dts.size<=1 else np.sum(np.delete(dts,0))
	uc_features.average_delta_ts = np.mean(dts) if dts.size<=1 else np.mean(np.delete(dts,0))
	uc_features.std_delta_ts = np.std(dts) if dts.size<=1 else np.std(np.delete(dts,0))
	uc_features.min_delta_ts = np.amin(dts) if dts.size<=1 else np.amin(np.delete(dts,0))
	uc_features.max_delta_ts = np.amax(dts) if dts.size<=1 else np.amax(np.delete(dts,0))
	uc_features.length_concatenated_m = len(concatenatedm.split())
	
	UStats[uc_features.user] = UStats.get(uc_features.user, {'grouped_length_concatenated_m':0, 'grouped_length_delta_ts':0, 'count':0})
	UStats[uc_features.user]['grouped_length_concatenated_m'] += uc_features.length_concatenated_m
	UStats[uc_features.user]['grouped_length_delta_ts'] += uc_features.length_delta_ts
	UStats[uc_features.user]['count'] += 1
	
	CStats[uc_features.channel] = CStats.get(uc_features.channel, {'grouped_length_concatenated_m':0, 'grouped_length_delta_ts':0, 'count':0})
	CStats[uc_features.channel]['grouped_length_concatenated_m'] += uc_features.length_concatenated_m
	CStats[uc_features.channel]['grouped_length_delta_ts'] += uc_features.length_delta_ts
	CStats[uc_features.channel]['count'] += 1

def extract_features_grouped(UserChannels, UStats, CStats):

	for uc_features in UserChannels:
		uc_features.average_SizeMsgChannels_User = UStats[uc_features.user]['grouped_length_concatenated_m'] / UStats[uc_features.user]['count']
		uc_features.count_Channels_User = UStats[uc_features.user]['count']
		uc_features.average_SizeMsgUsers_Channel = CStats[uc_features.channel]['grouped_length_concatenated_m'] / CStats[uc_features.channel]['count']
		uc_features.count_Users_Channel = CStats[uc_features.channel]['count']

def iterative_prepare_data_from_json(json_filename):
	
	UserChannels = list()
	UStats = dict()
	CStats = dict()
	
	with open(json_filename, 'r') as f:
		for line in f:
			uc_json = json.loads(line)
			uc_features = UCFeatures()
			
			concatenatedm, delta_ts = base_process_json_line(uc_json, uc_features)
			concatenatedm = remove_stopwords_line(concatenatedm)
			
			extract_features_line(uc_features, concatenatedm, delta_ts, UStats, CStats)
			
			UserChannels.append(uc_features)
	
	extract_features_grouped(UserChannels, UStats, CStats)
	
	for uc_features in UserChannels:
		if uc_features.average_SizeMsgChannels_User == None:
			print("OPS")
			return
	
	return UserChannels
	
def write_csv_dataframe(json_filename, UserChannels, ground_truth, csv_filename):
	
	with open(csv_filename, 'w') as f2:	
		
		f2.write(',channel,user,concatenated_m,delta_ts,subscribed,length(delta_ts),sum(delta_ts),average(delta_ts),std(delta_ts),min(delta_ts),max(delta_ts),length(concatenated_m),average(SizeMsgChannels_User),count(Channels_User),average(SizeMsgUsers_Channel),count(Users_Channel)\n' )
		
		with open(json_filename, 'r') as f:
			cont = 0 
			for line in f:
				uc_json = json.loads(line)
				uc_features = UCFeatures()
				
				concatenatedm, delta_ts = base_process_json_line(uc_json, uc_features)
				concatenatedm = remove_stopwords_line(concatenatedm)
				
				if ground_truth:
					subscribed = ground_truth[uc_features.channel][uc_features.user]
					print_line = [
						cont,
						uc_features.channel,
						uc_features.user,
						concatenatedm,
						delta_ts,
						subscribed,
						uc_features.length_delta_ts,
						uc_features.sum_delta_ts,
						uc_features.average_delta_ts,
						uc_features.std_delta_ts,
						uc_features.min_delta_ts,
						uc_features.max_delta_ts,
						uc_features.length_concatenated_m,
						uc_features.average_SizeMsgChannels_User,
						uc_features.count_Channels_User,
						uc_features.average_SizeMsgUsers_Channel,
						uc_features.count_Users_Channel
						]
					cont += 1
				else:
					print_line = [
						cont,
						uc_features.channel,
						uc_features.user,
						concatenatedm,
						delta_ts,
						uc_features.length_delta_ts,
						uc_features.sum_delta_ts,
						uc_features.average_delta_ts,
						uc_features.std_delta_ts,
						uc_features.min_delta_ts,
						uc_features.max_delta_ts,
						uc_features.length_concatenated_m,
						uc_features.average_SizeMsgChannels_User,
						uc_features.count_Channels_User,
						uc_features.average_SizeMsgUsers_Channel,
						uc_features.count_Users_Channel
						]
					cont += 1
					
				f2.write( ','.join(feature for feature in print_line) + '\n' )

if __name__ == '__main__':

	mode = sys.argv[1]
	inpath = sys.argv[2]
	outpath = sys.argv[3]
	include_gt = sys.argv[4]
	
	if include_gt=='True':
		if len(sys.argv)>5:
			gt = read_ground_truth(sys.argv[5])
		else:
			gt = read_ground_truth('../ChAT/train_truth.csv')
	else:
		gt = None
	
	if mode == 'first':
		infile = data_fnames(inpath)[0]
		df = prepare_data_from_json(infile,gt)
		df.to_csv(outpath+'/'+infile.stem+'.csv')
	elif mode == 'file':
		infile = Path(inpath)
		df = prepare_data_from_json(infile,gt)
		df.to_csv(outpath+'/'+infile.stem+'.csv')
	elif mode == 'all':
		for infile in data_fnames(inpath):
			df = prepare_data_from_json(infile,gt)
			df.to_csv(outpath+'/'+infile.stem+'.csv')
	elif mode == 'iterate':
		infile = Path(inpath)
		UserChannels = iterative_prepare_data_from_json(infile)
		write_csv_dataframe(json_filename, UserChannels, gt, outpath+'/'+infile.stem+'.csv')
	else:
		print("ERROR: invalid mode.")
		sys.exit(1)
