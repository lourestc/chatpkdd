import sys
import string

from pathlib import Path
import json
import csv
import pandas as pd
import pickle

import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import TweetTokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import sklearn

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

import keras
#from keras.callbacks import EarlyStopping
#from keras import backend as K
#from keras.regularizers import l2
#from keras.utils.vis_utils import plot_model

from prepare_data import *

def read_data(csv_filename, skiprows=None, nrows=None):
	df = pd.read_csv(csv_filename, skiprows=idx*batch_size, nrows=batch_size)
	df.loc[:,'delta_ts'] = df['delta_ts'].apply(eval)
	df.loc[:,'delta_ts'] = df['delta_ts'].apply(np.asarray)
	return df
	
def clean_text(text):
	cleant = []
	punc = set(string.punctuation)
	for id, msg in enumerate(text):
		if str(msg) != "nan":
			text_tokens = word_tokenize(msg)
			msg = ''.join(w if set(w) <= punc else ' '+w+' ' for w in text_tokens)
			msg = ' '.join(s.split())
		cleant.append(str(msg))
	return cleant
	#tknzr = TweetTokenizer(reduce_len=True, preserve_case=False)
	#return [ ' '.join(tknzr.tokenize(str(m))) for m in text  ]

def train_tokenizer(text):

	cleaned_m = clean_text(text)
	tokenizer = Tokenizer(filters='\t\n', lower=True, oov_token = True)
	tokenizer.fit_on_texts(cleaned_m) #create a array of words with indices
	
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))
	
	return tokenizer, word_index

def run_tokenizer(tokenizer, text, MAX_WORDS):

	sequences = tokenizer.texts_to_sequences(clean_text(text)) #transform in an array of indices
	padded_seq = pad_sequences(sequences, maxlen=MAX_WORDS)
	
	return padded_seq

def create_embedding_matrix(word_index, EMBEDDING_DIM):

	#load google embeddings
	word_vectors = {}
	vocabulary_size = len(word_index)+1
	embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
	print( "Embedding matrix shape:", embedding_matrix.shape )

	#inicialize embedding matrix based on Google embeddings
	for word, i in word_index.items():
		try:
			embedding_vector = word_vectors[word]
			embedding_matrix[i] = embedding_vector
		except KeyError:
			embedding_matrix[i]=np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)

	return embedding_matrix

def build_model(X_train, feat_train, word_index, embedding_matrix, EMBEDDING_DIM, OPT, LR):

	sequence_length = X_train.shape[1]
	print("sequence_length", sequence_length)
	#filters = 150
	#kernel_size = 3
	drop = 0.5

	print('Build model...')
	
	text_input = keras.layers.Input(shape=(sequence_length,))
	text_model = keras.layers.Embedding(len(word_index)+1,
								EMBEDDING_DIM,
								weights=[embedding_matrix],
								input_length=sequence_length,
								trainable=True)( text_input )
	text_model = keras.layers.Dropout(0.5)( text_model )  # helps prevent overfitting
	text_model = keras.layers.LSTM(100)( text_model )
	text_model = keras.layers.Dropout(0.5)( text_model )
	
	feature_input = keras.layers.Input(shape=(feat_train.shape[1],))
	
	merged_model = keras.layers.concatenate([text_model, feature_input])
	merged_model = keras.layers.Dense(1, activation='sigmoid')(merged_model)
	
	model = keras.models.Model( inputs=[text_input,feature_input], outputs=merged_model )
	model.compile(loss='binary_crossentropy',#categoric_crossentropy
					optimizer=OPT(lr=LR), #, rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=0.5
					metrics=['accuracy'])
					
	img = keras.utils.vis_utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	model.summary()
					
	return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, savemodel=False):

	val_loss = None
	val_acc = 0
	for e in range(0,epochs):
		print('epochs', e)

		hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=34, epochs=1)


		print(model.layers[2].name)
		print(model.layers[2].input.shape)
		print(model.layers[2].output.shape)


		print(model.layers[3].name)
		print(model.layers[3].input.shape)
		print(model.layers[3].output.shape)


		if val_loss==None or hist.history['val_loss'][0] < val_loss['loss']:
			val_acc = hist.history['val_accuracy'][0]
			val_loss = {'loss': hist.history['val_loss'][0], 'epoch': e}
			if savemodel:
				model.save_weights('modelos/model_weights.h5', overwrite=True)
				print('epochs_save', e)
	#callbacks = [EarlyStopping(monitor='val_loss')]
	#hist = model.fit(X_train, y_train, batch_size=50, epochs=1, validation_split=0.2, callbacks=callbacks)

def test_model(model, X_test, y_test):

	#predicted_classes = model.predict_classes(X_test)
	#np.savetxt('predictions_classes/' + name_writer + "_predictions_classes.csv", predicted_classes, delimiter=",")

	predicted_test = model.predict(X_test)
	#np.savetxt('predictions_original/' + name_writer + "_predictions_original.csv", predicted_test, delimiter=",")

	predictions = []
	for i, predicted2 in enumerate(predicted_test):
		if predicted2[0] >= 0.50:
			predictions.append(1)
		else:
			predictions.append(0)

	#print("iguais:", predicted_classes == predictions)
	#np.savetxt('predictions/' + name_writer + "_predictions.csv", predictions, delimiter=",")
	metrics = sklearn.metrics.precision_recall_fscore_support(y_true=np.asarray(test_data.subscribed), y_pred=predictions)

	df_metrics = pd.DataFrame(list(metrics), index=['Precision', 'Recall', "F-Score", "Support"])
	#df.transpose().to_csv('metrics/' + 'result' + name_writer + '_metrics.csv', sep='\t', encoding='utf-8')
	
	print(df_metrics)
	return predictions

def save_predictions( predictions, data, out_filename ):

	odict = {}
	
	for i,pred in enumerate(predictions):
		odict[i] = [ data.loc[i,'user'], data.loc[i,'channel'], pred ]
		
	out = pd.DataFrame.from_dict(odict, columns=['user', 'channel', 'subscribed'], orient='index')
	out.to_csv(out_filename)

def hyperparemeter_search(inpath, outpath, feature_list):
	
	# hyperparameters
	max_words = [300, 400, 500]
	embedding_dims = [100, 200, 300]
	optimizers = [keras.optimizers.RMSprop, keras.optimizers.Adam, keras.optimizers.Adagrad]
	learning_rates = [0.001, 0.0001]

	print("Reading data...")
	#data = read_data('timestamps/train_22_291_184_80_shuffle.csv')
	data = read_data(data_fnames(inpath)[0])
	train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
	train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

	print("Training tokenizer...")
	tokenizer, word_index = train_tokenizer(train_data.concatenated_m)
	
	for max_w in max_words:
		for embedding_d in embedding_dims:
		
			print("Preprocessing text... max_w="+str(max_w))
			X_train = run_tokenizer(tokenizer, train_data.concatenated_m, max_w)
			X_val = run_tokenizer(tokenizer, val_data.concatenated_m, max_w)
			X_test = run_tokenizer(tokenizer, test_data.concatenated_m, max_w)

			print("Creating embedding matrix... embedding_d="+str(embedding_d))
			embedding_matrix = create_embedding_matrix(word_index, embedding_d)
			
			for opt in optimizers:
				for lr in learning_rates:
				
					print("Optmizer model... opt="+str(opt.__name__)+"; lr="+str(lr))
					model = build_model(X_train, train_data[feature_list], word_index, embedding_matrix, embedding_d, opt, lr)
					
					print("Training model...")
					train_model(model, [X_train, train_data[feature_list]], train_data.subscribed, [X_val, val_data[feature_list]], val_data.subscribed, epochs=1)
					
					print(f"Evaluating with {max_w} max_words, {embedding_d} dimension of embedding, {opt.__name__} optimizer and {lr} learning rate")
					predictions = test_model(model, [X_test, test_data[feature_list]], test_data.subscribed)
					
					outfile = outpath+'/preds-maxw_'+str(max_w)+'-edim_'+str(embedding_d)+'-opt_'+str(opt.__name__)+'-lr_'+str(lr)+'.csv'
					save_predictions( predictions, data, outfile )

def train_all_files( inpath, outpath, feature_list ):

	max_w = 300
	embedding_d = 300
	opt = keras.optimizers.Adagrad
	lr = 0.001
	
	print("Reading data...")
	train_data = read_data(data_fnames(inpath)[0])
	
	print("Training tokenizer...")
	tokenizer, word_index = train_tokenizer(train_data.concatenated_m)
	X_train = run_tokenizer(tokenizer, train_data.concatenated_m, max_w)
	
	print("Creating embedding matrix... embedding_d="+str(embedding_d))
	embedding_matrix = create_embedding_matrix(word_index, embedding_d)
	
	print("Optmizer model... opt="+str(opt.__name__)+"; lr="+str(lr))
	model = build_model(X_train, train_data[feature_list], word_index, embedding_matrix, embedding_d, opt, lr)
	model.save_weights('modelos/model_weights.h5', overwrite=True)
	
	for in_f in data_fnames(inpath):
	
		print("===============================================")
		print("FILE:", in_f)
	
		print("Reading data...")
		data = read_data(in_f)
		
		for train_kf, val_kf in sklearn.model_selection.KFold(n_splits = 5, shuffle = True, random_state = 1).split(data):
		
			train_data = data.iloc[train_kf]
			val_data = data.iloc[val_kf]
			
			print("Preprocessing text... max_w="+str(max_w))
			X_train = run_tokenizer(tokenizer, train_data.concatenated_m, max_w)
			X_val = run_tokenizer(tokenizer, val_data.concatenated_m, max_w)

			print("Creating embedding matrix... embedding_d="+str(embedding_d))
			embedding_matrix = create_embedding_matrix(word_index, embedding_d)
			
			print("Optmizer model... opt="+str(opt.__name__)+"; lr="+str(lr))
			model.load_weights('modelos/model_weights.h5')
			
			print("Training model...")
			train_model(model, [X_train, train_data[feature_list]], train_data.subscribed, [X_val, val_data[feature_list]], val_data.subscribed, epochs=10, savemodel=True)

if __name__ == '__main__':
	
	mode = sys.argv[1]
	inpath = sys.argv[2]
	outpath = sys.argv[3]
	innlines = sys.argv[4]
	
	feature_list = [ 'sum(delta_ts)', 'average(delta_ts)', 'std(delta_ts)', 'min(delta_ts)', 'max(delta_ts)', 'length(delta_ts)', 'length(concatenated_m)', 'average(SizeMsgChannels_User)', 'count(Channels_User)', 'average(SizeMsgUsers_Channel)', 'count(Users_Channel)' ]
	
	if mode == 'hsearch':
		hyperparemeter_search( inpath, outpath, feature_list )
	elif mode == 'train':
		train_all_files( inpath, outpath, feature_list )
	elif mode == 'test':
		test_file( inpath, outpath, feature_list )
	else:
		print("ERROR: Unkown execution mode.")
		sys.exit(1)
	
	print("Done.")