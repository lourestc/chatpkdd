import sys
import string

from pathlib import Path
import json
import csv
import pandas as pd
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import sklearn

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

import nltk

import keras
#from keras.callbacks import EarlyStopping
#from keras import backend as K
#from keras.regularizers import l2
#from keras.utils.vis_utils import plot_model

from prepare_data import *

batch_size = 128

def read_data(csv_filename, skiprows=None, nrows=None):
	if skiprows>0:
		df = pd.read_csv(csv_filename, skiprows=range(1,skiprows+1), nrows=nrows)
	else:
		df = pd.read_csv(csv_filename)
	df.loc[:,'delta_ts'] = df['delta_ts'].apply(eval)
	df.loc[:,'delta_ts'] = df['delta_ts'].apply(np.asarray)
	return df
	
def clean_text(text):
	cleant = []
	punc = set(string.punctuation)
	for id, msg in enumerate(text):
		if str(msg) != "nan":
			text_tokens = nltk.tokenize.word_tokenize(msg)
			msg = ''.join(w if set(w) <= punc else ' '+w+' ' for w in text_tokens)
			msg = ' '.join(msg.split())
		cleant.append(str(msg).lower())
	return cleant

def load_text_batch(Train_df, idx, batch_size):
	df = read_data(Train_df, skiprows=idx*batch_size, nrows=batch_size)
	return( clean_text(df.concatenated_m) )

def batch_text_generator(Train_df, batch_size, steps):
	idx=0
	while idx<steps: 
		yield load_text_batch(Train_df,idx,batch_size)
		idx += 1

def train_tokenizer( infile, batch_size, innlines ):

	tokenizer_steps=np.ceil(innlines/batch_size)
	
	tokenizer = keras.preprocessing.text.Tokenizer(filters='\t\n', lower=True, oov_token = True)
	tokenizer_generator = batch_text_generator( infile, batch_size, tokenizer_steps )
	tokenizer.fit_on_texts(tokenizer_generator) #create a array of words with indices
	
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))
	
	with open('modelos/tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
	return tokenizer, word_index
	
def load_tokenizer():

	with open('modelos/tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	
	return tokenizer, tokenizer.word_index

def run_tokenizer(tokenizer, text, MAX_WORDS):

	sequences = tokenizer.texts_to_sequences(clean_text(text)) #transform in an array of indices
	padded_seq = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_WORDS)
	
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

def build_model(sequence_length, feature_list, word_index, embedding_matrix, EMBEDDING_DIM, OPT, LR):

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
	
	feature_input = keras.layers.Input(shape=(len(feature_list),))
	
	merged_model = keras.layers.concatenate([text_model, feature_input])
	merged_model = keras.layers.Dense(1, activation='sigmoid')(merged_model)
	
	model = keras.models.Model( inputs=[text_input,feature_input], outputs=merged_model )
	model.compile(loss='binary_crossentropy',#categoric_crossentropy
					optimizer=OPT(lr=LR), #, rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=0.5
					metrics=['accuracy'])
					
	img = keras.utils.vis_utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	model.summary()
					
	return model

def load_data_batch(Train_df, idx, batch_size, tokenizer, max_w, feature_list):
	df = read_data(Train_df, skiprows=idx*batch_size, nrows=batch_size)
	X_train = run_tokenizer(tokenizer, df.concatenated_m, max_w)
	return [X_train, df[feature_list]], df.subscribed

def batch_generator(Train_df, batch_size, steps, tokenizer, max_w, feature_list):
	idx=0
	while True: 
		yield load_data_batch(Train_df,idx,batch_size, tokenizer, max_w, feature_list) #Yields data
		if idx<steps:
			idx += 1
		else:
			idx = 0
			
def train_model(model, tokenizer, max_w, trainfile, trainlines, valfile, vallines, batch_size, epochs, savemodel=False):
	
	steps_per_epoch = np.ceil(trainlines / batch_size)
	validation_steps = np.ceil(vallines / batch_size)
	
	print(steps_per_epoch,validation_steps)
	input("Press Enter to continue...")
	
	training_batch_generator = batch_generator(trainfile, batch_size, steps_per_epoch, tokenizer, max_w, feature_list)
	validation_batch_generator = batch_generator(valfile, batch_size, validation_steps, tokenizer, max_w, feature_list)
	
	val_loss = None
	val_acc = 0
	for e in range(0,epochs):
		print('epochs', e)

		hist = model.fit_generator( training_batch_generator, epochs=1, steps_per_epoch=steps_per_epoch,
													validation_data=validation_batch_generator, validation_steps=validation_steps,
													verbose=1, workers=0 ) # , use_multiprocessing=True

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

def train_batched( trainfile, trainlines, valfile, vallines, feature_list ):
	
	max_w = 300
	embedding_d = 100
	opt = keras.optimizers.Adagrad
	lr = 0.001
	
	print("Training tokenizer...")
	tokenizer, word_index = train_tokenizer( trainfile, batch_size, trainlines )
	
	print("Creating embedding matrix... embedding_d="+str(embedding_d))
	embedding_matrix = create_embedding_matrix(word_index, embedding_d)
	
	print("Building model... opt="+str(opt.__name__)+"; lr="+str(lr))
	model = build_model(max_w, feature_list, word_index, embedding_matrix, embedding_d, opt, lr)
	
	print("Training model...")
	train_model(model, tokenizer, max_w, trainfile, trainlines, valfile, vallines, batch_size, epochs=1, savemodel=True)

def load_test_batch(Test_df, idx, batch_size, tokenizer, max_w, feature_list):
	df = read_data(Test_df, skiprows=idx*batch_size, nrows=batch_size)
	X_test = run_tokenizer(tokenizer, df.concatenated_m, max_w)
	return [X_test, df[feature_list]]

def test_generator(Test_df, batch_size, steps, tokenizer, max_w, feature_list):
	idx=1
	while True: 
		yield load_test_batch(Test_df,idx-1,batch_size, tokenizer, max_w, feature_list) #Yields data
		if idx<steps:
			idx += 1
		else:
			idx = 1

def test_model(model, tokenizer, max_w, testfile, testlines, batch_size):

	test_steps=np.ceil(testlines/batch_size)
	
	testing_batch_generator = test_generator(testfile, batch_size, test_steps, tokenizer, max_w)
	
	predicted_test = model.predict_generator(testing_batch_generator, steps=test_steps, verbose=1)
	
	predictions = []
	for i, predicted2 in enumerate(predicted_test):
		if predicted2[0] >= 0.50:
			predictions.append(1)
		else:
			predictions.append(0)
			
	return predictions

def test_batched( testfile, testlines, outpath, feature_list ):
	
	max_w = 300
	embedding_d = 100
	opt = keras.optimizers.Adagrad
	lr = 0.001
	
	print("Loading tokenizer...")
	tokenizer, word_index = load_tokenizer()
	
	print("Creating embedding matrix... embedding_d="+str(embedding_d))
	embedding_matrix = create_embedding_matrix(word_index, embedding_d)
	
	print("Building model... opt="+str(opt.__name__)+"; lr="+str(lr))
	model = build_model(max_w, feature_list, word_index, embedding_matrix, embedding_d, opt, lr)
	
	print("Loading model weights...")
	model.load_weights('modelos/model_weights.h5')
	
	print("Generating predictions model...")
	predictions = test_model(model, tokenizer, max_w, testfile, testlines, batch_size)

def test_model_simple(model, tokenizer, max_w, test_data):

	X_test = [ run_tokenizer(tokenizer, test_data.concatenated_m, max_w), test_data[feature_list] ]
	
	predicted_test = model.predict(X_test)
	
	predictions = []
	for i, predicted2 in enumerate(predicted_test):
		if predicted2[0] >= 0.50:
			predictions.append(1)
		else:
			predictions.append(0)
			
	return predictions
	
def save_predictions( predictions, data, out_filename ):

	odict = {}
	
	for i,pred in enumerate(predictions):
		odict[i] = [ data.loc[i,'user'], data.loc[i,'channel'], pred ]
		
	out = pd.DataFrame.from_dict(odict, columns=['user', 'channel', 'subscribed'], orient='index')
	out.to_csv(out_filename, mode='a')

def test_simple( testpath, outpath, feature_list ):
	
	out = pd.DataFrame.from_dict({}, columns=['user', 'channel', 'subscribed'], orient='index')
	out.to_csv(outpath+'/preds.csv')
	
	max_w = 300
	embedding_d = 100
	opt = keras.optimizers.Adagrad
	lr = 0.001
	
	print("Loading tokenizer...")
	tokenizer, word_index = load_tokenizer()
	
	print("Creating embedding matrix... embedding_d="+str(embedding_d))
	embedding_matrix = create_embedding_matrix(word_index, embedding_d)
	
	print("Building model... opt="+str(opt.__name__)+"; lr="+str(lr))
	model = build_model(max_w, feature_list, word_index, embedding_matrix, embedding_d, opt, lr)
	
	for testfile in data_fnames(testpath):
	
		print("Loading model weights...")
		model.load_weights('modelos/model_weights.h5')
		
		print("Predicting...")
		test_data = read_data(testfile)
		predictions = test_model_simple(model, tokenizer, max_w, test_data)
		save_predictions( predictions, test_data, outpath+'/preds.csv' )

if __name__ == '__main__':
	
	feature_list = [ 'sum(delta_ts)', 'average(delta_ts)', 'std(delta_ts)', 'min(delta_ts)', 'max(delta_ts)', 'length(delta_ts)', 'length(concatenated_m)', 'average(SizeMsgChannels_User)', 'count(Channels_User)', 'average(SizeMsgUsers_Channel)', 'count(Users_Channel)' ]
	
	mode = sys.argv[1]
	
	if mode == 'train':
		trainfile = sys.argv[2]
		trainlines = int(sys.argv[3])
		valfile = sys.argv[4]
		vallines = int(sys.argv[5])
		train_batched( trainfile, trainlines, valfile, vallines, feature_list )
	elif mode == 'test':
		testpath = sys.argv[2]
		outpath = sys.argv[3]
		test_simple( testpath, outpath, feature_list )
		#test_batched( testfile, testlines, outpath, feature_list )
	else:
		print("ERROR: Unkown execution mode.")
		sys.exit(1)
	
	print("Done.")