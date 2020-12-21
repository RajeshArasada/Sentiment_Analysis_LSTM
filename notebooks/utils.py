import csv
import os
import time
import datetime
import re
from utils import * 

# Data Analysis
import numpy as np
import pandas as pd

# Data Saving
import pickle
from pickle import dump
from pickle import load
from functools import partial



from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import confusion_matrix, roc_curve,auc, classification_report

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
print(f"Tensorflow Version: {tf.__version__}")


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
# Import nltk stop codons
stop = set(stopwords.words('english'))
import numpy as np


# print date and time for given type of representation
def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today()

def pretty_print(reviews, labels, idx):
    print("Sentiment {}: \t {}"  .format(labels[idx], reviews[idx][:100]))

def to_lower(text):
    """
    convert all characters to lower case
    """
    return text.lower()

def clean_text(x):
    stop = set(stopwords.words('english'))
    # split into tokens by white space
    tokens = str(x).split()
    # filter out stop words
    tokens = [tok for tok in tokens if not tok.lower() in stop]
    # filter out short words
    tokens = [tok for tok in tokens if len(tok)>1]
    # join the tokens
    return ' '.join(tokens)

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '', x)
    x = re.sub('[0-9]{4,}', '', x)
    x = re.sub('[0-9]{3,}', '', x)
    x = re.sub('[0-9]{2,}', '', x)
    return x

# save a dataset to file
def save_dataset(dataset, filename):
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s' % filename)
    
def load_dataset(filename):
    return load(open(filename, 'rb'))

def compute_ecdf(x, xlabel):
    n = len(x)
    x = np.sort(x)
    y = np.arange(1, n+1)/n
    plt.subplot()
    plt.plot(x, y, '.')
    plt.xlabel(xlabel)
    plt.ylabel("Empirical Cumulative Distribution")
    plt.show()
    
def compute_class_freqs(labels):
    """
    Compute positive and negative frequencies for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    N = labels.shape[0]
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = np.sum(labels == 0, axis=0) / N
    return positive_frequencies, negative_frequencies

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        loss = 0.0

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += -1 * K.mean((pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) + 
                     neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)))
        return loss

    return weighted_loss

def plot_performance(hist, model_name):
    """
    Plots the accuracy and loss during the training process.
    Args: 
    history: variable assigned to model.fit()
    model_name: A string identifying the model
    """
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    fig.suptitle("{} Training".format(model_name), fontsize=20)

    max_epoch = len(hist.history['accuracy'])+1
    epochs_list = list(range(1, max_epoch))

    ax1.plot(epochs_list, hist.history['accuracy'], color='b', linestyle='-', label='Training Data')
    ax1.plot(epochs_list, hist.history['val_accuracy'], color='r', linestyle='-', label ='Validation Data')
    ax1.set_title('Accuracy', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.legend(frameon=False, loc='lower center', ncol=2)

    ax2.plot(epochs_list, hist.history['loss'], color='b', linestyle='-', label='Training Data')
    ax2.plot(epochs_list, hist.history['val_loss'], color='r', linestyle='-', label ='Validation Data')
    ax2.set_title('Loss', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.legend(frameon=False, loc='upper center', ncol=2)
    
    if figure_directory:
        plt.savefig(figure_directory+"/history")
    plt.show()
    
def create_bidirectional_LSTM_model(num_class):
    model = Sequential()
   
    model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)))
    for units in [128, 64]:
        model.add(tf.keras.layers.Dense(units, activation = 'relu'))
        model.add(tf.keras.layers.Dropout(0.5))
    
    if num_class>=2:
        model.add(tf.keras.layers.Dense(num_class, activation ='softmax'))
    else:
        model.add(tf.keras.layers.Dense(1, activation ='sigmoid'))
    
    model.summary()  
    return model

def create_cnn_model(num_class):   
    model = Sequential()
    
    model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN))
    
    model.add(tf.keras.layers.Conv1D(1024, 3, padding='valid', activation='relu', strides=1))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2048, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    
    if num_class>=2:
        model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.summary()
    return model

def create_multibranch_CNN(length, vocab_size, num_class):
    """
    define three different CNN architectures using Functional API
    Each CNN uses kernels of different sizes: 4,6,8
    Args:
    length - length of the input data for example length of the input sequence
    vocab_size - Size of the vocabulary - with Tokenizer it's len of the word_index + 1
    num_class - number of classes the model is trying to classify.
    
    """
    #branch 01
    inputs_01 = tf.keras.layers.Input(shape=(length, ))         #   max length of the input data
    embedding_01 = Embedding(VOCAB_SIZE, EMBED_DIM)(inputs_01)
    conv_01 = tf.keras.layers.Conv1D(filters=256, kernel_size=4, activation='relu')(embedding_01)
    drop_01 = tf.keras.layers.Dropout(0.5)(conv_01)
    pool_01 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop_01)
    flat_01 = tf.keras.layers.Flatten()(pool_01)
    
    #branch 02
    inputs_02 = tf.keras.layers.Input(shape=(length, ))
    embedding_02 = Embedding(VOCAB_SIZE, EMBED_DIM)(inputs_02)
    conv_02 = tf.keras.layers.Conv1D(filters=256, kernel_size=8, activation='relu')(embedding_02)
    drop_02 = tf.keras.layers.Dropout(0.5)(conv_02)
    pool_02 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop_02)
    flat_02 = tf.keras.layers.Flatten()(pool_02)
    
    #branch 03
    inputs_03 = tf.keras.layers.Input(shape=(length, ))
    embedding_03 = Embedding(VOCAB_SIZE, EMBED_DIM)(inputs_03)
    conv_03 = tf.keras.layers.Conv1D(filters=128, kernel_size=11, activation='relu')(embedding_03)
    drop_03 = tf.keras.layers.Dropout(0.5)(conv_03)
    pool_03 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop_03)
    flat_03 = tf.keras.layers.Flatten()(pool_03)
    
    #merge
    merged = tf.keras.layers.concatenate([flat_01, flat_02, flat_03])
    
    # interpretation
    dense_01 = tf.keras.layers.Dense(200, activation='relu')(merged)
    
    if num_class>2:
        outputs = tf.keras.layers.Dense(num_class, activation='softmax')(dense_01)
    else:
        outputs = tf.keras.layers.Dense(num_class, activation='sigmoid')(dense_01)
    
    model = tf.keras.Model(inputs=[inputs_01, inputs_02, inputs_03], outputs=outputs)
    return model

def create_LSTM_model(num_classes):
    model = Sequential()
    model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(num_classes, activation ='softmax'))
    model.summary()
    return model
      
def create_Bidirectional_LSTM_model(num_classes):
    model = Sequential()
    model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))
    model.add(Bidirectional(tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def create_cnn_bidirectional_LSTM_model(num_class):
    model = Sequential()
    model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN))
    model.add(tf.keras.layers.Conv1D(256, kernel_size=8, strides=2, padding="valid",
                     activation='relu', name='conv1d'))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)))
    # Multiple Dense layers
    for units in [128, 64]:
        model.add(tf.keras.layers.Dense(units, activation = 'relu'))
        model.add(tf.keras.layers.Dropout(0.5))
    
    if num_class>=2:
        model.add(tf.keras.layers.Dense(num_class, activation ='softmax'))
    else:
        model.add(tf.keras.layers.Dense(1, activation ='sigmoid'))
    return model

def create_cnn_bidirectional_LSTM_model_02(num_class):
    model = Sequential()
    model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))
    model.add(tf.keras.layers.Conv1D(256, kernel_size=8, strides=2, padding="valid",
                     activation='relu', name='conv1d'))
    model.add(tf.keras.layers.BatchNormalization(name='bn_conv_1d'))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)))
    # Multiple Dense layers
    for idx, units in enumerate([128, 64]):
        model.add(tf.keras.layers.Dense(units, activation = 'relu', name=f'dense_{idx}'))
        model.add(tf.keras.layers.BatchNormalization(name=f'bn_dense_{idx}'))
        model.add(tf.keras.layers.Dropout(0.5))
    
    if num_class>=2:
        model.add(tf.keras.layers.Dense(num_class, activation ='softmax'))
    else:
        model.add(tf.keras.layers.Dense(1, activation ='sigmoid'))
    
    return model
        
        