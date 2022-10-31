# Import required packages

# Python Standard Packages

import re
import pickle

# External Packages

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# For model building
import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Activation, Embedding
from keras.utils import np_utils

# For model selection & evaluation
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split

# For text cleaning & preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize, word_tokenize, pos_tag
from bs4 import BeautifulSoup

# To generate word2vec embeddings
from gensim.models import word2vec, Word2Vec
from gensim.models.keyedvectors import KeyedVectors

