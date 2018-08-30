from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np


def train_and_score_by_model(model, Xtrain, Ytrain, Xtest, Ytest):
    _model = model()
    _model.fit(Xtrain, Ytrain)
    print("Classification rate for {}:{}".format(str(model), _model.score(Xtest, Ytest)))


data = pd.read_csv('../resources/spambase/spambase.data').values
# shuffle does in place shuffle of the data
# purpose of randomly splitting the data into a train and test sets, and want to be different every time
np.random.shuffle(data)

# first 48 columns word_freq_WORD
X = data[:, :48]
# last column indicates spam(1) or not(0)
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

train_and_score_by_model(MultinomialNB, Xtrain, Ytrain, Xtest, Ytest)
train_and_score_by_model(AdaBoostClassifier, Xtrain, Ytrain, Xtest, Ytest)
