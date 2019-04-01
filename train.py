from sklearn.externals import joblib

# Natural Language Processing

def filt(q):
    que = re.sub('[^a-zA-Z]', ' ', q)
    que = que.lower()
    que = que.split()
    ps = PorterStemmer()
    que = [ps.stem(word) for word in que]
    que = ' '.join(que)
    return que
    

# Importing the libraries
import re
import numpy as np

import pandas as pd


# Importing the dataset
dataset = pd.read_csv('chatbot.tsv', delimiter = '-')
ans= pd.read_csv('chat_answers.tsv', delimiter = '-')

joblib.dump(dataset, 'dataset.pkl')
joblib.dump(ans, 'ans.pkl')


# Cleaning the texts
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, len(dataset)):
    que=filt(dataset['que'][i])
    corpus.append(que)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
joblib.dump(cv, 'cv.pkl')
joblib.dump(X, 'X.pkl')


# Fitting models to the Training set
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(max_depth=5,n_estimators=300)
RFC.fit(X, y)
joblib.dump(RFC, 'RFC.pkl')



from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=5,weights='distance')
KNN.fit(X,y)
joblib.dump(KNN, 'KNN.pkl')



from sklearn.naive_bayes import GaussianNB
NaiveBayes=GaussianNB()
NaiveBayes.fit(X,y)
joblib.dump(NaiveBayes, 'NaiveBayes.pkl')



from xgboost.sklearn import XGBClassifier
XGB=XGBClassifier(n_estimators=300,objective='multi:softmax',max_depth=5)
XGB.fit(X,y)
joblib.dump(XGB,'XGB.pkl')


from xgboost.sklearn import XGBClassifier
XGB_Proba=XGBClassifier(n_estimators=300,objective='multi:softprob',max_depth=5)
XGB_Proba.fit(X,y)
joblib.dump(XGB, 'XGB_Proba.pkl')


from sklearn.ensemble import GradientBoostingClassifier
GBM=GradientBoostingClassifier(n_estimators=200,max_depth=5)
GBM.fit(X,y)
joblib.dump(GBM, 'GBM.pkl')

