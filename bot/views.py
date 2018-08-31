from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib






# Natural Language Processing

def sub(a):
	n=''
	for i in a:
		if i.isalpha() or i.isspace():
			n=n+i
	return n

def filt(q):
    que = sub(q)
    que = que.lower()
    que = que.split()
    ps = PorterStemmer()
    que = [ps.stem(word) for word in que]
    que = ' '.join(que)
    return que
    


# Create your views here.
@csrf_exempt
def bot(request):
	que=request.POST.get('que',None)
	ans=avartanbot(que)
	print("adarsh")
	print(request.POST)
	print(que)
	print(ans)
	return JsonResponse({'ans':ans})







def avartanbot(que1):
	dataset = joblib.load('dataset.pkl')
	ans= joblib.load('ans.pkl')

	# Cleaning the texts
	corpus = []

	for i in range(0, len(dataset)):
	    que=filt(dataset['que'][i])
	    corpus.append(que)

	# Creating the Bag of Words model
	cv = joblib.load('cv.pkl')
	X = joblib.load('X.pkl')
	y = dataset.iloc[:, 1].values


	# Fitting models to the Training set
	RFC = joblib.load('RFC.pkl')


	KNN=joblib.load('KNN.pkl')


	NaiveBayes=joblib.load('NaiveBayes.pkl')

	XGB=joblib.load('XGB.pkl')

	XGB_Proba=joblib.load('XGB_Proba.pkl')

	GBM=joblib.load('GBM.pkl')


	while True:
	    final_pred=-1
	    user=que1
	    user=filt(user)
	    user=cv.transform([user]).toarray()
	    rfc_pred=RFC.predict(user)
	    knn_pred=KNN.predict(user)
	    naive_pred=NaiveBayes.predict(user)
	    xgb_pred=XGB.predict(user)
	    gbm_pred=GBM.predict(user)
	    pred=np.array([rfc_pred[0],knn_pred[0],naive_pred[0],xgb_pred[0],gbm_pred[0]])
	    pred_set=set(pred)
	    
	            
	            
	    rfc_prob=RFC.predict_proba(user)
	    #knn_prob=KNN.predict_proba(user)
	    #naive_prob=NaiveBayes.predict_proba(user)
	    xgb_prob=XGB.predict_proba(user)
	    gbm_prob=GBM.predict_proba(user)
	    proba=[rfc_prob,xgb_prob,gbm_prob]
	    
	    if rfc_prob[0][final_pred]<0.20 or xgb_prob[0][final_pred]<.40:
	        final_pred=-1
	    
	    for i in pred_set:
	        if np.sum(pred==i)>=3:
	            final_pred=i
	        else:
	            final_pred=-1

	    print(pred)
	    print(proba)
	    
	    if final_pred!=-1:
	        return(ans['ans'][final_pred])
	    else:
	        return('Sorry I cant undertand. Please try to be more clear')


