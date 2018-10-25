# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os 
from utils import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from ggplot import *

from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import logger
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
#def lassoSelection(X,y,)

def lassoSelection(X_train, y_train, n):
	'''
	Lasso feature selection.  Select n features. 
	'''
	#lasso feature selection
	#print (X_train)
	clf = LassoCV()
	sfm = SelectFromModel(clf, threshold=0)
	sfm.fit(X_train, y_train)
	X_transform = sfm.transform(X_train)
	n_features = X_transform.shape[1]
	
	#print(n_features)
	while n_features > n:
		sfm.threshold += 0.01
		X_transform = sfm.transform(X_train)
		n_features = X_transform.shape[1]
	features = [index for index,value in enumerate(sfm.get_support()) if value == True  ]
	logger.info("selected features are {}".format(features))
	return features


def specificity_score(y_true, y_predict):
	'''
	true_negative rate
	'''
	true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0 ])
	real_negative = len(y_true) - sum(y_true)
	return true_negative / real_negative 

def model_fit_predict(X_train,X_test,y_train,y_test):

	np.random.seed(2018)
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.svm import SVC
	from sklearn.neural_network import MLPClassifier
	from sklearn.metrics import precision_score
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import f1_score
	from sklearn.metrics import recall_score
	models = {
		# 'LogisticRegression': LogisticRegression(),
		# 'ExtraTreesClassifier': ExtraTreesClassifier(),
		# 'RandomForestClassifier': RandomForestClassifier(),
  #   	'AdaBoostClassifier': AdaBoostClassifier(),
  #   	'GradientBoostingClassifier': GradientBoostingClassifier(),
  #   	'SVC': SVC(),
    	'MLPClassifier': MLPClassifier()
	}
	tuned_parameters = {
		# 'LogisticRegression':{'C': [1, 10]},
		# 'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
		# 'RandomForestClassifier': { 'n_estimators': [16, 32] },
  #   	'AdaBoostClassifier': { 'n_estimators': [16, 32] },
  #   	'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
  #   	'SVC': {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    	'MLPClassifier' : {'alpha': [1]}
	}
	scores= {}
	for key in models:
		clf = GridSearchCV(models[key], tuned_parameters[key], scoring=None,  refit=True, cv=10)
		clf.fit(X_train,y_train)
		# if key == 'MLPClassifier':
			# classifier = OneVsRestClassifier(models[key])
			# y_score = classifier.fit(X_train, y_train)#.decision_function(X_test)
		y_test_predict = clf.predict(X_test)
		precision = precision_score(y_test, y_test_predict, average = 'micro')
		accuracy = accuracy_score(y_test, y_test_predict)
		f1 = f1_score(y_test, y_test_predict, average = 'micro')
		recall = recall_score(y_test, y_test_predict, average = 'micro')
		specificity = specificity_score(y_test, y_test_predict)
		scores[key] = [precision,accuracy,f1,recall,specificity]
	#print(scores)
	return scores, y_test_predict



def draw(scores):
	'''
	draw scores.
	'''

	import matplotlib.pyplot as plt
	logger.info("scores are {}".format(scores))
	ax = plt.subplot(111)
	precisions = []
	accuracies =[]
	f1_scores = []
	recalls = []
	categories = []
	specificities = []
	N = len(scores)
	ind = np.arange(N)  # set the x locations for the groups
	width = 0.1        # the width of the bars
	for key in scores:
		categories.append(key)
		precisions.append(scores[key][0])
		accuracies.append(scores[key][1])
		f1_scores.append(scores[key][2])
		recalls.append(scores[key][3])
		specificities.append(scores[key][4])

	precision_bar = ax.bar(ind, precisions,width=0.1,color='b',align='center')
	accuracy_bar = ax.bar(ind+1*width, accuracies,width=0.1,color='g',align='center')
	f1_bar = ax.bar(ind+2*width, f1_scores,width=0.1,color='r',align='center')
	recall_bar = ax.bar(ind+3*width, recalls,width=0.1,color='y',align='center')
	specificity_bar = ax.bar(ind+4*width,specificities,width=0.1,color='purple',align='center')

	print(categories)
	ax.set_xticks(np.arange(N))
	ax.set_xticklabels(categories)
	ax.legend((precision_bar[0], accuracy_bar[0],f1_bar[0],recall_bar[0],specificity_bar[0]), ('precision', 'accuracy','f1','sensitivity','specificity'))
	ax.grid()
	plt.show()

def pca_calculation(X, Y):
	
	pca = PCA(n_components = 2)
	pca_result = pca.fit_transform(X)
	target = pd.DataFrame(data = Y, columns = ['label'])
	principaldf = pd.DataFrame(data = pca_result, columns = ['pca-one', 'pca-two'])
	finaldf = pd.concat([principaldf, target], axis = 1)

	chart = ggplot( finaldf.loc[:,:], aes(x='pca-one', y='pca-two',color='label') ) \
	        + geom_point(size=75,alpha=0.8) \
	        + ggtitle("First and Second Principal Components colored by digit")
	print(chart)

def tsne(X, Y):
	pca_50 = PCA(n_components=50)
	pca_result_50 = pca_50.fit_transform(X)
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_pca_results = tsne.fit_transform(pca_result_50)
	df_tsne = pd.DataFrame(data = tsne_pca_results, columns = ['x-tsne-pca', 'y-tsne-pca'])
	target = pd.DataFrame(data = Y, columns = ['label'])
	finaldf = pd.concat([df_tsne, target], axis = 1)
	
	chart1 = ggplot( finaldf, aes(x='x-tsne-pca', y='y-tsne-pca', color='label') ) \
	        + geom_point(size=70,alpha=0.1) \
	        + ggtitle("tSNE dimensions colored by Digit (PCA)")
	print(chart1)

def draw_roc(scores, Y):
	y_score = scores
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	# for i in range(2):
	fpr, tpr, _ = roc_curve(Y, y_score)
	roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
	# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()


if __name__ == '__main__':


	data_dir ="/home/sukruth/Lab10/"

	data_file = data_dir + "miRNA_matrix.csv"

	df = pd.read_csv(data_file)
	# print(df)
	y_data = df.pop('label').values

	df.pop('file_id')

	columns =df.columns
	#print (columns)
	X_data = df.values
	
	'''for i in range(0,X_data.shape[0]):
		if (y_data[i] == '0') or (y_data[i] == '1'):
			continue
		else:
			print(y_data[i])
			np.delete(y_data, i, 0)
			np.delete(X_data, i, 0)
	print(y_data)'''
	# split the data to train and test set
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
	

	#standardize the data.
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	print(type(y_train))

	# check the distribution of tumor and normal sampels in traing and test data set.
	# print("Percentage of tumor cases in training set is " + str(sum(y_train)/len(y_train)))
	# print("Percentage of tumor cases in test set is " + str(sum(y_test)/len(y_test)))
	logger.info("Percentage of tumor cases in training set is {}".format(sum(y_train)/len(y_train)))
	logger.info("Percentage of tumor cases in test set is {}".format(sum(y_test)/len(y_test)))
	
	n = 7
	feaures_columns = lassoSelection(X_train, y_train, n)



	scores, y_test_predict = model_fit_predict(X_train[:,feaures_columns],X_test[:,feaures_columns],y_train,y_test)
	pca_calculation(X_train, y_train)
	tsne(X_train, y_train)
	draw(scores)
	draw_roc(y_test_predict, y_test)
	#lasso cross validation
	# lassoreg = Lasso(random_state=0)
	# alphas = np.logspace(-4, -0.5, 30)
	# tuned_parameters = [{'alpha': alphas}]
	# n_fold = 10
	# clf = GridSearchCV(lassoreg,tuned_parameters,cv=10, refit = False)
	# clf.fit(X_train,y_train)




 




