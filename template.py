#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
from pandas import Series, DataFrame

def load_dataset(dataset_path):
	data_df = pd.read_csv(sys.argv[1])
def dataset_stat(dataset_df):
	n_feats = len(data_df)
	n_class0 = len(data_df.loc[data_df['target'] == 0])
	n_class1 = len(data_df.loc[data_df['target'] == 1])

def split_dataset(dataset_df, testset_size):
	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(data_df.data, data_df.target, test_size=sys.argv[2])

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import precision_score
	from sklearn.metrics import recall_score

	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)

	acc = accuracy_score(y_test, dt_cls.predict(x_test))
	prec = precision_score(y_test, dt_cls.predict(x_test))
	recall = recall_score(y_test, dt_cls.predict(x_test))

def random_forest_train_test(x_train, x_test, y_train, y_test):
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import precision_score
	from sklearn.metrics import recall_score

	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)

	acc = accuracy_score(y_test, rf_cls.predict(x_test))
	prec = precision_score(y_test, rf_cls.predict(x_test))
	recall = recall_score(y_test, rf_cls.predict(x_test))

def svm_train_test(x_train, x_test, y_train, y_test):
	from sklearn.svm import SVC
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import precision_score
	from sklearn.metrics import recall_score

	svm_cls = SVC()
	svm_cls.fit(x_train, y_train)

	acc = accuracy_score(y_test, svm_cls.predict(x_test))
	prec = precision_score(y_test, svm_cls.predict(x_test))
	recall = recall_score(y_test, svm_cls.predict(x_test))

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
