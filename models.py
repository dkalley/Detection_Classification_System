import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.cluster import SpectralClustering
from sklearn.tree import DecisionTreeClassifier

######################################################################
# Base Models
######################################################################
def dt(X,Y):
    clf = DecisionTreeClassifier(max_depth=4,max_features='auto')
    return clf.fit(X,Y)

def knn(X,Y):
    clf = KNeighborsClassifier(n_neighbors=7, n_jobs=12)
    return clf.fit(X,Y)

def lr(X,Y):
    clf = LogisticRegression(random_state=0, solver='newton-cg', tol=1e-4, n_jobs=12, max_iter=1000000)
    return clf.fit(X,Y)

def nb(X,Y):
    # clf = GaussianNB()
    clf = ComplementNB()
    # clf = CategoricalNB()
    return clf.fit(X,Y)

def svm(X,Y):
    clf = SVC(kernel='poly',degree=8, gamma='auto')
    return clf.fit(X,Y)

######################################################################
# Anomoly Detectors
######################################################################
def IF(X):
    clf = IsolationForest(random_state=0)
    return clf

######################################################################
# Ensemble Learning
######################################################################
def gb(X, Y):
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate=0.05,
                                    max_depth=8, max_features='auto',
                                    min_samples_leaf=15, min_samples_split=10, 
                                    random_state=1)
    return gb.fit(X,Y)

def rf(X,Y):
    clf = RandomForestClassifier(n_estimators=20, max_depth=8, 
                                 random_state=1, n_jobs=12, verbose=0)
    return clf.fit(X,Y)

######################################################################
# Neural Networks
######################################################################
def mlp(X, Y):
    clf = MLPClassifier(hidden_layer_sizes=(100,80,50,25), max_iter=300000,activation='tanh',solver='lbfgs',random_state=1, alpha=0.0005)  
    return clf.fit(X,Y) 

######################################################################
# Clustering
######################################################################

from sklearn.cluster import KMeans
def kmeans(X,Y,n=2):
    # clf = DBSCAN(n_jobs=8)
    clf = KMeans(n_clusters=n)
    return clf.fit(X,Y)

def isofor(X,Y,n=2):
    clf = IsolationForest(random_state=0)
    return clf.fit(X,Y)
