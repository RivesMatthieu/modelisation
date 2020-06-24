import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('./dc_modelisation.csv', sep=';')
clean_df = df.loc[df["NB"] != 0 , :  ]
clean_df = clean_df.drop(columns=['CMD'])
dataset = clean_df
dataset.shape
list_column = []

for x in clean_df:
    if len(x) > 1:
        list_column.append(x)
        

class model_logisticRegression(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        logreg = LogisticRegression(penalty='none',solver='newton-cg')
        logreg.fit(self.X,self.y)
        retour = round(float(1- logreg.score(self.X, self.y))*100 , 3)
        print('Modele de Régression Logistique :  il y aura',retour,'% de retour')

class model_SVM(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
        svclassifier = SVC(kernel='linear') #linéaire -> linear / Noyau Gaussien -> rbf / Noyau sigmoîde -> sigmoid
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        confusion_matrix(y_test,y_pred)
        print(classification_report(y_test,y_pred))

model_logisticRegression(clean_df[list_column], clean_df['y'])
model_SVM(clean_df[list_column], clean_df['y'])