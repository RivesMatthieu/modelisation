import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('./dc_modelisation.csv', sep=';')
clean_df = df.loc[df["NB"] != 0 , :  ]
list_column = []

for x in clean_df:
    list_column.append(x)

class App:  
    def __init__(self,X, Y, Z) :
        self.X = X
        self.Y = Y
        self.Z = Z 

    def model_logisticRegression(X, Y) :
       logreg = LogisticRegression()
       logreg.fit(X,Y)
       print(logreg.score(X, Y))
       X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)   
       
    def model_randomForest() :
        print('Hello worl !')
        
    def model_SVM() :
        print('Hello world !')
        
    model_logisticRegression(clean_df[['NB', 'a2']], clean_df['y'])