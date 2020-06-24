import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split

df = pd.read_csv('./dc_modelisation.csv', sep=';')
clean_df = df.loc[df["NB"] != 0 , :  ]
clean_df = clean_df.drop(columns=['CMD'])
list_column = []

for x in clean_df:
    if len(x) > 1:
        list_column.append(x)
        

class model_logisticRegression(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        logreg = LogisticRegression(penalty='none',solver='newton-cg')
        logreg.fit(self.X,self.Y)
        print(1  - logreg.score(self.X, self.Y))

model_logisticRegression(clean_df[list_column], clean_df['y'])


