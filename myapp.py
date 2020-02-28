import pandas as pd

df = pd.read_csv('./dc_modelisation.csv', sep=';')
clean_df = df.loc[df["NB"] != 0 , :  ]
list_column = []

for x in clean_df:
    list_column.append(x)

class App:  
    def __init__(self,X, Y) :
        self.X = X
        self.Y = Y
        
    def model_Dorian(self, X, Y) :
       logreg = LogisticRegression()
       logreg.fit(self.X,self.Y)
       print(logreg.predict(self.X))
       print(logreg.score(self.X,self.Y))
       X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2)   
       
    def model_Lothaire(self) :
        print('Modele de Lothaire')
        
    def model_Matthieu() :
        print('Hello world !')
        
    model_Dorian(self, clean_df[['NB', 'a2']], clean_df['y'])