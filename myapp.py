import pandas as pd

df = pd.read_csv('./dc_modelisation.csv', sep=';')
clean_df = df.loc[df["NB"] != 0 , :  ]
list_column = []

for x in clean_df:
    list_column.append(x)

