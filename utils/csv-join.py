import numpy as np
import pandas as pd

#read 3 csv files 
df1 = pd.read_csv('./dataset/compostagem_adjust.csv')
df2 = pd.read_csv('./dataset/alho_adjust.csv')
df3 = pd.read_csv('./dataset/racao_adjust.csv')

#join the 3 csv files in one
df = pd.concat([df1, df2, df3])
df.to_csv('./dataset/All_adjust.csv', index=False)