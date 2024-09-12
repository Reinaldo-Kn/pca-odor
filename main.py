import numpy as np
import pandas as pd

df = pd.read_csv('./dataset/Coffee2.csv')
#if the column is unnamed, it will be removed
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print(df.head())