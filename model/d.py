import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib , os
import matplotlib.pyplot as plt
import numpy as np  


dataset_patch = os.path.join('..','dataset_new')
df_atual_df = os.path.join(dataset_patch,'concat_resampled.csv')
df= pd.read_csv(df_atual_df)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# df = df.drop(['D20','D21','D22','D23','D24','D25'], axis=1)
df = df.dropna()
features = df.columns[1:-1]
df_features = df.iloc[:, 1:-1].copy()
df_labels = df.iloc[:,-1].copy()
print(df.columns)