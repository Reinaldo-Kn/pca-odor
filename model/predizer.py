import pandas as pd
from sklearn.decomposition import PCA
import joblib , os


dataset_patch = os.path.join('..','dataset_new')
df_atual_df = os.path.join(dataset_patch,'5-Perfil_Racao.csv')
# df_pca_original = os.path.join(dataset_patch,'pca_transformed.csv')
df= pd.read_csv(df_atual_df)

#preprocessamento
df = df.drop(['TIME'], axis=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(['D20','D21','D22','D23','D24','D25'], axis=1)
df = df.dropna()



# Preveja as classes
model = joblib.load('modelo_classificacao_dirty.pkl')
y_pred = model.predict(df)
# printa a classe predominante
print(pd.Series(y_pred).value_counts())