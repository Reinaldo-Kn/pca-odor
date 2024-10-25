import numpy as np
import pandas as pd
import os


dataset_patch = os.path.join('..','dataset_new')
df1_atual = os.path.join(dataset_patch,'4-Perfil_Racao.csv')
df1 = pd.read_csv(df1_atual)
df2_atual = os.path.join(dataset_patch,'6-Perfil_Racao.csv')
df2 = pd.read_csv(df2_atual)
df3_atual = os.path.join(dataset_patch,'7-Perfil_Compostagem(meio).csv')
df3 = pd.read_csv(df3_atual)
df4_atual = os.path.join(dataset_patch,'10-Perfil_Compostagem(meio).csv')
df4 = pd.read_csv(df4_atual)

df1['Class'] = 'Racao'
df2['Class'] = 'Racao'
df3['Class'] = 'Compostagem(meio)'
df4['Class'] = 'Compostagem(meio)'

df = pd.concat([df1, df2, df3,df4])
df_concat = os.path.join(dataset_patch,'concat_dirty.csv')
df.to_csv(df_concat, index=False)
