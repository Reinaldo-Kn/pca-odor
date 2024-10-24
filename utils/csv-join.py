import numpy as np
import pandas as pd
import os
#read 3 csv files 
dataset_patch = os.path.join('..','dataset_new')
df_1_atual_df = os.path.join(dataset_patch,'4-Perfil_Racao_adjust.csv')
df_1 = pd.read_csv(df_1_atual_df)
df_2_atual_df = os.path.join(dataset_patch,'6-Perfil_Racao_adjust.csv')
df_2 = pd.read_csv(df_2_atual_df)
df_3_atual_df = os.path.join(dataset_patch,'7-Perfil_Compostagem(meio)_adjust.csv')
df_3 = pd.read_csv(df_3_atual_df)
df_4_atual_df = os.path.join(dataset_patch,'10-Perfil_Compostagem(meio)_adjust.csv')
df_4 = pd.read_csv(df_4_atual_df)


#join the 3 csv files in one
df = pd.concat([df_1, df_2, df_3,df_4])
df_concat = os.path.join(dataset_patch,'concat.csv')
df.to_csv(df_concat, index=False)