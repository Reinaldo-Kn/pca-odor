import numpy as np
import pandas as pd
import os
dataset_patch = os.path.join('..','dataset_new')
atual_df = os.path.join(dataset_patch,'6-Perfil_Racao.csv')
df = pd.read_csv(atual_df)

print(df.columns)