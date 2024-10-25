import numpy as np
import pandas as pd
import os
dataset_patch = os.path.join('..','dataset_new')
atual_df = os.path.join(dataset_patch,'5-Perfil_Racao.csv')
df = pd.read_csv(atual_df)

# create a new column 
df['Class'] = 'Compostagem(meio)'


#delete the unused columns
df = df.drop(['D20','D21','D22','D23','D24','D25'], axis=1)

#rename the columns
df = df.rename(columns={
    'D0': 'VOC (ppb)', 
    'D1': 'VOC (ug m-3)', 
    'D2': 'VOC (Temperatura)', 
    'D3': 'VOC (Umidade)',
    'D4': 'H2S (ppb)', 
    'D5': 'H2S (ug m-3)', 
    'D6': 'H2S (Temperatura)', 
    'D7': 'H2S (Umidade)',
    'D8': 'SO2 (ppb)', 
    'D9': 'SO2 (ug m-3)', 
    'D10': 'SO2 (Temperatura)', 
    'D11': 'SO2 (Umidade)',
    'D12': 'NH3 (ppb)', 
    'D13': 'NH3 (ug m-3)', 
    'D14': 'NH3 (Temperatura)', 
    'D15': 'NH3 (Umidade)',
    'D16': 'CH3SH (ppb)', 
    'D17': 'CH3SH (ug m-3)', 
    'D18': 'CH3SH (Temperatura)', 
    'D19': 'CH3SH (Umidade)',
    'D26': 'BME280 - TEMPERATURA', 
    'D27': 'BME280 - PRESSAO', 
    'D28': 'BME280 - UMIDADE'
})


new_df = os.path.join(dataset_patch,'5-Perfil_Racao_adjust.csv')
#save as new csv
df.to_csv(new_df, index=False)