import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('./dataset/All.csv')
#if the column is unnamed, it will be removed
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#padronizando os dados
features = df.columns[4:-1]
features_data = df[features]

features_label = df.iloc[:,-1]
odor_dataset = pd.concat([features_data, features_label], axis=1)

x = odor_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)
print(np.mean(x), np.std(x))

#normalizando os dados
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]   
normalizado = pd.DataFrame(x,columns=feat_cols)

#PCA
pca_odor = PCA(n_components=2)
principalComponents_odor = pca_odor.fit_transform(x)
principal_odor_Df = pd.DataFrame(data = principalComponents_odor, columns = ['principal component 1', 'principal component 2'])
print('Explained variation per principal component: {}'.format(pca_odor.explained_variance_ratio_))


#plotando o gráfico
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("PCA of Odor Dataset",fontsize=20)
targets = ['Coffee 1', 'Coffee 2', 'Fresh Garlic', 'Gasoline']
colors = ['brown', 'black', 'green', 'blue']
for target, color in zip(targets,colors):
    indicesToKeep = odor_dataset['Sampe/Class'] == target
    plt.scatter(principal_odor_Df.loc[indicesToKeep, 'principal component 1']
               , principal_odor_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
    
plt.legend(targets,prop={'size': 15})
plt.show()