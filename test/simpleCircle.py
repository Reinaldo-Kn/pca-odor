import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Leitura do dataset
df = pd.read_csv('./dataset/All.csv')

# Remover colunas não nomeadas, se existirem
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Padronizando os dados
features = df.columns[4:-1]
features_data = df[features]
features_label = df.iloc[:, -1]
odor_dataset = pd.concat([features_data, features_label], axis=1)

x = odor_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)

# PCA
pca_odor = PCA(n_components=2)
principalComponents_odor = pca_odor.fit_transform(x)

# Normalizando os componentes principais 

norm_principalComponents_odor = principalComponents_odor / np.max(np.abs(principalComponents_odor), axis=0)

# Colocando os componentes principais em um DataFrame
principal_odor_Df = pd.DataFrame(data=norm_principalComponents_odor, columns=['principal component 1', 'principal component 2'])

print('Explained variation per principal component: {}'.format(pca_odor.explained_variance_ratio_))

# Plotando o gráfico circular
plt.figure(figsize=(10, 10))
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Desenhando o círculo
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
plt.gca().add_artist(circle)

# Definindo os alvos e as cores
targets = ['Coffee 1', 'Coffee 2', 'Fresh Garlic', 'Gasoline']
colors = ['brown', 'black', 'green', 'blue']

# Plotando os dados
for target, color in zip(targets, colors):
    indicesToKeep = odor_dataset['Sampe/Class'] == target
    plt.scatter(principal_odor_Df.loc[indicesToKeep, 'principal component 1'],
                principal_odor_Df.loc[indicesToKeep, 'principal component 2'], 
                c=color, s=50,label = target)

# Adicionando as setas dos componentes principais
for i, (pc1, pc2) in enumerate(zip(pca_odor.components_[0], pca_odor.components_[1])):
    plt.arrow(0, 0, pc1, pc2, color='r', alpha=0.5, head_width=0.05)
    plt.text(pc1 * 1.15, pc2 * 1.15, features[i], color='r', ha='center', va='center')

# Labels e título
plt.xlabel(f"Component 1 ({pca_odor.explained_variance_ratio_[0] * 100:.2f}%)")
plt.ylabel(f"Component 2 ({pca_odor.explained_variance_ratio_[1] * 100:.2f}%)")
plt.title("PCA of Odor Dataset", fontsize=20)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend(prop={'size': 15})
plt.grid()
plt.show()
