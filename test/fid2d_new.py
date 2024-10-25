import plotly.express as px
from sklearn.decomposition import PCA
import os
import pandas as pd

dataset_patch = os.path.join('..','dataset_new')
df_atual_df = os.path.join(dataset_patch,'concat.csv')
df_main = pd.read_csv(df_atual_df)

#separe column first and the last column from the rest
features = df_main.columns[1:-1]
df_features = df_main.iloc[:, 1:-1].copy()
df_labels = df_main.iloc[:,-1].copy()

df_odor = pd.concat([df_features, df_labels], axis=1) # get rid of the time column


pca = PCA(n_components=2)
principalComponents_odor = pca.fit_transform(df_features)


df_pca = pd.DataFrame(data=principalComponents_odor, columns=['PC1', 'PC2'])
df_pca['label'] = df_labels.values  # Adiciona as labels

# Exportar o DataFrame para um arquivo CSV
csv_path = os.path.join('..', 'dataset_new', 'pca_transformed.csv')
df_pca.to_csv(csv_path, index=False)

total_var = pca.explained_variance_ratio_.sum() * 100  # Multiplica por 100 para obter a porcentagem

# Plotar o gráfico de dispersão utilizando as componentes principais
fig = px.scatter(
    x=principalComponents_odor[:, 0],  # Primeira componente
    y=principalComponents_odor[:, 1],  # Segunda componente
    color=df_labels,  # As labels para colorir os pontos
    title=f'PCA Variância total: {total_var:.2f}%'  # Título com a variância explicada
)
fig.show()