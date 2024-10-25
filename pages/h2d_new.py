import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dash import dcc, html
import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import os

dash.register_page(__name__, path="/fig2dNEW", name="Novo PCA em 2D")

load_figure_template(["yeti","yeti_dark"])

dataset_patch = os.path.join('.','dataset_new')
df_atual_df = os.path.join(dataset_patch,'concat.csv')
df_main = pd.read_csv(df_atual_df)

#separe column first and the last column from the rest
features = df_main.columns[1:-1]
df_features = df_main.iloc[:, 1:-1].copy()
df_labels = df_main.iloc[:,-1].copy()

df_odor = pd.concat([df_features, df_labels], axis=1) # get rid of the time column


pca_odor = PCA(n_components=2)
principalComponents_odor = pca_odor.fit_transform(df_features)

# save pca analysis
# df_pca = pd.DataFrame(data=principalComponents_odor, columns=['PC1', 'PC2'])
# df_pca['label'] = df_labels.values  # Adiciona as labels

# # Exportar o DataFrame para um arquivo CSV
# csv_path = os.path.join('..', 'dataset_new', 'pca_transformed.csv')
# df_pca.to_csv(csv_path, index=False)

total_var = pca_odor.explained_variance_ratio_.sum() * 100  # Multiplica por 100 para obter a porcentagem

loadings = pca_odor.components_.T * np.sqrt(pca_odor.explained_variance_)



# Plotar o gráfico de dispersão utilizando as componentes principais
fig = px.scatter(
    x=principalComponents_odor[:, 0],  # Primeira componente
    y=principalComponents_odor[:, 1],  # Segunda componente
    color=df_labels,  # As labels para colorir os pontos
    title=f'PCA Variância total: {total_var:.2f}%'  # Título com a variância explicada
)


for i, feature in enumerate(features):
    x_start = loadings[i, 0]
    y_start = loadings[i, 1]
    
    # Adicionando um deslocamento para separar as setas
  # Desloca as setas um pouco em y, variando com o índice
    fig.add_annotation(
        ax=0, ay=0,
        axref='x', ayref='y',
        x=x_start ,
        y=y_start + 0.8 * i,
        showarrow=True,
        arrowsize=5,  # Aumentar o tamanho das setas
        arrowwidth=1,  # Aumentar a largura da seta
        arrowhead=2,
        xanchor='right',
        yanchor='top',
    )
    fig.add_annotation(
        x=x_start ,
        y=y_start + 0.8 * i,
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
        yshift=5,
        textangle=0  # Rotacionar o texto se necessário
    )
layout = html.Div([
    html.H1("NOVA Análise PCA em 2D"),
    dcc.Graph(figure=fig)  # Insere o gráfico no layout
])
