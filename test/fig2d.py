import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

pca_odor = PCA(n_components=2)
principalComponents_odor = pca_odor.fit_transform(x)

loadings = pca_odor.components_.T * np.sqrt(pca_odor.explained_variance_)

total_var = pca_odor.explained_variance_ratio_.sum() * 100

fig = px.scatter(
    x=principalComponents_odor[:,0], 
    y=principalComponents_odor[:,1], 
    color=odor_dataset['Sampe/Class'],
    title=f'PCA Variância total: {total_var:.2f}%'
    )

for i, feature in enumerate(features):
    fig.add_annotation(
        ax=0, ay=0,
        axref='x', ayref='y',
        x=loadings[i, 0],
        y=loadings[i, 1],
        showarrow=True,
        arrowsize=2,
        arrowhead=2,
        xanchor='right',
        yanchor='top',
    )
    fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
        yshift=5,
    )

fig.show()