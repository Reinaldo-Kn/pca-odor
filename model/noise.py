import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Carregando o dataset
dataset_patch = os.path.join('..', 'dataset_new')
df_atual_df = os.path.join(dataset_patch, 'concat_dirty.csv')
df = pd.read_csv(df_atual_df)

# Separando a coluna TIME para preservá-la
time_column = df.iloc[:, 0].copy()

# Separando as características e os rótulos
X = df.iloc[:, 1:-1].copy()  # Características (ignorando TIME e Class)
y = df.iloc[:, -1].copy()    # Coluna de rótulos (Class)

# Codificando os rótulos de 'Class' para SMOTE
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 1. Imprimindo a quantidade de dados antes da adição de ruído
print("Quantidade de dados antes da adição de ruído:", X.shape[0])

# 2. Adicionando ruído ao dataset
noise_factor = 0.05  # Ajuste este valor conforme necessário
X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

# 3. Combinando o dataset original com o dataset com ruído
X_combined = np.vstack((X, X_noisy))
y_combined = np.hstack((y_encoded, y_encoded))

# 4. Imprimindo a quantidade de dados após a adição de ruído
print("Quantidade de dados após a adição de ruído:", X_combined.shape[0])

# 5. Aplicando SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

# 6. Convertendo de volta para um DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)

# Convertendo as labels numéricas de volta para os rótulos originais
y_resampled_labels = label_encoder.inverse_transform(y_resampled)

# 7. Adicionando as labels convertidas e a coluna TIME ao DataFrame
df_resampled['label'] = y_resampled_labels
df_resampled['TIME'] = np.resize(time_column.values, df_resampled.shape[0])

# 8. Reorganizando as colunas para garantir que TIME seja a primeira e label a última
columns_order = ['TIME'] + [col for col in df_resampled.columns if col != 'TIME' and col != 'label'] + ['label']
df_resampled = df_resampled[columns_order]

# 9. Salvando o novo dataset
df_resampled.to_csv(os.path.join(dataset_patch, 'concat_resampled.csv'), index=False)

# 10. Imprimindo a quantidade de dados após a aplicação do SMOTE
print("Quantidade de dados após a aplicação do SMOTE:", df_resampled.shape[0])
