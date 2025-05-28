import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# === 1. Importação dos dados ===
df = pd.DataFrame(pd.read_pickle('x_scaled.pickle'))  # Substitua pelo caminho correto se necessário
print("Dados carregados. Dimensões:", df.shape)

# === 2. Gráfico de K-distância para definir 'eps' ===
k = 10
nn = NearestNeighbors(n_neighbors=k)
nn_fit = nn.fit(df)
distances, _ = nn_fit.kneighbors(df)

# Ordena as distâncias do k-ésimo vizinho
distances = np.sort(distances[:, k - 1])
plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.title(f'Gráfico de K-distância (k={k})')
plt.xlabel('Pontos ordenados')
plt.ylabel(f'{k}-ésima menor distância')
plt.grid(True)
plt.show()

# === 3. Aplicação do DBSCAN ===
# Com base no gráfico acima, ajuste eps se necessário
db = DBSCAN(eps=0.7, min_samples=10, n_jobs=-1).fit(df)
labels = db.labels_

# === 4. PCA para visualização ===
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df)

# Separação de outliers e clusters
outliers = labels == -1
clusters_points = labels != -1

# === 5. Visualização dos clusters ===
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[clusters_points, 0], pca_result[clusters_points, 1], c='blue', label='Clusters')
plt.scatter(pca_result[outliers, 0], pca_result[outliers, 1], c='red', marker='x', label='Outliers')
plt.title("Visualização dos Clusters via DBSCAN + PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid(True)
plt.show()

# === 6. Métricas ===
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
num_outliers = np.sum(outliers)

print("\n=== RESULTADOS ===")
print("Estimativa de número de clusters:", num_clusters)
print("Número de outliers detectados:", num_outliers)

# Silhouette Score (se possível)
if num_clusters > 1:
    sil_score = silhouette_score(df, labels)
    print(f"Silhouette Score: {sil_score:.2f}")
else:
    print("Silhouette Score não calculado (menos de 2 clusters).")

    
