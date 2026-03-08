from sklearn.decomposition import PCA
import umap


def compute_pca(features, n_components=2):
    return PCA(n_components=n_components, random_state=42).fit_transform(features)


def compute_umap(features, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(features)
