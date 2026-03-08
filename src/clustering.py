from sklearn.cluster import KMeans


def cluster_states(features, k=4):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    return model.fit_predict(features)
