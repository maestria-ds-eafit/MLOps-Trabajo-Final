from sklearn.neighbors import KNeighborsClassifier

knn_config = {
    "n_neighbors": 8,
    "weights": "distance",
}
knn_model = KNeighborsClassifier(**knn_config)
