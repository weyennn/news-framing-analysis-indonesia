from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

def evaluate_clustering(docs, labels):
    """
    Evaluasi clustering menggunakan silhouette score berbasis TF-IDF.
    """
    if len(set(labels)) < 2:
        return -1  # Tidak bisa dihitung jika hanya ada 1 cluster
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    return silhouette_score(X, labels)
