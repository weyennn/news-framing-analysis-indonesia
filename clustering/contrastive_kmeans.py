import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def cluster_documents_kmeans(docs, n_clusters=2, max_features=1000):
    """
    Cluster dokumen menggunakan KMeans berdasarkan representasi TF-IDF.
    :param docs: List of raw text documents
    :param n_clusters: Jumlah cluster (default 2)
    :param max_features: Jumlah maksimal fitur TF-IDF
    :return: labels, vectorizer, kmeans_model
    """
    cleaned_docs = [re.sub(r"[^\w\s]", "", doc.lower()) for doc in docs]
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(cleaned_docs)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)

    return labels, vectorizer, kmeans


def get_top_terms_per_cluster(kmeans_model, vectorizer, top_n=10):
    """
    Ambil top-N terms untuk setiap cluster berdasarkan centroid.
    :param kmeans_model: Trained KMeans object
    :param vectorizer: Fitted TfidfVectorizer
    :param top_n: Jumlah kata topik yang ingin diambil
    :return: Dict cluster -> list of top words
    """
    centroids = kmeans_model.cluster_centers_
    terms = vectorizer.get_feature_names_out()

    top_terms = {}
    for i, center in enumerate(centroids):
        top_idx = center.argsort()[::-1][:top_n]
        top_terms[i] = [terms[j] for j in top_idx]

    return top_terms
