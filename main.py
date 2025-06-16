import json
import numpy as np
from rank_bm25 import BM25Okapi
from clustering.contrastive_kmeans import cluster_documents_kmeans, get_top_terms_per_cluster
from llm.call_gpt import get_response_from_gpt
from llm.prompt_contrastive import build_contrastive_prompt
from utils.config_loader import load_config
from evaluation.ir_metrics import (
    precision_at_k, recall_at_k, average_precision
)

# === LOAD CONFIG ===
config = load_config()
TOP_K = config["retriever"]["top_k"]  # Misalnya 100
MODEL_NAME = config["llm"]["model"]
API_KEY = config["openrouter"]["api_key"]

# === LOAD DATA ===
with open("data/splitted/berita_1_10000.json", encoding="utf-8") as f:
    data = json.load(f)

documents = [item["content"] for item in data]
doc_ids = [item["id"] for item in data]
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# === LOAD RELEVANCE JUDGMENTS ===
with open("data/ground_truth_multiquery_10000.json", encoding="utf-8") as f:
    raw_judgments = json.load(f)

relevance_judgments = {
    q.strip().lower(): [int(i) for i in ids]
    for q, ids in raw_judgments.items()
}

# === QUERY TUNGGAL ===
query = "Apa alasan di balik penundaan pemilu dan wacana tiga periode?"
normalized_query = query.strip().lower()
tokenized_query = normalized_query.split()

# === RETRIEVE TOP-K INDEXES ===
scores = bm25.get_scores(tokenized_query)
ranked_indices = np.argsort(scores)[::-1][:TOP_K]

retrieved_doc_ids = [doc_ids[i] for i in ranked_indices]
retrieved_docs = [documents[i] for i in ranked_indices]

print(f"\nDitemukan {len(retrieved_docs)} dokumen dari BM25.")
print("Index dokumen hasil retrieve:", retrieved_doc_ids[:10])

# === EVALUASI IR ===
relevant_ids = relevance_judgments.get(normalized_query, [])

if relevant_ids:
    print("Ground truth tersedia untuk query ini.")
    print(f"→ Jumlah dokumen relevan: {len(relevant_ids)}")
    print(f"→ Relevan yang ditemukan dalam retrieved: {set(retrieved_doc_ids) & set(relevant_ids)}")

    prec = precision_at_k(retrieved_doc_ids, relevant_ids, k=10)
    rec = recall_at_k(retrieved_doc_ids, relevant_ids, k=10)
    ap = average_precision(retrieved_doc_ids, relevant_ids)

    print("\nEvaluasi IR:")
    print(f"Precision@10      : {prec:.5f}")
    print(f"Recall@10         : {rec:.5f}")
    print(f"Average Precision : {ap:.5f}")
else:
    print("Query ini tidak memiliki relevance judgment.")
    exit()

# === CLUSTERING ===
labels, vectorizer, model = cluster_documents_kmeans(retrieved_docs, n_clusters=2)
top_terms = get_top_terms_per_cluster(model, vectorizer, top_n=7)

clusters = {i: [] for i in set(labels)}
for idx, label in enumerate(labels):
    clusters[label].append(retrieved_docs[idx])

# === BUILD PROMPT & CALL GPT ===
prompt = build_contrastive_prompt(query, clusters, top_terms)
print("\nPrompt dikirim ke LLM...")
response = get_response_from_gpt(prompt, api_key=API_KEY, model=MODEL_NAME)

print("\nJawaban dari LLM:")
print(response)
