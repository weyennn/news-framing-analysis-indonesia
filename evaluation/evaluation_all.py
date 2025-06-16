import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import os

# === Load Berita ===
with open('data/splitted/berita_1_10000.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

doc_texts = [doc["content"] for doc in documents]
tokenized_corpus = [doc.lower().split() for doc in doc_texts]
bm25okapi = BM25Okapi(tokenized_corpus)

# === Load Ground Truth ===
with open('data/ground_truth_multiquery_10000.json', 'r', encoding='utf-8') as f:
    query_gt = json.load(f)

queries = list(query_gt.keys())
cutoffs = [10, 20, 30, 40, 50]
results = []
first_ranks = []

# === Fungsi IR Metrics ===
def average_precision(relevant, retrieved):
    score = 0.0
    num_hits = 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not relevant:
        return 0.0
    return score / min(len(relevant), len(retrieved))

def first_relevant_rank(relevant, retrieved):
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return i + 1
    return None

# === Evaluasi Setiap Query ===
for query in tqdm(queries, desc="Evaluating Queries"):
    tokenized_query = query.lower().split()
    scores = bm25okapi.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1][:500]

    relevant = set(query_gt[query])
    retrieved = list(ranked_indices)

    metrics = {"query": query}
    for k in cutoffs:
        top_k = retrieved[:k]
        rel_in_topk = [1 if doc_id in relevant else 0 for doc_id in top_k]
        metrics[f"precision@{k}"] = sum(rel_in_topk) / k
        metrics[f"recall@{k}"] = sum(rel_in_topk) / len(relevant) if relevant else 0

    metrics["average_precision"] = average_precision(relevant, retrieved)
    metrics["first_rank"] = first_relevant_rank(relevant, retrieved) or 0
    results.append(metrics)
    first_ranks.append(metrics["first_rank"])

# === Simpan Hasil Evaluasi ke TXT ===
df = pd.DataFrame(results)
os.makedirs("output", exist_ok=True)
txt_lines = []

for k in cutoffs:
    txt_lines.append("=" * 70)
    txt_lines.append(f"Evaluasi IR Metrics @K={k}")
    txt_lines.append("=" * 70)
    txt_lines.append(f"{'Query':<6} {'Precision':>10} {'Recall':>10}")
    txt_lines.append("-" * 70)
    for idx, row in df.iterrows():
        txt_lines.append(f"Q{idx+1:<5} {row[f'precision@{k}']:.4f}{row[f'recall@{k}']:>10.4f}")

    # Tambahkan rata-rata metrik per @K
    avg_prec = df[f'precision@{k}'].mean()
    avg_rec = df[f'recall@{k}'].mean()
    avg_ap = df['average_precision'].mean()
    txt_lines.append("-" * 70)
    txt_lines.append(f"{'Mean Precision':<18}: {avg_prec:.4f}")
    txt_lines.append(f"{'Mean Recall':<18}: {avg_rec:.4f}")
    txt_lines.append(f"{'Mean MAP':<18}: {avg_ap:.4f}")
    txt_lines.append("")

# === Ringkasan
txt_lines.append("=" * 70)
txt_lines.append("RINGKASAN METRIK")
txt_lines.append("=" * 70)
txt_lines.append(f"{'Total Query Evaluated'        :<30}: {len(queries):>6}")
txt_lines.append(f"{'Mean Average Precision (MAP)' :<30}: {np.mean(df['average_precision']):>6.4f}")
txt_lines.append(f"{'Mean First Relevant Rank'     :<30}: {np.mean([r for r in first_ranks if r > 0]):>6.2f}")
txt_lines.append("=" * 70)

with open("output/ir_metrics_detailed.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(txt_lines))

# === Visualisasi: Bar Chart per K dengan Legend + Nilai
labels = [f"Q{i+1}" for i in range(len(queries))]
os.makedirs("output/charts", exist_ok=True)

for k in cutoffs:
    plt.figure(figsize=(12, 6))

    x = np.arange(len(labels))
    bar_width = 0.25

    precision = df[f"precision@{k}"]
    recall = df[f"recall@{k}"]
    ap = df["average_precision"]

    bars1 = plt.bar(x - bar_width, precision, width=bar_width, label=f'Precision@{k} (avg={precision.mean():.2f})', color='cornflowerblue')
    bars2 = plt.bar(x, recall, width=bar_width, label=f'Recall@{k} (avg={recall.mean():.2f})', color='lightgreen')
    bars3 = plt.bar(x + bar_width, ap, width=bar_width, label=f'MAP (avg={ap.mean():.2f})', color='salmon')

    # Anotasi nilai precision di atas bar
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                     f"{height:.2f}", ha='center', va='bottom', fontsize=9)

    plt.xticks(ticks=x, labels=labels)
    plt.xlabel("Query")
    plt.ylabel("Score")
    plt.title(f"IR Metrics per Query @K={k}")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/charts/ir_metrics_at_{k}_with_avg_legend.png")
    plt.close()

print("Semua file disimpan ke folder 'output/' dan 'output/charts/'")
