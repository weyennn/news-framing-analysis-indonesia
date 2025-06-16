import math

def precision_at_k(retrieved, relevant, k):
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    true_positives = [doc for doc in retrieved_k if doc in relevant]
    return len(true_positives) / k

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    true_positives = [doc for doc in retrieved_k if doc in relevant]
    return len(true_positives) / len(relevant)

def average_precision(retrieved, relevant):
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant)

def mean_average_precision(all_retrieved, all_relevant):
    valid_queries = [q for q in all_retrieved if q in all_relevant]
    if not valid_queries:
        return 0.0
    total_ap = sum(
        average_precision(all_retrieved[q], all_relevant[q]) for q in valid_queries
    )
    return total_ap / len(valid_queries)

def reciprocal_rank(retrieved, relevant):
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    def dcg(docs):
        return sum([
            1 / math.log2(i + 2) if doc in relevant else 0
            for i, doc in enumerate(docs)
        ])
    
    retrieved_k = retrieved[:k]
    ideal_k = relevant[:k]  # asumsi relevansi biner, urutan tidak di-sort
    ideal_dcg = dcg(ideal_k)
    return dcg(retrieved_k) / ideal_dcg if ideal_dcg > 0 else 0.0
