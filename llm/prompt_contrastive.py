def build_contrastive_prompt(query, clusters, top_terms, max_examples=2):
    prompt = f"Pertanyaan pengguna:\n{query}\n\nSistem mendeteksi dua kelompok narasi utama dari berita:\n"

    for cluster_id, docs in clusters.items():
        terms = ", ".join(top_terms.get(cluster_id, [])[:5])
        prompt += f"\n---\nCluster {cluster_id + 1} – Topik: {terms}\n\nContoh isi berita:\n"
        for i, doc in enumerate(docs[:max_examples]):
            excerpt = doc.strip().replace("\n", " ")[:200]
            prompt += f"{i+1}. \"{excerpt}...\"\n"

    prompt += ("""
---

Tolong lakukan hal berikut:
1. Rangkum masing-masing narasi secara netral.
2. Jelaskan perbedaan sudut pandang yang muncul dari dua kelompok tersebut.
3. Terakhir, simpulkan framing utama dari masing-masing cluster dalam satu kalimat pendek.
   Contohnya: “politik elektoral”, “kepercayaan publik terhadap hukum”, “ideologi partai”, “politik bansos”, dll.

Jaga agar format jawaban tetap seperti ini:
- Ringkasan
- Perbedaan sudut pandang
- Framing utama:
  - Cluster 1: [label framing]
  - Cluster 2: [label framing]
""")

    return prompt
