# Eksplorasi Framing Berita Politik Indonesia Berbasis Contrastive Retrieval

## Latar Belakang Masalah
Pemberitaan politik daring di Indonesia sering menyajikan isu yang sama
dengan sudut pandang dan narasi yang berbeda. Perbedaan framing ini
dapat memengaruhi cara publik memahami suatu kebijakan atau peristiwa.

Namun, eksplorasi framing berita umumnya masih dilakukan secara manual
dan sulit diterapkan pada skala besar. Diperlukan pendekatan berbasis data
untuk membantu mengidentifikasi dan membandingkan narasi yang kontras
secara sistematis.

Proyek ini bertujuan untuk mengeksplorasi framing berita politik Indonesia
menggunakan pendekatan **contrastive retrieval**, sehingga perbedaan
narasi dapat dianalisis secara lebih terstruktur dan berbasis data.

---

## Data
- **Sumber:** Portal berita daring berbahasa Indonesia
- **Dataset:** [iqballx/indonesian_news_datasets](https://huggingface.co/datasets/iqballx/indonesian_news_datasets)
- **Unit analisis:** Artikel berita
- **Konten utama:** Judul dan isi berita

---

## Pendekatan Analisis

### 1. Pengambilan Dokumen
- Dokumen berita diambil berdasarkan **query pengguna**
  menggunakan metode **BM25** untuk memperoleh artikel yang relevan
  dengan isu tertentu.

### 2. Representasi dan Segmentasi Narasi
- Artikel direpresentasikan menggunakan **TF-IDF**
- Dokumen dikelompokkan ke dalam **dua klaster narasi**
  menggunakan **KMeans** untuk memisahkan sudut pandang yang kontras

### 3. Eksplorasi Framing
- Setiap klaster dianalisis untuk mengekstraksi kata kunci
  dan topik dominan
- Perbedaan fokus dan gaya narasi digunakan sebagai dasar
  identifikasi framing

### 4. Ringkasan Narasi
- Setiap klaster dirangkum secara terpisah menggunakan
  model bahasa (GPT-3.5)
- Hasil berupa **dua ringkasan narasi kontras**
  yang merepresentasikan framing berbeda terhadap isu yang sama

---

## Hasil Analisis
- Sistem mampu memisahkan berita ke dalam dua kelompok narasi
  yang merepresentasikan framing berbeda.
- Ringkasan yang dihasilkan membantu menyederhanakan
  perbedaan sudut pandang media terhadap suatu isu.
- Pendekatan ini memudahkan eksplorasi framing tanpa
  membaca seluruh artikel satu per satu.

---

## Insight Utama
- Isu politik yang sama dapat dibingkai secara kontras
  antara sudut pandang teknokratik dan perspektif publik.
- Pendekatan contrastive retrieval efektif untuk
  mengungkap perbedaan narasi dalam kumpulan berita besar.
- Ringkasan narasi membantu mempercepat pemahaman
  terhadap dinamika framing media.

---

## Rekomendasi
- **Analisis Media & Kebijakan:**  
  Metode ini dapat digunakan untuk memantau perbedaan framing
  media terhadap kebijakan publik secara cepat dan sistematis.
- **Pemantauan Opini Publik:**  
  Hasil eksplorasi framing dapat menjadi indikator awal
  pergeseran persepsi publik.
- **Pengembangan Analisis Lanjutan:**  
  Sistem dapat dikombinasikan dengan analisis sentimen
  atau analisis temporal untuk studi framing yang lebih mendalam.

---

## Contoh Penggunaan
**Query:**

**Output yang diharapkan:**
- Narasi 1: Perspektif kebijakan dan argumentasi pemerintah
- Narasi 2: Perspektif kritik dan keresahan masyarakat

---

## Tools & Teknologi
- Python
- BM25 Information Retrieval
- TF-IDF & KMeans
- Natural Language Processing (NLP)
- Integrasi LLM (GPT-3.5)
- Matplotlib, Seaborn

---

## Struktur Proyek
```
news_framing_analysis/
├── data/
│   └── berita.json
├── retriever/
│   └── bm25_retriever.py
├── clustering/
│   └── contrastive_kmeans.py
├── evaluation/
│   ├── ir_metrics.py
│   └── rouge_eval.py
├── llm/
│   ├── call_gpt.py
│   └── prompt_contrastive.py
├── utils/
│   ├── config_loader.py
│   ├── save_output.py
│   └── text_preprocessing.py
├── main.py
├── requirements.txt
└── README.md
```
---

## Penulis
Yayang Matira | Mahasiswa Magister Ilmu Komputer | Universitas Gadjah Mada
