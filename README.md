# Sistem Contrastive Retrieval untuk Eksplorasi Framing Berita

Prototipe penelitian ini dirancang untuk mengeksplorasi framing dalam berita daring berbahasa Indonesia menggunakan pendekatan contrastive information retrieval. Sistem ini mengambil dokumen relevan berdasarkan query pengguna, mengelompokkannya ke dalam dua kelompok narasi menggunakan KMeans, lalu merangkum masing-masing narasi menggunakan GPT-3.5.

## Ringkasan Sistem
Sistem ini dirancang untuk:
- Mengambil dokumen berita yang relevan menggunakan BM25
- Merepresentasikan dokumen dengan TF-IDF dan mengelompokkan ke dalam dua klaster narasi (KMeans)
- Mengekstraksi topik dominan dari masing-masing klaster
- Menyusun prompt dan mengirimkannya ke GPT-3.5 untuk menghasilkan ringkasan
- Menampilkan dua ringkasan narasi kontras yang mencerminkan framing yang berbeda

## Fitur
- Pengambilan dokumen berbasis BM25
- Representasi TF-IDF dan clustering dengan KMeans
- Ekstraksi topik otomatis
- Penyusunan prompt kontras
- Integrasi dengan GPT-3.5 melalui API OpenRouter

## Struktur Proyek
```
news_insight_project/
│
├── data/
│   └── berita.json                 # Data berita mentah
│
├── retriever/
│   └── bm25_retriever.py          # Modul pengambilan dokumen
│
├── clustering/
│   └── contrastive_kmeans.py      # Modul clustering dan ekstraksi topik
│ 
├── evaluation/
│   └── ir_metrics.py      # Modul clustering dan ekstraksi topik
│   └── rouge_eval.py
│
├── llm/
│   ├── call_gpt.py                # Pemanggilan API GPT
│   └── prompt_contrastive.py     # Penyusun prompt kontras
│
├── utils/
│   └── config_loader.py           # Pemuat konfigurasi sistem
│   └── save_output.py 
│   └── text_preprocessing.py 
│
├── main.py           # Pipeline utama
└── README.md
└── requirements.txt
```

## Cara Instalasi
1. Kloning repositori
2. Install dependensi:
```bash
pip install -r requirements.txt
```
3. Tambahkan API key di file `config.yaml`:
```yaml
openrouter:
  api_key: "masukkan_api_key_anda"
```

## Cara Menjalankan
```bash
python main_contrastive.py
```
Masukkan query saat diminta. Sistem akan menghasilkan dua ringkasan narasi dari dokumen yang terkelompok.

## Contoh Query
```
Apa reaksi publik terhadap kenaikan harga BBM?
```

Output yang diharapkan:
- Klaster 1: Narasi teknokratik dari sisi pemerintah
- Klaster 2: Narasi keresahan masyarakat umum

## Dataset
Dataset yang digunakan: [iqballx/indonesian_news_datasets](https://huggingface.co/datasets/iqballx/indonesian_news_datasets)

## Pengembang
- Yayang Matira  
- Magister Ilmu Komputer UGM  

## Lisensi
MIT License

## Pengembangan Selanjutnya
- Antarmuka interaktif untuk eksplorasi multi-query
- Penambahan filter emosi atau sentimen
- Visualisasi hasil clustering dengan PCA atau UMAP
