import json
import os

# Load data
with open('data/berita.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Fungsi untuk membagi data ke dalam chunk
def split_json(data, chunk_sizes, output_dir="data/splitted"):
    os.makedirs(output_dir, exist_ok=True)
    start = 0
    for i, size in enumerate(chunk_sizes):
        end = start + size
        chunk = data[start:end]
        if chunk:
            out_path = os.path.join(output_dir, f'berita_{start+1}_{end}.json')
            with open(out_path, 'w', encoding='utf-8') as out_f:
                json.dump(chunk, out_f, ensure_ascii=False, indent=2)
            print(f"Saved: {out_path} ({len(chunk)} records)")
        start = end

# Contoh pemecahan: 10rb, 20rb, 20rb, 18rb (total 68k misal)
split_json(data, chunk_sizes=[10000, 20000, 20000, 18000])
