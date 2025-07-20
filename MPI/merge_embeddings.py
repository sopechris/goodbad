import os
import numpy as np
import faiss

EMBEDDINGS_DIR = './embeddings_output'
OUTPUT_MERGED_FILE = 'philosophy_embeddings_merged.npz' # or religion_embeddings_merged.npz
FAISS_INDEX_FILE = 'philosophy_faiss.index' # or religion_faiss.index

# Gather all npz files
files = sorted([f for f in os.listdir(EMBEDDINGS_DIR) if f.startswith('embeddings_rank_') and f.endswith('.npz')])

all_embeddings = []
all_metadata = []

print(f"Loading {len(files)} embedding chunks...")

for file in files:
    data = np.load(os.path.join(EMBEDDINGS_DIR, file), allow_pickle=True)
    embeddings = data['embeddings']
    metadata = data['metadata']
    all_embeddings.append(embeddings)
    all_metadata.extend(metadata)

# Concatenate all embeddings (shape: total_sentences x embedding_dim)
all_embeddings = np.vstack(all_embeddings).astype('float32')

print(f"Total embeddings shape: {all_embeddings.shape}")
print(f"Total metadata items: {len(all_metadata)}")

# Save merged embeddings and metadata
np.savez_compressed(OUTPUT_MERGED_FILE, embeddings=all_embeddings, metadata=all_metadata)

print(f"Merged embeddings and metadata saved to {OUTPUT_MERGED_FILE}")

# Build FAISS index for similarity search
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # inner product = cosine similarity if vectors normalized

# Optional: normalize vectors for cosine similarity
faiss.normalize_L2(all_embeddings)

print("Adding embeddings to FAISS index...")
index.add(all_embeddings)

faiss.write_index(index, FAISS_INDEX_FILE)
print(f"FAISS index saved to {FAISS_INDEX_FILE}")

print("All done!")
