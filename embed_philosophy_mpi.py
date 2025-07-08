from mpi4py import MPI
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Config
MODEL_NAME = 'intfloat/e5-large-v2'  # or 'bge-large-en-v1.5'
INPUT_FILE = 'philosophy_data.csv'
TEXT_COLUMN = 'sentence_str'  # use original sentence
OUTPUT_DIR = 'embeddings_output'

# Ensure output directory exists
if rank == 0 and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
comm.Barrier()  # Sync all processes

# Load data
df = pd.read_csv(INPUT_FILE)
chunks = np.array_split(df, size)
local_df = chunks[rank].reset_index(drop=True)

# Add prefix for e5-style models
sentences = ['passage: ' + s for s in local_df[TEXT_COLUMN].tolist()]

# Load model
model = SentenceTransformer(MODEL_NAME)

# Encode
embeddings = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)

# Save embeddings and metadata
output_path = os.path.join(OUTPUT_DIR, f'embeddings_rank_{rank}.npz')
np.savez_compressed(
    output_path,
    embeddings=embeddings,
    indices=local_df.index.values,
    metadata=local_df[['title', 'author', 'school', 'sentence_str']].to_dict('records')
)

if rank == 0:
    print(f"Embedding completed with model {MODEL_NAME}. Outputs saved to {OUTPUT_DIR}/embeddings_rank_*.npz")

