import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import umap.umap_ as umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import defaultdict
import faiss
from sentence_transformers import SentenceTransformer
import time

# --------- MPI Setup ---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --------- Topics for Thematic Analysis ---------
TOPICS = [
    "query: What is justice?",
    "query: What is virtue?",
    "query: What is truth?",
    "query: What is wisdom?",
    "query: How do we deal with suffering?",
    "query: What is happiness?",
    "query: What is our duty?",
    "query: What does it mean to be free?",
    "query: What is compassion?",
    "query: What is evil?",
    "query: How should we face death?",
    "query: What is the meaning of life?",
    "query: How should we handle relationships?",
    "query: What is friendship?",
    "query: What is love?",
    "query: How do we find our identity?",
    "query: How should we approach mental health?",
    "query: What is the impact of technology on life?",
    "query: How does social media affect us?",
    "query: What should we do about climate change?",
    "query: How should we address inequality?",
    "query: How should we respond to racism?",
    "query: What is gender?",
    "query: What is the meaning of work?",
    "query: How do we find purpose?",
    "query: How do we deal with anxiety?",
    "query: How do we build self-esteem?",
    "query: How do we overcome addiction?",
    "query: How do we prevent violence?",
    "query: What is forgiveness?",
    "query: What is good leadership?",
    "query: What is community?",
    "query: What is privacy?"
]

# --------- Helper Functions ---------
def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    return data['embeddings'], data['metadata']

def group_by_school(metadata, embeddings):
    schools = defaultdict(list)
    for i, meta in enumerate(metadata):
        school = meta.get('school', 'Unknown')
        schools[school].append(i)
    return schools

def average_school_embeddings(schools, embeddings):
    school_vecs = {}
    for school, idxs in schools.items():
        school_vecs[school] = np.mean(embeddings[idxs], axis=0)
    return school_vecs

def split_work(items):
    # Split a list of items across MPI ranks
    n = len(items)
    chunk = n // size
    start = rank * chunk
    end = (rank + 1) * chunk if rank < size - 1 else n
    return items[start:end]

def save_pickle(obj, fname):
    if rank == 0:
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)

# --------- Load Data (on rank 0, then broadcast) ---------
if rank == 0:
    print("Loading embeddings...")
    emb_phil, meta_phil = load_embeddings('philosophy_embeddings_merged.npz')
    emb_reli, meta_reli = load_embeddings('religion_embeddings_merged.npz')
    emb_unified, meta_unified = load_embeddings('religion_philosophy_embeddings_merged.npz')
else:
    emb_phil = emb_reli = emb_unified = None
    meta_phil = meta_reli = meta_unified = None

# Broadcast data to all ranks
emb_phil = comm.bcast(emb_phil, root=0)
meta_phil = comm.bcast(meta_phil, root=0)
emb_reli = comm.bcast(emb_reli, root=0)
meta_reli = comm.bcast(meta_reli, root=0)
emb_unified = comm.bcast(emb_unified, root=0)
meta_unified = comm.bcast(meta_unified, root=0)

# --------- School-level Embedding Averages ---------
schools_phil = group_by_school(meta_phil, emb_phil)
schools_reli = group_by_school(meta_reli, emb_reli)
schools_unified = group_by_school(meta_unified, emb_unified)

school_vecs_phil = average_school_embeddings(schools_phil, emb_phil)
school_vecs_reli = average_school_embeddings(schools_reli, emb_reli)
school_vecs_unified = average_school_embeddings(schools_unified, emb_unified)

# Gather school names for each domain
school_names_phil = sorted(school_vecs_phil.keys())
school_names_reli = sorted(school_vecs_reli.keys())
school_names_unified = sorted(school_vecs_unified.keys())

# --------- Pairwise Similarity Matrices ---------
if rank == 0:
    print("Computing similarity matrices...")
    mat_phil = np.array([school_vecs_phil[k] for k in school_names_phil])
    mat_reli = np.array([school_vecs_reli[k] for k in school_names_reli])
    mat_unified = np.array([school_vecs_unified[k] for k in school_names_unified])

    sim_phil_vs_phil = cosine_similarity(mat_phil)
    sim_reli_vs_reli = cosine_similarity(mat_reli)
    sim_phil_vs_reli = cosine_similarity(mat_phil, mat_reli)
    sim_unified = cosine_similarity(mat_unified)
else:
    sim_phil_vs_phil = sim_reli_vs_reli = sim_phil_vs_reli = sim_unified = None

# --------- Hierarchical Clustering ---------
def do_clustering(mat, method='ward'):
    # Ward requires euclidean, others can use cosine
    if method == 'ward':
        from scipy.spatial.distance import pdist
        dist = pdist(mat, metric='euclidean')
    else:
        from scipy.spatial.distance import pdist
        dist = pdist(mat, metric='cosine')
    return linkage(dist, method=method)

if rank == 0:
    print("Clustering...")
    Z_phil_ward = do_clustering(mat_phil, 'ward')
    Z_reli_ward = do_clustering(mat_reli, 'ward')
    Z_unified_ward = do_clustering(mat_unified, 'ward')
    # You can add 'average', 'complete', etc. if desired

# --------- Thematic Analysis (Distributed) ---------
if rank == 0:
    print("Starting thematic analysis...")
model = SentenceTransformer("e5-large-v2")

my_topics = split_work(TOPICS)
thematic_results = {}

topic_times = []
for idx, topic in enumerate(my_topics):
    t_start = time.time()
    qvec = model.encode([topic], normalize_embeddings=True)
    results = {}
    for domain, emb, meta, schools, school_vecs in [
        ('philosophy', emb_phil, meta_phil, schools_phil, school_vecs_phil),
        ('religion', emb_reli, meta_reli, schools_reli, school_vecs_reli)
    ]:
        school_hits = {}
        for school, idxs in schools.items():
            sims_texts = []
            for i in idxs:
                sim = float(np.dot(qvec, emb[i].reshape(-1)).item() / (np.linalg.norm(qvec) * np.linalg.norm(emb[i])))
                sims_texts.append((sim, meta[i].get('sentence_str', '')))
            top_n = sorted(sims_texts, reverse=True)[:5]
            school_hits[school] = [{'similarity': sim, 'text': text} for sim, text in top_n]
        results[domain] = school_hits
    thematic_results[topic] = results
    t_end = time.time()
    topic_times.append(t_end - t_start)
    avg_time = np.mean(topic_times)
    remaining = (len(my_topics) - (idx + 1)) * avg_time
    print(f"[Rank {rank}] Finished topic {idx+1}/{len(my_topics)}: '{topic}' in {t_end - t_start:.2f}s. "
          f"Avg/topic: {avg_time:.2f}s. Est. remaining: {remaining/60:.1f} min.")

# Gather all thematic results at rank 0
all_thematic_results = comm.gather(thematic_results, root=0)

if rank == 0:
    thematic_results_merged = {}
    for d in all_thematic_results:
        thematic_results_merged.update(d)

# --------- Save All Results ---------
if rank == 0:
    print("Saving results...")
    save_pickle({
        'school_names_phil': school_names_phil,
        'school_names_reli': school_names_reli,
        'sim_phil_vs_phil': sim_phil_vs_phil,
        'sim_reli_vs_reli': sim_reli_vs_reli,
        'sim_phil_vs_reli': sim_phil_vs_reli,
        'Z_phil_ward': Z_phil_ward,
        'Z_reli_ward': Z_reli_ward,
        'Z_unified_ward': Z_unified_ward,
        'thematic_results': thematic_results_merged,
        'topics': TOPICS
    }, 'deep_analysis_results.pkl')
    print("Done.")

# --------- Visualization (to run later, not in MPI) ---------
# Use the saved pickle to plot heatmaps, dendrograms, UMAP, etc.