from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# STEP 7: EMBEDDING + FAISS
# =========================

def build_faiss(dataset):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    texts = [item["noi_dung"] for item in dataset]
    
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


# =========================
# STEP 8: SAVE FAISS
# =========================

def save_faiss(index, path):
    faiss.write_index(index, path)
