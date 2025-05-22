import faiss 
import numpy as np 
import os
import sys
import openai
import pickle
import tiktoken 
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import OPENAI_API_KEY, VECTOR_DB_PATH, VECTOR_DB_PATH2, USER_EMB_PATH, TOKEN_LIMIT

openai.api_key = OPENAI_API_KEY



def get_embeddings_batch(texts, embedding_model):
    enc = tiktoken.encoding_for_model(embedding_model)
    batches = []
    current_batch = []
    current_tokens = 0

    for t in texts:
        if not isinstance(t, str) or not t.strip():
            continue
        tokens = len(enc.encode(t))
        if tokens > TOKEN_LIMIT:
            continue  # skip overly long individual entries
        if current_tokens + tokens > TOKEN_LIMIT:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(t)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    all_embeddings = []
    for batch in tqdm(batches, desc="Embedding chunks"):
        response = openai.embeddings.create(input=batch, model=embedding_model)
        all_embeddings.extend([np.array(res.embedding, dtype='float32') for res in response.data])

    return all_embeddings

def build_vector_store(chunks,emb_model):
    embeddings = get_embeddings_batch(chunks,emb_model)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    faiss.write_index(index, VECTOR_DB_PATH)
    os.makedirs(os.path.dirname(VECTOR_DB_PATH2), exist_ok=True)
    with open(VECTOR_DB_PATH2, "wb") as f:
        pickle.dump(chunks, f)    
    with open(USER_EMB_PATH, "w") as f:
        f.write(emb_model)
    return 

def get_embedding(text,emb_model):
    response = openai.embeddings.create(input=[text], model=emb_model)
    return np.array(response.data[0].embedding, dtype='float32')

def load_vector_store():
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(f"Index file not found: {VECTOR_DB_PATH}")
    index = faiss.read_index(VECTOR_DB_PATH)
    if not index:
            raise ValueError("Index file is empty.")
    
    if not os.path.exists(VECTOR_DB_PATH2):
        raise FileNotFoundError(f"Compressed text file not found: {VECTOR_DB_PATH2}")
    with open(VECTOR_DB_PATH2, "rb") as f:
        all_chunks = pickle.load(f)
        if not all_chunks:
            raise ValueError("Compressed text file is empty.")
    return index, all_chunks

def get_stored_embedding_model():
    if not os.path.exists(USER_EMB_PATH):
        raise FileNotFoundError(f"User embedding model file not found: {USER_EMB_PATH}")
    with open(USER_EMB_PATH, "r") as f:
        model = f.read().strip()
        if not model:
            raise ValueError("User embedding model file is empty.")
        return model

def search_index(index, query, emb_model, k):
    query_vec = get_embedding(query,emb_model)
    D, I = index.search(np.array([query_vec]), k)
    top_distance = D[0][0]
    alpha = 0.5  # can be tuned
    relevance = float(np.clip(100 * np.exp(-alpha * top_distance), 0, 100))
    return I[0], D[0], relevance
