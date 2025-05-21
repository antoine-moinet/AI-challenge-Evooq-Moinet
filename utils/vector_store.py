import faiss 
import numpy as np 
import os
import sys
import openai 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import OPENAI_API_KEY, VECTOR_DB_PATH

openai.api_key = OPENAI_API_KEY

def get_embedding(text,emb_model):
    response = openai.embeddings.create(input=[text], model=emb_model)
    return np.array(response.data[0].embedding, dtype='float32')

def build_vector_store(chunks,emb_model):
    print('start embedding')
    embeddings = [get_embedding(chunk,emb_model) for chunk in chunks]
    print('embeddings done')
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    faiss.write_index(index, VECTOR_DB_PATH)
    return 

def load_vector_store():
    index = faiss.read_index(VECTOR_DB_PATH)
    return index


def search_index(index, query, emb_model, k):
    query_vec = get_embedding(query,emb_model)
    D, I = index.search(np.array([query_vec]), k)
    return I[0], D[0]
