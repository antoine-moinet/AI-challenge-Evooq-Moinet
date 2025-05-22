import faiss 
import numpy as np 
import os
import sys
import openai
import pickle
import tiktoken 
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import OPENAI_API_KEY, VECTOR_DB_PATH, VECTOR_DB_PATH2, USER_EMB_PATH, USER_TOK_LIM_PATH
from utils.chunk_utils import batch_chunks

openai.api_key = OPENAI_API_KEY

def get_embeddings_batch(chunks,embedding_model,token_limit):
    """
    Makes batches of text chunks that do not exceed token limit, 
    makes an embedding request for each batch
    and returns a list of embeddings for all chunks

    Args:
        chunks (List[str]): a list of chunks extracted from all the PDFs in the folder
        emb_model (str): the embedding model
        token_limit (int): the maximum number of tokens allowed in a single embedding request
    """
    batches = batch_chunks(chunks,embedding_model,token_limit)
    all_embeddings = []
    for batch in tqdm(batches, desc="Embedding chunks"):
        response = openai.embeddings.create(input=batch, model=embedding_model)
        all_embeddings.extend([np.array(res.embedding, dtype='float32') for res in response.data])
    return all_embeddings

def build_vector_store(chunks,emb_model,token_limit):
    """
    Maps the chunks to a list of embeddings and stores a vector index and a pickle file for future retrieval

    Args:
        chunks (List[str]): a list of chunks extracted from all the PDFs in the folder
        emb_model (str): the embedding model
        token_limit (int): the maximum number of tokens allowed in a single embedding request
    """
    embeddings = get_embeddings_batch(chunks,emb_model,token_limit)
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

def get_embedding(text,emb_model,token_limit):
    """
    Returns an openai embedding for a single chunk of text
    """
    enc = tiktoken.encoding_for_model(emb_model)
    tokens = len(enc.encode(text))
    if tokens > token_limit:
        raise ValueError("Text exceeds token limit.")
    response = openai.embeddings.create(input=[text], model=emb_model)
    return np.array(response.data[0].embedding, dtype='float32')

def load_vector_store():
    """
    Loads and returns the previously saved index and list of text chunks
    """
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
    """
    Reads and returns the saved embedding model provided by the user at ingestion
    """
    if not os.path.exists(USER_EMB_PATH):
        raise FileNotFoundError(f"User embedding model file not found: {USER_EMB_PATH}")
    with open(USER_EMB_PATH, "r") as f:
        model = f.read().strip()
        if not model:
            raise ValueError("User embedding model file is empty.")
        return model
    
def get_stored_token_limit():
    """
    Reads and returns the saved embedding model token limit provided by the user at ingestion
    """
    if not os.path.exists(USER_TOK_LIM_PATH):
        raise FileNotFoundError(f"User token limit file not found: {USER_TOK_LIM_PATH}")
    with open(USER_TOK_LIM_PATH, "r") as f:
        model = f.read().strip()
        if not model:
            raise ValueError("User token limit file is empty.")
        return model

def search_index(index, query, emb_model, token_limit, k):
    """
    Embeds the query and returns the k (int) closest chunks of text in the index, the corresponding distances.
    Returns also the relevance: similarity between query and closest chunk on a 0-100% scale
    """
    query_vec = get_embedding(query,emb_model,token_limit)
    D, I = index.search(np.array([query_vec]), k)
    top_distance = D[0][0]
    alpha = 0.5  # can be tuned
    relevance = float(np.clip(100 * np.exp(-alpha * top_distance), 0, 100))
    return I[0], D[0], relevance
