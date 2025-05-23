import faiss 
import numpy as np 
import os
import sys
import openai
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

class FAISSIndexer:
    def __init__(self,index_path):
        self.index_path = index_path
    
    def save_index(self,embeddings,chunks):
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        FAISS_PATH = os.path.join(self.index_path,'index.faiss')
        PKL_PATH = os.path.join(self.index_path,'chunks.pkl')

        os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
        faiss.write_index(index, FAISS_PATH)
        os.makedirs(os.path.dirname(PKL_PATH), exist_ok=True)
        with open(PKL_PATH, "wb") as f:
            pickle.dump(chunks, f)   

    def load_index(self):
        FAISS_PATH = os.path.join(self.index_path,'index.faiss')        
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(f"Index file not found: {FAISS_PATH}")
        index = faiss.read_index(FAISS_PATH)
        if not index:
            raise ValueError("Index file is empty.")        
        PKL_PATH = os.path.join(self.index_path,'chunks.pkl')
        if not os.path.exists(PKL_PATH):
            raise FileNotFoundError(f"pkl file not found: {PKL_PATH}")
        with open(PKL_PATH, "rb") as f:
            all_chunks = pickle.load(f)
            if not all_chunks:
                raise ValueError("pkl file is empty.")
        return index, all_chunks
    
    def search_index(self, query_emb, k):
        """
        the query must be previously embedded into query_emb.
        this function finds the k closest embeddings in the index 
        and returns the corresponding chunks as a context
        """
        index, chunks = self.load_index()
        D, I = index.search(np.array([query_emb]), k)
        context = "\n\n".join([chunks[i] for i in I[0]])
        return context
