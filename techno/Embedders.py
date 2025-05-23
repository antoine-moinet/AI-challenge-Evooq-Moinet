import numpy as np 
import os
import sys
import openai
import tiktoken 
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import OPENAI_API_KEY
from utils.chunk_utils import batch_chunks

openai.api_key = OPENAI_API_KEY


class OpenAIEmbedder:
    def __init__(self,embedding_model,token_limit):
        self.emb_mod = embedding_model
        self.token_limit = token_limit
        
    def embed_chunks(self,chunks):
        """
        Makes batches of text chunks that do not exceed token limit, 
        makes an embedding request for each batch
        and returns a list of embeddings for all chunks
        """
        batches = batch_chunks(chunks,self.tokenize,self.token_limit)
        all_embeddings = []
        for batch in tqdm(batches, desc="Embedding chunks"):
            response = openai.embeddings.create(input=batch, model=self.emb_mod)
            all_embeddings.extend([np.array(res.embedding, dtype='float32') for res in response.data])
        return all_embeddings

    
    def embed_text(self,text):
        """
        Returns an openai embedding for a single chunk of text
        """
        tokens = len(self.tokenize(text))
        if tokens > self.token_limit:
            raise ValueError("Text exceeds token limit.")
        response = openai.embeddings.create(input=[text], model=self.emb_mod)
        return np.array(response.data[0].embedding, dtype='float32')
    
    def tokenize(self,text):
        encoder = tiktoken.encoding_for_model(self.emb_mod)
        return encoder.encode(text)
    
    



