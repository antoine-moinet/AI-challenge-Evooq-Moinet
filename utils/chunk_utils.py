import sys
import os
import nltk
import tiktoken  

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#nltk.download('punkt_tab')

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize 

def chunk_text(text, chunk_size, overlap):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = []
    length = 0
    for sentence in sentences:
        if length + len(sentence.split()) > chunk_size:
            chunks.append(" ".join(chunk))
            chunk = chunk[-overlap:]  # retain overlap
            length = len(" ".join(chunk).split())
        chunk.append(sentence)
        length += len(sentence.split())
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def batch_chunks(chunks,embedding_model,token_limit):
    enc = tiktoken.encoding_for_model(embedding_model)
    batches = []
    current_batch = []
    current_tokens = 0

    for chunk in chunks:
        if not isinstance(chunk, str) or not chunk.strip():
            continue
        tokens = len(enc.encode(chunk))
        if tokens > token_limit:
            continue  # skip overly long individual chunks. With default CHUNK_SIZE = 500 words and TOKEN_LIMIT = 8192 we're safe.
        if current_tokens + tokens > token_limit:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(chunk)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)
    return batches

