import sys
import os
import nltk
import tiktoken  

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize 

def chunk_text(text, chunk_size, overlap):
    """
    Extract sentences from the text (str) and returns a list of text chunks 
    with the provided chunk size (number of words) and overlap (number of sentences)
    """
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
    """
    Takes a list of chunks and returns a list of batches of chunks
    such that each batch does not exceed the token limit of the embedding model
    """
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

