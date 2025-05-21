import sys
import os
import nltk 

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
nltk.download('punkt_tab')

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

