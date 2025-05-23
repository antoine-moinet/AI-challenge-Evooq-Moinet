import os
import nltk
import fitz  

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize 

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF and returns a single string 
    """
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    return text

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

def get_pdf_chunks(folder_path,chunk_size,chunk_overlap):
    """
    extract and returns a list of chunks from all the PDFs contained in a folder
    """
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"No PDF files found in folder: {folder_path}")
    if len(pdf_files) > 100:
        raise ValueError("Number of PDF files exceeds limit")
    all_chunks = []
    for filename in pdf_files:
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            print(f"Extracting from {filename}...")
            text = extract_text_from_pdf(full_path)
            chunks = chunk_text(text,chunk_size,chunk_overlap) 
            all_chunks.extend(chunks)
    return all_chunks

def batch_chunks(chunks,tokenize,token_limit):
    """
    Takes a list of chunks, a tokenize method and a token limit from the embedder,
    and returns a list of batches of chunks such that each batch does not exceed 
    the token limit of the embedder
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for chunk in chunks:
        if not isinstance(chunk, str) or not chunk.strip():
            continue
        tokens = len(tokenize(chunk))
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

