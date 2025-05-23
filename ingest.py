import os
import argparse
from config import EMBEDDING_MODEL, CHUNK_OVERLAP, CHUNK_SIZE, TOKEN_LIMIT
from utils.pdf_utils import extract_text_from_pdf
from utils.chunk_utils import chunk_text
from utils.vector_store import build_vector_store

def ingest_folder(folder_path, embedding_model, token_limit, chunk_size, chunk_overlap):
    """
    Reads and chunks the PDFs according to the overlap and chunk size provided,
    embeds the chunks according to the embedding model and token limit provided, 
    stores a vector index of the embeddings for later comparison with the query,
    stores a pickle file of the chunks to later retrieve top k similar chunks,
    and stores the embedding model and token limit in text files
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
    build_vector_store(all_chunks,embedding_model,token_limit) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_folder", required=True, help="Path to folder with PDFs")
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL, help="Embedding model to use")
    parser.add_argument("--token_limit", default=TOKEN_LIMIT, help="Token limit of the embedding model")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Chunk size in words")
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP, help="Overlap between chunks")
    args = parser.parse_args()
    ingest_folder(args.pdf_folder, args.embedding_model, args.token_limit, args.chunk_size, args.chunk_overlap)
    
    