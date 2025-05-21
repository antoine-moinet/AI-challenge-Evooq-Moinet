import os
import argparse
import pickle
from config import EMBEDDING_MODEL,CHUNK_OVERLAP,CHUNK_SIZE
from utils.pdf_utils import extract_text_from_pdf
from utils.chunk_utils import chunk_text
from utils.vector_store import build_vector_store



all_chunks = []

def ingest_folder(folder_path, embedding_model, chunk_size, chunk_overlap):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"No PDF files found in folder: {folder_path}")
    
    for filename in pdf_files:
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            print(f"Extracting from {filename}...")
            text = extract_text_from_pdf(full_path)
            print('text extracted')
            chunks = chunk_text(text,chunk_size,chunk_overlap)
            print('text chunked')
            all_chunks.extend(chunks)
    build_vector_store(all_chunks,embedding_model)
    with open("vector_store/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_folder", required=True, help="Path to folder with PDFs")
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL, help="Embedding model to use")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Chunk size in words")
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP, help="Overlap between chunks")
    args = parser.parse_args()
    ingest_folder(args.pdf_folder, args.embedding_model, args.chunk_size, args.chunk_overlap)
    with open('embedding_model.txt', "w") as f:
        f.write(args.embedding_model)
    