import argparse
from config import EMBEDDING_MODEL, CHUNK_OVERLAP, CHUNK_SIZE, TOKEN_LIMIT, INDEX_PATH
from utils.chunk_utils import get_pdf_chunks   
from techno.Embedders import OpenAIEmbedder 
from techno.Indexers import FAISSIndexer
from user_model.user_defined_model import store_user_model
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_folder", required=True, help="Path to folder with PDFs")
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL, help="Embedding model to use")
    parser.add_argument("--token_limit", default=TOKEN_LIMIT, help="Token limit of the embedding model")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Chunk size in words")
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP, help="Overlap between chunks")
    args = parser.parse_args()

    embedder = OpenAIEmbedder(args.embedding_model,args.token_limit)
    store_user_model(args.embedding_model,args.token_limit)   
    all_chunks = get_pdf_chunks(args.pdf_folder,args.chunk_size,args.chunk_overlap)
    embeddings = embedder.embed_chunks(all_chunks)
     
    indexer = FAISSIndexer(INDEX_PATH)
    indexer.save_index(embeddings,all_chunks)
    
    