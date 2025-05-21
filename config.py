import os

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 3   # 3 sentences overlap
SIMILAR_CHUNKS = 5

# define absolute path for vector database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store", "index.faiss")