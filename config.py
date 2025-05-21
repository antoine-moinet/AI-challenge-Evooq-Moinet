import os

OPENAI_API_KEY = "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"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SIMILAR_CHUNKS = 5

# define absolute path for vector database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store", "index.faiss")