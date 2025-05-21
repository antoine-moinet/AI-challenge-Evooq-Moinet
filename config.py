import os

OPENAI_API_KEY = "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"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 3   # 3 sentences overlap
SIMILAR_CHUNKS = 5

# define absolute path for vector database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store", "index.faiss")